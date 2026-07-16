//! Private tensor lifecycle orchestration.
//!
//! This machine replaces the pointer-retaining lifecycle in
//! `emel.cpp/src/emel/model/tensor/` with actor-owned Rust storage. Guards own
//! every runtime validation and outcome choice; actions only execute the
//! behavior selected by the transition table.

#![allow(
    clippy::derive_partial_eq_without_eq,
    reason = "stateforward-sml generated state tokens intentionally derive PartialEq"
)]

use core::cell::{Cell, RefCell};
use core::mem;

use emel_io::{mmap, read, staged_read};
use sml::sml;

use super::actor::MAX_TENSORS;
use super::dependency::{Mapper, Reader, Stager, TensorDependencies};
use super::event::{
    ApplyBoundEffectResultsError, ApplyOwnedEffectResultsError, BindStorageDone, BindStorageError,
    BindTensorDone, EffectBuffer, EffectError, EffectRequest, Error, EvictTensorDone, Lifecycle,
    MappedLoad, MappedLoadDone, OwnedLoadDone, PlanLoadDone, PlanLoadError, ReadLoad,
    ReleaseMapped, ReleaseMappedDone, StagedLoad, StorageBatch, StrategyKind, TensorMetadata,
    TensorOperation, TensorState,
};

const MAX_READ_FILE_PATH_BYTES: usize = 4_095;

#[derive(Debug)]
struct Slot {
    lifecycle: Lifecycle,
    bytes: Option<Box<[u8]>>,
    mapping_owner_tensor_id: Option<i32>,
    mapping_handle: Option<u32>,
    buffer_bytes: u64,
    metadata: TensorMetadata,
}

impl Default for Slot {
    fn default() -> Self {
        Self {
            lifecycle: Lifecycle::Unbound,
            bytes: None,
            mapping_owner_tensor_id: None,
            mapping_handle: None,
            buffer_bytes: 0,
            metadata: TensorMetadata::default(),
        }
    }
}

pub(super) struct Context {
    slots: Vec<Slot>,
    active_extent: usize,
}

impl Context {
    pub(super) fn allocate(tensor_capacity: usize) -> Result<Self, Error> {
        if tensor_capacity == 0 || tensor_capacity > MAX_TENSORS {
            return Err(Error::Capacity);
        }
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(tensor_capacity)
            .map_err(|_| Error::Capacity)?;
        for _ in 0..tensor_capacity {
            slots.push(Slot::default());
        }
        Ok(Self {
            slots,
            active_extent: 0,
        })
    }

    fn tensor_id_valid(&self, tensor_id: i32) -> bool {
        usize::try_from(tensor_id).is_ok_and(|index| index < self.slots.len())
    }

    fn active_tensor_id_valid(&self, tensor_id: i32) -> bool {
        usize::try_from(tensor_id).is_ok_and(|index| index < self.active_extent)
    }

    fn slot(&self, tensor_id: i32) -> &Slot {
        &self.slots[usize::try_from(tensor_id).expect("tensor-id guard accepted the identifier")]
    }

    fn slot_mut(&mut self, tensor_id: i32) -> &mut Slot {
        &mut self.slots
            [usize::try_from(tensor_id).expect("tensor-id guard accepted the identifier")]
    }

    fn plan_input_valid(&self, event: &PlanLoadRuntime<'_>) -> bool {
        let result = event.result.borrow();
        let effects = &result
            .as_ref()
            .expect_err("plan wrapper starts with caller-owned effects")
            .effects;
        self.active_extent > 0
            && effects.0.len() >= self.active_extent
            && effects
                .0
                .iter()
                .all(|effect| *effect == EffectRequest::Empty)
    }

    fn guard_plan_strategy_valid(
        &self,
        event: &PlanLoadRuntime<'_>,
        strategy: StrategyKind,
    ) -> bool {
        event.strategy == strategy
            && self.plan_input_valid(event)
            && !self.mapping_cleanup_pending()
    }

    fn mapping_ownership_retained(&self) -> bool {
        self.slots.iter().any(|slot| slot.mapping_handle.is_some())
    }

    fn mapping_cleanup_pending(&self) -> bool {
        self.slots
            .iter()
            .any(|slot| slot.lifecycle == Lifecycle::MappedCleanupPending)
    }

    fn effect_plan<const STRATEGY: u8>(&self, event: PlanLoadRuntime<'_>) {
        let mut effects = take_effect_buffer(event.result);
        for (tensor_id, (effect, slot)) in effects
            .0
            .iter_mut()
            .zip(self.slots.iter())
            .take(self.active_extent)
            .enumerate()
        {
            *effect = planned_effect::<STRATEGY>(
                i32::try_from(tensor_id).expect("actor capacity fits i32"),
                slot,
            );
        }
        *event.result.borrow_mut() = Ok(PlanLoadDone::new(effects, self.active_extent));
    }
}

#[derive(Clone, Copy)]
pub(super) struct BindRuntime<'dispatch> {
    pub(super) tensor_id: i32,
    pub(super) metadata: TensorMetadata,
    pub(super) bytes: &'dispatch RefCell<Option<Box<[u8]>>>,
    pub(super) result: &'dispatch Cell<Result<BindTensorDone, Error>>,
}

#[derive(Clone, Copy)]
pub(super) struct EvictRuntime<'dispatch> {
    pub(super) tensor_id: i32,
    pub(super) result: &'dispatch RefCell<Result<EvictTensorDone, Error>>,
}

#[derive(Clone, Copy)]
pub(super) struct CaptureRuntime<'dispatch> {
    pub(super) tensor_id: i32,
    pub(super) result: &'dispatch Cell<Result<TensorState, Error>>,
}

#[derive(Clone, Copy)]
pub(super) struct BindStorageRuntime<'dispatch> {
    pub(super) result: &'dispatch RefCell<Result<BindStorageDone, BindStorageError>>,
}

#[derive(Clone, Copy)]
pub(super) struct PlanLoadRuntime<'dispatch> {
    pub(super) strategy: StrategyKind,
    pub(super) result: &'dispatch RefCell<Result<PlanLoadDone, PlanLoadError>>,
}

#[derive(Clone, Copy)]
pub(super) struct ApplyBoundRuntime<'dispatch> {
    pub(super) result: &'dispatch RefCell<Result<(), ApplyBoundEffectResultsError>>,
}

#[derive(Clone, Copy)]
pub(super) struct ApplyOwnedRuntime<'dispatch> {
    pub(super) result: &'dispatch RefCell<Result<(), ApplyOwnedEffectResultsError>>,
}

#[derive(Clone, Copy)]
pub(super) struct ApplyEffectErrorRuntime<'dispatch> {
    pub(super) tensor_id: i32,
    pub(super) error: EffectError,
    pub(super) result: &'dispatch Cell<Result<(), Error>>,
}

fn take_effect_buffer(result: &RefCell<Result<PlanLoadDone, PlanLoadError>>) -> EffectBuffer {
    let previous = mem::replace(
        &mut *result.borrow_mut(),
        Err(PlanLoadError::new(
            Error::Internal,
            EffectBuffer(Box::default()),
        )),
    );
    previous
        .expect_err("plan guard selected caller-owned effect buffer")
        .effects
}

const PLAN_NONE: u8 = 0;
const PLAN_MAPPED_FILE: u8 = 1;
const PLAN_READ_COPY: u8 = 2;
const PLAN_EXTERNAL_BUFFER: u8 = 3;
const PLAN_STAGED_READ: u8 = 4;

// `STRATEGY` is fixed by each explicit transition action. Runtime strategy
// selection remains in the guards; this helper only removes duplicate scans.
fn planned_effect<const STRATEGY: u8>(tensor_id: i32, slot: &Slot) -> EffectRequest {
    let file_index = slot.metadata.file_index();
    let offset = slot.metadata.file_offset();
    let size = slot.metadata.data_size();
    match STRATEGY {
        PLAN_NONE => EffectRequest::None {
            tensor_id,
            file_index,
            offset,
            size,
        },
        PLAN_MAPPED_FILE => EffectRequest::MappedFile {
            tensor_id,
            file_index,
            offset,
            size,
        },
        PLAN_READ_COPY => EffectRequest::ReadCopy {
            tensor_id,
            file_index,
            offset,
            size,
        },
        PLAN_EXTERNAL_BUFFER => EffectRequest::ExternalBuffer {
            tensor_id,
            file_index,
            offset,
            size,
        },
        PLAN_STAGED_READ => EffectRequest::StagedRead {
            tensor_id,
            file_index,
            offset,
            size,
        },
        _ => unreachable!("plan actions instantiate only supported strategy constants"),
    }
}

sml! {
    ModelTensor {
        // Bulk storage ownership transfer.
        "state_ready"_s <= *"state_ready"_s + BindStorage(BindStorageRuntime<'dispatch>) [guard_storage_bind_valid] / effect_bind_storage,
        "state_ready"_s <= "state_ready"_s + BindStorage(BindStorageRuntime<'dispatch>) [guard_storage_bind_invalid] / effect_bind_storage_invalid,
        "state_ready"_s <= "state_ready"_s + BindStorage(BindStorageRuntime<'dispatch>) [guard_storage_bind_mapped] / effect_bind_storage_invalid,

        // Strategy-explicit load planning.
        "state_awaiting_bound_results"_s <= "state_ready"_s + PlanLoad(PlanLoadRuntime<'dispatch>) [guard_plan_none_valid] / effect_plan_none,
        "state_awaiting_owned_results"_s <= "state_ready"_s + PlanLoad(PlanLoadRuntime<'dispatch>) [guard_plan_read_copy_valid] / effect_plan_read_copy,
        "state_awaiting_owned_results"_s <= "state_ready"_s + PlanLoad(PlanLoadRuntime<'dispatch>) [guard_plan_external_buffer_valid] / effect_plan_external_buffer,
        "state_awaiting_owned_results"_s <= "state_ready"_s + PlanLoad(PlanLoadRuntime<'dispatch>) [guard_plan_staged_read_valid] / effect_plan_staged_read,
        "state_awaiting_mapped_results"_s <= "state_ready"_s + PlanLoad(PlanLoadRuntime<'dispatch>) [guard_plan_mapped_file_valid] / effect_plan_mapped_file,
        "state_ready"_s <= "state_ready"_s + PlanLoad(PlanLoadRuntime<'dispatch>) [guard_plan_cleanup_pending] / effect_plan_cleanup_pending,
        "state_ready"_s <= "state_ready"_s + PlanLoad(PlanLoadRuntime<'dispatch>) [guard_plan_invalid_request] / effect_plan_invalid_request,
        "state_ready"_s <= "state_ready"_s + PlanLoad(PlanLoadRuntime<'dispatch>) [guard_plan_capacity] / effect_plan_capacity,
        "state_ready"_s <= "state_ready"_s + PlanLoad(PlanLoadRuntime<'dispatch>) [guard_plan_unsupported_strategy] / effect_plan_unsupported_strategy,

        // Complete-batch result validation and application.
        "state_ready"_s <= "state_awaiting_bound_results"_s + ApplyBound(ApplyBoundRuntime<'dispatch>) [guard_bound_results_mapping_owned] / effect_apply_bound_results_mapping_owned,
        "state_ready"_s <= "state_awaiting_bound_results"_s + ApplyBound(ApplyBoundRuntime<'dispatch>) [guard_bound_results_valid] / effect_apply_bound_results,
        "state_ready"_s <= "state_awaiting_bound_results"_s + ApplyBound(ApplyBoundRuntime<'dispatch>) [guard_bound_results_invalid] / effect_apply_bound_results_invalid,
        "state_ready"_s <= "state_awaiting_owned_results"_s + ApplyBound(ApplyBoundRuntime<'dispatch>) / effect_apply_bound_results_invalid,
        "state_ready"_s <= "state_awaiting_mapped_results"_s + ApplyBound(ApplyBoundRuntime<'dispatch>) / effect_apply_bound_results_invalid,
        "state_ready"_s <= "state_awaiting_owned_results"_s + ApplyOwned(ApplyOwnedRuntime<'dispatch>) [guard_owned_results_mapping_owned] / effect_apply_owned_results_mapping_owned,
        "state_ready"_s <= "state_awaiting_owned_results"_s + ApplyOwned(ApplyOwnedRuntime<'dispatch>) [guard_owned_results_valid] / effect_apply_owned_results,
        "state_ready"_s <= "state_awaiting_owned_results"_s + ApplyOwned(ApplyOwnedRuntime<'dispatch>) [guard_owned_results_invalid] / effect_apply_owned_results_invalid,
        "state_ready"_s <= "state_awaiting_bound_results"_s + ApplyOwned(ApplyOwnedRuntime<'dispatch>) / effect_apply_owned_results_invalid,
        "state_ready"_s <= "state_awaiting_mapped_results"_s + ApplyOwned(ApplyOwnedRuntime<'dispatch>) / effect_apply_owned_results_invalid,
        "state_ready"_s <= "state_awaiting_bound_results"_s + ApplyError(ApplyEffectErrorRuntime<'dispatch>) / effect_apply_backend_error,
        "state_ready"_s <= "state_awaiting_owned_results"_s + ApplyError(ApplyEffectErrorRuntime<'dispatch>) / effect_apply_backend_error,
        "state_ready"_s <= "state_awaiting_mapped_results"_s + ApplyError(ApplyEffectErrorRuntime<'dispatch>) / effect_apply_backend_error,
        "state_ready"_s <= "state_ready"_s + ApplyBound(ApplyBoundRuntime<'dispatch>) / effect_apply_bound_results_invalid,
        "state_ready"_s <= "state_ready"_s + ApplyOwned(ApplyOwnedRuntime<'dispatch>) / effect_apply_owned_results_invalid,
        "state_ready"_s <= "state_ready"_s + ApplyError(ApplyEffectErrorRuntime<'dispatch>) / effect_apply_error_invalid,

        // Awaiting states reject conflicting work without losing their phase.
        "state_awaiting_bound_results"_s <= "state_awaiting_bound_results"_s + BindStorage(BindStorageRuntime<'dispatch>) / effect_bind_storage_busy,
        "state_awaiting_bound_results"_s <= "state_awaiting_bound_results"_s + PlanLoad(PlanLoadRuntime<'dispatch>) / effect_plan_busy,
        "state_awaiting_bound_results"_s <= "state_awaiting_bound_results"_s + Bind(BindRuntime<'dispatch>) / effect_bind_busy,
        "state_awaiting_bound_results"_s <= "state_awaiting_bound_results"_s + Evict(EvictRuntime<'dispatch>) / effect_evict_busy,
        "state_awaiting_bound_results"_s <= "state_awaiting_bound_results"_s + Capture(CaptureRuntime<'dispatch>) / effect_capture_busy,
        "state_awaiting_owned_results"_s <= "state_awaiting_owned_results"_s + BindStorage(BindStorageRuntime<'dispatch>) / effect_bind_storage_busy,
        "state_awaiting_owned_results"_s <= "state_awaiting_owned_results"_s + PlanLoad(PlanLoadRuntime<'dispatch>) / effect_plan_busy,
        "state_awaiting_owned_results"_s <= "state_awaiting_owned_results"_s + Bind(BindRuntime<'dispatch>) / effect_bind_busy,
        "state_awaiting_owned_results"_s <= "state_awaiting_owned_results"_s + Evict(EvictRuntime<'dispatch>) / effect_evict_busy,
        "state_awaiting_owned_results"_s <= "state_awaiting_owned_results"_s + Capture(CaptureRuntime<'dispatch>) / effect_capture_busy,
        "state_awaiting_mapped_results"_s <= "state_awaiting_mapped_results"_s + BindStorage(BindStorageRuntime<'dispatch>) / effect_bind_storage_busy,
        "state_awaiting_mapped_results"_s <= "state_awaiting_mapped_results"_s + PlanLoad(PlanLoadRuntime<'dispatch>) / effect_plan_busy,
        "state_awaiting_mapped_results"_s <= "state_awaiting_mapped_results"_s + Bind(BindRuntime<'dispatch>) / effect_bind_busy,
        "state_awaiting_mapped_results"_s <= "state_awaiting_mapped_results"_s + Evict(EvictRuntime<'dispatch>) / effect_evict_busy,
        "state_awaiting_mapped_results"_s <= "state_awaiting_mapped_results"_s + Capture(CaptureRuntime<'dispatch>) / effect_capture_busy,

        // Bind validation and ownership transfer.
        "state_bind_id_decision"_s <= "state_ready"_s + Bind(BindRuntime<'dispatch>) / effect_begin_bind,
        "state_bind_residency_decision"_s <= "state_bind_id_decision"_s + completion<Bind>(BindRuntime<'dispatch>) [guard_tensor_id_valid_bind],
        "state_ready"_s <= "state_bind_id_decision"_s + completion<Bind>(BindRuntime<'dispatch>) [guard_tensor_id_invalid_bind] / effect_bind_invalid,
        "state_bind_payload_decision"_s <= "state_bind_residency_decision"_s + completion<Bind>(BindRuntime<'dispatch>) [guard_slot_bindable],
        "state_ready"_s <= "state_bind_residency_decision"_s + completion<Bind>(BindRuntime<'dispatch>) [guard_slot_resident] / effect_bind_already_resident,
        "state_ready"_s <= "state_bind_residency_decision"_s + completion<Bind>(BindRuntime<'dispatch>) [guard_slot_mapped] / effect_mapped_requires_release_bind,
        "state_ready"_s <= "state_bind_residency_decision"_s + completion<Bind>(BindRuntime<'dispatch>) [guard_slot_internal_bind] / effect_bind_internal,
        "state_ready"_s <= "state_bind_payload_decision"_s + completion<Bind>(BindRuntime<'dispatch>) [guard_bind_payload_valid] / effect_bind,
        "state_ready"_s <= "state_bind_payload_decision"_s + completion<Bind>(BindRuntime<'dispatch>) [guard_bind_payload_invalid] / effect_bind_invalid,

        // Eviction validation and ownership return.
        "state_evict_id_decision"_s <= "state_ready"_s + Evict(EvictRuntime<'dispatch>) / effect_begin_evict,
        "state_evict_residency_decision"_s <= "state_evict_id_decision"_s + completion<Evict>(EvictRuntime<'dispatch>) [guard_tensor_id_valid_evict],
        "state_ready"_s <= "state_evict_id_decision"_s + completion<Evict>(EvictRuntime<'dispatch>) [guard_tensor_id_invalid_evict] / effect_evict_invalid,
        "state_ready"_s <= "state_evict_residency_decision"_s + completion<Evict>(EvictRuntime<'dispatch>) [guard_evict_resident] / effect_evict,
        "state_ready"_s <= "state_evict_residency_decision"_s + completion<Evict>(EvictRuntime<'dispatch>) [guard_evict_unbound] / effect_evict_unbound,
        "state_ready"_s <= "state_evict_residency_decision"_s + completion<Evict>(EvictRuntime<'dispatch>) [guard_evict_mapped] / effect_mapped_requires_release_evict,
        "state_ready"_s <= "state_evict_residency_decision"_s + completion<Evict>(EvictRuntime<'dispatch>) [guard_evict_internal] / effect_evict_internal,

        // Metadata-only state capture.
        "state_capture_id_decision"_s <= "state_ready"_s + Capture(CaptureRuntime<'dispatch>) / effect_begin_capture,
        "state_ready"_s <= "state_capture_id_decision"_s + completion<Capture>(CaptureRuntime<'dispatch>) [guard_tensor_id_valid_capture] / effect_capture,
        "state_ready"_s <= "state_capture_id_decision"_s + completion<Capture>(CaptureRuntime<'dispatch>) [guard_tensor_id_invalid_capture] / effect_capture_invalid,

        // Unexpected events fail closed instead of being silently discarded.
        "state_ready"_s <= "state_ready"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_bind_id_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_bind_residency_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_bind_payload_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_evict_id_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_evict_residency_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_capture_id_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_awaiting_bound_results"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_awaiting_owned_results"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_awaiting_mapped_results"_s + unexpected_event<_> / effect_unexpected,
    }
}

impl ModelTensorStateMachineContext for Context {
    fn guard_storage_bind_valid(&self, event: &BindStorageRuntime<'_>) -> Result<bool, ()> {
        let result = event.result.borrow();
        let storage = &result
            .as_ref()
            .expect_err("bind wrapper starts with caller-owned storage")
            .storage;
        Ok(!storage.0.is_empty()
            && storage.0.len() <= self.slots.len()
            && !self.slots.iter().any(|slot| {
                matches!(
                    slot.lifecycle,
                    Lifecycle::MappedResident | Lifecycle::MappedCleanupPending
                )
            }))
    }

    fn guard_storage_bind_invalid(&self, event: &BindStorageRuntime<'_>) -> Result<bool, ()> {
        let result = event.result.borrow();
        let storage = &result
            .as_ref()
            .expect_err("bind wrapper starts with caller-owned storage")
            .storage;
        Ok(storage.0.is_empty() || storage.0.len() > self.slots.len())
    }

    fn guard_storage_bind_mapped(&self, event: &BindStorageRuntime<'_>) -> Result<bool, ()> {
        let result = event.result.borrow();
        let storage = &result
            .as_ref()
            .expect_err("bind wrapper starts with caller-owned storage")
            .storage;
        Ok(!storage.0.is_empty()
            && storage.0.len() <= self.slots.len()
            && self.slots.iter().any(|slot| {
                matches!(
                    slot.lifecycle,
                    Lifecycle::MappedResident | Lifecycle::MappedCleanupPending
                )
            }))
    }

    fn effect_bind_storage(&mut self, event: BindStorageRuntime<'_>) -> Result<(), ()> {
        let previous = mem::replace(
            &mut *event.result.borrow_mut(),
            Err(BindStorageError::new(
                Error::Internal,
                StorageBatch(Box::default()),
            )),
        );
        let error = previous.expect_err("storage guard selected caller-owned batch");
        let StorageBatch(mut entries) = error.storage;
        let active_extent = entries.len();

        for slot in &mut self.slots {
            *slot = Slot::default();
        }
        for (slot, entry) in self.slots.iter_mut().zip(entries.iter_mut()) {
            slot.metadata = entry.metadata;
            slot.bytes = entry.bytes.take();
        }
        self.active_extent = active_extent;
        *event.result.borrow_mut() = Ok(BindStorageDone::new(active_extent));
        Ok(())
    }

    fn effect_bind_storage_invalid(&mut self, event: BindStorageRuntime<'_>) -> Result<(), ()> {
        event
            .result
            .borrow_mut()
            .as_mut()
            .expect_err("bind rejection retains caller-owned storage")
            .error = Error::InvalidRequest;
        Ok(())
    }

    fn guard_plan_none_valid(&self, event: &PlanLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(self.guard_plan_strategy_valid(event, StrategyKind::None))
    }

    fn guard_plan_read_copy_valid(&self, event: &PlanLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(self.guard_plan_strategy_valid(event, StrategyKind::ReadCopy))
    }

    fn guard_plan_external_buffer_valid(&self, event: &PlanLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(self.guard_plan_strategy_valid(event, StrategyKind::ExternalBuffer))
    }

    fn guard_plan_staged_read_valid(&self, event: &PlanLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(self.guard_plan_strategy_valid(event, StrategyKind::StagedRead))
    }

    fn guard_plan_mapped_file_valid(&self, event: &PlanLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(self.guard_plan_strategy_valid(event, StrategyKind::MappedFile))
    }

    fn guard_plan_invalid_request(&self, event: &PlanLoadRuntime<'_>) -> Result<bool, ()> {
        let result = event.result.borrow();
        let effects = &result
            .as_ref()
            .expect_err("plan wrapper starts with caller-owned effects")
            .effects;
        Ok(self.active_extent == 0
            || (effects.0.len() >= self.active_extent
                && effects
                    .0
                    .iter()
                    .any(|effect| *effect != EffectRequest::Empty)))
    }

    fn guard_plan_capacity(&self, event: &PlanLoadRuntime<'_>) -> Result<bool, ()> {
        let result = event.result.borrow();
        let effects = &result
            .as_ref()
            .expect_err("plan wrapper starts with caller-owned effects")
            .effects;
        Ok(self.active_extent > 0 && effects.0.len() < self.active_extent)
    }

    fn guard_plan_unsupported_strategy(&self, event: &PlanLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(self.plan_input_valid(event)
            && !self.mapping_cleanup_pending()
            && matches!(event.strategy, StrategyKind::Unknown(_)))
    }

    fn guard_plan_cleanup_pending(&self, event: &PlanLoadRuntime<'_>) -> Result<bool, ()> {
        Ok(self.plan_input_valid(event) && self.mapping_cleanup_pending())
    }

    fn effect_plan_none(&mut self, event: PlanLoadRuntime<'_>) -> Result<(), ()> {
        self.effect_plan::<PLAN_NONE>(event);
        Ok(())
    }

    fn effect_plan_read_copy(&mut self, event: PlanLoadRuntime<'_>) -> Result<(), ()> {
        self.effect_plan::<PLAN_READ_COPY>(event);
        Ok(())
    }

    fn effect_plan_external_buffer(&mut self, event: PlanLoadRuntime<'_>) -> Result<(), ()> {
        self.effect_plan::<PLAN_EXTERNAL_BUFFER>(event);
        Ok(())
    }

    fn effect_plan_staged_read(&mut self, event: PlanLoadRuntime<'_>) -> Result<(), ()> {
        self.effect_plan::<PLAN_STAGED_READ>(event);
        Ok(())
    }

    fn effect_plan_mapped_file(&mut self, event: PlanLoadRuntime<'_>) -> Result<(), ()> {
        self.effect_plan::<PLAN_MAPPED_FILE>(event);
        Ok(())
    }

    fn effect_plan_cleanup_pending(&mut self, event: PlanLoadRuntime<'_>) -> Result<(), ()> {
        event
            .result
            .borrow_mut()
            .as_mut()
            .expect_err("mapped-ownership rejection retains caller-owned effects")
            .error = Error::MappedTensorRequiresRelease;
        Ok(())
    }

    fn effect_plan_invalid_request(&mut self, event: PlanLoadRuntime<'_>) -> Result<(), ()> {
        event
            .result
            .borrow_mut()
            .as_mut()
            .expect_err("invalid plan retains caller-owned effects")
            .error = Error::InvalidRequest;
        Ok(())
    }

    fn effect_plan_capacity(&mut self, event: PlanLoadRuntime<'_>) -> Result<(), ()> {
        event
            .result
            .borrow_mut()
            .as_mut()
            .expect_err("capacity failure retains caller-owned effects")
            .error = Error::Capacity;
        Ok(())
    }

    fn effect_plan_unsupported_strategy(&mut self, event: PlanLoadRuntime<'_>) -> Result<(), ()> {
        event
            .result
            .borrow_mut()
            .as_mut()
            .expect_err("unsupported plan retains caller-owned effects")
            .error = Error::UnsupportedStrategy;
        Ok(())
    }

    fn guard_bound_results_valid(&self, event: &ApplyBoundRuntime<'_>) -> Result<bool, ()> {
        let result = event.result.borrow();
        let tensor_ids = &result
            .as_ref()
            .expect_err("bound-result wrapper starts with caller-owned ids")
            .tensor_ids;
        Ok(!self.mapping_ownership_retained()
            && tensor_ids.len() == self.active_extent
            && tensor_ids.iter().enumerate().all(|(index, tensor_id)| {
                *tensor_id == i32::try_from(index).expect("actor capacity fits i32")
                    && self.slots[index].bytes.as_ref().is_some_and(|bytes| {
                        u64::try_from(bytes.len())
                            .is_ok_and(|length| length >= self.slots[index].metadata.data_size())
                    })
            }))
    }

    fn guard_bound_results_invalid(&self, event: &ApplyBoundRuntime<'_>) -> Result<bool, ()> {
        let result = event.result.borrow();
        let tensor_ids = &result
            .as_ref()
            .expect_err("bound-result wrapper starts with caller-owned ids")
            .tensor_ids;
        Ok(!self.mapping_ownership_retained()
            && (tensor_ids.len() != self.active_extent
                || tensor_ids.iter().enumerate().any(|(index, tensor_id)| {
                    *tensor_id != i32::try_from(index).expect("actor capacity fits i32")
                        || !self.slots[index].bytes.as_ref().is_some_and(|bytes| {
                            u64::try_from(bytes.len()).is_ok_and(|length| {
                                length >= self.slots[index].metadata.data_size()
                            })
                        })
                })))
    }

    fn guard_bound_results_mapping_owned(&self, _: &ApplyBoundRuntime<'_>) -> Result<bool, ()> {
        Ok(self.mapping_ownership_retained())
    }

    fn effect_apply_bound_results(&mut self, event: ApplyBoundRuntime<'_>) -> Result<(), ()> {
        for slot in self.slots.iter_mut().take(self.active_extent) {
            slot.lifecycle = Lifecycle::Resident;
            slot.buffer_bytes = slot.metadata.data_size();
        }
        *event.result.borrow_mut() = Ok(());
        Ok(())
    }

    fn effect_apply_bound_results_invalid(
        &mut self,
        event: ApplyBoundRuntime<'_>,
    ) -> Result<(), ()> {
        event
            .result
            .borrow_mut()
            .as_mut()
            .expect_err("invalid bound results retain caller-owned ids")
            .error = Error::InvalidRequest;
        Ok(())
    }

    fn effect_apply_bound_results_mapping_owned(
        &mut self,
        event: ApplyBoundRuntime<'_>,
    ) -> Result<(), ()> {
        event
            .result
            .borrow_mut()
            .as_mut()
            .expect_err("mapped-ownership rejection retains caller-owned ids")
            .error = Error::MappedTensorRequiresRelease;
        Ok(())
    }

    fn guard_owned_results_valid(&self, event: &ApplyOwnedRuntime<'_>) -> Result<bool, ()> {
        let result = event.result.borrow();
        let results = &result
            .as_ref()
            .expect_err("owned-result wrapper starts with caller-owned results")
            .results;
        Ok(!self.mapping_ownership_retained()
            && results.len() == self.active_extent
            && results.iter().enumerate().all(|(index, result)| {
                result.tensor_id == i32::try_from(index).expect("actor capacity fits i32")
                    && u64::try_from(result.bytes.len())
                        .is_ok_and(|length| length >= self.slots[index].metadata.data_size())
            }))
    }

    fn guard_owned_results_invalid(&self, event: &ApplyOwnedRuntime<'_>) -> Result<bool, ()> {
        let result = event.result.borrow();
        let results = &result
            .as_ref()
            .expect_err("owned-result wrapper starts with caller-owned results")
            .results;
        Ok(!self.mapping_ownership_retained()
            && (results.len() != self.active_extent
                || results.iter().enumerate().any(|(index, result)| {
                    result.tensor_id != i32::try_from(index).expect("actor capacity fits i32")
                        || !u64::try_from(result.bytes.len())
                            .is_ok_and(|length| length >= self.slots[index].metadata.data_size())
                })))
    }

    fn guard_owned_results_mapping_owned(&self, _: &ApplyOwnedRuntime<'_>) -> Result<bool, ()> {
        Ok(self.mapping_ownership_retained())
    }

    fn effect_apply_owned_results(&mut self, event: ApplyOwnedRuntime<'_>) -> Result<(), ()> {
        let mut outcome = event.result.borrow_mut();
        let results = &mut outcome
            .as_mut()
            .expect_err("owned-result guard selected caller-owned results")
            .results;
        for (slot, result) in self
            .slots
            .iter_mut()
            .zip(results.iter_mut())
            .take(self.active_extent)
        {
            slot.lifecycle = Lifecycle::Resident;
            slot.buffer_bytes = slot.metadata.data_size();
            slot.bytes = Some(mem::take(&mut result.bytes));
        }
        *outcome = Ok(());
        Ok(())
    }

    fn effect_apply_owned_results_invalid(
        &mut self,
        event: ApplyOwnedRuntime<'_>,
    ) -> Result<(), ()> {
        event
            .result
            .borrow_mut()
            .as_mut()
            .expect_err("invalid owned results retain caller-owned buffers")
            .error = Error::InvalidRequest;
        Ok(())
    }

    fn effect_apply_owned_results_mapping_owned(
        &mut self,
        event: ApplyOwnedRuntime<'_>,
    ) -> Result<(), ()> {
        event
            .result
            .borrow_mut()
            .as_mut()
            .expect_err("mapped-ownership rejection retains caller-owned results")
            .error = Error::MappedTensorRequiresRelease;
        Ok(())
    }

    fn effect_apply_backend_error(&mut self, event: ApplyEffectErrorRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::BackendError));
        Ok(())
    }

    fn effect_apply_error_invalid(&mut self, event: ApplyEffectErrorRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::InvalidRequest));
        Ok(())
    }

    fn effect_bind_storage_busy(&mut self, event: BindStorageRuntime<'_>) -> Result<(), ()> {
        event
            .result
            .borrow_mut()
            .as_mut()
            .expect_err("busy bind retains caller-owned storage")
            .error = Error::Busy;
        Ok(())
    }

    fn effect_plan_busy(&mut self, event: PlanLoadRuntime<'_>) -> Result<(), ()> {
        event
            .result
            .borrow_mut()
            .as_mut()
            .expect_err("busy plan retains caller-owned effects")
            .error = Error::Busy;
        Ok(())
    }

    fn effect_bind_busy(&mut self, event: BindRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::Busy));
        Ok(())
    }

    fn effect_evict_busy(&mut self, event: EvictRuntime<'_>) -> Result<(), ()> {
        let _previous = event.result.replace(Err(Error::Busy));
        Ok(())
    }

    fn effect_capture_busy(&mut self, event: CaptureRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::Busy));
        Ok(())
    }

    fn effect_begin_bind(&mut self, event: BindRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::Internal));
        Ok(())
    }

    fn guard_tensor_id_valid_bind(&self, event: &BindRuntime<'_>) -> Result<bool, ()> {
        Ok(self.tensor_id_valid(event.tensor_id))
    }

    fn guard_tensor_id_invalid_bind(&self, event: &BindRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.tensor_id_valid(event.tensor_id))
    }

    fn guard_slot_bindable(&self, event: &BindRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(
            self.slot(event.tensor_id).lifecycle,
            Lifecycle::Unbound | Lifecycle::Evicted
        ))
    }

    fn guard_slot_resident(&self, event: &BindRuntime<'_>) -> Result<bool, ()> {
        Ok(self.slot(event.tensor_id).lifecycle == Lifecycle::Resident)
    }

    fn guard_slot_mapped(&self, event: &BindRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(
            self.slot(event.tensor_id).lifecycle,
            Lifecycle::MappedResident | Lifecycle::MappedCleanupPending
        ))
    }

    fn guard_slot_internal_bind(&self, event: &BindRuntime<'_>) -> Result<bool, ()> {
        Ok(self.slot(event.tensor_id).lifecycle == Lifecycle::InternalError)
    }

    fn guard_bind_payload_valid(&self, event: &BindRuntime<'_>) -> Result<bool, ()> {
        Ok(event.metadata.data_size() > 0
            && event.bytes.borrow().as_ref().is_some_and(|bytes| {
                !bytes.is_empty()
                    && u64::try_from(bytes.len())
                        .is_ok_and(|length| length >= event.metadata.data_size())
            }))
    }

    fn guard_bind_payload_invalid(&self, event: &BindRuntime<'_>) -> Result<bool, ()> {
        Ok(!(event.metadata.data_size() > 0
            && event.bytes.borrow().as_ref().is_some_and(|bytes| {
                !bytes.is_empty()
                    && u64::try_from(bytes.len())
                        .is_ok_and(|length| length >= event.metadata.data_size())
            })))
    }

    fn effect_bind(&mut self, event: BindRuntime<'_>) -> Result<(), ()> {
        let bytes = event
            .bytes
            .borrow_mut()
            .take()
            .expect("payload guard selected actor-owned bytes");
        let buffer_bytes =
            u64::try_from(bytes.len()).expect("payload guard selected a supported byte length");
        let slot = self.slot_mut(event.tensor_id);
        slot.lifecycle = Lifecycle::Resident;
        slot.buffer_bytes = buffer_bytes;
        slot.metadata = event.metadata;
        slot.bytes = Some(bytes);
        event
            .result
            .set(Ok(BindTensorDone::new(event.tensor_id, buffer_bytes)));
        Ok(())
    }

    fn effect_bind_invalid(&mut self, event: BindRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::InvalidRequest));
        Ok(())
    }

    fn effect_bind_already_resident(&mut self, event: BindRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::TensorAlreadyResident));
        Ok(())
    }

    fn effect_mapped_requires_release_bind(&mut self, event: BindRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::MappedTensorRequiresRelease));
        Ok(())
    }

    fn effect_bind_internal(&mut self, event: BindRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::Internal));
        Ok(())
    }

    fn guard_tensor_id_valid_evict(&self, event: &EvictRuntime<'_>) -> Result<bool, ()> {
        Ok(self.tensor_id_valid(event.tensor_id))
    }

    fn effect_begin_evict(&mut self, event: EvictRuntime<'_>) -> Result<(), ()> {
        let _previous = event.result.replace(Err(Error::Internal));
        Ok(())
    }

    fn guard_tensor_id_invalid_evict(&self, event: &EvictRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.tensor_id_valid(event.tensor_id))
    }

    fn guard_evict_resident(&self, event: &EvictRuntime<'_>) -> Result<bool, ()> {
        Ok(self.slot(event.tensor_id).lifecycle == Lifecycle::Resident)
    }

    fn guard_evict_unbound(&self, event: &EvictRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(
            self.slot(event.tensor_id).lifecycle,
            Lifecycle::Unbound | Lifecycle::Evicted
        ))
    }

    fn guard_evict_mapped(&self, event: &EvictRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(
            self.slot(event.tensor_id).lifecycle,
            Lifecycle::MappedResident | Lifecycle::MappedCleanupPending
        ))
    }

    fn guard_evict_internal(&self, event: &EvictRuntime<'_>) -> Result<bool, ()> {
        Ok(self.slot(event.tensor_id).lifecycle == Lifecycle::InternalError)
    }

    fn effect_evict(&mut self, event: EvictRuntime<'_>) -> Result<(), ()> {
        let slot = self.slot_mut(event.tensor_id);
        let bytes = slot
            .bytes
            .take()
            .expect("resident guard selected actor-owned bytes");
        slot.lifecycle = Lifecycle::Evicted;
        slot.buffer_bytes = 0;
        let _previous = event
            .result
            .replace(Ok(EvictTensorDone::new(event.tensor_id, bytes)));
        Ok(())
    }

    fn effect_evict_invalid(&mut self, event: EvictRuntime<'_>) -> Result<(), ()> {
        let _previous = event.result.replace(Err(Error::InvalidRequest));
        Ok(())
    }

    fn effect_evict_unbound(&mut self, event: EvictRuntime<'_>) -> Result<(), ()> {
        let _previous = event.result.replace(Err(Error::TensorUnbound));
        Ok(())
    }

    fn effect_mapped_requires_release_evict(&mut self, event: EvictRuntime<'_>) -> Result<(), ()> {
        let _previous = event
            .result
            .replace(Err(Error::MappedTensorRequiresRelease));
        Ok(())
    }

    fn effect_evict_internal(&mut self, event: EvictRuntime<'_>) -> Result<(), ()> {
        let _previous = event.result.replace(Err(Error::Internal));
        Ok(())
    }

    fn guard_tensor_id_valid_capture(&self, event: &CaptureRuntime<'_>) -> Result<bool, ()> {
        Ok(self.tensor_id_valid(event.tensor_id))
    }

    fn effect_begin_capture(&mut self, event: CaptureRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::Internal));
        Ok(())
    }

    fn guard_tensor_id_invalid_capture(&self, event: &CaptureRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.tensor_id_valid(event.tensor_id))
    }

    fn effect_capture(&mut self, event: CaptureRuntime<'_>) -> Result<(), ()> {
        let slot = self.slot(event.tensor_id);
        event.result.set(Ok(TensorState::new(
            slot.lifecycle,
            slot.buffer_bytes,
            slot.metadata,
        )));
        Ok(())
    }

    fn effect_capture_invalid(&mut self, event: CaptureRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::InvalidRequest));
        Ok(())
    }

    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}

#[derive(Clone, Copy)]
pub(super) struct MappedRuntime<'dispatch, 'source> {
    request: &'source MappedLoad,
    owner_ready: bool,
    child: &'dispatch Cell<Option<Result<mmap::event::MapDone, mmap::event::Error>>>,
    cleanup: &'dispatch Cell<Option<Result<(), mmap::event::Error>>>,
    result: &'dispatch Cell<Result<MappedLoadDone, Error>>,
}

#[derive(Clone, Copy)]
pub(super) struct ReadRuntime<'dispatch, 'source> {
    request: ReadLoad<'source>,
    owner_ready: bool,
    child: &'dispatch Cell<Option<Result<read::event::ReadTensorDone, read::event::Error>>>,
    result: &'dispatch Cell<Result<OwnedLoadDone, Error>>,
}

#[derive(Clone, Copy)]
pub(super) struct StagedRuntime<'dispatch, 'source> {
    request: StagedLoad<'source>,
    owner_ready: bool,
    child: &'dispatch Cell<
        Option<Result<staged_read::event::StageWindowDone, staged_read::event::Error>>,
    >,
    result: &'dispatch Cell<Result<OwnedLoadDone, Error>>,
}

#[derive(Clone, Copy)]
pub(super) struct ReleaseRuntime<'dispatch> {
    request: ReleaseMapped,
    owner_ready: bool,
    child: &'dispatch Cell<Option<Result<(), mmap::event::Error>>>,
    result: &'dispatch Cell<Result<ReleaseMappedDone, Error>>,
}

struct IoContext<'actor, Dependencies> {
    tensor: &'actor mut Context,
    dependencies: &'actor mut Dependencies,
}

sml! {
    ModelTensorIo {
        // Caller-created mapping capability route.
        "state_mapped_owner_decision"_s <= *"state_ready"_s + Mapped(MappedRuntime<'dispatch, 'source>),
        X <= "state_mapped_owner_decision"_s + completion<Mapped>(MappedRuntime<'dispatch, 'source>) [guard_mapped_owner_busy] / effect_mapped_busy,
        "state_mapped_capability_decision"_s <= "state_mapped_owner_decision"_s + completion<Mapped>(MappedRuntime<'dispatch, 'source>) [guard_mapped_owner_ready],
        X <= "state_mapped_capability_decision"_s + completion<Mapped>(MappedRuntime<'dispatch, 'source>) [guard_mapper_absent] / effect_mapped_unavailable,
        "state_mapped_request_decision"_s <= "state_mapped_capability_decision"_s + completion<Mapped>(MappedRuntime<'dispatch, 'source>) [guard_mapper_present],
        X <= "state_mapped_request_decision"_s + completion<Mapped>(MappedRuntime<'dispatch, 'source>) [guard_mapped_request_invalid] / effect_invalid_mapped,
        "state_mapped_residency_decision"_s <= "state_mapped_request_decision"_s + completion<Mapped>(MappedRuntime<'dispatch, 'source>) [guard_mapped_request_valid],
        X <= "state_mapped_residency_decision"_s + completion<Mapped>(MappedRuntime<'dispatch, 'source>) [guard_mapped_already_resident] / effect_mapped_already_resident,
        X <= "state_mapped_residency_decision"_s + completion<Mapped>(MappedRuntime<'dispatch, 'source>) [guard_mapped_internal] / effect_mapped_internal,
        "state_mapped_child_decision"_s <= "state_mapped_residency_decision"_s + completion<Mapped>(MappedRuntime<'dispatch, 'source>) [guard_mapped_loadable] / effect_dispatch_mapped,
        X <= "state_mapped_child_decision"_s + completion<Mapped>(MappedRuntime<'dispatch, 'source>) [guard_mapped_child_done] / effect_commit_mapped,
        "state_mapped_cleanup_decision"_s <= "state_mapped_child_decision"_s + completion<Mapped>(MappedRuntime<'dispatch, 'source>) [guard_mapped_child_malformed] / effect_release_malformed_mapping,
        X <= "state_mapped_cleanup_decision"_s + completion<Mapped>(MappedRuntime<'dispatch, 'source>) [guard_mapped_cleanup_done] / effect_mapped_child_malformed,
        X <= "state_mapped_cleanup_decision"_s + completion<Mapped>(MappedRuntime<'dispatch, 'source>) [guard_mapped_cleanup_error] / effect_retain_malformed_mapping,
        X <= "state_mapped_child_decision"_s + completion<Mapped>(MappedRuntime<'dispatch, 'source>) [guard_mapped_child_error] / effect_mapped_child_error,

        // Read/copy into actor-owned setup storage.
        "state_read_owner_decision"_s <= "state_ready"_s + Read(ReadRuntime<'dispatch, 'source>),
        X <= "state_read_owner_decision"_s + completion<Read>(ReadRuntime<'dispatch, 'source>) [guard_read_owner_busy] / effect_read_busy,
        "state_read_capability_decision"_s <= "state_read_owner_decision"_s + completion<Read>(ReadRuntime<'dispatch, 'source>) [guard_read_owner_ready],
        X <= "state_read_capability_decision"_s + completion<Read>(ReadRuntime<'dispatch, 'source>) [guard_reader_absent] / effect_read_unavailable,
        "state_read_request_decision"_s <= "state_read_capability_decision"_s + completion<Read>(ReadRuntime<'dispatch, 'source>) [guard_reader_present],
        X <= "state_read_request_decision"_s + completion<Read>(ReadRuntime<'dispatch, 'source>) [guard_read_request_invalid] / effect_invalid_read,
        "state_read_residency_decision"_s <= "state_read_request_decision"_s + completion<Read>(ReadRuntime<'dispatch, 'source>) [guard_read_request_valid],
        X <= "state_read_residency_decision"_s + completion<Read>(ReadRuntime<'dispatch, 'source>) [guard_read_already_resident] / effect_read_already_resident,
        X <= "state_read_residency_decision"_s + completion<Read>(ReadRuntime<'dispatch, 'source>) [guard_read_internal] / effect_read_internal,
        "state_read_capacity_decision"_s <= "state_read_residency_decision"_s + completion<Read>(ReadRuntime<'dispatch, 'source>) [guard_read_loadable],
        X <= "state_read_capacity_decision"_s + completion<Read>(ReadRuntime<'dispatch, 'source>) [guard_read_capacity_missing] / effect_read_capacity,
        "state_read_child_decision"_s <= "state_read_capacity_decision"_s + completion<Read>(ReadRuntime<'dispatch, 'source>) [guard_read_capacity_available] / effect_dispatch_read,
        X <= "state_read_child_decision"_s + completion<Read>(ReadRuntime<'dispatch, 'source>) [guard_read_child_done] / effect_commit_read,
        X <= "state_read_child_decision"_s + completion<Read>(ReadRuntime<'dispatch, 'source>) [guard_read_child_malformed] / effect_read_child_malformed,
        X <= "state_read_child_decision"_s + completion<Read>(ReadRuntime<'dispatch, 'source>) [guard_read_child_error] / effect_read_child_error,

        // Staged copy into actor-owned setup storage.
        "state_staged_owner_decision"_s <= "state_ready"_s + Staged(StagedRuntime<'dispatch, 'source>),
        X <= "state_staged_owner_decision"_s + completion<Staged>(StagedRuntime<'dispatch, 'source>) [guard_staged_owner_busy] / effect_staged_busy,
        "state_staged_capability_decision"_s <= "state_staged_owner_decision"_s + completion<Staged>(StagedRuntime<'dispatch, 'source>) [guard_staged_owner_ready],
        X <= "state_staged_capability_decision"_s + completion<Staged>(StagedRuntime<'dispatch, 'source>) [guard_stager_absent] / effect_staged_unavailable,
        "state_staged_request_decision"_s <= "state_staged_capability_decision"_s + completion<Staged>(StagedRuntime<'dispatch, 'source>) [guard_stager_present],
        X <= "state_staged_request_decision"_s + completion<Staged>(StagedRuntime<'dispatch, 'source>) [guard_staged_request_invalid] / effect_invalid_staged,
        "state_staged_residency_decision"_s <= "state_staged_request_decision"_s + completion<Staged>(StagedRuntime<'dispatch, 'source>) [guard_staged_request_valid],
        X <= "state_staged_residency_decision"_s + completion<Staged>(StagedRuntime<'dispatch, 'source>) [guard_staged_already_resident] / effect_staged_already_resident,
        X <= "state_staged_residency_decision"_s + completion<Staged>(StagedRuntime<'dispatch, 'source>) [guard_staged_internal] / effect_staged_internal,
        "state_staged_capacity_decision"_s <= "state_staged_residency_decision"_s + completion<Staged>(StagedRuntime<'dispatch, 'source>) [guard_staged_loadable],
        X <= "state_staged_capacity_decision"_s + completion<Staged>(StagedRuntime<'dispatch, 'source>) [guard_staged_capacity_missing] / effect_staged_capacity,
        "state_staged_child_decision"_s <= "state_staged_capacity_decision"_s + completion<Staged>(StagedRuntime<'dispatch, 'source>) [guard_staged_capacity_available] / effect_dispatch_staged,
        X <= "state_staged_child_decision"_s + completion<Staged>(StagedRuntime<'dispatch, 'source>) [guard_staged_child_done] / effect_commit_staged,
        X <= "state_staged_child_decision"_s + completion<Staged>(StagedRuntime<'dispatch, 'source>) [guard_staged_child_malformed] / effect_staged_child_malformed,
        X <= "state_staged_child_decision"_s + completion<Staged>(StagedRuntime<'dispatch, 'source>) [guard_staged_child_error] / effect_staged_child_error,

        // Mapped release preserves ownership on child failure.
        "state_release_owner_decision"_s <= "state_ready"_s + Release(ReleaseRuntime<'dispatch>),
        X <= "state_release_owner_decision"_s + completion<Release>(ReleaseRuntime<'dispatch>) [guard_release_owner_busy] / effect_release_busy,
        "state_release_capability_decision"_s <= "state_release_owner_decision"_s + completion<Release>(ReleaseRuntime<'dispatch>) [guard_release_owner_ready],
        X <= "state_release_capability_decision"_s + completion<Release>(ReleaseRuntime<'dispatch>) [guard_release_mapper_absent] / effect_release_unavailable,
        "state_release_request_decision"_s <= "state_release_capability_decision"_s + completion<Release>(ReleaseRuntime<'dispatch>) [guard_release_mapper_present],
        X <= "state_release_request_decision"_s + completion<Release>(ReleaseRuntime<'dispatch>) [guard_release_id_invalid] / effect_release_invalid,
        "state_release_residency_decision"_s <= "state_release_request_decision"_s + completion<Release>(ReleaseRuntime<'dispatch>) [guard_release_id_valid],
        X <= "state_release_residency_decision"_s + completion<Release>(ReleaseRuntime<'dispatch>) [guard_release_unmapped] / effect_release_unmapped,
        X <= "state_release_residency_decision"_s + completion<Release>(ReleaseRuntime<'dispatch>) [guard_release_handle_mismatch] / effect_release_invalid,
        "state_release_child_decision"_s <= "state_release_residency_decision"_s + completion<Release>(ReleaseRuntime<'dispatch>) [guard_release_handle_matches] / effect_dispatch_release,
        X <= "state_release_child_decision"_s + completion<Release>(ReleaseRuntime<'dispatch>) [guard_release_child_done] / effect_commit_release,
        X <= "state_release_child_decision"_s + completion<Release>(ReleaseRuntime<'dispatch>) [guard_release_child_error] / effect_release_child_error,

        // Every reachable decision state fails closed on an unexpected event.
        X <= "state_ready"_s + unexpected_event<_> / effect_io_unexpected,
        X <= "state_mapped_owner_decision"_s + unexpected_event<_> / effect_io_unexpected,
        X <= "state_mapped_capability_decision"_s + unexpected_event<_> / effect_io_unexpected,
        X <= "state_mapped_request_decision"_s + unexpected_event<_> / effect_io_unexpected,
        X <= "state_mapped_residency_decision"_s + unexpected_event<_> / effect_io_unexpected,
        X <= "state_mapped_child_decision"_s + unexpected_event<_> / effect_io_unexpected,
        X <= "state_mapped_cleanup_decision"_s + unexpected_event<_> / effect_io_unexpected,
        X <= "state_read_owner_decision"_s + unexpected_event<_> / effect_io_unexpected,
        X <= "state_read_capability_decision"_s + unexpected_event<_> / effect_io_unexpected,
        X <= "state_read_request_decision"_s + unexpected_event<_> / effect_io_unexpected,
        X <= "state_read_residency_decision"_s + unexpected_event<_> / effect_io_unexpected,
        X <= "state_read_capacity_decision"_s + unexpected_event<_> / effect_io_unexpected,
        X <= "state_read_child_decision"_s + unexpected_event<_> / effect_io_unexpected,
        X <= "state_staged_owner_decision"_s + unexpected_event<_> / effect_io_unexpected,
        X <= "state_staged_capability_decision"_s + unexpected_event<_> / effect_io_unexpected,
        X <= "state_staged_request_decision"_s + unexpected_event<_> / effect_io_unexpected,
        X <= "state_staged_residency_decision"_s + unexpected_event<_> / effect_io_unexpected,
        X <= "state_staged_capacity_decision"_s + unexpected_event<_> / effect_io_unexpected,
        X <= "state_staged_child_decision"_s + unexpected_event<_> / effect_io_unexpected,
        X <= "state_release_owner_decision"_s + unexpected_event<_> / effect_io_unexpected,
        X <= "state_release_capability_decision"_s + unexpected_event<_> / effect_io_unexpected,
        X <= "state_release_request_decision"_s + unexpected_event<_> / effect_io_unexpected,
        X <= "state_release_residency_decision"_s + unexpected_event<_> / effect_io_unexpected,
        X <= "state_release_child_decision"_s + unexpected_event<_> / effect_io_unexpected,
    }
}

pub(super) struct IoDispatcher<'actor, Dependencies>
where
    Dependencies: TensorDependencies,
{
    machine: ModelTensorIoStateMachine<IoContext<'actor, Dependencies>>,
}

impl<'actor, Dependencies> IoDispatcher<'actor, Dependencies>
where
    Dependencies: TensorDependencies,
{
    pub(super) const fn new(
        tensor: &'actor mut Context,
        dependencies: &'actor mut Dependencies,
    ) -> Self {
        Self {
            machine: ModelTensorIoStateMachine::new(IoContext {
                tensor,
                dependencies,
            }),
        }
    }

    #[allow(
        clippy::needless_pass_by_value,
        reason = "the caller-created mapping capability is consumed by one top-level tensor event"
    )]
    pub(super) fn process_mapped(
        self,
        request: MappedLoad,
        owner_ready: bool,
    ) -> Result<MappedLoadDone, Error> {
        let child = Cell::new(None);
        let cleanup = Cell::new(None);
        let result = Cell::new(Err(Error::Internal));
        let mut machine = self.machine;
        machine
            .process_event(ModelTensorIoEvents::Mapped(MappedRuntime {
                request: &request,
                owner_ready,
                child: &child,
                cleanup: &cleanup,
                result: &result,
            }))
            .expect("tensor I/O SML callbacks are infallible");
        result.get()
    }

    pub(super) fn process_read(
        self,
        request: ReadLoad<'_>,
        owner_ready: bool,
    ) -> Result<OwnedLoadDone, Error> {
        let child = Cell::new(None);
        let result = Cell::new(Err(Error::Internal));
        let mut machine = self.machine;
        machine
            .process_event(ModelTensorIoEvents::Read(ReadRuntime {
                request,
                owner_ready,
                child: &child,
                result: &result,
            }))
            .expect("tensor I/O SML callbacks are infallible");
        result.get()
    }

    pub(super) fn process_staged(
        self,
        request: StagedLoad<'_>,
        owner_ready: bool,
    ) -> Result<OwnedLoadDone, Error> {
        let child = Cell::new(None);
        let result = Cell::new(Err(Error::Internal));
        let mut machine = self.machine;
        machine
            .process_event(ModelTensorIoEvents::Staged(StagedRuntime {
                request,
                owner_ready,
                child: &child,
                result: &result,
            }))
            .expect("tensor I/O SML callbacks are infallible");
        result.get()
    }

    pub(super) fn process_release(
        self,
        request: ReleaseMapped,
        owner_ready: bool,
    ) -> Result<ReleaseMappedDone, Error> {
        let child = Cell::new(None);
        let result = Cell::new(Err(Error::Internal));
        let mut machine = self.machine;
        machine
            .process_event(ModelTensorIoEvents::Release(ReleaseRuntime {
                request,
                owner_ready,
                child: &child,
                result: &result,
            }))
            .expect("tensor I/O SML callbacks are infallible");
        result.get()
    }
}

impl<Dependencies> ModelTensorIoStateMachineContext for IoContext<'_, Dependencies>
where
    Dependencies: TensorDependencies,
{
    fn guard_mapped_owner_busy(&self, event: &MappedRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(!event.owner_ready)
    }

    fn guard_mapped_owner_ready(&self, event: &MappedRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.owner_ready)
    }

    fn effect_mapped_busy(&mut self, event: MappedRuntime<'_, '_>) -> Result<(), ()> {
        event.result.set(Err(Error::Busy));
        Ok(())
    }

    fn guard_mapper_absent(&self, _: &MappedRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(!<Dependencies as Mapper>::AVAILABLE)
    }

    fn guard_mapper_present(&self, _: &MappedRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(<Dependencies as Mapper>::AVAILABLE)
    }

    fn guard_mapped_request_invalid(&self, event: &MappedRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(!self.mapped_request_valid(event.request))
    }

    fn guard_mapped_request_valid(&self, event: &MappedRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(self.mapped_request_valid(event.request))
    }

    fn guard_mapped_loadable(&self, event: &MappedRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(self.slot_loadable(event.request.tensor_id))
    }

    fn guard_mapped_already_resident(&self, event: &MappedRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(matches!(
            self.tensor.slot(event.request.tensor_id).lifecycle,
            Lifecycle::Resident | Lifecycle::MappedResident | Lifecycle::MappedCleanupPending
        ))
    }

    fn guard_mapped_internal(&self, event: &MappedRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(self.tensor.slot(event.request.tensor_id).lifecycle == Lifecycle::InternalError)
    }

    fn effect_dispatch_mapped(&mut self, event: MappedRuntime<'_, '_>) -> Result<(), ()> {
        let request = event.request;
        let child = mmap::event::MapTensor::new(
            request.tensor_id,
            request.source.clone(),
            request.offset,
            request.len,
        )
        .with_file_index(request.file_index);
        event.child.set(Some(self.dependencies.map_tensor(child)));
        Ok(())
    }

    fn guard_mapped_child_done(&self, event: &MappedRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.child.get().is_some_and(|outcome| {
            outcome.is_ok_and(|done| Self::mapped_success_valid(event.request, done))
        }))
    }

    fn guard_mapped_child_malformed(&self, event: &MappedRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.child.get().is_some_and(|outcome| {
            outcome.is_ok_and(|done| !Self::mapped_success_valid(event.request, done))
        }))
    }

    fn guard_mapped_child_error(&self, event: &MappedRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(matches!(event.child.get(), Some(Err(_))))
    }

    fn effect_commit_mapped(&mut self, event: MappedRuntime<'_, '_>) -> Result<(), ()> {
        let done = event
            .child
            .get()
            .expect("mapped child action records a result")
            .expect("mapped success guard selected success");
        let slot = self.tensor.slot_mut(event.request.tensor_id);
        slot.lifecycle = Lifecycle::MappedResident;
        slot.mapping_owner_tensor_id = Some(done.tensor_id());
        slot.mapping_handle = Some(done.handle());
        slot.buffer_bytes = done.len();
        event.result.set(Ok(MappedLoadDone::new(
            done.tensor_id(),
            done.handle(),
            done.len(),
        )));
        Ok(())
    }

    fn effect_mapped_child_error(&mut self, event: MappedRuntime<'_, '_>) -> Result<(), ()> {
        let error = event
            .child
            .get()
            .expect("mapped child action records a result")
            .expect_err("mapped error guard selected failure");
        event.result.set(Err(Error::Mmap(error)));
        Ok(())
    }

    fn effect_mapped_child_malformed(&mut self, event: MappedRuntime<'_, '_>) -> Result<(), ()> {
        event.result.set(Err(Error::DependencyContract));
        Ok(())
    }

    fn effect_release_malformed_mapping(&mut self, event: MappedRuntime<'_, '_>) -> Result<(), ()> {
        let done = event
            .child
            .get()
            .expect("mapped child action records a result")
            .expect("malformed-success guard selected a successful child outcome");
        event.cleanup.set(Some(self.dependencies.release_mapping(
            mmap::event::ReleaseMapping::new(done.tensor_id(), done.handle()),
        )));
        Ok(())
    }

    fn guard_mapped_cleanup_done(&self, event: &MappedRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(matches!(event.cleanup.get(), Some(Ok(()))))
    }

    fn guard_mapped_cleanup_error(&self, event: &MappedRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(matches!(event.cleanup.get(), Some(Err(_))))
    }

    fn effect_retain_malformed_mapping(&mut self, event: MappedRuntime<'_, '_>) -> Result<(), ()> {
        let done = event
            .child
            .get()
            .expect("mapped child action records a result")
            .expect("malformed-success guard selected a successful child outcome");
        let error = event
            .cleanup
            .get()
            .expect("malformed mapping cleanup records a result")
            .expect_err("cleanup-error guard selected a failed cleanup");
        let slot = self.tensor.slot_mut(event.request.tensor_id);
        slot.lifecycle = Lifecycle::MappedCleanupPending;
        slot.mapping_owner_tensor_id = Some(done.tensor_id());
        slot.mapping_handle = Some(done.handle());
        slot.buffer_bytes = done.len();
        event.result.set(Err(Error::DependencyContractCleanup {
            mapping_handle: done.handle(),
            error,
        }));
        Ok(())
    }

    fn effect_mapped_unavailable(&mut self, event: MappedRuntime<'_, '_>) -> Result<(), ()> {
        event.result.set(Err(Error::MmapUnavailable));
        Ok(())
    }

    fn effect_invalid_mapped(&mut self, event: MappedRuntime<'_, '_>) -> Result<(), ()> {
        event.result.set(Err(Error::InvalidRequest));
        Ok(())
    }

    fn effect_mapped_already_resident(&mut self, event: MappedRuntime<'_, '_>) -> Result<(), ()> {
        event.result.set(Err(Error::TensorAlreadyResident));
        Ok(())
    }

    fn effect_mapped_internal(&mut self, event: MappedRuntime<'_, '_>) -> Result<(), ()> {
        event.result.set(Err(Error::Internal));
        Ok(())
    }

    fn guard_reader_absent(&self, _: &ReadRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(!<Dependencies as Reader>::AVAILABLE)
    }

    fn guard_read_owner_busy(&self, event: &ReadRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(!event.owner_ready)
    }

    fn guard_read_owner_ready(&self, event: &ReadRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.owner_ready)
    }

    fn effect_read_busy(&mut self, event: ReadRuntime<'_, '_>) -> Result<(), ()> {
        event.result.set(Err(Error::Busy));
        Ok(())
    }

    fn guard_reader_present(&self, _: &ReadRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(<Dependencies as Reader>::AVAILABLE)
    }

    fn guard_read_request_invalid(&self, event: &ReadRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(!self.read_request_valid(&event.request))
    }

    fn guard_read_request_valid(&self, event: &ReadRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(self.read_request_valid(&event.request))
    }

    fn guard_read_loadable(&self, event: &ReadRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(self.slot_loadable(event.request.tensor_id))
    }

    fn guard_read_already_resident(&self, event: &ReadRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(matches!(
            self.tensor.slot(event.request.tensor_id).lifecycle,
            Lifecycle::Resident | Lifecycle::MappedResident | Lifecycle::MappedCleanupPending
        ))
    }

    fn guard_read_internal(&self, event: &ReadRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(self.tensor.slot(event.request.tensor_id).lifecycle == Lifecycle::InternalError)
    }

    fn guard_read_capacity_missing(&self, event: &ReadRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(!self.target_capacity_available(event.request.tensor_id, event.request.len))
    }

    fn guard_read_capacity_available(&self, event: &ReadRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(self.target_capacity_available(event.request.tensor_id, event.request.len))
    }

    fn effect_dispatch_read(&mut self, event: ReadRuntime<'_, '_>) -> Result<(), ()> {
        let request = event.request;
        let slot = self.tensor.slot_mut(request.tensor_id);
        let bytes = slot
            .bytes
            .as_mut()
            .expect("capacity guard selected target storage");
        let target = read::event::Target::new(bytes);
        let child = read::event::ReadTensor::new(
            request.tensor_id,
            request.file_path,
            request.source,
            &target,
        )
        .with_file_index(request.file_index)
        .with_range(request.offset, request.len)
        .with_source_error_option(request.source_error);
        event.child.set(Some(self.dependencies.read_tensor(child)));
        Ok(())
    }

    fn guard_read_child_done(&self, event: &ReadRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.child.get().is_some_and(|outcome| {
            outcome.is_ok_and(|done| Self::read_success_valid(&event.request, done))
        }))
    }

    fn guard_read_child_malformed(&self, event: &ReadRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.child.get().is_some_and(|outcome| {
            outcome.is_ok_and(|done| !Self::read_success_valid(&event.request, done))
        }))
    }

    fn guard_read_child_error(&self, event: &ReadRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(matches!(event.child.get(), Some(Err(_))))
    }

    fn effect_commit_read(&mut self, event: ReadRuntime<'_, '_>) -> Result<(), ()> {
        let done = event
            .child
            .get()
            .expect("read child action records a result")
            .expect("read success guard selected success");
        let slot = self.tensor.slot_mut(event.request.tensor_id);
        slot.lifecycle = Lifecycle::Resident;
        slot.buffer_bytes = done.bytes_copied();
        event.result.set(Ok(OwnedLoadDone::new(
            done.tensor_id(),
            done.bytes_copied(),
        )));
        Ok(())
    }

    fn effect_read_child_error(&mut self, event: ReadRuntime<'_, '_>) -> Result<(), ()> {
        let error = event
            .child
            .get()
            .expect("read child action records a result")
            .expect_err("read error guard selected failure");
        event.result.set(Err(Error::Read(error)));
        Ok(())
    }

    fn effect_read_child_malformed(&mut self, event: ReadRuntime<'_, '_>) -> Result<(), ()> {
        event.result.set(Err(Error::DependencyContract));
        Ok(())
    }

    fn effect_read_unavailable(&mut self, event: ReadRuntime<'_, '_>) -> Result<(), ()> {
        event.result.set(Err(Error::ReadUnavailable));
        Ok(())
    }

    fn effect_invalid_read(&mut self, event: ReadRuntime<'_, '_>) -> Result<(), ()> {
        event.result.set(Err(Error::InvalidRequest));
        Ok(())
    }

    fn effect_read_already_resident(&mut self, event: ReadRuntime<'_, '_>) -> Result<(), ()> {
        event.result.set(Err(Error::TensorAlreadyResident));
        Ok(())
    }

    fn effect_read_internal(&mut self, event: ReadRuntime<'_, '_>) -> Result<(), ()> {
        event.result.set(Err(Error::Internal));
        Ok(())
    }

    fn effect_read_capacity(&mut self, event: ReadRuntime<'_, '_>) -> Result<(), ()> {
        event.result.set(Err(Error::Capacity));
        Ok(())
    }

    fn guard_stager_absent(&self, _: &StagedRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(!<Dependencies as Stager>::AVAILABLE)
    }

    fn guard_staged_owner_busy(&self, event: &StagedRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(!event.owner_ready)
    }

    fn guard_staged_owner_ready(&self, event: &StagedRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.owner_ready)
    }

    fn effect_staged_busy(&mut self, event: StagedRuntime<'_, '_>) -> Result<(), ()> {
        event.result.set(Err(Error::Busy));
        Ok(())
    }

    fn guard_stager_present(&self, _: &StagedRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(<Dependencies as Stager>::AVAILABLE)
    }

    fn guard_staged_request_invalid(&self, event: &StagedRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(!self.staged_request_valid(&event.request))
    }

    fn guard_staged_request_valid(&self, event: &StagedRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(self.staged_request_valid(&event.request))
    }

    fn guard_staged_loadable(&self, event: &StagedRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(self.slot_loadable(event.request.tensor_id))
    }

    fn guard_staged_already_resident(&self, event: &StagedRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(matches!(
            self.tensor.slot(event.request.tensor_id).lifecycle,
            Lifecycle::Resident | Lifecycle::MappedResident | Lifecycle::MappedCleanupPending
        ))
    }

    fn guard_staged_internal(&self, event: &StagedRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(self.tensor.slot(event.request.tensor_id).lifecycle == Lifecycle::InternalError)
    }

    fn guard_staged_capacity_missing(&self, event: &StagedRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(!self.target_capacity_available(event.request.tensor_id, event.request.len))
    }

    fn guard_staged_capacity_available(&self, event: &StagedRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(self.target_capacity_available(event.request.tensor_id, event.request.len))
    }

    fn effect_dispatch_staged(&mut self, event: StagedRuntime<'_, '_>) -> Result<(), ()> {
        let request = event.request;
        let source = request
            .source
            .expect("staged request guard selected a present source");
        let source_start = usize::try_from(request.offset)
            .expect("staged request guard selected a representable source offset");
        let source_end = usize::try_from(
            request
                .offset
                .checked_add(request.len)
                .expect("staged request guard selected a bounded source range"),
        )
        .expect("staged request guard selected a representable source end");
        let source_window = &source[source_start..source_end];
        let slot = self.tensor.slot_mut(request.tensor_id);
        let bytes = slot
            .bytes
            .as_mut()
            .expect("capacity guard selected target storage");
        let target = staged_read::event::Target::new(bytes);
        let done = Cell::new(None);
        let error = Cell::new(None);
        let child = staged_read::event::StageWindow::new(
            request.offset,
            request.len,
            request.stage_chunk_bytes,
            Some(source_window),
            &target,
        )
        .on_done(staged_read::event::Callback::store(&done))
        .on_error(staged_read::event::Callback::store(&error));
        event.child.set(Some(self.dependencies.stage_tensor(child)));
        Ok(())
    }

    fn guard_staged_child_done(&self, event: &StagedRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.child.get().is_some_and(|outcome| {
            outcome.is_ok_and(|done| Self::staged_success_valid(&event.request, done))
        }))
    }

    fn guard_staged_child_malformed(&self, event: &StagedRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(event.child.get().is_some_and(|outcome| {
            outcome.is_ok_and(|done| !Self::staged_success_valid(&event.request, done))
        }))
    }

    fn guard_staged_child_error(&self, event: &StagedRuntime<'_, '_>) -> Result<bool, ()> {
        Ok(matches!(event.child.get(), Some(Err(_))))
    }

    fn effect_commit_staged(&mut self, event: StagedRuntime<'_, '_>) -> Result<(), ()> {
        let done = event
            .child
            .get()
            .expect("staged child action records a result")
            .expect("staged success guard selected success");
        let slot = self.tensor.slot_mut(event.request.tensor_id);
        slot.lifecycle = Lifecycle::Resident;
        slot.buffer_bytes = done.bytes_committed();
        event.result.set(Ok(OwnedLoadDone::new(
            event.request.tensor_id,
            done.bytes_committed(),
        )));
        Ok(())
    }

    fn effect_staged_child_error(&mut self, event: StagedRuntime<'_, '_>) -> Result<(), ()> {
        let error = event
            .child
            .get()
            .expect("staged child action records a result")
            .expect_err("staged error guard selected failure");
        event.result.set(Err(Error::Staged(error)));
        Ok(())
    }

    fn effect_staged_child_malformed(&mut self, event: StagedRuntime<'_, '_>) -> Result<(), ()> {
        event.result.set(Err(Error::DependencyContract));
        Ok(())
    }

    fn effect_staged_unavailable(&mut self, event: StagedRuntime<'_, '_>) -> Result<(), ()> {
        event.result.set(Err(Error::StagerUnavailable));
        Ok(())
    }

    fn effect_invalid_staged(&mut self, event: StagedRuntime<'_, '_>) -> Result<(), ()> {
        event.result.set(Err(Error::InvalidRequest));
        Ok(())
    }

    fn effect_staged_already_resident(&mut self, event: StagedRuntime<'_, '_>) -> Result<(), ()> {
        event.result.set(Err(Error::TensorAlreadyResident));
        Ok(())
    }

    fn effect_staged_internal(&mut self, event: StagedRuntime<'_, '_>) -> Result<(), ()> {
        event.result.set(Err(Error::Internal));
        Ok(())
    }

    fn effect_staged_capacity(&mut self, event: StagedRuntime<'_, '_>) -> Result<(), ()> {
        event.result.set(Err(Error::Capacity));
        Ok(())
    }

    fn guard_release_mapper_absent(&self, _: &ReleaseRuntime<'_>) -> Result<bool, ()> {
        Ok(!<Dependencies as Mapper>::AVAILABLE)
    }

    fn guard_release_owner_busy(&self, event: &ReleaseRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.owner_ready)
    }

    fn guard_release_owner_ready(&self, event: &ReleaseRuntime<'_>) -> Result<bool, ()> {
        Ok(event.owner_ready)
    }

    fn effect_release_busy(&mut self, event: ReleaseRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::Busy));
        Ok(())
    }

    fn guard_release_mapper_present(&self, _: &ReleaseRuntime<'_>) -> Result<bool, ()> {
        Ok(<Dependencies as Mapper>::AVAILABLE)
    }

    fn guard_release_id_invalid(&self, event: &ReleaseRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.tensor.active_tensor_id_valid(event.request.tensor_id))
    }

    fn guard_release_id_valid(&self, event: &ReleaseRuntime<'_>) -> Result<bool, ()> {
        Ok(self.tensor.active_tensor_id_valid(event.request.tensor_id))
    }

    fn guard_release_unmapped(&self, event: &ReleaseRuntime<'_>) -> Result<bool, ()> {
        Ok(!matches!(
            self.tensor.slot(event.request.tensor_id).lifecycle,
            Lifecycle::MappedResident | Lifecycle::MappedCleanupPending
        ))
    }

    fn guard_release_handle_mismatch(&self, event: &ReleaseRuntime<'_>) -> Result<bool, ()> {
        let slot = self.tensor.slot(event.request.tensor_id);
        Ok(matches!(
            slot.lifecycle,
            Lifecycle::MappedResident | Lifecycle::MappedCleanupPending
        ) && slot.mapping_handle != Some(event.request.mapping_handle))
    }

    fn guard_release_handle_matches(&self, event: &ReleaseRuntime<'_>) -> Result<bool, ()> {
        let slot = self.tensor.slot(event.request.tensor_id);
        Ok(matches!(
            slot.lifecycle,
            Lifecycle::MappedResident | Lifecycle::MappedCleanupPending
        ) && slot.mapping_handle == Some(event.request.mapping_handle))
    }

    fn effect_dispatch_release(&mut self, event: ReleaseRuntime<'_>) -> Result<(), ()> {
        let mapping_owner_tensor_id = self
            .tensor
            .slot(event.request.tensor_id)
            .mapping_owner_tensor_id
            .expect("mapped lifecycle retains the dependency's owner tensor identifier");
        event.child.set(Some(self.dependencies.release_mapping(
            mmap::event::ReleaseMapping::new(mapping_owner_tensor_id, event.request.mapping_handle),
        )));
        Ok(())
    }

    fn guard_release_child_done(&self, event: &ReleaseRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.child.get(), Some(Ok(()))))
    }

    fn guard_release_child_error(&self, event: &ReleaseRuntime<'_>) -> Result<bool, ()> {
        Ok(matches!(event.child.get(), Some(Err(_))))
    }

    fn effect_commit_release(&mut self, event: ReleaseRuntime<'_>) -> Result<(), ()> {
        let slot = self.tensor.slot_mut(event.request.tensor_id);
        slot.lifecycle = Lifecycle::Evicted;
        slot.mapping_owner_tensor_id = None;
        slot.mapping_handle = None;
        slot.buffer_bytes = 0;
        event
            .result
            .set(Ok(ReleaseMappedDone::new(event.request.tensor_id)));
        Ok(())
    }

    fn effect_release_child_error(&mut self, event: ReleaseRuntime<'_>) -> Result<(), ()> {
        let error = event
            .child
            .get()
            .expect("release child action records a result")
            .expect_err("release error guard selected failure");
        event.result.set(Err(Error::Mmap(error)));
        Ok(())
    }

    fn effect_release_unavailable(&mut self, event: ReleaseRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::MmapUnavailable));
        Ok(())
    }

    fn effect_release_invalid(&mut self, event: ReleaseRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::InvalidRequest));
        Ok(())
    }

    fn effect_release_unmapped(&mut self, event: ReleaseRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(Error::TensorUnbound));
        Ok(())
    }

    fn effect_io_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}

impl<Dependencies> IoContext<'_, Dependencies>
where
    Dependencies: TensorDependencies,
{
    fn mapped_request_valid(&self, request: &MappedLoad) -> bool {
        self.tensor.active_tensor_id_valid(request.tensor_id)
            && request.len > 0
            && request.offset.checked_add(request.len).is_some()
            && usize::try_from(request.len).is_ok()
    }

    fn owned_request_valid(&self, tensor_id: i32, offset: u64, len: u64) -> bool {
        self.tensor.active_tensor_id_valid(tensor_id)
            && len > 0
            && offset.checked_add(len).is_some()
            && usize::try_from(len).is_ok()
            && self.tensor.slot(tensor_id).metadata.data_size() >= len
    }

    fn read_request_valid(&self, request: &ReadLoad<'_>) -> bool {
        !request.file_path.is_empty()
            && request.file_path.len() <= MAX_READ_FILE_PATH_BYTES
            && !request.file_path.as_bytes().contains(&0)
            && self.owned_request_valid(request.tensor_id, request.offset, request.len)
    }

    fn staged_request_valid(&self, request: &StagedLoad<'_>) -> bool {
        request.stage_chunk_bytes > 0
            && request.source.is_some_and(|source| {
                u64::try_from(source.len()).is_ok_and(|source_len| {
                    request
                        .offset
                        .checked_add(request.len)
                        .is_some_and(|end| end <= source_len)
                })
            })
            && self.owned_request_valid(request.tensor_id, request.offset, request.len)
    }

    fn slot_loadable(&self, tensor_id: i32) -> bool {
        matches!(
            self.tensor.slot(tensor_id).lifecycle,
            Lifecycle::Unbound | Lifecycle::Evicted
        )
    }

    fn target_capacity_available(&self, tensor_id: i32, len: u64) -> bool {
        self.tensor
            .slot(tensor_id)
            .bytes
            .as_ref()
            .is_some_and(|bytes| u64::try_from(bytes.len()).is_ok_and(|size| size >= len))
    }

    const fn mapped_success_valid(request: &MappedLoad, done: mmap::event::MapDone) -> bool {
        done.tensor_id() == request.tensor_id && done.len() == request.len
    }

    const fn read_success_valid(request: &ReadLoad<'_>, done: read::event::ReadTensorDone) -> bool {
        done.tensor_id() == request.tensor_id && done.bytes_copied() == request.len
    }

    const fn staged_success_valid(
        request: &StagedLoad<'_>,
        done: staged_read::event::StageWindowDone,
    ) -> bool {
        done.bytes_committed() == request.len
    }
}

#[derive(Clone, Copy)]
pub(super) struct AccessRequest {
    pub(super) tensor_id: i32,
}

pub(super) struct AccessRuntime<'dispatch, Operation>
where
    Operation: TensorOperation,
{
    request: AccessRequest,
    owner_ready: bool,
    operation: &'dispatch RefCell<&'dispatch mut Operation>,
    child: &'dispatch RefCell<Option<Result<Operation::Output, mmap::event::Error>>>,
    result: &'dispatch RefCell<Option<Result<Operation::Output, Error>>>,
}

impl<Operation> Clone for AccessRuntime<'_, Operation>
where
    Operation: TensorOperation,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<Operation> Copy for AccessRuntime<'_, Operation> where Operation: TensorOperation {}

struct AccessContext<'actor, Dependencies>
where
    Dependencies: TensorDependencies,
{
    tensor: &'actor mut Context,
    dependencies: &'actor mut Dependencies,
}

struct MappingAdapter<'dispatch, Operation>
where
    Operation: TensorOperation,
{
    operation: &'dispatch RefCell<&'dispatch mut Operation>,
}

impl<Operation> mmap::event::MappingOperation for MappingAdapter<'_, Operation>
where
    Operation: TensorOperation,
{
    type Output = Operation::Output;

    fn apply(&mut self, bytes: &[u8]) -> Self::Output {
        self.operation.borrow_mut().apply(bytes)
    }
}

sml! {
    ModelTensorAccess<'dispatch, Operation>
    where
        Operation: TensorOperation + 'dispatch,
    {
        "state_owner_decision"_s <= *"state_ready"_s + Access(AccessRuntime<'dispatch, Operation>),
        X <= "state_owner_decision"_s + completion<Access>(AccessRuntime<'dispatch, Operation>) [guard_access_owner_busy] / effect_access_busy,
        "state_id_decision"_s <= "state_owner_decision"_s + completion<Access>(AccessRuntime<'dispatch, Operation>) [guard_access_owner_ready],
        X <= "state_id_decision"_s + completion<Access>(AccessRuntime<'dispatch, Operation>) [guard_access_id_invalid] / effect_access_invalid,
        "state_residency_decision"_s <= "state_id_decision"_s + completion<Access>(AccessRuntime<'dispatch, Operation>) [guard_access_id_valid],
        X <= "state_residency_decision"_s + completion<Access>(AccessRuntime<'dispatch, Operation>) [guard_access_unbound] / effect_access_unbound,
        X <= "state_residency_decision"_s + completion<Access>(AccessRuntime<'dispatch, Operation>) [guard_access_internal] / effect_access_internal,
        X <= "state_residency_decision"_s + completion<Access>(AccessRuntime<'dispatch, Operation>) [guard_access_cleanup_pending] / effect_access_cleanup_pending,
        X <= "state_residency_decision"_s + completion<Access>(AccessRuntime<'dispatch, Operation>) [guard_access_owned] / effect_access_owned,
        X <= "state_residency_decision"_s + completion<Access>(AccessRuntime<'dispatch, Operation>) [guard_access_mapped_without_mapper] / effect_access_mapper_unavailable,
        "state_mapped_child_decision"_s <= "state_residency_decision"_s + completion<Access>(AccessRuntime<'dispatch, Operation>) [guard_access_mapped_with_mapper] / effect_access_mapped,
        X <= "state_mapped_child_decision"_s + completion<Access>(AccessRuntime<'dispatch, Operation>) [guard_access_mapped_done] / effect_access_mapped_done,
        X <= "state_mapped_child_decision"_s + completion<Access>(AccessRuntime<'dispatch, Operation>) [guard_access_mapped_error] / effect_access_mapped_error,
        X <= "state_ready"_s + unexpected_event<_> / effect_access_unexpected,
        X <= "state_owner_decision"_s + unexpected_event<_> / effect_access_unexpected,
        X <= "state_id_decision"_s + unexpected_event<_> / effect_access_unexpected,
        X <= "state_residency_decision"_s + unexpected_event<_> / effect_access_unexpected,
        X <= "state_mapped_child_decision"_s + unexpected_event<_> / effect_access_unexpected,
    }
}

pub(super) struct TensorAccessDispatcher<'actor, Dependencies>
where
    Dependencies: TensorDependencies,
{
    machine: ModelTensorAccessStateMachine<AccessContext<'actor, Dependencies>>,
}

impl<'actor, Dependencies> TensorAccessDispatcher<'actor, Dependencies>
where
    Dependencies: TensorDependencies,
{
    pub(super) const fn new(
        tensor: &'actor mut Context,
        dependencies: &'actor mut Dependencies,
    ) -> Self {
        Self {
            machine: ModelTensorAccessStateMachine::new(AccessContext {
                tensor,
                dependencies,
            }),
        }
    }

    pub(super) fn process_event<Operation>(
        mut self,
        request: AccessRequest,
        operation: &mut Operation,
        owner_ready: bool,
    ) -> Result<Operation::Output, Error>
    where
        Operation: TensorOperation,
    {
        let operation = RefCell::new(operation);
        let child = RefCell::new(None);
        let result = RefCell::new(None);
        self.machine
            .process_event(ModelTensorAccessEvents::Access(AccessRuntime {
                request,
                owner_ready,
                operation: &operation,
                child: &child,
                result: &result,
            }))
            .expect("tensor access SML callbacks are infallible");
        result
            .into_inner()
            .expect("tensor access SML stores one typed outcome")
    }
}

impl<Dependencies> ModelTensorAccessStateMachineContext for AccessContext<'_, Dependencies>
where
    Dependencies: TensorDependencies,
{
    fn guard_access_owner_busy<'dispatch, Operation>(
        &self,
        event: &AccessRuntime<'dispatch, Operation>,
    ) -> Result<bool, ()>
    where
        Operation: TensorOperation + 'dispatch,
    {
        Ok(!event.owner_ready)
    }

    fn guard_access_owner_ready<'dispatch, Operation>(
        &self,
        event: &AccessRuntime<'dispatch, Operation>,
    ) -> Result<bool, ()>
    where
        Operation: TensorOperation + 'dispatch,
    {
        Ok(event.owner_ready)
    }

    fn effect_access_busy<'dispatch, Operation>(
        &mut self,
        event: AccessRuntime<'dispatch, Operation>,
    ) -> Result<(), ()>
    where
        Operation: TensorOperation + 'dispatch,
    {
        event.result.replace(Some(Err(Error::Busy)));
        Ok(())
    }

    fn guard_access_id_invalid<'dispatch, Operation>(
        &self,
        event: &AccessRuntime<'dispatch, Operation>,
    ) -> Result<bool, ()>
    where
        Operation: TensorOperation + 'dispatch,
    {
        Ok(!self.tensor.tensor_id_valid(event.request.tensor_id))
    }

    fn guard_access_id_valid<'dispatch, Operation>(
        &self,
        event: &AccessRuntime<'dispatch, Operation>,
    ) -> Result<bool, ()>
    where
        Operation: TensorOperation + 'dispatch,
    {
        Ok(self.tensor.tensor_id_valid(event.request.tensor_id))
    }

    fn guard_access_unbound<'dispatch, Operation>(
        &self,
        event: &AccessRuntime<'dispatch, Operation>,
    ) -> Result<bool, ()>
    where
        Operation: TensorOperation + 'dispatch,
    {
        Ok(matches!(
            self.tensor.slot(event.request.tensor_id).lifecycle,
            Lifecycle::Unbound | Lifecycle::Evicted
        ))
    }

    fn guard_access_internal<'dispatch, Operation>(
        &self,
        event: &AccessRuntime<'dispatch, Operation>,
    ) -> Result<bool, ()>
    where
        Operation: TensorOperation + 'dispatch,
    {
        Ok(self.tensor.slot(event.request.tensor_id).lifecycle == Lifecycle::InternalError)
    }

    fn guard_access_cleanup_pending<'dispatch, Operation>(
        &self,
        event: &AccessRuntime<'dispatch, Operation>,
    ) -> Result<bool, ()>
    where
        Operation: TensorOperation + 'dispatch,
    {
        Ok(self.tensor.slot(event.request.tensor_id).lifecycle == Lifecycle::MappedCleanupPending)
    }

    fn guard_access_owned<'dispatch, Operation>(
        &self,
        event: &AccessRuntime<'dispatch, Operation>,
    ) -> Result<bool, ()>
    where
        Operation: TensorOperation + 'dispatch,
    {
        Ok(self.tensor.slot(event.request.tensor_id).lifecycle == Lifecycle::Resident)
    }

    fn guard_access_mapped_without_mapper<'dispatch, Operation>(
        &self,
        event: &AccessRuntime<'dispatch, Operation>,
    ) -> Result<bool, ()>
    where
        Operation: TensorOperation + 'dispatch,
    {
        Ok(
            self.tensor.slot(event.request.tensor_id).lifecycle == Lifecycle::MappedResident
                && !<Dependencies as Mapper>::AVAILABLE,
        )
    }

    fn guard_access_mapped_with_mapper<'dispatch, Operation>(
        &self,
        event: &AccessRuntime<'dispatch, Operation>,
    ) -> Result<bool, ()>
    where
        Operation: TensorOperation + 'dispatch,
    {
        Ok(
            self.tensor.slot(event.request.tensor_id).lifecycle == Lifecycle::MappedResident
                && <Dependencies as Mapper>::AVAILABLE,
        )
    }

    fn effect_access_owned<'dispatch, Operation>(
        &mut self,
        event: AccessRuntime<'dispatch, Operation>,
    ) -> Result<(), ()>
    where
        Operation: TensorOperation + 'dispatch,
    {
        let slot = self.tensor.slot(event.request.tensor_id);
        let bytes = slot
            .bytes
            .as_deref()
            .expect("resident lifecycle is established only with actor-owned bytes");
        let resident_len = usize::try_from(slot.buffer_bytes)
            .expect("resident byte count was validated before commit");
        let output = event.operation.borrow_mut().apply(&bytes[..resident_len]);
        event.result.replace(Some(Ok(output)));
        Ok(())
    }

    fn effect_access_cleanup_pending<'dispatch, Operation>(
        &mut self,
        event: AccessRuntime<'dispatch, Operation>,
    ) -> Result<(), ()>
    where
        Operation: TensorOperation + 'dispatch,
    {
        event
            .result
            .replace(Some(Err(Error::MappedTensorRequiresRelease)));
        Ok(())
    }

    fn effect_access_mapped<'dispatch, Operation>(
        &mut self,
        event: AccessRuntime<'dispatch, Operation>,
    ) -> Result<(), ()>
    where
        Operation: TensorOperation + 'dispatch,
    {
        let slot = self.tensor.slot(event.request.tensor_id);
        let mapping_owner_tensor_id = slot
            .mapping_owner_tensor_id
            .expect("mapped lifecycle retains the dependency's owner tensor identifier");
        let handle = slot
            .mapping_handle
            .expect("mapped lifecycle retains a mapping handle");
        let mut adapter = MappingAdapter {
            operation: event.operation,
        };
        let child = self
            .dependencies
            .with_mapping(mmap::event::WithMapping::new(
                mapping_owner_tensor_id,
                handle,
                &mut adapter,
            ));
        event.child.replace(Some(child));
        Ok(())
    }

    fn guard_access_mapped_done<'dispatch, Operation>(
        &self,
        event: &AccessRuntime<'dispatch, Operation>,
    ) -> Result<bool, ()>
    where
        Operation: TensorOperation + 'dispatch,
    {
        Ok(event.child.borrow().as_ref().is_some_and(Result::is_ok))
    }

    fn guard_access_mapped_error<'dispatch, Operation>(
        &self,
        event: &AccessRuntime<'dispatch, Operation>,
    ) -> Result<bool, ()>
    where
        Operation: TensorOperation + 'dispatch,
    {
        Ok(event.child.borrow().as_ref().is_some_and(Result::is_err))
    }

    fn effect_access_mapped_done<'dispatch, Operation>(
        &mut self,
        event: AccessRuntime<'dispatch, Operation>,
    ) -> Result<(), ()>
    where
        Operation: TensorOperation + 'dispatch,
    {
        let output = event
            .child
            .borrow_mut()
            .take()
            .expect("mapped access action records a result")
            .expect("mapped access success guard selected success");
        event.result.replace(Some(Ok(output)));
        Ok(())
    }

    fn effect_access_mapped_error<'dispatch, Operation>(
        &mut self,
        event: AccessRuntime<'dispatch, Operation>,
    ) -> Result<(), ()>
    where
        Operation: TensorOperation + 'dispatch,
    {
        let error = event
            .child
            .borrow_mut()
            .take()
            .expect("mapped access action records a result")
            .err()
            .expect("mapped access error guard selected failure");
        event.result.replace(Some(Err(Error::Mmap(error))));
        Ok(())
    }

    fn effect_access_invalid<'dispatch, Operation>(
        &mut self,
        event: AccessRuntime<'dispatch, Operation>,
    ) -> Result<(), ()>
    where
        Operation: TensorOperation + 'dispatch,
    {
        event.result.replace(Some(Err(Error::InvalidRequest)));
        Ok(())
    }

    fn effect_access_unbound<'dispatch, Operation>(
        &mut self,
        event: AccessRuntime<'dispatch, Operation>,
    ) -> Result<(), ()>
    where
        Operation: TensorOperation + 'dispatch,
    {
        event.result.replace(Some(Err(Error::TensorUnbound)));
        Ok(())
    }

    fn effect_access_internal<'dispatch, Operation>(
        &mut self,
        event: AccessRuntime<'dispatch, Operation>,
    ) -> Result<(), ()>
    where
        Operation: TensorOperation + 'dispatch,
    {
        event.result.replace(Some(Err(Error::Internal)));
        Ok(())
    }

    fn effect_access_mapper_unavailable<'dispatch, Operation>(
        &mut self,
        event: AccessRuntime<'dispatch, Operation>,
    ) -> Result<(), ()>
    where
        Operation: TensorOperation + 'dispatch,
    {
        event.result.replace(Some(Err(Error::MmapUnavailable)));
        Ok(())
    }

    fn effect_access_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}
