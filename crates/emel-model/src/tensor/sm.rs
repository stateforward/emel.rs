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

use sml::sml;

use super::actor::MAX_TENSORS;
use super::event::{
    ApplyBoundEffectResultsError, ApplyOwnedEffectResultsError, BindStorageDone, BindStorageError,
    BindTensorDone, EffectBuffer, EffectError, EffectRequest, Error, EvictTensorDone, Lifecycle,
    PlanLoadDone, PlanLoadError, StorageBatch, StrategyKind, TensorMetadata, TensorState,
};

#[derive(Debug)]
struct Slot {
    lifecycle: Lifecycle,
    bytes: Option<Box<[u8]>>,
    buffer_bytes: u64,
    metadata: TensorMetadata,
}

impl Default for Slot {
    fn default() -> Self {
        Self {
            lifecycle: Lifecycle::Unbound,
            bytes: None,
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
        event.strategy == strategy && self.plan_input_valid(event)
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
        "state_ready"_s <= "state_ready"_s + PlanLoad(PlanLoadRuntime<'dispatch>) [guard_plan_invalid_request] / effect_plan_invalid_request,
        "state_ready"_s <= "state_ready"_s + PlanLoad(PlanLoadRuntime<'dispatch>) [guard_plan_capacity] / effect_plan_capacity,
        "state_ready"_s <= "state_ready"_s + PlanLoad(PlanLoadRuntime<'dispatch>) [guard_plan_unsupported_strategy] / effect_plan_unsupported_strategy,

        // Complete-batch result validation and application.
        "state_ready"_s <= "state_awaiting_bound_results"_s + ApplyBound(ApplyBoundRuntime<'dispatch>) [guard_bound_results_valid] / effect_apply_bound_results,
        "state_ready"_s <= "state_awaiting_bound_results"_s + ApplyBound(ApplyBoundRuntime<'dispatch>) [guard_bound_results_invalid] / effect_apply_bound_results_invalid,
        "state_ready"_s <= "state_awaiting_owned_results"_s + ApplyBound(ApplyBoundRuntime<'dispatch>) / effect_apply_bound_results_invalid,
        "state_ready"_s <= "state_awaiting_mapped_results"_s + ApplyBound(ApplyBoundRuntime<'dispatch>) / effect_apply_bound_results_invalid,
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
            && !self
                .slots
                .iter()
                .any(|slot| slot.lifecycle == Lifecycle::MappedResident))
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
            && self
                .slots
                .iter()
                .any(|slot| slot.lifecycle == Lifecycle::MappedResident))
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
        Ok(self.plan_input_valid(event) && matches!(event.strategy, StrategyKind::Unknown(_)))
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
        Ok(tensor_ids.len() == self.active_extent
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
        Ok(tensor_ids.len() != self.active_extent
            || tensor_ids.iter().enumerate().any(|(index, tensor_id)| {
                *tensor_id != i32::try_from(index).expect("actor capacity fits i32")
                    || !self.slots[index].bytes.as_ref().is_some_and(|bytes| {
                        u64::try_from(bytes.len())
                            .is_ok_and(|length| length >= self.slots[index].metadata.data_size())
                    })
            }))
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

    fn guard_owned_results_valid(&self, event: &ApplyOwnedRuntime<'_>) -> Result<bool, ()> {
        let result = event.result.borrow();
        let results = &result
            .as_ref()
            .expect_err("owned-result wrapper starts with caller-owned results")
            .results;
        Ok(results.len() == self.active_extent
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
        Ok(results.len() != self.active_extent
            || results.iter().enumerate().any(|(index, result)| {
                result.tensor_id != i32::try_from(index).expect("actor capacity fits i32")
                    || !u64::try_from(result.bytes.len())
                        .is_ok_and(|length| length >= self.slots[index].metadata.data_size())
            }))
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
        Ok(self.slot(event.tensor_id).lifecycle == Lifecycle::MappedResident)
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
        Ok(self.slot(event.tensor_id).lifecycle == Lifecycle::MappedResident)
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
