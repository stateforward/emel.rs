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

use sml::sml;

use super::actor::MAX_TENSORS;
use super::event::{
    BindTensorDone, Error, EvictTensorDone, Lifecycle, TensorMetadata, TensorState,
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
        Ok(Self { slots })
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

sml! {
    ModelTensor {
        // Bind validation and ownership transfer.
        "state_bind_id_decision"_s <= *"state_ready"_s + Bind(BindRuntime<'dispatch>) / effect_begin_bind,
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
    }
}

impl ModelTensorStateMachineContext for Context {
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
