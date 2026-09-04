//! Allocation-free core tensor-window lifecycle actor.

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::elidable_lifetime_names)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::unused_self)]
#![allow(clippy::large_stack_arrays)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::needless_pass_by_ref_mut)]

use super::detail::{self, WindowState};
use super::event::{self, Acquire, Bind, Unbind};
use crate::tensor::dependency::Stager;
use emel_io::staged_read::event::Target;

/// Owns one bounded tensor-window lifecycle.
#[derive(Debug, Default)]
pub struct Window {
    state: WindowState,
}

impl Window {
    /// Creates an unbound window. All dispatch state is inline and reusable.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state: WindowState {
                plan: [detail::LayerDescriptor {
                    weights: [detail::WeightExtent {
                        tensor_id: 0,
                        file_offset: 0,
                        byte_size: 0,
                        slot_offset: 0,
                    }; detail::MAX_WEIGHTS_PER_LAYER],
                    weight_count: 0,
                    file_begin: 0,
                    file_span: 0,
                    slot_bytes: 0,
                }; detail::MAX_STREAM_LAYERS],
                layer_count: 0,
                slots: [detail::WindowSlot {
                    layer: -1,
                    lifecycle: detail::SlotLifecycle::Vacant,
                }; detail::MAX_WINDOW_SLOTS],
                slot_count: 0,
                slot_capacity_bytes: 0,
                total_stream_bytes: 0,
                source_bytes: 0,
                budget_bytes: 0,
                prefetch_depth: 0,
                stage_chunk_bytes: detail::DEFAULT_STREAM_CHUNK_BYTES,
                next_prefetch_layer: -1,
                streaming_active: false,
                bound: false,
            },
        }
    }

    pub fn bind<'a>(&mut self, request: Bind<'a>) -> Result<event::BindDone, event::BindError> {
        if self.state.bound {
            return self.bind_error(request, event::Error::AlreadyBound);
        }
        if request.file_size_bytes == 0
            || request.layer_weight_counts.is_empty()
            || usize::try_from(request.window_slots)
                .ok()
                .is_none_or(|slots| slots > detail::MAX_WINDOW_SLOTS)
            || (request.window_slots == 0 && request.budget_bytes != 0)
            || request.stage_chunk_bytes < detail::MIN_STREAM_CHUNK_BYTES
            || request.stage_chunk_bytes > detail::MAX_STREAM_CHUNK_BYTES
            || request.layer_weight_counts.len() > detail::MAX_STREAM_LAYERS
            || request.extents.len() > detail::MAX_STREAM_LAYERS * detail::MAX_WEIGHTS_PER_LAYER
        {
            return self.bind_error(request, event::Error::InvalidRequest);
        }
        for &count in request.layer_weight_counts {
            if count == 0 || usize::from(count) > detail::MAX_WEIGHTS_PER_LAYER {
                return self.bind_error(request, event::Error::InvalidRequest);
            }
        }
        let mut candidate = WindowState::default();
        for extent in request.extents {
            if extent.byte_size == 0
                || extent.file_offset > request.file_size_bytes
                || extent.byte_size > request.file_size_bytes - extent.file_offset
            {
                return self.bind_error(request, event::Error::InvalidRequest);
            }
        }
        if !detail::scan_layer_descriptors(
            request.extents,
            request.layer_weight_counts,
            &mut candidate,
        ) {
            return self.bind_error(request, event::Error::InvalidRequest);
        }
        let streaming =
            request.budget_bytes != 0 && candidate.total_stream_bytes > request.budget_bytes;
        if streaming {
            if request.window_slots < 2
                || request.prefetch_depth == 0
                || request.prefetch_depth >= request.window_slots
            {
                return self.bind_error(request, event::Error::InvalidRequest);
            }
            let Some(slots_bytes) = detail::slot_bytes_needed(&candidate, request.window_slots)
            else {
                return self.bind_error(request, event::Error::BudgetTooSmall);
            };
            if slots_bytes > request.budget_bytes {
                return self.bind_error(request, event::Error::BudgetTooSmall);
            }
            if !detail::slot_storage_sufficient(
                &candidate,
                request.slot_storage,
                request.window_slots,
            ) {
                return self.bind_error(request, event::Error::SlotStorageTooSmall);
            }
            candidate.slot_count = request.window_slots;
            candidate.streaming_active = true;
        }
        candidate.source_bytes = request.file_size_bytes;
        candidate.budget_bytes = request.budget_bytes;
        candidate.prefetch_depth = request.prefetch_depth;
        candidate.stage_chunk_bytes = request.stage_chunk_bytes;
        candidate.bound = true;
        self.state = candidate;
        let result = event::BindDone::new(streaming, candidate.source_bytes, candidate.slot_count);
        if let Some(callback) = request.on_done {
            callback.publish(result);
        }
        Ok(result)
    }

    fn bind_error<'a>(
        &mut self,
        request: Bind<'a>,
        error: event::Error,
    ) -> Result<event::BindDone, event::BindError> {
        let result = event::BindError::new(error);
        if let Some(callback) = request.on_error {
            callback.publish(result);
        }
        Err(result)
    }

    /// Dispatches an acquire request and synchronously settles slot residency.
    pub fn acquire<'a>(
        &mut self,
        request: Acquire<'a>,
    ) -> Result<event::AcquireDone, event::AcquireError> {
        if !self.state.bound {
            return self.acquire_error(request, event::Error::NotBound);
        }
        if !self.state.streaming_active {
            return self.acquire_error(request, event::Error::NotStreaming);
        }
        if request.layer_index < 0
            || u32::try_from(request.layer_index)
                .ok()
                .is_none_or(|layer| layer >= self.state.layer_count)
        {
            return self.acquire_error(request, event::Error::LayerOutOfRange);
        }
        let slot = u32::try_from(request.layer_index).expect("validated non-negative layer")
            % self.state.slot_count;
        let result = event::AcquireDone::new(
            request.layer_index,
            slot,
            self.state.plan[usize::try_from(request.layer_index).expect("validated layer")],
        );
        if let Some(callback) = request.on_done {
            callback.publish(result);
        }
        Ok(result)
    }

    fn acquire_error<'a>(
        &mut self,
        request: Acquire<'a>,
        error: event::Error,
    ) -> Result<event::AcquireDone, event::AcquireError> {
        let result = event::AcquireError::new(error);
        if let Some(callback) = request.on_error {
            callback.publish(result);
        }
        Err(result)
    }

    /// Unbinds and resets the actor-owned lifecycle state.
    pub fn unbind<'a>(
        &mut self,
        request: Unbind<'a>,
    ) -> Result<event::UnbindDone, event::UnbindError> {
        if !self.state.bound {
            let result = event::UnbindError::new(event::Error::NotBound);
            if let Some(callback) = request.on_error {
                callback.publish(result);
            }
            return Err(result);
        }
        detail::reset(&mut self.state);
        let result = event::UnbindDone;
        if let Some(callback) = request.on_done {
            callback.publish(result);
        }
        Ok(result)
    }
}

/// Window actor variant with an injected synchronous staged-read actor.
/// Slot storage is caller-owned for the actor lifetime; this type never allocates it.
#[derive(Debug)]
pub struct WindowWithStager<'arena, S> {
    window: Window,
    stager: S,
    slots: &'arena mut [u8],
}

impl<'arena, S: Stager> WindowWithStager<'arena, S> {
    /// Creates a window over caller-provided reusable slot storage.
    #[must_use]
    pub const fn new(stager: S, slots: &'arena mut [u8]) -> Self {
        Self {
            window: Window::new(),
            stager,
            slots,
        }
    }

    pub fn bind<'a>(&mut self, request: Bind<'a>) -> Result<event::BindDone, event::BindError> {
        let result = self.window.bind(request);
        let Ok(done) = result else {
            return result;
        };
        if done.streaming_active()
            && (!detail::slot_storage_sufficient(
                &self.window.state,
                self.slots,
                done.window_slots(),
            ) || self.window.state.slot_capacity_bytes == 0)
        {
            let _ = self.window.unbind(Unbind::new());
            return Err(event::BindError::new(event::Error::SlotStorageTooSmall));
        }
        Ok(done)
    }

    /// Acquires a streamed layer and synchronously settles slot residency.
    pub fn acquire<'a>(
        &mut self,
        request: Acquire<'a>,
        source: Option<&'a [u8]>,
        target_bytes: &'a mut [u8],
    ) -> Result<event::AcquireDone, event::AcquireError> {
        if !self.window.state.bound {
            return self.window.acquire_error(request, event::Error::NotBound);
        }
        if !self.window.state.streaming_active {
            return self
                .window
                .acquire_error(request, event::Error::NotStreaming);
        }
        if request.layer_index < 0
            || u32::try_from(request.layer_index)
                .ok()
                .is_none_or(|layer| layer >= self.window.state.layer_count)
        {
            return self
                .window
                .acquire_error(request, event::Error::LayerOutOfRange);
        }
        let descriptor =
            self.window.state.plan[usize::try_from(request.layer_index).expect("validated layer")];
        let layer = u32::try_from(request.layer_index).expect("validated non-negative layer");
        let slot = layer % self.window.state.slot_count;
        let existing = self.window.state.slots[usize::try_from(slot).expect("slot is bounded")];
        if existing.layer == request.layer_index
            && existing.lifecycle == detail::SlotLifecycle::Resident
        {
            let result = event::AcquireDone::new(request.layer_index, slot, descriptor);
            if let Some(callback) = request.on_done {
                callback.publish(result);
            }
            return Ok(result);
        }
        self.acquire_slot(request, source, target_bytes, &descriptor, slot)
    }

    fn acquire_slot<'a>(
        &mut self,
        request: Acquire<'a>,
        source: Option<&'a [u8]>,
        _target_bytes: &'a mut [u8],
        descriptor: &detail::LayerDescriptor,
        slot: u32,
    ) -> Result<event::AcquireDone, event::AcquireError> {
        let Ok(slot_capacity) = usize::try_from(self.window.state.slot_capacity_bytes) else {
            return self
                .window
                .acquire_error(request, event::Error::SlotStorageTooSmall);
        };
        let Some(slot_start) = (slot as usize).checked_mul(slot_capacity) else {
            return self
                .window
                .acquire_error(request, event::Error::SlotStorageTooSmall);
        };
        let Ok(slot_bytes) = usize::try_from(descriptor.slot_bytes) else {
            return self
                .window
                .acquire_error(request, event::Error::SlotStorageTooSmall);
        };
        let Some(slot_end) = slot_start.checked_add(slot_bytes) else {
            return self
                .window
                .acquire_error(request, event::Error::SlotStorageTooSmall);
        };
        if self.slots.get(slot_start..slot_end).is_none() {
            return self
                .window
                .acquire_error(request, event::Error::SlotStorageTooSmall);
        }
        self.window.state.slots[slot as usize] = detail::WindowSlot {
            layer: request.layer_index,
            lifecycle: detail::SlotLifecycle::Loading,
        };
        for extent in descriptor
            .weights
            .iter()
            .copied()
            .take(descriptor.weight_count as usize)
        {
            let offset = usize::try_from(extent.slot_offset)
                .map_err(|_| event::AcquireError::new(event::Error::SlotStorageTooSmall));
            let length = usize::try_from(extent.byte_size)
                .map_err(|_| event::AcquireError::new(event::Error::SlotStorageTooSmall));
            let (Ok(offset), Ok(length)) = (offset, length) else {
                self.window.state.slots[slot as usize].lifecycle = detail::SlotLifecycle::Failed;
                return self
                    .window
                    .acquire_error(request, event::Error::SlotStorageTooSmall);
            };
            let Some(end) = offset.checked_add(length) else {
                self.window.state.slots[slot as usize].lifecycle = detail::SlotLifecycle::Failed;
                return self
                    .window
                    .acquire_error(request, event::Error::SlotStorageTooSmall);
            };
            let Some(target_slice) = self.slots.get_mut(slot_start + offset..slot_start + end)
            else {
                self.window.state.slots[slot as usize].lifecycle = detail::SlotLifecycle::Failed;
                return self
                    .window
                    .acquire_error(request, event::Error::SlotStorageTooSmall);
            };
            let extent_target = Target::new(target_slice);
            let stage_chunk = self.window.state.stage_chunk_bytes.min(extent.byte_size);
            let staged = emel_io::staged_read::event::StageWindow::new(
                extent.file_offset,
                extent.byte_size,
                stage_chunk,
                source,
                &extent_target,
            );
            let Ok(done) = self.stager.stage_tensor(staged) else {
                self.window.state.slots[slot as usize].lifecycle = detail::SlotLifecycle::Failed;
                return self
                    .window
                    .acquire_error(request, event::Error::SlotCopyFailed);
            };
            if done.bytes_committed() != extent.byte_size {
                self.window.state.slots[slot as usize].lifecycle = detail::SlotLifecycle::Failed;
                return self
                    .window
                    .acquire_error(request, event::Error::SlotCopyFailed);
            }
        }
        self.window.state.slots[slot as usize].lifecycle = detail::SlotLifecycle::Resident;
        self.window.state.next_prefetch_layer =
            detail::prefetch_layer(&self.window.state, request.layer_index).unwrap_or(-1);
        let result = event::AcquireDone::new(request.layer_index, slot, *descriptor);
        if let Some(callback) = request.on_done {
            callback.publish(result);
        }
        Ok(result)
    }

    /// Unbinds and resets lifecycle state while retaining caller-owned storage.
    pub fn unbind<'a>(
        &mut self,
        request: Unbind<'a>,
    ) -> Result<event::UnbindDone, event::UnbindError> {
        self.window.unbind(request)
    }

    /// Reads resident slot bytes without exposing mutable storage.
    #[must_use]
    pub fn slot_bytes(&self, slot: u32, length: usize) -> Option<&[u8]> {
        let slot_index = usize::try_from(slot).ok()?;
        if !self.window.state.bound
            || slot >= self.window.state.slot_count
            || self.window.state.slots.get(slot_index)?.lifecycle != detail::SlotLifecycle::Resident
        {
            return None;
        }
        let capacity = self.window.state.slot_capacity_bytes as usize;
        let start = slot_index.checked_mul(capacity)?;
        let end = start.checked_add(length)?;
        self.slots.get(start..end)
    }

    /// Returns the next layer marked for prefetch.
    #[must_use]
    pub const fn next_prefetch_layer(&self) -> Option<i32> {
        if self.window.state.next_prefetch_layer >= 0 {
            Some(self.window.state.next_prefetch_layer)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::dependency::Stager;
    use crate::tensor::window::detail::WeightExtent;

    #[derive(Debug, Default)]
    struct CopyStager;
    /// Allocation-free test storage whose base address satisfies the slot contract.
    #[repr(align(64))]
    #[derive(Debug)]
    struct AlignedBytes<const N: usize> {
        bytes: [u8; N],
    }

    impl<const N: usize> AlignedBytes<N> {
        const fn new() -> Self {
            Self { bytes: [0; N] }
        }

        fn as_mut_slice(&mut self) -> &mut [u8] {
            &mut self.bytes
        }
    }

    impl Stager for CopyStager {
        const AVAILABLE: bool = true;

        fn stage_tensor(
            &mut self,
            event: emel_io::staged_read::event::StageWindow<'_>,
        ) -> Result<emel_io::staged_read::event::StageWindowDone, emel_io::staged_read::event::Error>
        {
            let source = event
                .source()
                .ok_or(emel_io::staged_read::event::Error::NullSourceSpan)?;
            let start = event.file_offset() as usize;
            let end = start + event.logical_byte_length() as usize;
            event
                .target()
                .try_copy_from(&source[start..end])
                .map_err(|_| emel_io::staged_read::event::Error::InvalidTargetWindow)?;
            Ok(emel_io::staged_read::event::StageWindowDone::new(
                event.logical_byte_length(),
            ))
        }
    }

    #[derive(Debug)]
    struct FlakyStager {
        fail: bool,
    }

    impl Stager for FlakyStager {
        const AVAILABLE: bool = true;

        fn stage_tensor(
            &mut self,
            event: emel_io::staged_read::event::StageWindow<'_>,
        ) -> Result<emel_io::staged_read::event::StageWindowDone, emel_io::staged_read::event::Error>
        {
            if self.fail {
                self.fail = false;
                return Err(emel_io::staged_read::event::Error::InternalError);
            }
            CopyStager.stage_tensor(event)
        }
    }

    #[test]
    fn lifecycle_rejects_invalid_bind_and_unbound_acquire() {
        let mut window = Window::new();
        let mut storage = AlignedBytes::<64>::new();
        let invalid = Bind::new(0, &[], &[], 0, storage.as_mut_slice(), 0, 0);
        assert_eq!(
            window.bind(invalid).unwrap_err().error(),
            event::Error::InvalidRequest
        );
        assert_eq!(
            window.acquire(Acquire::new(0)).unwrap_err().error(),
            event::Error::NotBound
        );
    }

    #[test]
    fn fitting_bind_acquire_is_allocation_free_and_unbinds() {
        let mut window = Window::new();
        let mut storage = AlignedBytes::<64>::new();
        let extents = [WeightExtent {
            tensor_id: 7,
            file_offset: 8,
            byte_size: 8,
            slot_offset: 0,
        }];
        let done = window
            .bind(Bind::new(
                128,
                &extents,
                &[1],
                0,
                storage.as_mut_slice(),
                0,
                0,
            ))
            .unwrap();
        assert!(!done.streaming_active());
        assert_eq!(
            window.acquire(Acquire::new(0)).unwrap_err().error(),
            event::Error::NotStreaming
        );
        window.unbind(Unbind::new()).unwrap();
        assert_eq!(
            window.unbind(Unbind::new()).unwrap_err().error(),
            event::Error::NotBound
        );
    }

    #[test]
    fn streaming_bind_requires_slot_storage() {
        let mut window = Window::new();
        let mut storage = AlignedBytes::<127>::new();
        let extents = [
            WeightExtent {
                tensor_id: 7,
                file_offset: 8,
                byte_size: 65,
                slot_offset: 0,
            },
            WeightExtent {
                tensor_id: 8,
                file_offset: 80,
                byte_size: 65,
                slot_offset: 0,
            },
            WeightExtent {
                tensor_id: 9,
                file_offset: 152,
                byte_size: 65,
                slot_offset: 0,
            },
        ];
        let error = window
            .bind(Bind::new(
                384,
                &extents,
                &[1, 1, 1],
                256,
                storage.as_mut_slice(),
                2,
                1,
            ))
            .unwrap_err();
        assert_eq!(error.error(), event::Error::SlotStorageTooSmall);
    }

    #[test]
    fn streaming_bind_rejects_unaligned_slot_storage() {
        let extents = [
            WeightExtent {
                tensor_id: 7,
                file_offset: 8,
                byte_size: 65,
                slot_offset: 0,
            },
            WeightExtent {
                tensor_id: 8,
                file_offset: 80,
                byte_size: 65,
                slot_offset: 0,
            },
            WeightExtent {
                tensor_id: 9,
                file_offset: 152,
                byte_size: 65,
                slot_offset: 0,
            },
        ];
        let mut storage = AlignedBytes::<320>::new();
        let unaligned_storage = &mut storage.as_mut_slice()[1..];
        let error = Window::new()
            .bind(Bind::new(
                384,
                &extents,
                &[1, 1, 1],
                256,
                unaligned_storage,
                2,
                1,
            ))
            .unwrap_err();
        assert_eq!(error.error(), event::Error::SlotStorageTooSmall);
    }

    #[test]
    fn streaming_bind_classifies_budget_and_configuration_errors() {
        let extents = [
            WeightExtent {
                tensor_id: 7,
                file_offset: 8,
                byte_size: 65,
                slot_offset: 0,
            },
            WeightExtent {
                tensor_id: 8,
                file_offset: 80,
                byte_size: 65,
                slot_offset: 0,
            },
            WeightExtent {
                tensor_id: 9,
                file_offset: 152,
                byte_size: 65,
                slot_offset: 0,
            },
        ];
        let mut storage = AlignedBytes::<256>::new();
        let mut window = Window::new();
        let error = window
            .bind(Bind::new(
                256,
                &extents,
                &[1, 1, 1],
                1,
                storage.as_mut_slice(),
                2,
                1,
            ))
            .unwrap_err();
        assert_eq!(error.error(), event::Error::BudgetTooSmall);
        let mut window = Window::new();
        let error = window
            .bind(Bind::new(
                256,
                &extents,
                &[1, 1, 1],
                1,
                storage.as_mut_slice(),
                1,
                0,
            ))
            .unwrap_err();
        assert_eq!(error.error(), event::Error::InvalidRequest);
    }

    #[test]
    fn bind_reports_streaming_for_layer_larger_than_budget() {
        let extents = [
            WeightExtent {
                tensor_id: 7,
                file_offset: 8,
                byte_size: 65,
                slot_offset: 0,
            },
            WeightExtent {
                tensor_id: 8,
                file_offset: 80,
                byte_size: 65,
                slot_offset: 0,
            },
            WeightExtent {
                tensor_id: 9,
                file_offset: 152,
                byte_size: 65,
                slot_offset: 0,
            },
        ];
        let mut storage = AlignedBytes::<256>::new();
        let result = Window::new().bind(Bind::new(
            256,
            &extents,
            &[1, 1, 1],
            256,
            storage.as_mut_slice(),
            2,
            1,
        ));
        assert!(result.is_ok(), "unexpected bind result: {result:?}");
        assert!(result.unwrap().streaming_active());
    }

    #[test]
    fn injected_stager_copies_selected_extent_and_reports_result() {
        let mut slots = AlignedBytes::<512>::new();
        let mut window = WindowWithStager::new(CopyStager, slots.as_mut_slice());
        let extents = [
            WeightExtent {
                tensor_id: 7,
                file_offset: 2,
                byte_size: 4,
                slot_offset: 0,
            },
            WeightExtent {
                tensor_id: 8,
                file_offset: 2,
                byte_size: 4,
                slot_offset: 0,
            },
            WeightExtent {
                tensor_id: 9,
                file_offset: 2,
                byte_size: 4,
                slot_offset: 0,
            },
        ];
        let mut bind_storage = AlignedBytes::<128>::new();
        window
            .bind(Bind::new(
                16,
                &extents,
                &[1, 1, 1],
                128,
                bind_storage.as_mut_slice(),
                2,
                1,
            ))
            .unwrap();
        let source = [9u8, 8, 7, 6, 5, 4];
        let mut target = AlignedBytes::<64>::new();
        let result = window
            .acquire(Acquire::new(0), Some(&source), target.as_mut_slice())
            .unwrap();
        assert_eq!(result.layer_index(), 0);
        assert_eq!(window.slot_bytes(0, 4), Some(&[7, 6, 5, 4][..]));
        assert_eq!(window.next_prefetch_layer(), Some(1));
    }

    #[test]
    fn resident_acquire_reuses_slot_without_second_copy() {
        let mut slots = AlignedBytes::<128>::new();
        let mut window = WindowWithStager::new(CopyStager, slots.as_mut_slice());
        let extents = [
            WeightExtent {
                tensor_id: 7,
                file_offset: 2,
                byte_size: 4,
                slot_offset: 0,
            },
            WeightExtent {
                tensor_id: 8,
                file_offset: 2,
                byte_size: 4,
                slot_offset: 0,
            },
            WeightExtent {
                tensor_id: 9,
                file_offset: 2,
                byte_size: 4,
                slot_offset: 0,
            },
        ];
        let mut bind_storage = AlignedBytes::<128>::new();
        window
            .bind(Bind::new(
                16,
                &extents,
                &[1, 1, 1],
                128,
                bind_storage.as_mut_slice(),
                2,
                1,
            ))
            .unwrap();
        let mut source = [9u8, 8, 7, 6, 5, 4];
        let mut target = AlignedBytes::<64>::new();
        window
            .acquire(Acquire::new(0), Some(&source), target.as_mut_slice())
            .unwrap();
        source[2..].copy_from_slice(&[1, 1, 1, 1]);
        window
            .acquire(Acquire::new(0), Some(&source), target.as_mut_slice())
            .unwrap();
        assert_eq!(window.slot_bytes(0, 4), Some(&[7, 6, 5, 4][..]));
    }

    #[test]
    fn unbind_clears_residency_and_allows_clean_rebind() {
        let mut slots = AlignedBytes::<128>::new();
        let mut window = WindowWithStager::new(CopyStager, slots.as_mut_slice());
        let extents = [
            WeightExtent {
                tensor_id: 7,
                file_offset: 2,
                byte_size: 4,
                slot_offset: 0,
            },
            WeightExtent {
                tensor_id: 8,
                file_offset: 2,
                byte_size: 4,
                slot_offset: 0,
            },
            WeightExtent {
                tensor_id: 9,
                file_offset: 2,
                byte_size: 4,
                slot_offset: 0,
            },
        ];
        let mut bind_storage = AlignedBytes::<128>::new();
        window
            .bind(Bind::new(
                16,
                &extents,
                &[1, 1, 1],
                128,
                bind_storage.as_mut_slice(),
                2,
                1,
            ))
            .unwrap();
        let source = [9u8, 8, 7, 6, 5, 4];
        let mut target = AlignedBytes::<64>::new();
        window
            .acquire(Acquire::new(0), Some(&source), target.as_mut_slice())
            .unwrap();
        window.unbind(Unbind::new()).unwrap();
        assert_eq!(window.slot_bytes(0, 4), None);
        assert_eq!(
            window
                .acquire(Acquire::new(0), Some(&source), target.as_mut_slice())
                .unwrap_err()
                .error(),
            event::Error::NotBound
        );
        window
            .bind(Bind::new(
                16,
                &extents,
                &[1, 1, 1],
                128,
                bind_storage.as_mut_slice(),
                2,
                1,
            ))
            .unwrap();
    }

    #[test]
    fn failed_slot_retries_and_commits_on_next_acquire() {
        let mut slots = AlignedBytes::<128>::new();
        let mut window = WindowWithStager::new(FlakyStager { fail: true }, slots.as_mut_slice());
        let extents = [
            WeightExtent {
                tensor_id: 7,
                file_offset: 2,
                byte_size: 4,
                slot_offset: 0,
            },
            WeightExtent {
                tensor_id: 8,
                file_offset: 2,
                byte_size: 4,
                slot_offset: 0,
            },
            WeightExtent {
                tensor_id: 9,
                file_offset: 2,
                byte_size: 4,
                slot_offset: 0,
            },
        ];
        let mut bind_storage = AlignedBytes::<128>::new();
        window
            .bind(Bind::new(
                16,
                &extents,
                &[1, 1, 1],
                128,
                bind_storage.as_mut_slice(),
                2,
                1,
            ))
            .unwrap();
        let source = [9u8, 8, 7, 6, 5, 4];
        let mut target = AlignedBytes::<64>::new();
        assert_eq!(
            window
                .acquire(Acquire::new(0), Some(&source), target.as_mut_slice())
                .unwrap_err()
                .error(),
            event::Error::SlotCopyFailed
        );
        window
            .acquire(Acquire::new(0), Some(&source), target.as_mut_slice())
            .unwrap();
        assert_eq!(window.slot_bytes(0, 4), Some(&[7, 6, 5, 4][..]));
    }
}
