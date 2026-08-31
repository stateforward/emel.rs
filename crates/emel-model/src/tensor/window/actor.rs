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

    /// Dispatches a bind request synchronously.
    pub fn bind<'a>(&mut self, request: Bind<'a>) -> Result<event::BindDone, event::BindError> {
        if self.state.bound {
            return self.bind_error(request, event::Error::AlreadyBound);
        }
        if request.file_size_bytes == 0
            || request.window_slots as usize > detail::MAX_WINDOW_SLOTS
            || request.window_slots == 0 && request.budget_bytes != 0
            || request.stage_chunk_bytes < detail::MIN_STREAM_CHUNK_BYTES
            || request.stage_chunk_bytes > detail::MAX_STREAM_CHUNK_BYTES
            || !detail::scan_layer_descriptors(
                request.extents,
                request.layer_weight_counts,
                &mut self.state,
            )
        {
            return self.bind_error(request, event::Error::InvalidRequest);
        }
        self.state.source_bytes = request.file_size_bytes;
        self.state.budget_bytes = request.budget_bytes;
        self.state.prefetch_depth = request.prefetch_depth;
        self.state.stage_chunk_bytes = request.stage_chunk_bytes;
        let streaming =
            request.budget_bytes != 0 && self.state.total_stream_bytes > request.budget_bytes;
        if streaming {
            if request.window_slots < 2
                || request.window_slots as usize > detail::MAX_WINDOW_SLOTS
                || request.prefetch_depth == 0
                || request.prefetch_depth >= request.window_slots
            {
                return self.bind_error(request, event::Error::InvalidRequest);
            }
            let slots_bytes = request.window_slots as u64 * self.state.slot_capacity_bytes;
            if slots_bytes > request.budget_bytes {
                return self.bind_error(request, event::Error::BudgetTooSmall);
            }
            if request.slot_storage.len() < slots_bytes as usize {
                return self.bind_error(request, event::Error::SlotStorageTooSmall);
            }
            self.state.slot_count = request.window_slots;
            self.state.streaming_active = true;
        }
        self.state.bound = true;
        let result =
            event::BindDone::new(streaming, self.state.source_bytes, self.state.slot_count);
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

    /// Dispatches an acquire request. Residency loading is added by the staged-I/O slice.
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
        if request.layer_index < 0 || request.layer_index as u32 >= self.state.layer_count {
            return self.acquire_error(request, event::Error::LayerOutOfRange);
        }
        let slot = request.layer_index as u32 % self.state.slot_count;
        let result = event::AcquireDone::new(
            request.layer_index,
            slot,
            self.state.plan[request.layer_index as usize],
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
    pub fn new(stager: S, slots: &'arena mut [u8]) -> Self {
        Self { window: Window::new(), stager, slots }
    }

    /// Binds using the core lifecycle actor without allocating storage.
    pub fn bind<'a>(&mut self, request: Bind<'a>) -> Result<event::BindDone, event::BindError> {
        let result = self.window.bind(request);
        let Ok(done) = result else { return result };
        if done.streaming_active() {
            let Some(needed) = done.window_slots().checked_mul(self.window.state.slot_capacity_bytes as u32) else {
                let _ = self.window.unbind(Unbind::new());
                return Err(event::BindError::new(event::Error::SlotStorageTooSmall));
            };
            if needed == 0 || needed as usize > self.slots.len() {
                let _ = self.window.unbind(Unbind::new());
                return Err(event::BindError::new(event::Error::SlotStorageTooSmall));
            }
        }
        Ok(done)
    }

    /// Acquires a layer and stages each extent into caller-owned storage.
    /// `target_bytes` is validated before any staged write.
    pub fn acquire<'a>(&mut self, request: Acquire<'a>, source: Option<&'a [u8]>, target_bytes: &'a mut [u8]) -> Result<event::AcquireDone, event::AcquireError> {
        if !self.window.state.bound { return self.window.acquire_error(request, event::Error::NotBound); }
        if !self.window.state.streaming_active { return self.window.acquire_error(request, event::Error::NotStreaming); }
        if request.layer_index < 0 || request.layer_index as u32 >= self.window.state.layer_count { return self.window.acquire_error(request, event::Error::LayerOutOfRange); }
        let descriptor = self.window.state.plan[request.layer_index as usize];
        if target_bytes.len() < descriptor.slot_bytes as usize { return self.window.acquire_error(request, event::Error::SlotStorageTooSmall); }
        let slot = request.layer_index as u32 % self.window.state.slot_count;
        let existing = self.window.state.slots[slot as usize];
        if existing.layer == request.layer_index && existing.lifecycle == detail::SlotLifecycle::Resident {
            let result = event::AcquireDone::new(request.layer_index, slot, descriptor);
            if let Some(callback) = request.on_done { callback.publish(result); }
            return Ok(result);
        }
        self.window.state.slots[slot as usize].layer = request.layer_index;
        self.window.state.slots[slot as usize].lifecycle = detail::SlotLifecycle::Loading;
        let slot_start = slot as usize * self.window.state.slot_capacity_bytes as usize;
        let slot_end = slot_start + descriptor.slot_bytes as usize;
        let target = Target::new(&mut self.slots[slot_start..slot_end]);
        for index in 0..descriptor.weight_count as usize {
            let extent = descriptor.weights[index];
            let staged = emel_io::staged_read::event::StageWindow::new(extent.file_offset, extent.byte_size, self.window.state.stage_chunk_bytes, source, &target);
            if self.stager.stage_tensor(staged).is_err() {
                self.window.state.slots[slot as usize].lifecycle = detail::SlotLifecycle::Failed;
                return self.window.acquire_error(request, event::Error::SlotCopyFailed);
            }
        }
        self.window.state.slots[slot as usize].lifecycle = detail::SlotLifecycle::Resident;
        if let Some(prefetch) = detail::prefetch_layer(&self.window.state, request.layer_index) { self.window.state.next_prefetch_layer = prefetch; }
        let result = event::AcquireDone::new(request.layer_index, slot, descriptor);
        if let Some(callback) = request.on_done { callback.publish(result); }
        Ok(result)
    }

    /// Unbinds and resets lifecycle state while retaining caller-owned storage.
    pub fn unbind<'a>(&mut self, request: Unbind<'a>) -> Result<event::UnbindDone, event::UnbindError> {
        self.window.unbind(request)
    }

    /// Reads resident slot bytes without exposing mutable storage.
    #[must_use]
    pub fn slot_bytes(&self, slot: u32, length: usize) -> Option<&[u8]> {
        let slot_index = usize::try_from(slot).ok()?;
        if !self.window.state.bound
            || slot >= self.window.state.slot_count
            || self.window.state.slots.get(slot_index)?.lifecycle
                != detail::SlotLifecycle::Resident
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
        if self.window.state.next_prefetch_layer >= 0 { Some(self.window.state.next_prefetch_layer) } else { None }
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::dependency::Stager;
    use crate::tensor::window::detail::WeightExtent;

    #[derive(Debug, Default)]
    struct CopyStager;

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
        let mut storage = [0u8; 64];
        let invalid = Bind::new(0, &[], &[], 0, &mut storage, 0, 0);
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
        let mut storage = [0u8; 64];
        let extents = [WeightExtent {
            tensor_id: 7,
            file_offset: 8,
            byte_size: 8,
            slot_offset: 0,
        }];
        let done = window
            .bind(Bind::new(128, &extents, &[1], 0, &mut storage, 0, 0))
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
        let mut storage = [0u8; 127];
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
                &mut storage,
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
        let mut storage = [0u8; 256];
        let mut window = Window::new();
        let error = window
            .bind(Bind::new(256, &extents, &[1, 1, 1], 1, &mut storage, 2, 1))
            .unwrap_err();
        assert_eq!(error.error(), event::Error::BudgetTooSmall);
        let mut window = Window::new();
        let error = window
            .bind(Bind::new(256, &extents, &[1, 1, 1], 1, &mut storage, 1, 0))
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
        let mut storage = [0u8; 256];
        let result = Window::new().bind(Bind::new(
            256,
            &extents,
            &[1, 1, 1],
            256,
            &mut storage,
            2,
            1,
        ));
        assert!(result.is_ok(), "unexpected bind result: {result:?}");
        assert!(result.unwrap().streaming_active());
    }

    #[test]
    fn injected_stager_copies_selected_extent_and_reports_result() {
        let mut slots = [0u8; 512];
        let mut window = WindowWithStager::new(CopyStager, &mut slots);
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
        window
            .bind(Bind::new(16, &extents, &[1, 1, 1], 128, &mut slots, 2, 1))
            .unwrap();
        let source = [9u8, 8, 7, 6, 5, 4];
        let mut target = [0u8; 64];
        let result = window
            .acquire(Acquire::new(0), Some(&source), &mut target)
            .unwrap();
        assert_eq!(result.layer_index(), 0);
        assert_eq!(window.slot_bytes(0, 4), Some(&[7, 6, 5, 4][..]));
        assert_eq!(window.next_prefetch_layer(), Some(1));
    }

    #[test]
    fn resident_acquire_reuses_slot_without_second_copy() {
        let mut slots = [0u8; 128];
        let mut window = WindowWithStager::new(CopyStager, &mut slots);
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
        window
            .bind(Bind::new(16, &extents, &[1, 1, 1], 128, &mut slots, 2, 1))
            .unwrap();
        let mut source = [9u8, 8, 7, 6, 5, 4];
        let mut target = [0u8; 64];
        window
            .acquire(Acquire::new(0), Some(&source), &mut target)
            .unwrap();
        source[2..].copy_from_slice(&[1, 1, 1, 1]);
        window
            .acquire(Acquire::new(0), Some(&source), &mut target)
            .unwrap();
        assert_eq!(window.slot_bytes(0, 4), Some(&[7, 6, 5, 4][..]));
    }

    #[test]
    fn unbind_clears_residency_and_allows_clean_rebind() {
        let mut slots = [0u8; 128];
        let mut window = WindowWithStager::new(CopyStager, &mut slots);
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
        window
            .bind(Bind::new(16, &extents, &[1, 1, 1], 128, &mut slots, 2, 1))
            .unwrap();
        let source = [9u8, 8, 7, 6, 5, 4];
        let mut target = [0u8; 64];
        window
            .acquire(Acquire::new(0), Some(&source), &mut target)
            .unwrap();
        window.unbind(Unbind::new()).unwrap();
        assert_eq!(window.slot_bytes(0, 4), None);
        assert_eq!(
            window
                .acquire(Acquire::new(0), Some(&source), &mut target)
                .unwrap_err()
                .error(),
            event::Error::NotBound
        );
        window
            .bind(Bind::new(16, &extents, &[1, 1, 1], 128, &mut slots, 2, 1))
            .unwrap();
    }

    #[test]
    fn failed_slot_retries_and_commits_on_next_acquire() {
        let mut slots = [0u8; 128];
        let mut window = WindowWithStager::new(FlakyStager { fail: true }, &mut slots);
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
        window
            .bind(Bind::new(16, &extents, &[1, 1, 1], 128, &mut slots, 2, 1))
            .unwrap();
        let source = [9u8, 8, 7, 6, 5, 4];
        let mut target = [0u8; 64];
        assert_eq!(
            window
                .acquire(Acquire::new(0), Some(&source), &mut target)
                .unwrap_err()
                .error(),
            event::Error::SlotCopyFailed
        );
        window
            .acquire(Acquire::new(0), Some(&source), &mut target)
            .unwrap();
        assert_eq!(window.slot_bytes(0, 4), Some(&[7, 6, 5, 4][..]));
    }
}
