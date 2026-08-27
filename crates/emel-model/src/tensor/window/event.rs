//! Typed tensor-window requests and synchronous outcomes.

#![allow(clippy::enum_variant_names)]
#![allow(clippy::large_types_passed_by_value)]

use core::cell::Cell;

use super::detail::{DEFAULT_STREAM_CHUNK_BYTES, WeightExtent};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum Error {
    InvalidRequest,
    NotBound,
    AlreadyBound,
    SourceMapFailed,
    BudgetTooSmall,
    LayerOutOfRange,
    SlotCopyFailed,
    NotStreaming,
    InternalError,
    SlotStorageTooSmall,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BindDone {
    streaming_active: bool,
    source_bytes: u64,
    window_slots: u32,
}

impl BindDone {
    pub(crate) const fn new(streaming_active: bool, source_bytes: u64, window_slots: u32) -> Self {
        Self {
            streaming_active,
            source_bytes,
            window_slots,
        }
    }

    #[must_use]
    pub const fn streaming_active(self) -> bool {
        self.streaming_active
    }

    #[must_use]
    pub const fn source_bytes(self) -> u64 {
        self.source_bytes
    }

    #[must_use]
    pub const fn window_slots(self) -> u32 {
        self.window_slots
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BindError {
    error: Error,
}

impl BindError {
    pub(crate) const fn new(error: Error) -> Self {
        Self { error }
    }

    #[must_use]
    pub const fn error(self) -> Error {
        self.error
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AcquireDone {
    layer_index: i32,
    slot_index: u32,
    layout: super::detail::LayerDescriptor,
}

impl AcquireDone {
    pub(crate) const fn new(
        layer_index: i32,
        slot_index: u32,
        layout: super::detail::LayerDescriptor,
    ) -> Self {
        Self {
            layer_index,
            slot_index,
            layout,
        }
    }

    #[must_use]
    pub const fn layer_index(self) -> i32 {
        self.layer_index
    }

    #[must_use]
    pub const fn slot_index(self) -> u32 {
        self.slot_index
    }

    #[must_use]
    pub const fn layout(self) -> super::detail::LayerDescriptor {
        self.layout
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AcquireError {
    error: Error,
}

impl AcquireError {
    pub(crate) const fn new(error: Error) -> Self {
        Self { error }
    }

    #[must_use]
    pub const fn error(self) -> Error {
        self.error
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UnbindDone;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UnbindError {
    error: Error,
}

impl UnbindError {
    pub(crate) const fn new(error: Error) -> Self {
        Self { error }
    }

    #[must_use]
    pub const fn error(self) -> Error {
        self.error
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Callback<'a, T: Copy> {
    slot: &'a Cell<Option<T>>,
}

impl<'a, T: Copy> Callback<'a, T> {
    #[must_use]
    pub const fn store(slot: &'a Cell<Option<T>>) -> Self {
        Self { slot }
    }

    pub(crate) fn publish(self, value: T) {
        self.slot.set(Some(value));
    }
}

#[derive(Debug)]
pub struct Bind<'a> {
    pub(crate) file_size_bytes: u64,
    pub(crate) extents: &'a [WeightExtent],
    pub(crate) layer_weight_counts: &'a [u16],
    pub(crate) budget_bytes: u64,
    pub(crate) slot_storage: &'a mut [u8],
    pub(crate) window_slots: u32,
    pub(crate) prefetch_depth: u32,
    pub(crate) stage_chunk_bytes: u64,
    pub(crate) on_done: Option<Callback<'a, BindDone>>,
    pub(crate) on_error: Option<Callback<'a, BindError>>,
}

impl<'a> Bind<'a> {
    #[must_use]
    pub const fn new(
        file_size_bytes: u64,
        extents: &'a [WeightExtent],
        layer_weight_counts: &'a [u16],
        budget_bytes: u64,
        slot_storage: &'a mut [u8],
        window_slots: u32,
        prefetch_depth: u32,
    ) -> Self {
        Self {
            file_size_bytes,
            extents,
            layer_weight_counts,
            budget_bytes,
            slot_storage,
            window_slots,
            prefetch_depth,
            stage_chunk_bytes: DEFAULT_STREAM_CHUNK_BYTES,
            on_done: None,
            on_error: None,
        }
    }

    #[must_use]
    pub const fn stage_chunk_bytes(mut self, bytes: u64) -> Self {
        self.stage_chunk_bytes = bytes;
        self
    }

    #[must_use]
    pub const fn on_done(mut self, callback: Callback<'a, BindDone>) -> Self {
        self.on_done = Some(callback);
        self
    }

    #[must_use]
    pub const fn on_error(mut self, callback: Callback<'a, BindError>) -> Self {
        self.on_error = Some(callback);
        self
    }
}

#[derive(Debug)]
pub struct Acquire<'a> {
    pub(crate) layer_index: i32,
    pub(crate) on_done: Option<Callback<'a, AcquireDone>>,
    pub(crate) on_error: Option<Callback<'a, AcquireError>>,
}

impl<'a> Acquire<'a> {
    #[must_use]
    pub const fn new(layer_index: i32) -> Self {
        Self {
            layer_index,
            on_done: None,
            on_error: None,
        }
    }

    #[must_use]
    pub const fn on_done(mut self, callback: Callback<'a, AcquireDone>) -> Self {
        self.on_done = Some(callback);
        self
    }

    #[must_use]
    pub const fn on_error(mut self, callback: Callback<'a, AcquireError>) -> Self {
        self.on_error = Some(callback);
        self
    }
}

#[derive(Debug)]
pub struct Unbind<'a> {
    pub(crate) on_done: Option<Callback<'a, UnbindDone>>,
    pub(crate) on_error: Option<Callback<'a, UnbindError>>,
}

impl<'a> Unbind<'a> {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            on_done: None,
            on_error: None,
        }
    }

    #[must_use]
    pub const fn on_done(mut self, callback: Callback<'a, UnbindDone>) -> Self {
        self.on_done = Some(callback);
        self
    }

    #[must_use]
    pub const fn on_error(mut self, callback: Callback<'a, UnbindError>) -> Self {
        self.on_error = Some(callback);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::{Acquire, Bind, Callback, Unbind};
    use core::cell::Cell;

    #[test]
    fn event_builders_preserve_all_request_configuration() {
        let mut storage = [0u8; 64];
        let bind_done = Cell::new(None);
        let bind_error = Cell::new(None);
        let bind = Bind::new(1024, &[], &[2, 3], 512, &mut storage, 2, 1)
            .stage_chunk_bytes(128)
            .on_done(Callback::store(&bind_done))
            .on_error(Callback::store(&bind_error));
        assert_eq!(bind.file_size_bytes, 1024);
        assert_eq!(bind.budget_bytes, 512);
        assert_eq!(bind.window_slots, 2);
        assert_eq!(bind.prefetch_depth, 1);
        assert_eq!(bind.stage_chunk_bytes, 128);
        assert!(bind.on_done.is_some());
        assert!(bind.on_error.is_some());

        let acquire_done = Cell::new(None);
        let acquire_error = Cell::new(None);
        let acquire = Acquire::new(7)
            .on_done(Callback::store(&acquire_done))
            .on_error(Callback::store(&acquire_error));
        assert_eq!(acquire.layer_index, 7);
        assert!(acquire.on_done.is_some());
        assert!(acquire.on_error.is_some());

        let unbind_done = Cell::new(None);
        let unbind_error = Cell::new(None);
        let unbind = Unbind::new()
            .on_done(Callback::store(&unbind_done))
            .on_error(Callback::store(&unbind_error));
        assert!(unbind.on_done.is_some());
        assert!(unbind.on_error.is_some());
    }
}
