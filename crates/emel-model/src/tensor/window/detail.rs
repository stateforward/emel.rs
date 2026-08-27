//! Bounded tensor-window data owned by the window actor.

#![allow(clippy::redundant_pub_crate)]
#![allow(clippy::large_stack_arrays)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::field_reassign_with_default)]
#![allow(clippy::missing_const_for_fn)]

pub(crate) const MAX_WINDOW_SLOTS: usize = 8;
pub(crate) const MAX_STREAM_LAYERS: usize = 160;
pub(crate) const MAX_WEIGHTS_PER_LAYER: usize = 12;
pub(crate) const MIN_STREAM_CHUNK_BYTES: u64 = 1_024 * 1_024;
pub(crate) const DEFAULT_STREAM_CHUNK_BYTES: u64 = 8 * 1_024 * 1_024;
pub(crate) const MAX_STREAM_CHUNK_BYTES: u64 = 16 * 1_024 * 1_024;
pub(crate) const SLOT_ALIGNMENT_BYTES: u64 = 64;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct WeightExtent {
    pub(crate) tensor_id: i32,
    pub(crate) file_offset: u64,
    pub(crate) byte_size: u64,
    pub(crate) slot_offset: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct LayerDescriptor {
    pub(crate) weights: [WeightExtent; MAX_WEIGHTS_PER_LAYER],
    pub(crate) weight_count: u32,
    pub(crate) file_begin: u64,
    pub(crate) file_span: u64,
    pub(crate) slot_bytes: u64,
}

impl Default for LayerDescriptor {
    fn default() -> Self {
        Self {
            weights: [WeightExtent::default(); MAX_WEIGHTS_PER_LAYER],
            weight_count: 0,
            file_begin: 0,
            file_span: 0,
            slot_bytes: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum SlotLifecycle {
    #[default]
    Vacant,
    Loading,
    Resident,
    Failed,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct WindowSlot {
    pub(crate) layer: i32,
    pub(crate) lifecycle: SlotLifecycle,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct WindowState {
    pub(crate) plan: [LayerDescriptor; MAX_STREAM_LAYERS],
    pub(crate) layer_count: u32,
    pub(crate) slots: [WindowSlot; MAX_WINDOW_SLOTS],
    pub(crate) slot_count: u32,
    pub(crate) slot_capacity_bytes: u64,
    pub(crate) total_stream_bytes: u64,
    pub(crate) source_bytes: u64,
    pub(crate) budget_bytes: u64,
    pub(crate) prefetch_depth: u32,
    pub(crate) stage_chunk_bytes: u64,
    pub(crate) next_prefetch_layer: i32,
    pub(crate) streaming_active: bool,
    pub(crate) bound: bool,
}

impl Default for WindowState {
    fn default() -> Self {
        Self {
            plan: [LayerDescriptor::default(); MAX_STREAM_LAYERS],
            layer_count: 0,
            slots: [WindowSlot::default(); MAX_WINDOW_SLOTS],
            slot_count: 0,
            slot_capacity_bytes: 0,
            total_stream_bytes: 0,
            source_bytes: 0,
            budget_bytes: 0,
            prefetch_depth: 0,
            stage_chunk_bytes: DEFAULT_STREAM_CHUNK_BYTES,
            next_prefetch_layer: -1,
            streaming_active: false,
            bound: false,
        }
    }
}

pub(crate) const fn aligned_bytes(bytes: u64) -> Option<u64> {
    match bytes.checked_add(SLOT_ALIGNMENT_BYTES - 1) {
        Some(value) => Some(value & !(SLOT_ALIGNMENT_BYTES - 1)),
        None => None,
    }
}

pub(crate) fn reset(window: &mut WindowState) {
    *window = WindowState::default();
}

pub(crate) fn prefetch_layer(window: &WindowState, published_layer: i32) -> Option<i32> {
    if !window.streaming_active || window.layer_count == 0 || window.slot_count == 0 {
        return None;
    }
    let count = window.layer_count as i32;
    let candidate = (published_layer + window.prefetch_depth as i32).rem_euclid(count);
    let published_slot = published_layer.rem_euclid(count) as u32 % window.slot_count;
    let candidate_slot = candidate as u32 % window.slot_count;
    if candidate_slot == published_slot && window.slots[candidate_slot as usize].layer >= 0 {
        return None;
    }
    let slot = window.slots[candidate_slot as usize];
    if slot.layer == candidate
        && matches!(
            slot.lifecycle,
            SlotLifecycle::Loading | SlotLifecycle::Resident
        )
    {
        return None;
    }
    Some(candidate)
}

pub(crate) fn scan_layer_descriptors(
    extents: &[WeightExtent],
    layer_weight_counts: &[u16],
    window: &mut WindowState,
) -> bool {
    if layer_weight_counts.len() > MAX_STREAM_LAYERS
        || extents.len() > MAX_STREAM_LAYERS * MAX_WEIGHTS_PER_LAYER
    {
        return false;
    }
    let mut cursor = 0usize;
    let mut maximum = 0u64;
    let mut total_stream_bytes = 0u64;
    for (layer, &count) in layer_weight_counts.iter().enumerate() {
        let count = count as usize;
        if count > MAX_WEIGHTS_PER_LAYER || cursor.checked_add(count).is_none() {
            return false;
        }
        let end = cursor + count;
        if end > extents.len() {
            return false;
        }
        let mut descriptor = LayerDescriptor::default();
        descriptor.weight_count = count as u32;
        let mut offset = 0u64;
        let mut file_begin = u64::MAX;
        let mut file_end = 0u64;
        for (index, source) in extents[cursor..end].iter().copied().enumerate() {
            let Some(aligned) = aligned_bytes(source.byte_size) else {
                return false;
            };
            let Some(next_offset) = offset.checked_add(aligned) else {
                return false;
            };
            let Some(next_file_end) = source.file_offset.checked_add(source.byte_size) else {
                return false;
            };
            let mut extent = source;
            extent.slot_offset = offset;
            descriptor.weights[index] = extent;
            offset = next_offset;
            file_begin = file_begin.min(source.file_offset);
            file_end = file_end.max(next_file_end);
        }
        descriptor.file_begin = if count == 0 { 0 } else { file_begin };
        descriptor.file_span = if count == 0 { 0 } else { file_end - file_begin };
        descriptor.slot_bytes = offset;
        maximum = maximum.max(offset);
        let Some(total) = total_stream_bytes.checked_add(offset) else {
            return false;
        };
        total_stream_bytes = total;
        window.plan[layer] = descriptor;
        cursor = end;
    }
    if cursor != extents.len() {
        return false;
    }
    window.layer_count = layer_weight_counts.len() as u32;
    window.slot_capacity_bytes = maximum;
    window.total_stream_bytes = total_stream_bytes;
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_constants_and_layout_are_bounded() {
        assert_eq!(MAX_WINDOW_SLOTS, 8);
        assert_eq!(MAX_STREAM_LAYERS, 160);
        assert_eq!(MAX_WEIGHTS_PER_LAYER, 12);
        assert_eq!(aligned_bytes(1), Some(64));
    }

    #[test]
    fn scan_computes_offsets_and_rejects_inconsistent_input() {
        let extents = [
            WeightExtent {
                tensor_id: 1,
                file_offset: 10,
                byte_size: 65,
                slot_offset: 0,
            },
            WeightExtent {
                tensor_id: 2,
                file_offset: 100,
                byte_size: 8,
                slot_offset: 0,
            },
        ];
        let mut window = WindowState::default();
        assert!(scan_layer_descriptors(&extents, &[2], &mut window));
        assert_eq!(window.plan[0].weights[1].slot_offset, 128);
        assert_eq!(window.plan[0].slot_bytes, 192);
        assert!(!scan_layer_descriptors(&extents, &[1], &mut window));
    }

    #[test]
    fn prefetch_wraps_and_never_targets_published_slot() {
        let mut window = WindowState::default();
        window.streaming_active = true;
        window.layer_count = 5;
        window.slot_count = 4;
        window.prefetch_depth = 2;
        assert_eq!(prefetch_layer(&window, 4), Some(1));
        window.slot_count = 3;
        window.prefetch_depth = 2;
        assert_eq!(prefetch_layer(&window, 4), None);
    }
}
