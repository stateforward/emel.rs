//! Source-aligned `MemoryKv` state machine port.

#![allow(
    clippy::enum_variant_names,
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::missing_const_for_fn,
    dead_code,
    missing_docs
)]

use core::cell::RefCell;
use sml::sml;

pub const DEFAULT_BLOCK_TOKENS: i32 = 16;
pub const MAX_SEQUENCES: usize = 256;
pub const MAX_BLOCKS: usize = 32_768;
pub const MAX_BLOCKS_PER_SEQUENCE: usize = 4_096;
const MAX_SEQUENCES_I32: i32 = 256;
const MAX_BLOCKS_I32: i32 = 32_768;
const MAX_BLOCKS_PER_SEQUENCE_I32: i32 = 4_096;

#[repr(i32)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum KvError {
    #[default]
    None = 0,
    InvalidRequest = 1 << 0,
    BackendError = 1 << 1,
    InternalError = 1 << 2,
    OutOfMemory = 1 << 3,
    Untracked = 1 << 4,
}

impl KvError {
    const fn code(self) -> i32 {
        self as i32
    }
}

#[derive(Debug)]
pub struct MemoryKvContext {
    pub max_sequences: i32,
    pub max_blocks: i32,
    pub block_tokens: i32,
    pub block_refs: Box<[u16]>,
    pub sequence_active: Box<[bool]>,
    pub sequence_length: Box<[i32]>,
    pub sequence_block_count: Box<[i32]>,
    pub seq_to_blocks: Box<[[u16; MAX_BLOCKS_PER_SEQUENCE]]>,
    pub free_stack: Box<[u16]>,
    pub free_count: i32,
}

impl Default for MemoryKvContext {
    fn default() -> Self {
        Self {
            max_sequences: MAX_SEQUENCES_I32,
            max_blocks: MAX_BLOCKS_I32,
            block_tokens: DEFAULT_BLOCK_TOKENS,
            block_refs: vec![0; MAX_BLOCKS].into_boxed_slice(),
            sequence_active: vec![false; MAX_SEQUENCES].into_boxed_slice(),
            sequence_length: vec![0; MAX_SEQUENCES].into_boxed_slice(),
            sequence_block_count: vec![0; MAX_SEQUENCES].into_boxed_slice(),
            seq_to_blocks: vec![[0; MAX_BLOCKS_PER_SEQUENCE]; MAX_SEQUENCES].into_boxed_slice(),
            free_stack: vec![0; MAX_BLOCKS].into_boxed_slice(),
            free_count: 0,
        }
    }
}

fn index_in(value: i32, bound: usize) -> Option<usize> {
    usize::try_from(value).ok().filter(|index| *index < bound)
}

fn bounded_count(value: i32, maximum: i32) -> Option<usize> {
    (0..=maximum)
        .contains(&value)
        .then(|| usize::try_from(value).ok())
        .flatten()
}

fn free_stack_block(ctx: &MemoryKvContext, stack_index: i32) -> Option<u16> {
    index_in(stack_index, MAX_BLOCKS).and_then(|index| ctx.free_stack.get(index).copied())
}

fn valid_sequence_id(ctx: &MemoryKvContext, seq_id: i32) -> bool {
    seq_id >= 0 && seq_id < ctx.max_sequences && index_in(seq_id, MAX_SEQUENCES).is_some()
}

fn blocks_for_length(block_tokens: i32, token_count: i32) -> i32 {
    if block_tokens <= 0 || token_count <= 0 {
        return 0;
    }
    let wide = (i64::from(token_count) + i64::from(block_tokens) - 1) / i64::from(block_tokens);
    i32::try_from(wide).expect("positive i32 token geometry always fits an i32 block count")
}

fn resolved_or_default(value: i32, fallback: i32) -> i32 {
    if value > 0 { value } else { fallback }
}

fn set_error(output: Option<&RefCell<i32>>, error: KvError) {
    if let Some(output) = output {
        *output.borrow_mut() = error.code();
    }
}

fn set_count(output: Option<&RefCell<i32>>, count: i32) {
    if let Some(output) = output {
        *output.borrow_mut() = count;
    }
}

fn reset_runtime(ctx: &mut MemoryKvContext) {
    ctx.sequence_active.fill(false);
    ctx.sequence_length.fill(0);
    ctx.sequence_block_count.fill(0);
    for row in &mut ctx.seq_to_blocks {
        row.fill(0);
    }
    ctx.block_refs.fill(0);
    ctx.free_count = ctx.max_blocks;
    let free_count = usize::try_from(ctx.free_count).expect("reserved block count is positive");
    for (index, block) in ctx.free_stack.iter_mut().enumerate().take(free_count) {
        *block = u16::try_from(free_count - 1 - index)
            .expect("the fixed block pool uses u16 physical block identifiers");
    }
}

fn fill_snapshot(ctx: &MemoryKvContext, snapshot: &mut Snapshot) {
    snapshot.sequence_active.fill(0);
    snapshot.sequence_length_values.fill(0);
    snapshot.sequence_kv_block_count.fill(0);
    for row in &mut snapshot.sequence_kv_blocks {
        row.fill(0);
    }
    snapshot.sequence_recurrent_slot.fill(0);
    snapshot.max_sequences = 0;
    snapshot.block_tokens = DEFAULT_BLOCK_TOKENS;
    snapshot.max_sequences = ctx.max_sequences;
    snapshot.block_tokens = ctx.block_tokens;
    let Some(max_sequences) = usize::try_from(ctx.max_sequences)
        .ok()
        .filter(|count| *count <= MAX_SEQUENCES)
    else {
        return;
    };
    for seq_index in 0..max_sequences {
        let active = ctx.sequence_active[seq_index];
        snapshot.sequence_active[seq_index] = u8::from(active);
        snapshot.sequence_recurrent_slot[seq_index] = -1;
        snapshot.sequence_length_values[seq_index] =
            ctx.sequence_length[seq_index] * i32::from(active);
        let block_count = ctx.sequence_block_count[seq_index] * i32::from(active);
        snapshot.sequence_kv_block_count[seq_index] = block_count;
        let Some(count) = usize::try_from(block_count)
            .ok()
            .filter(|count| *count <= MAX_BLOCKS_PER_SEQUENCE)
        else {
            return;
        };
        for block_index in 0..MAX_BLOCKS_PER_SEQUENCE {
            snapshot.sequence_kv_blocks[seq_index][block_index] = if block_index < count {
                ctx.seq_to_blocks[seq_index][block_index]
            } else {
                0
            };
        }
    }
}

fn block_index(block_id: u16) -> Option<usize> {
    index_in(i32::from(block_id), MAX_BLOCKS)
}

fn block_linkable(ctx: &MemoryKvContext, block_id: u16) -> bool {
    block_index(block_id).is_some_and(|index| ctx.block_refs[index] < u16::MAX)
}

fn block_unlinkable(ctx: &MemoryKvContext, block_id: u16) -> bool {
    block_index(block_id).is_some_and(|index| ctx.block_refs[index] > 0)
}

fn link_block(ctx: &mut MemoryKvContext, block_id: u16) -> bool {
    if let Some(index) = block_index(block_id)
        && ctx.block_refs[index] < u16::MAX
    {
        ctx.block_refs[index] += 1;
        true
    } else {
        false
    }
}

fn unlink_block(ctx: &mut MemoryKvContext, block_id: u16) -> bool {
    if let Some(index) = block_index(block_id)
        && ctx.block_refs[index] > 0
    {
        ctx.block_refs[index] -= 1;
        true
    } else {
        false
    }
}

fn shared_tail_split_needed(ctx: &MemoryKvContext, seq_index: usize) -> bool {
    let old_length = ctx.sequence_length[seq_index];
    let old_blocks = blocks_for_length(ctx.block_tokens, old_length);
    if old_length <= 0
        || ctx.block_tokens <= 0
        || old_length % ctx.block_tokens == 0
        || old_blocks <= 0
    {
        return false;
    }
    let Some(tail_index) = usize::try_from(old_blocks - 1).ok() else {
        return false;
    };
    let tail_block = ctx.seq_to_blocks[seq_index][tail_index];
    block_unlinkable(ctx, tail_block)
        && block_index(tail_block).is_some_and(|index| ctx.block_refs[index] > 1)
}

fn slots_shape_valid(ctx: &MemoryKvContext, event: &EventAllocateSlotsRuntime<'_>) -> bool {
    valid_sequence_id(ctx, event.seq_id)
        && event.token_count > 0
        && ctx.block_tokens > 0
        && index_in(event.seq_id, MAX_SEQUENCES).is_some_and(|index| ctx.sequence_active[index])
}

fn slots_length_valid(ctx: &MemoryKvContext, event: &EventAllocateSlotsRuntime<'_>) -> bool {
    if !slots_shape_valid(ctx, event) {
        return false;
    }
    let Some(index) = index_in(event.seq_id, MAX_SEQUENCES) else {
        return false;
    };
    let new_length = i64::from(ctx.sequence_length[index]) + i64::from(event.token_count);
    new_length > 0 && new_length <= i64::from(i32::MAX)
}

fn slots_layout_valid(ctx: &MemoryKvContext, event: &EventAllocateSlotsRuntime<'_>) -> bool {
    if !slots_length_valid(ctx, event) {
        return false;
    }
    let Some(index) = index_in(event.seq_id, MAX_SEQUENCES) else {
        return false;
    };
    let old_length = ctx.sequence_length[index];
    let Some(new_length) = old_length.checked_add(event.token_count) else {
        return false;
    };
    let old_blocks = blocks_for_length(ctx.block_tokens, old_length);
    let new_blocks = blocks_for_length(ctx.block_tokens, new_length);
    ctx.sequence_block_count[index] >= old_blocks && new_blocks >= old_blocks
}

fn slots_capacity_valid(ctx: &MemoryKvContext, event: &EventAllocateSlotsRuntime<'_>) -> bool {
    if !slots_layout_valid(ctx, event) {
        return false;
    }
    let Some(index) = index_in(event.seq_id, MAX_SEQUENCES) else {
        return false;
    };
    let old_length = ctx.sequence_length[index];
    let Some(new_length) = old_length.checked_add(event.token_count) else {
        return false;
    };
    let old_blocks = blocks_for_length(ctx.block_tokens, old_length);
    let new_blocks = blocks_for_length(ctx.block_tokens, new_length);
    let blocks_needed = new_blocks - old_blocks;
    let split = i32::from(shared_tail_split_needed(ctx, index));
    i64::from(ctx.sequence_block_count[index]) + i64::from(blocks_needed)
        <= i64::from(MAX_BLOCKS_PER_SEQUENCE_I32)
        && i64::from(ctx.free_count) >= i64::from(blocks_needed) + i64::from(split)
}

fn branch_request_valid(ctx: &MemoryKvContext, event: &EventBranchSequenceRuntime<'_>) -> bool {
    let parent = index_in(event.parent_seq_id, MAX_SEQUENCES);
    let child = index_in(event.child_seq_id, MAX_SEQUENCES);
    valid_sequence_id(ctx, event.parent_seq_id)
        && valid_sequence_id(ctx, event.child_seq_id)
        && event.parent_seq_id != event.child_seq_id
        && parent.is_some_and(|index| ctx.sequence_active[index])
        && child.is_some_and(|index| !ctx.sequence_active[index])
}

fn rollback_request_valid(ctx: &MemoryKvContext, event: &EventRollbackSlotsRuntime<'_>) -> bool {
    valid_sequence_id(ctx, event.seq_id)
        && event.token_count > 0
        && ctx.block_tokens > 0
        && index_in(event.seq_id, MAX_SEQUENCES).is_some_and(|index| ctx.sequence_active[index])
}

#[allow(clippy::too_many_lines)]
fn effect_allocate_slots(
    ctx: &mut MemoryKvContext,
    event: &EventAllocateSlotsRuntime<'_>,
    split: bool,
) {
    let Some(index) = index_in(event.seq_id, MAX_SEQUENCES) else {
        let mut op = event.context.borrow_mut();
        op.accepted = false;
        op.operation_error = KvError::InternalError;
        return;
    };
    let mut op = event.context.borrow_mut();
    op.old_length = ctx.sequence_length[index];
    let Some(new_length) = op.old_length.checked_add(event.token_count) else {
        op.accepted = false;
        op.operation_error = KvError::InternalError;
        return;
    };
    op.new_length = new_length;
    op.existing_block_count = ctx.sequence_block_count[index];
    op.old_blocks = blocks_for_length(ctx.block_tokens, op.old_length);
    op.new_blocks = blocks_for_length(ctx.block_tokens, op.new_length);
    let Some(existing_block_count) =
        bounded_count(op.existing_block_count, MAX_BLOCKS_PER_SEQUENCE_I32)
    else {
        op.accepted = false;
        op.operation_error = KvError::InternalError;
        return;
    };
    let Some(old_block_count) = bounded_count(op.old_blocks, MAX_BLOCKS_PER_SEQUENCE_I32) else {
        op.accepted = false;
        op.operation_error = KvError::InternalError;
        return;
    };
    let Some(_new_block_count) = bounded_count(op.new_blocks, MAX_BLOCKS_PER_SEQUENCE_I32) else {
        op.accepted = false;
        op.operation_error = KvError::InternalError;
        return;
    };
    let Some(blocks_needed) = op.new_blocks.checked_sub(op.old_blocks) else {
        op.accepted = false;
        op.operation_error = KvError::InternalError;
        return;
    };
    let Some(physical_blocks_needed) = blocks_needed.checked_add(i32::from(split)) else {
        op.accepted = false;
        op.operation_error = KvError::InternalError;
        return;
    };
    op.blocks_needed = blocks_needed;
    op.tail_split_needed = split;
    op.physical_blocks_needed = physical_blocks_needed;
    op.linked_count = 0;
    op.unlinked_count = 0;

    let original_free_count = ctx.free_count;
    let mut old_tail_block = 0;
    let mut new_tail_block = 0;
    let mut tail_block_index = 0;
    let mut refs_can_update = if split {
        let Some(tail_index) = old_block_count.checked_sub(1) else {
            op.accepted = false;
            op.operation_error = KvError::InternalError;
            return;
        };
        let Some(tail_index_i32) = i32::try_from(tail_index).ok() else {
            op.accepted = false;
            op.operation_error = KvError::InternalError;
            return;
        };
        tail_block_index = tail_index_i32;
        old_tail_block = ctx.seq_to_blocks[index][tail_index];
        let Some(block) = free_stack_block(ctx, original_free_count - 1) else {
            op.accepted = false;
            op.operation_error = KvError::InternalError;
            return;
        };
        new_tail_block = block;
        block_linkable(ctx, new_tail_block) && block_unlinkable(ctx, old_tail_block)
    } else {
        true
    };
    for offset in 0..op.blocks_needed {
        let stack_index = original_free_count - 1 - i32::from(split) - offset;
        let Some(block) = free_stack_block(ctx, stack_index) else {
            op.accepted = false;
            op.operation_error = KvError::InternalError;
            return;
        };
        refs_can_update &= block_linkable(ctx, block);
    }
    if refs_can_update {
        if split {
            op.linked_count += i32::from(link_block(ctx, new_tail_block));
            op.unlinked_count += i32::from(unlink_block(ctx, old_tail_block));
        }
        for offset in 0..op.blocks_needed {
            let stack_index = original_free_count - 1 - i32::from(split) - offset;
            let Some(block) = free_stack_block(ctx, stack_index) else {
                op.accepted = false;
                op.operation_error = KvError::InternalError;
                return;
            };
            op.linked_count += i32::from(link_block(ctx, block));
        }
    }
    let linked_all =
        op.linked_count == op.physical_blocks_needed && op.unlinked_count == i32::from(split);
    if split {
        let Some(tail_index) = index_in(tail_block_index, MAX_BLOCKS_PER_SEQUENCE) else {
            op.accepted = false;
            op.operation_error = KvError::InternalError;
            return;
        };
        ctx.seq_to_blocks[index][tail_index] = if linked_all {
            new_tail_block
        } else {
            old_tail_block
        };
    }
    for offset in 0..op.blocks_needed {
        let block_index = existing_block_count
            + usize::try_from(offset).expect("a block allocation offset is non-negative");
        let stack_index = original_free_count - 1 - i32::from(split) - offset;
        let Some(block) = free_stack_block(ctx, stack_index) else {
            op.accepted = false;
            op.operation_error = KvError::InternalError;
            return;
        };
        let old_block = ctx.seq_to_blocks[index][block_index];
        ctx.seq_to_blocks[index][block_index] = if linked_all { block } else { old_block };
    }
    if linked_all {
        ctx.free_count = original_free_count - op.physical_blocks_needed;
        ctx.sequence_block_count[index] = op.new_blocks;
        ctx.sequence_length[index] = op.new_length;
    }
    op.block_count = if linked_all {
        op.physical_blocks_needed
    } else {
        0
    };
    op.accepted = linked_all;
    op.operation_error = KvError::None;
}

macro_rules! outcome_guards {
    ($(($with:ident, $without:ident, $success:ident, $event:ident)),+ $(,)?) => {
        $(
            fn $with(&self, event: &$event<'_>) -> Result<bool, ()> {
                let op = event.context.borrow();
                Ok(!op.accepted && op.operation_error != KvError::None)
            }
            fn $without(&self, event: &$event<'_>) -> Result<bool, ()> {
                let op = event.context.borrow();
                Ok(!op.accepted && op.operation_error == KvError::None)
            }
            fn $success(&self, event: &$event<'_>) -> Result<bool, ()> {
                Ok(event.context.borrow().accepted)
            }
        )+
    };
}

macro_rules! mark_error_actions {
    ($(($backend:ident, $invalid:ident, $from_operation:ident, $event:ident)),+ $(,)?) => {
        $(
            fn $backend(&mut self, event: &$event<'_>) -> Result<(), ()> {
                event.context.borrow_mut().err = KvError::BackendError;
                set_error(event.error_out, KvError::BackendError);
                Ok(())
            }
            fn $invalid(&mut self, event: &$event<'_>) -> Result<(), ()> {
                event.context.borrow_mut().err = KvError::InvalidRequest;
                set_error(event.error_out, KvError::InvalidRequest);
                Ok(())
            }
            fn $from_operation(&mut self, event: &$event<'_>) -> Result<(), ()> {
                let error = event.context.borrow().operation_error;
                event.context.borrow_mut().err = error;
                set_error(event.error_out, error);
                Ok(())
            }
        )+
    };
}

macro_rules! publish_without_count {
    ($($name:ident, $event:ident),+ $(,)?) => {
        $(
            fn $name(&mut self, event: &$event<'_>) -> Result<(), ()> {
                event.context.borrow_mut().err = KvError::None;
                set_error(event.error_out, KvError::None);
                Ok(())
            }
        )+
    };
}

macro_rules! publish_with_count {
    ($($name:ident, $event:ident),+ $(,)?) => {
        $(
            fn $name(&mut self, event: &$event<'_>) -> Result<(), ()> {
                let count = event.context.borrow().block_count;
                event.context.borrow_mut().err = KvError::None;
                set_count(event.block_count_out, count);
                set_error(event.error_out, KvError::None);
                Ok(())
            }
        )+
    };
}

macro_rules! publish_error_without_count {
    ($($name:ident, $event:ident),+ $(,)?) => {
        $(
            fn $name(&mut self, event: &$event<'_>) -> Result<(), ()> {
                let error = event.context.borrow().err;
                set_error(event.error_out, error);
                Ok(())
            }
        )+
    };
}

macro_rules! publish_error_with_count {
    ($($name:ident, $event:ident),+ $(,)?) => {
        $(
            fn $name(&mut self, event: &$event<'_>) -> Result<(), ()> {
                let error = event.context.borrow().err;
                set_count(event.block_count_out, 0);
                set_error(event.error_out, error);
                Ok(())
            }
        )+
    };
}

macro_rules! unexpected_actions {
    ($($name:ident),+ $(,)?) => {
        $(fn $name(&mut self) -> Result<(), ()> { Ok(()) })+
    };
}

impl MemoryKvStateMachineContext for MemoryKvContext {
    fn allocate_sequence_request_invalid(
        &self,
        event: &EventAllocateSequenceRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(!valid_sequence_id(self, event.seq_id))
    }

    fn allocate_sequence_request_valid(
        &self,
        event: &EventAllocateSequenceRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(valid_sequence_id(self, event.seq_id))
    }

    fn allocate_slots_request_block_layout_invalid(
        &self,
        event: &EventAllocateSlotsRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(!slots_layout_valid(self, event))
    }

    fn allocate_slots_request_block_layout_valid(
        &self,
        event: &EventAllocateSlotsRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(slots_layout_valid(self, event))
    }

    fn allocate_slots_request_capacity_invalid(
        &self,
        event: &EventAllocateSlotsRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(!slots_capacity_valid(self, event))
    }

    fn allocate_slots_request_capacity_valid(
        &self,
        event: &EventAllocateSlotsRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(slots_capacity_valid(self, event))
    }

    fn allocate_slots_request_length_invalid(
        &self,
        event: &EventAllocateSlotsRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(!slots_length_valid(self, event))
    }

    fn allocate_slots_request_length_valid(
        &self,
        event: &EventAllocateSlotsRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(slots_length_valid(self, event))
    }

    fn allocate_slots_request_shape_invalid(
        &self,
        event: &EventAllocateSlotsRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(!slots_shape_valid(self, event))
    }

    fn allocate_slots_request_shape_valid(
        &self,
        event: &EventAllocateSlotsRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(slots_shape_valid(self, event))
    }

    fn begin_allocate_sequence(
        &mut self,
        event: &EventAllocateSequenceRuntime<'_>,
    ) -> Result<(), ()> {
        *event.context.borrow_mut() = AllocateSequenceContext::default();
        set_error(event.error_out, KvError::None);
        Ok(())
    }

    fn begin_allocate_slots(&mut self, event: &EventAllocateSlotsRuntime<'_>) -> Result<(), ()> {
        *event.context.borrow_mut() = AllocateSlotsContext::default();
        set_count(event.block_count_out, 0);
        set_error(event.error_out, KvError::None);
        Ok(())
    }

    fn begin_branch_sequence(&mut self, event: &EventBranchSequenceRuntime<'_>) -> Result<(), ()> {
        *event.context.borrow_mut() = BranchSequenceContext::default();
        set_error(event.error_out, KvError::None);
        Ok(())
    }

    fn begin_capture_view(&mut self, event: &EventCaptureViewRuntime<'_>) -> Result<(), ()> {
        *event.context.borrow_mut() = CaptureViewContext::default();
        set_error(event.error_out, KvError::None);
        Ok(())
    }

    fn begin_free_sequence(&mut self, event: &EventFreeSequenceRuntime<'_>) -> Result<(), ()> {
        *event.context.borrow_mut() = FreeSequenceContext::default();
        set_error(event.error_out, KvError::None);
        Ok(())
    }

    fn begin_reserve(&mut self, event: &EventReserveRuntime<'_>) -> Result<(), ()> {
        *event.context.borrow_mut() = ReserveContext::default();
        set_error(event.error_out, KvError::None);
        Ok(())
    }

    fn begin_rollback_slots(&mut self, event: &EventRollbackSlotsRuntime<'_>) -> Result<(), ()> {
        *event.context.borrow_mut() = RollbackSlotsContext::default();
        set_count(event.block_count_out, 0);
        set_error(event.error_out, KvError::None);
        Ok(())
    }

    fn branch_sequence_request_invalid(
        &self,
        event: &EventBranchSequenceRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(!branch_request_valid(self, event))
    }

    fn branch_sequence_request_valid(
        &self,
        event: &EventBranchSequenceRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(branch_request_valid(self, event))
    }

    fn capture_request_invalid(&self, event: &EventCaptureViewRuntime<'_>) -> Result<bool, ()> {
        Ok(event.snapshot_out.is_none())
    }

    fn capture_request_valid(&self, event: &EventCaptureViewRuntime<'_>) -> Result<bool, ()> {
        Ok(event.snapshot_out.is_some())
    }

    fn effect_allocate_slots_with_tail_split(
        &mut self,
        event: &EventAllocateSlotsRuntime<'_>,
    ) -> Result<(), ()> {
        effect_allocate_slots(self, event, true);
        Ok(())
    }

    fn effect_allocate_slots_without_tail_split(
        &mut self,
        event: &EventAllocateSlotsRuntime<'_>,
    ) -> Result<(), ()> {
        effect_allocate_slots(self, event, false);
        Ok(())
    }

    fn effect_copy_shared_tail_block(
        &mut self,
        event: &EventAllocateSlotsRuntime<'_>,
    ) -> Result<(), ()> {
        let Some(index) = index_in(event.seq_id, MAX_SEQUENCES) else {
            let mut op = event.context.borrow_mut();
            op.accepted = false;
            op.operation_error = KvError::InternalError;
            return Ok(());
        };
        let old_blocks = blocks_for_length(self.block_tokens, self.sequence_length[index]);
        let Some(old_block_index) = bounded_count(old_blocks - 1, MAX_BLOCKS_PER_SEQUENCE_I32 - 1)
        else {
            let mut op = event.context.borrow_mut();
            op.accepted = false;
            op.operation_error = KvError::InternalError;
            return Ok(());
        };
        let Some(free_index) = index_in(self.free_count - 1, MAX_BLOCKS) else {
            let mut op = event.context.borrow_mut();
            op.accepted = false;
            op.operation_error = KvError::InternalError;
            return Ok(());
        };
        let old_block = self.seq_to_blocks[index][old_block_index];
        let new_block = self.free_stack[free_index];
        let mut op = event.context.borrow_mut();
        op.copy_error = KvError::None;
        if let Some(copier) = event.copy_block {
            match copier.copy_block(
                i32::from(old_block),
                i32::from(new_block),
                self.block_tokens,
            ) {
                Ok(accepted) => {
                    op.copy_accepted = accepted;
                    op.operation_error = KvError::None;
                }
                Err(error) => {
                    op.copy_accepted = false;
                    op.copy_error = error;
                    op.operation_error = error;
                }
            }
        } else {
            op.copy_accepted = false;
            op.operation_error = KvError::InvalidRequest;
        }
        op.accepted = op.copy_accepted && op.copy_error == KvError::None;
        Ok(())
    }

    fn exec_allocate_sequence(
        &mut self,
        event: &EventAllocateSequenceRuntime<'_>,
    ) -> Result<(), ()> {
        let Some(index) = index_in(event.seq_id, MAX_SEQUENCES) else {
            let mut op = event.context.borrow_mut();
            op.accepted = false;
            op.operation_error = KvError::InternalError;
            return Ok(());
        };
        let was_active = self.sequence_active[index];
        self.sequence_active[index] = true;
        if !was_active {
            self.sequence_length[index] = 0;
            self.sequence_block_count[index] = 0;
        }
        let mut op = event.context.borrow_mut();
        op.accepted = true;
        op.operation_error = KvError::None;
        Ok(())
    }

    fn exec_branch_sequence(&mut self, event: &EventBranchSequenceRuntime<'_>) -> Result<(), ()> {
        let (Some(parent), Some(child)) = (
            index_in(event.parent_seq_id, MAX_SEQUENCES),
            index_in(event.child_seq_id, MAX_SEQUENCES),
        ) else {
            let mut op = event.context.borrow_mut();
            op.accepted = false;
            op.operation_error = KvError::InternalError;
            return Ok(());
        };
        let mut op = event.context.borrow_mut();
        op.parent_blocks = self.sequence_block_count[parent];
        let Some(parent_block_count) = bounded_count(op.parent_blocks, MAX_BLOCKS_PER_SEQUENCE_I32)
        else {
            op.accepted = false;
            op.operation_error = KvError::InternalError;
            return Ok(());
        };
        op.linked_count = 0;
        for block_index in 0..parent_block_count {
            let block = self.seq_to_blocks[parent][block_index];
            self.seq_to_blocks[child][block_index] = block;
            op.linked_count += i32::from(link_block(self, block));
        }
        let linked_all = op.linked_count == op.parent_blocks;
        self.sequence_active[child] = linked_all;
        if linked_all {
            self.sequence_length[child] = self.sequence_length[parent];
            self.sequence_block_count[child] = op.parent_blocks;
        }
        op.accepted = linked_all;
        op.operation_error = KvError::None;
        Ok(())
    }

    fn exec_capture_view(&mut self, event: &EventCaptureViewRuntime<'_>) -> Result<(), ()> {
        let mut op = event.context.borrow_mut();
        if let Some(snapshot) = event.snapshot_out {
            fill_snapshot(self, &mut snapshot.borrow_mut());
            op.accepted = true;
            op.operation_error = KvError::None;
        } else {
            op.accepted = false;
            op.operation_error = KvError::InvalidRequest;
        }
        Ok(())
    }

    fn exec_free_sequence(&mut self, event: &EventFreeSequenceRuntime<'_>) -> Result<(), ()> {
        let Some(index) = index_in(event.seq_id, MAX_SEQUENCES) else {
            let mut op = event.context.borrow_mut();
            op.accepted = false;
            op.operation_error = KvError::InternalError;
            return Ok(());
        };
        let was_active = self.sequence_active[index];
        let mut op = event.context.borrow_mut();
        op.block_count = if was_active {
            self.sequence_block_count[index]
        } else {
            0
        };
        let Some(block_count) = bounded_count(op.block_count, MAX_BLOCKS_PER_SEQUENCE_I32) else {
            op.accepted = false;
            op.operation_error = KvError::InternalError;
            return Ok(());
        };
        op.unlinked_count = 0;
        for block_index in 0..block_count {
            op.unlinked_count +=
                i32::from(unlink_block(self, self.seq_to_blocks[index][block_index]));
        }
        let unlink_ok = op.unlinked_count == op.block_count;
        let allow_recycle = was_active && unlink_ok;
        let mut free_write = self.free_count;
        let mut recycle_ok = allow_recycle;
        for offset in 0..block_count {
            let sequence_block_index = block_count - 1 - offset;
            let block = self.seq_to_blocks[index][sequence_block_index];
            let is_free =
                block_index(block).is_some_and(|ref_index| self.block_refs[ref_index] == 0);
            if allow_recycle && is_free {
                if let Some(free_index) = index_in(free_write, MAX_BLOCKS) {
                    self.free_stack[free_index] = block;
                    free_write += 1;
                } else {
                    recycle_ok = false;
                }
            }
        }
        if allow_recycle && recycle_ok {
            self.free_count = free_write;
            self.sequence_active[index] = false;
            self.sequence_length[index] = 0;
            self.sequence_block_count[index] = 0;
        }
        op.accepted = !was_active || (allow_recycle && recycle_ok);
        op.operation_error = KvError::None;
        Ok(())
    }

    fn exec_reserve(&mut self, event: &EventReserveRuntime<'_>) -> Result<(), ()> {
        let max_sequences = resolved_or_default(event.max_sequences, MAX_SEQUENCES_I32);
        let max_blocks = resolved_or_default(event.max_blocks, MAX_BLOCKS_I32);
        let block_tokens = resolved_or_default(event.block_tokens, DEFAULT_BLOCK_TOKENS);
        {
            let mut op = event.context.borrow_mut();
            op.resolved_max_sequences = max_sequences;
            op.resolved_max_blocks = max_blocks;
            op.resolved_block_tokens = block_tokens;
        }
        self.max_sequences = max_sequences;
        self.max_blocks = max_blocks;
        self.block_tokens = block_tokens;
        reset_runtime(self);
        let mut op = event.context.borrow_mut();
        op.accepted = true;
        op.operation_error = KvError::None;
        Ok(())
    }

    fn exec_rollback_slots(&mut self, event: &EventRollbackSlotsRuntime<'_>) -> Result<(), ()> {
        let Some(index) = index_in(event.seq_id, MAX_SEQUENCES) else {
            let mut op = event.context.borrow_mut();
            op.accepted = false;
            op.operation_error = KvError::InternalError;
            return Ok(());
        };
        let mut op = event.context.borrow_mut();
        op.current_length = self.sequence_length[index];
        let new_length = (i64::from(op.current_length) - i64::from(event.token_count)).max(0);
        op.new_length = i32::try_from(new_length).expect("rollback length is bounded by i32");
        op.existing_block_count = self.sequence_block_count[index];
        op.new_blocks = blocks_for_length(self.block_tokens, op.new_length);
        let Some(existing_block_count) =
            bounded_count(op.existing_block_count, MAX_BLOCKS_PER_SEQUENCE_I32)
        else {
            op.accepted = false;
            op.operation_error = KvError::InternalError;
            return Ok(());
        };
        op.unlinked_count = 0;
        let plan_valid = op.current_length >= 0 && op.new_blocks <= op.existing_block_count;
        let effective_remove_count = if plan_valid {
            op.existing_block_count - op.new_blocks
        } else {
            0
        };
        op.remove_count = effective_remove_count;
        let Some(remove_count) = bounded_count(effective_remove_count, MAX_BLOCKS_PER_SEQUENCE_I32)
        else {
            op.accepted = false;
            op.operation_error = KvError::InternalError;
            return Ok(());
        };
        for offset in 0..remove_count {
            let block_index = existing_block_count - 1 - offset;
            op.unlinked_count +=
                i32::from(unlink_block(self, self.seq_to_blocks[index][block_index]));
        }
        let unlink_ok = op.unlinked_count == effective_remove_count;
        let apply_update = plan_valid && unlink_ok;
        let mut free_write = self.free_count;
        let mut recycle_ok = apply_update;
        for offset in 0..remove_count {
            let sequence_block_index = existing_block_count - 1 - offset;
            let block = self.seq_to_blocks[index][sequence_block_index];
            let is_free =
                block_index(block).is_some_and(|ref_index| self.block_refs[ref_index] == 0);
            if apply_update && is_free {
                if let Some(free_index) = index_in(free_write, MAX_BLOCKS) {
                    self.free_stack[free_index] = block;
                    free_write += 1;
                } else {
                    recycle_ok = false;
                }
            }
        }
        let apply_update = apply_update && recycle_ok;
        if apply_update {
            self.free_count = free_write;
            self.sequence_block_count[index] = op.new_blocks;
            self.sequence_length[index] = op.new_length;
        }
        op.block_count = if apply_update {
            effective_remove_count
        } else {
            0
        };
        op.accepted = apply_update;
        op.operation_error = KvError::None;
        Ok(())
    }

    fn free_sequence_request_invalid(
        &self,
        event: &EventFreeSequenceRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(!valid_sequence_id(self, event.seq_id))
    }

    fn free_sequence_request_valid(
        &self,
        event: &EventFreeSequenceRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(valid_sequence_id(self, event.seq_id))
    }

    fn guard_allocate_slots_shared_tail_copy_missing(
        &self,
        event: &EventAllocateSlotsRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(slots_capacity_valid(self, event)
            && index_in(event.seq_id, MAX_SEQUENCES)
                .is_some_and(|index| shared_tail_split_needed(self, index))
            && event.copy_block.is_none())
    }

    fn guard_allocate_slots_shared_tail_copy_ready(
        &self,
        event: &EventAllocateSlotsRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(slots_capacity_valid(self, event)
            && index_in(event.seq_id, MAX_SEQUENCES)
                .is_some_and(|index| shared_tail_split_needed(self, index))
            && event.copy_block.is_some())
    }

    fn guard_allocate_slots_shared_tail_split_not_required(
        &self,
        event: &EventAllocateSlotsRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(!(slots_capacity_valid(self, event)
            && index_in(event.seq_id, MAX_SEQUENCES)
                .is_some_and(|index| shared_tail_split_needed(self, index))))
    }

    fn mark_out_of_memory(&mut self, event: &EventAllocateSlotsRuntime<'_>) -> Result<(), ()> {
        event.context.borrow_mut().err = KvError::OutOfMemory;
        set_error(event.error_out, KvError::OutOfMemory);
        Ok(())
    }

    unexpected_actions!(
        on_unexpected_from_allocate_sequence_exec,
        on_unexpected_from_allocate_sequence_request_decision,
        on_unexpected_from_allocate_sequence_result_decision,
        on_unexpected_from_allocate_slots_request_block_layout_decision,
        on_unexpected_from_allocate_slots_request_capacity_decision,
        on_unexpected_from_allocate_slots_request_decision,
        on_unexpected_from_allocate_slots_request_length_decision,
        on_unexpected_from_allocate_slots_request_shape_decision,
        on_unexpected_from_allocate_slots_result_decision,
        on_unexpected_from_branch_sequence_exec,
        on_unexpected_from_branch_sequence_request_decision,
        on_unexpected_from_branch_sequence_result_decision,
        on_unexpected_from_capture_exec,
        on_unexpected_from_capture_request_decision,
        on_unexpected_from_capture_result_decision,
        on_unexpected_from_done,
        on_unexpected_from_errored,
        on_unexpected_from_free_sequence_exec,
        on_unexpected_from_free_sequence_request_decision,
        on_unexpected_from_free_sequence_result_decision,
        on_unexpected_from_ready,
        on_unexpected_from_reserve_exec,
        on_unexpected_from_reserve_request_decision,
        on_unexpected_from_reserve_result_decision,
        on_unexpected_from_rollback_slots_exec,
        on_unexpected_from_rollback_slots_request_decision,
        on_unexpected_from_rollback_slots_result_decision,
        on_unexpected_from_state_allocate_slots_direct_exec,
        on_unexpected_from_state_allocate_slots_tail_copy_exec,
        on_unexpected_from_state_allocate_slots_tail_copy_result_decision,
        on_unexpected_from_state_allocate_slots_tail_decision,
        on_unexpected_from_state_allocate_slots_tail_split_exec,
    );

    outcome_guards!(
        (
            operation_failed_with_error_event_allocate_sequence_runtime,
            operation_failed_without_error_event_allocate_sequence_runtime,
            operation_succeeded_event_allocate_sequence_runtime,
            EventAllocateSequenceRuntime
        ),
        (
            operation_failed_with_error_event_allocate_slots_runtime,
            operation_failed_without_error_event_allocate_slots_runtime,
            operation_succeeded_event_allocate_slots_runtime,
            EventAllocateSlotsRuntime
        ),
        (
            operation_failed_with_error_event_branch_sequence_runtime,
            operation_failed_without_error_event_branch_sequence_runtime,
            operation_succeeded_event_branch_sequence_runtime,
            EventBranchSequenceRuntime
        ),
        (
            operation_failed_with_error_event_capture_view_runtime,
            operation_failed_without_error_event_capture_view_runtime,
            operation_succeeded_event_capture_view_runtime,
            EventCaptureViewRuntime
        ),
        (
            operation_failed_with_error_event_free_sequence_runtime,
            operation_failed_without_error_event_free_sequence_runtime,
            operation_succeeded_event_free_sequence_runtime,
            EventFreeSequenceRuntime
        ),
        (
            operation_failed_with_error_event_reserve_runtime,
            operation_failed_without_error_event_reserve_runtime,
            operation_succeeded_event_reserve_runtime,
            EventReserveRuntime
        ),
        (
            operation_failed_with_error_event_rollback_slots_runtime,
            operation_failed_without_error_event_rollback_slots_runtime,
            operation_succeeded_event_rollback_slots_runtime,
            EventRollbackSlotsRuntime
        ),
    );

    mark_error_actions!(
        (
            mark_backend_error_event_allocate_sequence_runtime,
            mark_invalid_request_event_allocate_sequence_runtime,
            mark_error_from_operation_event_allocate_sequence_runtime,
            EventAllocateSequenceRuntime
        ),
        (
            mark_backend_error_event_allocate_slots_runtime,
            mark_invalid_request_event_allocate_slots_runtime,
            mark_error_from_operation_event_allocate_slots_runtime,
            EventAllocateSlotsRuntime
        ),
        (
            mark_backend_error_event_branch_sequence_runtime,
            mark_invalid_request_event_branch_sequence_runtime,
            mark_error_from_operation_event_branch_sequence_runtime,
            EventBranchSequenceRuntime
        ),
        (
            mark_backend_error_event_capture_view_runtime,
            mark_invalid_request_event_capture_view_runtime,
            mark_error_from_operation_event_capture_view_runtime,
            EventCaptureViewRuntime
        ),
        (
            mark_backend_error_event_free_sequence_runtime,
            mark_invalid_request_event_free_sequence_runtime,
            mark_error_from_operation_event_free_sequence_runtime,
            EventFreeSequenceRuntime
        ),
        (
            mark_backend_error_event_reserve_runtime,
            mark_invalid_request_event_reserve_runtime,
            mark_error_from_operation_event_reserve_runtime,
            EventReserveRuntime
        ),
        (
            mark_backend_error_event_rollback_slots_runtime,
            mark_invalid_request_event_rollback_slots_runtime,
            mark_error_from_operation_event_rollback_slots_runtime,
            EventRollbackSlotsRuntime
        ),
    );

    publish_without_count!(
        publish_done_event_allocate_sequence_runtime,
        EventAllocateSequenceRuntime,
        publish_done_event_branch_sequence_runtime,
        EventBranchSequenceRuntime,
        publish_done_event_capture_view_runtime,
        EventCaptureViewRuntime,
        publish_done_event_free_sequence_runtime,
        EventFreeSequenceRuntime,
        publish_done_event_reserve_runtime,
        EventReserveRuntime,
    );

    publish_with_count!(
        publish_done_event_allocate_slots_runtime,
        EventAllocateSlotsRuntime,
        publish_done_event_rollback_slots_runtime,
        EventRollbackSlotsRuntime,
    );

    publish_error_without_count!(
        publish_error_event_allocate_sequence_runtime,
        EventAllocateSequenceRuntime,
        publish_error_event_branch_sequence_runtime,
        EventBranchSequenceRuntime,
        publish_error_event_capture_view_runtime,
        EventCaptureViewRuntime,
        publish_error_event_free_sequence_runtime,
        EventFreeSequenceRuntime,
        publish_error_event_reserve_runtime,
        EventReserveRuntime,
    );

    publish_error_with_count!(
        publish_error_event_allocate_slots_runtime,
        EventAllocateSlotsRuntime,
        publish_error_event_rollback_slots_runtime,
        EventRollbackSlotsRuntime,
    );

    fn reserve_request_invalid(&self, event: &EventReserveRuntime<'_>) -> Result<bool, ()> {
        let max_sequences = resolved_or_default(event.max_sequences, MAX_SEQUENCES_I32);
        let max_blocks = resolved_or_default(event.max_blocks, MAX_BLOCKS_I32);
        let block_tokens = resolved_or_default(event.block_tokens, DEFAULT_BLOCK_TOKENS);
        Ok(!(max_sequences > 0
            && max_sequences <= MAX_SEQUENCES_I32
            && max_blocks > 0
            && max_blocks <= MAX_BLOCKS_I32
            && block_tokens > 0))
    }

    fn reserve_request_valid(&self, event: &EventReserveRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.reserve_request_invalid(event)?)
    }

    fn rollback_slots_request_invalid(
        &self,
        event: &EventRollbackSlotsRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(!rollback_request_valid(self, event))
    }

    fn rollback_slots_request_valid(
        &self,
        event: &EventRollbackSlotsRuntime<'_>,
    ) -> Result<bool, ()> {
        Ok(rollback_request_valid(self, event))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct CopyBlocks;

    impl BlockCopier for CopyBlocks {
        fn copy_block(
            &self,
            _source: i32,
            _destination: i32,
            _block_tokens: i32,
        ) -> Result<bool, KvError> {
            Ok(true)
        }
    }

    fn reserve_event<'event>(
        context: &'event RefCell<ReserveContext>,
        error: &'event RefCell<i32>,
    ) -> EventReserveRuntime<'event> {
        EventReserveRuntime {
            max_sequences: 4,
            max_blocks: 8,
            block_tokens: 4,
            error_out: Some(error),
            context,
        }
    }

    #[test]
    fn lifecycle_allocates_branches_rolls_back_and_frees_deterministically() {
        let mut machine = MemoryKvStateMachine::new(MemoryKvContext::default());
        let reserve_context = RefCell::new(ReserveContext::default());
        let error = RefCell::new(-1);
        machine
            .process_event(reserve_event(&reserve_context, &error))
            .unwrap();
        assert_eq!(*error.borrow(), 0);
        assert_eq!(machine.context().free_count, 8);

        let allocate_context = RefCell::new(AllocateSequenceContext::default());
        machine
            .process_event(EventAllocateSequenceRuntime {
                seq_id: 0,
                error_out: Some(&error),
                context: &allocate_context,
            })
            .unwrap();

        let slots_context = RefCell::new(AllocateSlotsContext::default());
        let block_count = RefCell::new(-1);
        machine
            .process_event(EventAllocateSlotsRuntime {
                seq_id: 0,
                token_count: 5,
                block_count_out: Some(&block_count),
                error_out: Some(&error),
                copy_block: None,
                context: &slots_context,
            })
            .unwrap();
        assert_eq!(*block_count.borrow(), 2);
        assert_eq!(machine.context().sequence_length[0], 5);
        assert_eq!(machine.context().free_count, 6);

        let child_context = RefCell::new(AllocateSequenceContext::default());
        machine
            .process_event(EventAllocateSequenceRuntime {
                seq_id: 1,
                error_out: Some(&error),
                context: &child_context,
            })
            .unwrap();
        let branch_context = RefCell::new(BranchSequenceContext::default());
        machine
            .process_event(EventBranchSequenceRuntime {
                parent_seq_id: 0,
                child_seq_id: 1,
                copy_state: None,
                error_out: Some(&error),
                context: &branch_context,
            })
            .unwrap();
        assert_eq!(machine.context().free_count, 6);

        let rollback_context = RefCell::new(RollbackSlotsContext::default());
        machine
            .process_event(EventRollbackSlotsRuntime {
                seq_id: 0,
                token_count: 4,
                block_count_out: Some(&block_count),
                error_out: Some(&error),
                context: &rollback_context,
            })
            .unwrap();
        assert_eq!(*block_count.borrow(), 1);
        assert_eq!(machine.context().sequence_length[0], 1);

        let free_context = RefCell::new(FreeSequenceContext::default());
        machine
            .process_event(EventFreeSequenceRuntime {
                seq_id: 1,
                error_out: Some(&error),
                context: &free_context,
            })
            .unwrap();
        machine
            .process_event(EventFreeSequenceRuntime {
                seq_id: 0,
                error_out: Some(&error),
                context: &free_context,
            })
            .unwrap();
        assert_eq!(machine.context().free_count, 8);
    }

    #[test]
    fn shared_tail_copy_requires_typed_copier_and_capture_is_caller_owned() {
        let mut machine = MemoryKvStateMachine::new(MemoryKvContext::default());
        let reserve_context = RefCell::new(ReserveContext::default());
        let error = RefCell::new(-1);
        machine
            .process_event(reserve_event(&reserve_context, &error))
            .unwrap();
        let sequence_context = RefCell::new(AllocateSequenceContext::default());
        machine
            .process_event(EventAllocateSequenceRuntime {
                seq_id: 0,
                error_out: Some(&error),
                context: &sequence_context,
            })
            .unwrap();
        let block_count = RefCell::new(0);
        let slots_context = RefCell::new(AllocateSlotsContext::default());
        machine
            .process_event(EventAllocateSlotsRuntime {
                seq_id: 0,
                token_count: 3,
                block_count_out: Some(&block_count),
                error_out: Some(&error),
                copy_block: None,
                context: &slots_context,
            })
            .unwrap();
        let branch_context = RefCell::new(BranchSequenceContext::default());
        machine
            .process_event(EventBranchSequenceRuntime {
                parent_seq_id: 0,
                child_seq_id: 1,
                copy_state: None,
                error_out: Some(&error),
                context: &branch_context,
            })
            .unwrap();
        let copier = CopyBlocks;
        machine
            .process_event(EventAllocateSlotsRuntime {
                seq_id: 0,
                token_count: 2,
                block_count_out: Some(&block_count),
                error_out: Some(&error),
                copy_block: Some(&copier),
                context: &slots_context,
            })
            .unwrap();
        assert_eq!(*block_count.borrow(), 2);

        let snapshot = RefCell::new(Snapshot::default());
        let capture_context = RefCell::new(CaptureViewContext::default());
        machine
            .process_event(EventCaptureViewRuntime {
                snapshot_out: Some(&snapshot),
                error_out: Some(&error),
                context: &capture_context,
            })
            .unwrap();
        assert_eq!(snapshot.borrow().max_sequences, 4);
        assert_eq!(snapshot.borrow().sequence_length_values[0], 5);
        assert!(snapshot.borrow().is_sequence_active(0));
        assert_eq!(snapshot.borrow().sequence_length(0), 5);
        assert!(snapshot.borrow().lookup_kv_block(0, 0) >= 0);
        assert_eq!(snapshot.borrow().lookup_recurrent_slot(0), -1);
    }
}

pub trait BlockCopier {
    fn copy_block(&self, source: i32, destination: i32, block_tokens: i32)
    -> Result<bool, KvError>;
}

pub trait StateCopier {
    fn copy_state(&self, source_slot: i32, destination_slot: i32) -> Result<(), KvError>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Snapshot {
    pub max_sequences: i32,
    pub block_tokens: i32,
    pub sequence_active: Box<[u8]>,
    pub sequence_length_values: Box<[i32]>,
    pub sequence_kv_block_count: Box<[i32]>,
    pub sequence_kv_blocks: Box<[[u16; MAX_BLOCKS_PER_SEQUENCE]]>,
    pub sequence_recurrent_slot: Box<[i32]>,
}

impl Default for Snapshot {
    fn default() -> Self {
        Self {
            max_sequences: 0,
            block_tokens: DEFAULT_BLOCK_TOKENS,
            sequence_active: vec![0; MAX_SEQUENCES].into_boxed_slice(),
            sequence_length_values: vec![0; MAX_SEQUENCES].into_boxed_slice(),
            sequence_kv_block_count: vec![0; MAX_SEQUENCES].into_boxed_slice(),
            sequence_kv_blocks: vec![[0; MAX_BLOCKS_PER_SEQUENCE]; MAX_SEQUENCES]
                .into_boxed_slice(),
            sequence_recurrent_slot: vec![0; MAX_SEQUENCES].into_boxed_slice(),
        }
    }
}

impl Snapshot {
    fn sequence_index(&self, seq_id: i32) -> Option<usize> {
        (seq_id < self.max_sequences)
            .then(|| index_in(seq_id, MAX_SEQUENCES))
            .flatten()
    }

    pub fn valid_seq_id(&self, seq_id: i32) -> bool {
        self.sequence_index(seq_id).is_some()
    }

    pub fn is_sequence_active(&self, seq_id: i32) -> bool {
        self.sequence_index(seq_id)
            .is_some_and(|index| self.sequence_active[index] != 0)
    }

    pub fn sequence_length(&self, seq_id: i32) -> i32 {
        self.sequence_index(seq_id)
            .filter(|&index| self.sequence_active[index] != 0)
            .map_or(0, |index| self.sequence_length_values[index])
    }

    pub fn lookup_kv_block(&self, seq_id: i32, pos: i32) -> i32 {
        if !self.is_sequence_active(seq_id) || pos < 0 || self.block_tokens <= 0 {
            return -1;
        }
        let Some(seq_index) = self.sequence_index(seq_id) else {
            return -1;
        };
        let length = self.sequence_length_values[seq_index];
        if pos >= length {
            return -1;
        }
        let block_count = self.sequence_kv_block_count[seq_index];
        if block_count <= 0 || block_count > MAX_BLOCKS_PER_SEQUENCE_I32 {
            return -1;
        }
        let logical_block = pos / self.block_tokens;
        if logical_block < 0 || logical_block >= block_count {
            return -1;
        }
        let Some(logical_block) = index_in(logical_block, MAX_BLOCKS_PER_SEQUENCE) else {
            return -1;
        };
        let block = self.sequence_kv_blocks[seq_index][logical_block];
        if block == u16::MAX {
            -1
        } else {
            i32::from(block)
        }
    }

    pub fn lookup_recurrent_slot(&self, seq_id: i32) -> i32 {
        self.sequence_index(seq_id)
            .filter(|&index| self.sequence_active[index] != 0)
            .map_or(-1, |index| self.sequence_recurrent_slot[index])
    }
}

pub type View = Snapshot;

#[derive(Clone, Copy, Debug, Default)]
pub struct ReserveContext {
    err: KvError,
    accepted: bool,
    operation_error: KvError,
    resolved_max_sequences: i32,
    resolved_max_blocks: i32,
    resolved_block_tokens: i32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AllocateSequenceContext {
    err: KvError,
    accepted: bool,
    operation_error: KvError,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AllocateSlotsContext {
    err: KvError,
    accepted: bool,
    operation_error: KvError,
    block_count: i32,
    old_length: i32,
    new_length: i32,
    old_blocks: i32,
    new_blocks: i32,
    existing_block_count: i32,
    blocks_needed: i32,
    tail_split_needed: bool,
    physical_blocks_needed: i32,
    copy_accepted: bool,
    copy_error: KvError,
    linked_count: i32,
    unlinked_count: i32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BranchSequenceContext {
    err: KvError,
    accepted: bool,
    operation_error: KvError,
    parent_blocks: i32,
    linked_count: i32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FreeSequenceContext {
    err: KvError,
    accepted: bool,
    operation_error: KvError,
    block_count: i32,
    unlinked_count: i32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RollbackSlotsContext {
    err: KvError,
    accepted: bool,
    operation_error: KvError,
    block_count: i32,
    current_length: i32,
    new_length: i32,
    existing_block_count: i32,
    new_blocks: i32,
    remove_count: i32,
    unlinked_count: i32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CaptureViewContext {
    err: KvError,
    accepted: bool,
    operation_error: KvError,
}

#[derive(Clone)]
pub struct EventAllocateSequenceRuntime<'event> {
    pub seq_id: i32,
    pub error_out: Option<&'event RefCell<i32>>,
    pub context: &'event RefCell<AllocateSequenceContext>,
}

#[derive(Clone)]
pub struct EventAllocateSlotsRuntime<'event> {
    pub seq_id: i32,
    pub token_count: i32,
    pub block_count_out: Option<&'event RefCell<i32>>,
    pub error_out: Option<&'event RefCell<i32>>,
    pub copy_block: Option<&'event dyn BlockCopier>,
    pub context: &'event RefCell<AllocateSlotsContext>,
}

#[derive(Clone)]
pub struct EventBranchSequenceRuntime<'event> {
    pub parent_seq_id: i32,
    pub child_seq_id: i32,
    pub copy_state: Option<&'event dyn StateCopier>,
    pub error_out: Option<&'event RefCell<i32>>,
    pub context: &'event RefCell<BranchSequenceContext>,
}

#[derive(Clone)]
pub struct EventCaptureViewRuntime<'event> {
    pub snapshot_out: Option<&'event RefCell<Snapshot>>,
    pub error_out: Option<&'event RefCell<i32>>,
    pub context: &'event RefCell<CaptureViewContext>,
}

#[derive(Clone)]
pub struct EventFreeSequenceRuntime<'event> {
    pub seq_id: i32,
    pub error_out: Option<&'event RefCell<i32>>,
    pub context: &'event RefCell<FreeSequenceContext>,
}

#[derive(Clone)]
pub struct EventReserveRuntime<'event> {
    pub max_sequences: i32,
    pub max_blocks: i32,
    pub block_tokens: i32,
    pub error_out: Option<&'event RefCell<i32>>,
    pub context: &'event RefCell<ReserveContext>,
}

#[derive(Clone)]
pub struct EventRollbackSlotsRuntime<'event> {
    pub seq_id: i32,
    pub token_count: i32,
    pub block_count_out: Option<&'event RefCell<i32>>,
    pub error_out: Option<&'event RefCell<i32>>,
    pub context: &'event RefCell<RollbackSlotsContext>,
}

sml! {
    MemoryKv<'event> {
        "reserve_request_decision"_s <= *"ready"_s + event<EventReserveRuntime<'event>> / begin_reserve,
        "reserve_exec"_s <= "reserve_request_decision"_s + completion<EventReserveRuntime>(EventReserveRuntime<'event>) [reserve_request_valid],
        "errored"_s <= "reserve_request_decision"_s + completion<EventReserveRuntime>(EventReserveRuntime<'event>) [reserve_request_invalid] / mark_invalid_request_event_reserve_runtime,
        "reserve_result_decision"_s <= "reserve_exec"_s + completion<EventReserveRuntime>(EventReserveRuntime<'event>) / exec_reserve,
        "done"_s <= "reserve_result_decision"_s + completion<EventReserveRuntime>(EventReserveRuntime<'event>) [operation_succeeded_event_reserve_runtime],
        "errored"_s <= "reserve_result_decision"_s + completion<EventReserveRuntime>(EventReserveRuntime<'event>) [operation_failed_with_error_event_reserve_runtime] / mark_error_from_operation_event_reserve_runtime,
        "errored"_s <= "reserve_result_decision"_s + completion<EventReserveRuntime>(EventReserveRuntime<'event>) [operation_failed_without_error_event_reserve_runtime] / mark_backend_error_event_reserve_runtime,
        "allocate_sequence_request_decision"_s <= "ready"_s + event<EventAllocateSequenceRuntime<'event>> / begin_allocate_sequence,
        "allocate_sequence_exec"_s <= "allocate_sequence_request_decision"_s + completion<EventAllocateSequenceRuntime>(EventAllocateSequenceRuntime<'event>) [allocate_sequence_request_valid],
        "errored"_s <= "allocate_sequence_request_decision"_s + completion<EventAllocateSequenceRuntime>(EventAllocateSequenceRuntime<'event>) [allocate_sequence_request_invalid] / mark_invalid_request_event_allocate_sequence_runtime,
        "allocate_sequence_result_decision"_s <= "allocate_sequence_exec"_s + completion<EventAllocateSequenceRuntime>(EventAllocateSequenceRuntime<'event>) / exec_allocate_sequence,
        "done"_s <= "allocate_sequence_result_decision"_s + completion<EventAllocateSequenceRuntime>(EventAllocateSequenceRuntime<'event>) [operation_succeeded_event_allocate_sequence_runtime],
        "errored"_s <= "allocate_sequence_result_decision"_s + completion<EventAllocateSequenceRuntime>(EventAllocateSequenceRuntime<'event>) [operation_failed_with_error_event_allocate_sequence_runtime] / mark_error_from_operation_event_allocate_sequence_runtime,
        "errored"_s <= "allocate_sequence_result_decision"_s + completion<EventAllocateSequenceRuntime>(EventAllocateSequenceRuntime<'event>) [operation_failed_without_error_event_allocate_sequence_runtime] / mark_backend_error_event_allocate_sequence_runtime,
        "allocate_slots_request_decision"_s <= "ready"_s + event<EventAllocateSlotsRuntime<'event>> / begin_allocate_slots,
        "allocate_slots_request_shape_decision"_s <= "allocate_slots_request_decision"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>),
        "allocate_slots_request_length_decision"_s <= "allocate_slots_request_shape_decision"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) [allocate_slots_request_shape_valid],
        "errored"_s <= "allocate_slots_request_shape_decision"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) [allocate_slots_request_shape_invalid] / mark_invalid_request_event_allocate_slots_runtime,
        "allocate_slots_request_block_layout_decision"_s <= "allocate_slots_request_length_decision"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) [allocate_slots_request_length_valid],
        "errored"_s <= "allocate_slots_request_length_decision"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) [allocate_slots_request_length_invalid] / mark_invalid_request_event_allocate_slots_runtime,
        "allocate_slots_request_capacity_decision"_s <= "allocate_slots_request_block_layout_decision"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) [allocate_slots_request_block_layout_valid],
        "errored"_s <= "allocate_slots_request_block_layout_decision"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) [allocate_slots_request_block_layout_invalid] / mark_backend_error_event_allocate_slots_runtime,
        "state_allocate_slots_tail_decision"_s <= "allocate_slots_request_capacity_decision"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) [allocate_slots_request_capacity_valid],
        "errored"_s <= "allocate_slots_request_capacity_decision"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) [allocate_slots_request_capacity_invalid] / mark_out_of_memory,
        "state_allocate_slots_tail_copy_exec"_s <= "state_allocate_slots_tail_decision"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) [guard_allocate_slots_shared_tail_copy_ready],
        "errored"_s <= "state_allocate_slots_tail_decision"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) [guard_allocate_slots_shared_tail_copy_missing] / mark_invalid_request_event_allocate_slots_runtime,
        "state_allocate_slots_direct_exec"_s <= "state_allocate_slots_tail_decision"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) [guard_allocate_slots_shared_tail_split_not_required],
        "state_allocate_slots_tail_copy_result_decision"_s <= "state_allocate_slots_tail_copy_exec"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) / effect_copy_shared_tail_block,
        "state_allocate_slots_tail_split_exec"_s <= "state_allocate_slots_tail_copy_result_decision"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) [operation_succeeded_event_allocate_slots_runtime],
        "errored"_s <= "state_allocate_slots_tail_copy_result_decision"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) [operation_failed_with_error_event_allocate_slots_runtime] / mark_error_from_operation_event_allocate_slots_runtime,
        "errored"_s <= "state_allocate_slots_tail_copy_result_decision"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) [operation_failed_without_error_event_allocate_slots_runtime] / mark_backend_error_event_allocate_slots_runtime,
        "allocate_slots_result_decision"_s <= "state_allocate_slots_tail_split_exec"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) / effect_allocate_slots_with_tail_split,
        "allocate_slots_result_decision"_s <= "state_allocate_slots_direct_exec"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) / effect_allocate_slots_without_tail_split,
        "done"_s <= "allocate_slots_result_decision"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) [operation_succeeded_event_allocate_slots_runtime],
        "errored"_s <= "allocate_slots_result_decision"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) [operation_failed_with_error_event_allocate_slots_runtime] / mark_error_from_operation_event_allocate_slots_runtime,
        "errored"_s <= "allocate_slots_result_decision"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) [operation_failed_without_error_event_allocate_slots_runtime] / mark_backend_error_event_allocate_slots_runtime,
        "branch_sequence_request_decision"_s <= "ready"_s + event<EventBranchSequenceRuntime<'event>> / begin_branch_sequence,
        "branch_sequence_exec"_s <= "branch_sequence_request_decision"_s + completion<EventBranchSequenceRuntime>(EventBranchSequenceRuntime<'event>) [branch_sequence_request_valid],
        "errored"_s <= "branch_sequence_request_decision"_s + completion<EventBranchSequenceRuntime>(EventBranchSequenceRuntime<'event>) [branch_sequence_request_invalid] / mark_invalid_request_event_branch_sequence_runtime,
        "branch_sequence_result_decision"_s <= "branch_sequence_exec"_s + completion<EventBranchSequenceRuntime>(EventBranchSequenceRuntime<'event>) / exec_branch_sequence,
        "done"_s <= "branch_sequence_result_decision"_s + completion<EventBranchSequenceRuntime>(EventBranchSequenceRuntime<'event>) [operation_succeeded_event_branch_sequence_runtime],
        "errored"_s <= "branch_sequence_result_decision"_s + completion<EventBranchSequenceRuntime>(EventBranchSequenceRuntime<'event>) [operation_failed_with_error_event_branch_sequence_runtime] / mark_error_from_operation_event_branch_sequence_runtime,
        "errored"_s <= "branch_sequence_result_decision"_s + completion<EventBranchSequenceRuntime>(EventBranchSequenceRuntime<'event>) [operation_failed_without_error_event_branch_sequence_runtime] / mark_backend_error_event_branch_sequence_runtime,
        "free_sequence_request_decision"_s <= "ready"_s + event<EventFreeSequenceRuntime<'event>> / begin_free_sequence,
        "free_sequence_exec"_s <= "free_sequence_request_decision"_s + completion<EventFreeSequenceRuntime>(EventFreeSequenceRuntime<'event>) [free_sequence_request_valid],
        "errored"_s <= "free_sequence_request_decision"_s + completion<EventFreeSequenceRuntime>(EventFreeSequenceRuntime<'event>) [free_sequence_request_invalid] / mark_invalid_request_event_free_sequence_runtime,
        "free_sequence_result_decision"_s <= "free_sequence_exec"_s + completion<EventFreeSequenceRuntime>(EventFreeSequenceRuntime<'event>) / exec_free_sequence,
        "done"_s <= "free_sequence_result_decision"_s + completion<EventFreeSequenceRuntime>(EventFreeSequenceRuntime<'event>) [operation_succeeded_event_free_sequence_runtime],
        "errored"_s <= "free_sequence_result_decision"_s + completion<EventFreeSequenceRuntime>(EventFreeSequenceRuntime<'event>) [operation_failed_with_error_event_free_sequence_runtime] / mark_error_from_operation_event_free_sequence_runtime,
        "errored"_s <= "free_sequence_result_decision"_s + completion<EventFreeSequenceRuntime>(EventFreeSequenceRuntime<'event>) [operation_failed_without_error_event_free_sequence_runtime] / mark_backend_error_event_free_sequence_runtime,
        "rollback_slots_request_decision"_s <= "ready"_s + event<EventRollbackSlotsRuntime<'event>> / begin_rollback_slots,
        "rollback_slots_exec"_s <= "rollback_slots_request_decision"_s + completion<EventRollbackSlotsRuntime>(EventRollbackSlotsRuntime<'event>) [rollback_slots_request_valid],
        "errored"_s <= "rollback_slots_request_decision"_s + completion<EventRollbackSlotsRuntime>(EventRollbackSlotsRuntime<'event>) [rollback_slots_request_invalid] / mark_invalid_request_event_rollback_slots_runtime,
        "rollback_slots_result_decision"_s <= "rollback_slots_exec"_s + completion<EventRollbackSlotsRuntime>(EventRollbackSlotsRuntime<'event>) / exec_rollback_slots,
        "done"_s <= "rollback_slots_result_decision"_s + completion<EventRollbackSlotsRuntime>(EventRollbackSlotsRuntime<'event>) [operation_succeeded_event_rollback_slots_runtime],
        "errored"_s <= "rollback_slots_result_decision"_s + completion<EventRollbackSlotsRuntime>(EventRollbackSlotsRuntime<'event>) [operation_failed_with_error_event_rollback_slots_runtime] / mark_error_from_operation_event_rollback_slots_runtime,
        "errored"_s <= "rollback_slots_result_decision"_s + completion<EventRollbackSlotsRuntime>(EventRollbackSlotsRuntime<'event>) [operation_failed_without_error_event_rollback_slots_runtime] / mark_backend_error_event_rollback_slots_runtime,
        "capture_request_decision"_s <= "ready"_s + event<EventCaptureViewRuntime<'event>> / begin_capture_view,
        "capture_exec"_s <= "capture_request_decision"_s + completion<EventCaptureViewRuntime>(EventCaptureViewRuntime<'event>) [capture_request_valid],
        "errored"_s <= "capture_request_decision"_s + completion<EventCaptureViewRuntime>(EventCaptureViewRuntime<'event>) [capture_request_invalid] / mark_invalid_request_event_capture_view_runtime,
        "capture_result_decision"_s <= "capture_exec"_s + completion<EventCaptureViewRuntime>(EventCaptureViewRuntime<'event>) / exec_capture_view,
        "done"_s <= "capture_result_decision"_s + completion<EventCaptureViewRuntime>(EventCaptureViewRuntime<'event>) [operation_succeeded_event_capture_view_runtime],
        "errored"_s <= "capture_result_decision"_s + completion<EventCaptureViewRuntime>(EventCaptureViewRuntime<'event>) [operation_failed_with_error_event_capture_view_runtime] / mark_error_from_operation_event_capture_view_runtime,
        "errored"_s <= "capture_result_decision"_s + completion<EventCaptureViewRuntime>(EventCaptureViewRuntime<'event>) [operation_failed_without_error_event_capture_view_runtime] / mark_backend_error_event_capture_view_runtime,
        "ready"_s <= "done"_s + completion<EventReserveRuntime>(EventReserveRuntime<'event>) / publish_done_event_reserve_runtime,
        "ready"_s <= "errored"_s + completion<EventReserveRuntime>(EventReserveRuntime<'event>) / publish_error_event_reserve_runtime,
        "ready"_s <= "done"_s + completion<EventAllocateSequenceRuntime>(EventAllocateSequenceRuntime<'event>) / publish_done_event_allocate_sequence_runtime,
        "ready"_s <= "errored"_s + completion<EventAllocateSequenceRuntime>(EventAllocateSequenceRuntime<'event>) / publish_error_event_allocate_sequence_runtime,
        "ready"_s <= "done"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) / publish_done_event_allocate_slots_runtime,
        "ready"_s <= "errored"_s + completion<EventAllocateSlotsRuntime>(EventAllocateSlotsRuntime<'event>) / publish_error_event_allocate_slots_runtime,
        "ready"_s <= "done"_s + completion<EventBranchSequenceRuntime>(EventBranchSequenceRuntime<'event>) / publish_done_event_branch_sequence_runtime,
        "ready"_s <= "errored"_s + completion<EventBranchSequenceRuntime>(EventBranchSequenceRuntime<'event>) / publish_error_event_branch_sequence_runtime,
        "ready"_s <= "done"_s + completion<EventFreeSequenceRuntime>(EventFreeSequenceRuntime<'event>) / publish_done_event_free_sequence_runtime,
        "ready"_s <= "errored"_s + completion<EventFreeSequenceRuntime>(EventFreeSequenceRuntime<'event>) / publish_error_event_free_sequence_runtime,
        "ready"_s <= "done"_s + completion<EventRollbackSlotsRuntime>(EventRollbackSlotsRuntime<'event>) / publish_done_event_rollback_slots_runtime,
        "ready"_s <= "errored"_s + completion<EventRollbackSlotsRuntime>(EventRollbackSlotsRuntime<'event>) / publish_error_event_rollback_slots_runtime,
        "ready"_s <= "done"_s + completion<EventCaptureViewRuntime>(EventCaptureViewRuntime<'event>) / publish_done_event_capture_view_runtime,
        "ready"_s <= "errored"_s + completion<EventCaptureViewRuntime>(EventCaptureViewRuntime<'event>) / publish_error_event_capture_view_runtime,
        "ready"_s <= "ready"_s + unexpected_event<_> / on_unexpected_from_ready,
        "ready"_s <= "reserve_request_decision"_s + unexpected_event<_> / on_unexpected_from_reserve_request_decision,
        "ready"_s <= "reserve_exec"_s + unexpected_event<_> / on_unexpected_from_reserve_exec,
        "ready"_s <= "reserve_result_decision"_s + unexpected_event<_> / on_unexpected_from_reserve_result_decision,
        "ready"_s <= "allocate_sequence_request_decision"_s + unexpected_event<_> / on_unexpected_from_allocate_sequence_request_decision,
        "ready"_s <= "allocate_sequence_exec"_s + unexpected_event<_> / on_unexpected_from_allocate_sequence_exec,
        "ready"_s <= "allocate_sequence_result_decision"_s + unexpected_event<_> / on_unexpected_from_allocate_sequence_result_decision,
        "ready"_s <= "allocate_slots_request_decision"_s + unexpected_event<_> / on_unexpected_from_allocate_slots_request_decision,
        "ready"_s <= "allocate_slots_request_shape_decision"_s + unexpected_event<_> / on_unexpected_from_allocate_slots_request_shape_decision,
        "ready"_s <= "allocate_slots_request_length_decision"_s + unexpected_event<_> / on_unexpected_from_allocate_slots_request_length_decision,
        "ready"_s <= "allocate_slots_request_block_layout_decision"_s + unexpected_event<_> / on_unexpected_from_allocate_slots_request_block_layout_decision,
        "ready"_s <= "allocate_slots_request_capacity_decision"_s + unexpected_event<_> / on_unexpected_from_allocate_slots_request_capacity_decision,
        "ready"_s <= "state_allocate_slots_tail_decision"_s + unexpected_event<_> / on_unexpected_from_state_allocate_slots_tail_decision,
        "ready"_s <= "state_allocate_slots_tail_copy_exec"_s + unexpected_event<_> / on_unexpected_from_state_allocate_slots_tail_copy_exec,
        "ready"_s <= "state_allocate_slots_tail_copy_result_decision"_s + unexpected_event<_> / on_unexpected_from_state_allocate_slots_tail_copy_result_decision,
        "ready"_s <= "state_allocate_slots_direct_exec"_s + unexpected_event<_> / on_unexpected_from_state_allocate_slots_direct_exec,
        "ready"_s <= "state_allocate_slots_tail_split_exec"_s + unexpected_event<_> / on_unexpected_from_state_allocate_slots_tail_split_exec,
        "ready"_s <= "allocate_slots_result_decision"_s + unexpected_event<_> / on_unexpected_from_allocate_slots_result_decision,
        "ready"_s <= "branch_sequence_request_decision"_s + unexpected_event<_> / on_unexpected_from_branch_sequence_request_decision,
        "ready"_s <= "branch_sequence_exec"_s + unexpected_event<_> / on_unexpected_from_branch_sequence_exec,
        "ready"_s <= "branch_sequence_result_decision"_s + unexpected_event<_> / on_unexpected_from_branch_sequence_result_decision,
        "ready"_s <= "free_sequence_request_decision"_s + unexpected_event<_> / on_unexpected_from_free_sequence_request_decision,
        "ready"_s <= "free_sequence_exec"_s + unexpected_event<_> / on_unexpected_from_free_sequence_exec,
        "ready"_s <= "free_sequence_result_decision"_s + unexpected_event<_> / on_unexpected_from_free_sequence_result_decision,
        "ready"_s <= "rollback_slots_request_decision"_s + unexpected_event<_> / on_unexpected_from_rollback_slots_request_decision,
        "ready"_s <= "rollback_slots_exec"_s + unexpected_event<_> / on_unexpected_from_rollback_slots_exec,
        "ready"_s <= "rollback_slots_result_decision"_s + unexpected_event<_> / on_unexpected_from_rollback_slots_result_decision,
        "ready"_s <= "capture_request_decision"_s + unexpected_event<_> / on_unexpected_from_capture_request_decision,
        "ready"_s <= "capture_exec"_s + unexpected_event<_> / on_unexpected_from_capture_exec,
        "ready"_s <= "capture_result_decision"_s + unexpected_event<_> / on_unexpected_from_capture_result_decision,
        "ready"_s <= "done"_s + unexpected_event<_> / on_unexpected_from_done,
        "ready"_s <= "errored"_s + unexpected_event<_> / on_unexpected_from_errored,
    }
}
