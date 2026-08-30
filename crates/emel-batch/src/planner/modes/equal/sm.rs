//! Equal fixed-size batch planning algorithm.

#![allow(
    clippy::enum_variant_names,
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::empty_structs_with_brackets,
    clippy::missing_const_for_fn,
    dead_code,
    unused_imports,
    missing_docs,
    private_interfaces
)]

use crate::planner::sm::{
    mask_equal, mask_overlaps, normalized_seq_mask, PlanRuntime, PlannerError, SeqMask,
    MAX_PLAN_STEPS, MAX_SEQ,
};
use sml::sml;

// --- machine BatchPlannerModesEqual from emel.cpp/src/emel/batch/planner/modes/equal/sm.hpp ---
pub(crate) type EventPlanRuntime = PlanRuntime;

sml! {
    BatchPlannerModesEqual {
        "state_planning"_s <= *"state_preparing"_s + Plan(PlanRuntime) / effect_begin_planning,
        "state_planning_mode_decision"_s <= "state_planning"_s + completion<Plan>(PlanRuntime),
        "state_planning_fast_input_decision"_s <= "state_planning_mode_decision"_s + completion<Plan>(PlanRuntime) [guard_mode_is_primary_fast_path],
        "state_planning_general_input_decision"_s <= "state_planning_mode_decision"_s + completion<Plan>(PlanRuntime) [guard_mode_is_general_path],
        "state_planning_general_capacity_decision"_s <= "state_planning_general_input_decision"_s + completion<Plan>(PlanRuntime) [guard_general_input_valid],
        "state_planning_failed"_s <= "state_planning_general_input_decision"_s + completion<Plan>(PlanRuntime) [guard_has_invalid_step_size] / effect_reject_invalid_step_size_from_state_planning_general_input_decision,
        "state_planning_failed"_s <= "state_planning_general_capacity_decision"_s + completion<Plan>(PlanRuntime) [guard_lacks_step_capacity] / effect_reject_output_steps_full_from_state_planning_general_capacity_decision,
        "state_planning_failed"_s <= "state_planning_general_capacity_decision"_s + completion<Plan>(PlanRuntime) [guard_lacks_index_capacity] / effect_reject_output_indices_full_from_state_planning_general_capacity_decision,
        "state_planning_general_execute"_s <= "state_planning_general_capacity_decision"_s + completion<Plan>(PlanRuntime) [guard_storage_capacity_valid],
        "state_planning_general_result_decision"_s <= "state_planning_general_execute"_s + completion<Plan>(PlanRuntime) / effect_plan_equal_batches,
        "state_planning_failed"_s <= "state_planning_fast_input_decision"_s + completion<Plan>(PlanRuntime) [guard_has_invalid_step_size] / effect_reject_invalid_step_size_from_state_planning_fast_input_decision,
        "state_planning_failed"_s <= "state_planning_fast_input_decision"_s + completion<Plan>(PlanRuntime) [guard_fast_path_missing_primary_ids] / effect_reject_invalid_sequence_id_from_state_planning_fast_input_decision,
        "state_planning_failed"_s <= "state_planning_fast_input_decision"_s + completion<Plan>(PlanRuntime) [guard_fast_path_primary_ids_invalid] / effect_reject_invalid_sequence_id_from_state_planning_fast_input_decision,
        "state_planning_fast_capacity_decision"_s <= "state_planning_fast_input_decision"_s + completion<Plan>(PlanRuntime) [guard_fast_path_input_valid],
        "state_planning_failed"_s <= "state_planning_fast_capacity_decision"_s + completion<Plan>(PlanRuntime) [guard_lacks_step_capacity] / effect_reject_output_steps_full_from_state_planning_fast_capacity_decision,
        "state_planning_failed"_s <= "state_planning_fast_capacity_decision"_s + completion<Plan>(PlanRuntime) [guard_lacks_index_capacity] / effect_reject_output_indices_full_from_state_planning_fast_capacity_decision,
        "state_planning_fast_execute"_s <= "state_planning_fast_capacity_decision"_s + completion<Plan>(PlanRuntime) [guard_storage_capacity_valid],
        "state_planning_fast_result_decision"_s <= "state_planning_fast_execute"_s + completion<Plan>(PlanRuntime) / effect_plan_equal_primary_batches,
        "state_planning_done"_s <= "state_planning_general_result_decision"_s + completion<Plan>(PlanRuntime) [guard_planning_succeeded] / effect_emit_plan_done_from_state_planning_general_result_decision,
        "state_planning_failed"_s <= "state_planning_general_result_decision"_s + completion<Plan>(PlanRuntime) [guard_planning_failed] / effect_reject_planning_progress_stalled_from_state_planning_general_result_decision,
        "state_planning_done"_s <= "state_planning_fast_result_decision"_s + completion<Plan>(PlanRuntime) [guard_planning_succeeded] / effect_emit_plan_done_from_state_planning_fast_result_decision,
        "state_planning_failed"_s <= "state_planning_fast_result_decision"_s + completion<Plan>(PlanRuntime) [guard_planning_failed] / effect_reject_planning_progress_stalled_from_state_planning_fast_result_decision,
        "state_planning_failed"_s <= "state_preparing"_s + unexpected_event<_> / effect_emit_internal_plan_error_from_state_preparing,
        "state_planning_failed"_s <= "state_planning"_s + unexpected_event<_> / effect_emit_internal_plan_error_from_state_planning,
        "state_planning_failed"_s <= "state_planning_mode_decision"_s + unexpected_event<_> / effect_emit_internal_plan_error_from_state_planning_mode_decision,
        "state_planning_failed"_s <= "state_planning_fast_input_decision"_s + unexpected_event<_> / effect_emit_internal_plan_error_from_state_planning_fast_input_decision,
        "state_planning_failed"_s <= "state_planning_fast_capacity_decision"_s + unexpected_event<_> / effect_emit_internal_plan_error_from_state_planning_fast_capacity_decision,
        "state_planning_failed"_s <= "state_planning_fast_execute"_s + unexpected_event<_> / effect_emit_internal_plan_error_from_state_planning_fast_execute,
        "state_planning_failed"_s <= "state_planning_general_input_decision"_s + unexpected_event<_> / effect_emit_internal_plan_error_from_state_planning_general_input_decision,
        "state_planning_failed"_s <= "state_planning_general_capacity_decision"_s + unexpected_event<_> / effect_emit_internal_plan_error_from_state_planning_general_capacity_decision,
        "state_planning_failed"_s <= "state_planning_general_execute"_s + unexpected_event<_> / effect_emit_internal_plan_error_from_state_planning_general_execute,
        "state_planning_failed"_s <= "state_planning_general_result_decision"_s + unexpected_event<_> / effect_emit_internal_plan_error_from_state_planning_general_result_decision,
        "state_planning_failed"_s <= "state_planning_fast_result_decision"_s + unexpected_event<_> / effect_emit_internal_plan_error_from_state_planning_fast_result_decision,
        "state_planning_failed"_s <= "state_planning_done"_s + unexpected_event<_> / effect_emit_internal_plan_error_from_state_planning_done,
        "state_planning_failed"_s <= "state_planning_failed"_s + unexpected_event<_> / effect_emit_internal_plan_error_from_state_planning_failed,
    }
}

#[derive(Debug, Default)]
pub struct BatchPlannerModesEqualContext;

fn fail(runtime: &PlanRuntime, error: PlannerError) {
    runtime.scratch.borrow_mut().fail(error);
}

fn finalize_offsets(runtime: &PlanRuntime) {
    let mut scratch = runtime.scratch.borrow_mut();
    let offset = scratch.step_token_indices.len();
    if !scratch.step_token_offsets.push(offset) {
        scratch.fail(PlannerError::OutputStepsFull);
    }
}

fn plan_equal_batches(runtime: &PlanRuntime) {
    let size = runtime.scratch.borrow().effective_step_size;
    let n = runtime.request.token_ids.len();
    if size == 0 {
        fail(runtime, PlannerError::InvalidStepSize);
        return;
    }

    let mut used = [false; MAX_PLAN_STEPS];
    let mut used_count = 0usize;
    while used_count < n {
        let mut groups = [SeqMask::default(); MAX_PLAN_STEPS];
        let mut group_count = 0usize;
        let mut last_primary = -1i32;

        for index in 0..n {
            if used[index] {
                continue;
            }
            let mask = normalized_seq_mask(&runtime.request, index);
            if (0..group_count).any(|group| mask_overlaps(groups[group], mask)) {
                continue;
            }
            if runtime.request.equal_sequential {
                if let Some(ids) = runtime.request.seq_primary_ids.as_ref() {
                    let primary = ids.get(index).copied().unwrap_or(-1);
                    if group_count > 0 && primary != last_primary + 1 {
                        continue;
                    }
                    last_primary = primary;
                }
            }
            groups[group_count] = mask;
            group_count += 1;
            if group_count >= size {
                break;
            }
        }

        if group_count == 0 {
            fail(runtime, PlannerError::PlanningProgressStalled);
            return;
        }

        let mut min_available = n + 1;
        for group in 0..group_count {
            let available = (0..n)
                .filter(|&index| {
                    !used[index]
                        && mask_equal(normalized_seq_mask(&runtime.request, index), groups[group])
                })
                .count();
            min_available = min_available.min(available);
        }
        let rows = (size / group_count).min(min_available);
        if rows == 0 {
            fail(runtime, PlannerError::PlanningProgressStalled);
            return;
        }

        {
            let mut scratch = runtime.scratch.borrow_mut();
            let offset = scratch.step_token_indices.len();
            if !scratch.step_token_offsets.push(offset) {
                scratch.fail(PlannerError::OutputStepsFull);
                return;
            }
        }
        for group in 0..group_count {
            let mut remaining = rows;
            for index in 0..n {
                if remaining == 0 {
                    break;
                }
                if used[index]
                    || !mask_equal(normalized_seq_mask(&runtime.request, index), groups[group])
                {
                    continue;
                }
                used[index] = true;
                used_count += 1;
                if !runtime.scratch.borrow_mut().step_token_indices.push(index) {
                    fail(runtime, PlannerError::OutputIndicesFull);
                    return;
                }
                remaining -= 1;
            }
            if remaining != 0 {
                fail(runtime, PlannerError::AlgorithmFailed);
                return;
            }
        }
        if !runtime
            .scratch
            .borrow_mut()
            .step_sizes
            .push(rows * group_count)
        {
            fail(runtime, PlannerError::OutputStepsFull);
            return;
        }
    }
    finalize_offsets(runtime);
}

fn plan_equal_primary_batches(runtime: &PlanRuntime) {
    let size = runtime.scratch.borrow().effective_step_size;
    let n = runtime.request.token_ids.len();
    let Some(ids) = runtime.request.seq_primary_ids.as_ref() else {
        fail(runtime, PlannerError::InvalidSequenceId);
        return;
    };
    if size == 0 {
        fail(runtime, PlannerError::InvalidStepSize);
        return;
    }
    let max_seq = runtime.request.seq_mask_words.saturating_mul(64);
    if max_seq == 0 || max_seq > MAX_SEQ || ids.len() < n {
        fail(runtime, PlannerError::InvalidSequenceId);
        return;
    }

    let mut counts = [0usize; MAX_SEQ];
    let mut offsets = [0usize; MAX_SEQ + 1];
    let mut used = [0usize; MAX_SEQ];
    let mut cursor = [0usize; MAX_SEQ];
    let mut sequence_indices = [0usize; MAX_PLAN_STEPS];
    for &id in &ids[..n] {
        if id < 0 || (id as usize) >= max_seq {
            fail(runtime, PlannerError::InvalidSequenceId);
            return;
        }
        counts[id as usize] += 1;
    }
    for sequence in 0..max_seq {
        offsets[sequence + 1] = offsets[sequence] + counts[sequence];
        cursor[sequence] = offsets[sequence];
    }
    for index in 0..n {
        let sequence = ids[index] as usize;
        let position = cursor[sequence];
        if position >= n {
            fail(runtime, PlannerError::AlgorithmFailed);
            return;
        }
        sequence_indices[position] = index;
        cursor[sequence] += 1;
    }

    let mut remaining = n;
    while remaining > 0 {
        let mut group_used = [false; MAX_SEQ];
        let mut group_ids = [0usize; MAX_SEQ];
        let mut group_count = 0usize;
        let mut last_primary = -1i32;
        for index in 0..n {
            let sequence = ids[index] as usize;
            if used[sequence] >= counts[sequence] || group_used[sequence] {
                continue;
            }
            if runtime.request.equal_sequential
                && group_count > 0
                && sequence as i32 != last_primary + 1
            {
                continue;
            }
            group_used[sequence] = true;
            group_ids[group_count] = sequence;
            group_count += 1;
            last_primary = sequence as i32;
            if group_count >= size {
                break;
            }
        }
        if group_count == 0 {
            fail(runtime, PlannerError::PlanningProgressStalled);
            return;
        }
        let min_available = (0..group_count)
            .map(|group| counts[group_ids[group]] - used[group_ids[group]])
            .min()
            .unwrap_or(0);
        let rows = (size / group_count).min(min_available);
        if rows == 0 {
            fail(runtime, PlannerError::PlanningProgressStalled);
            return;
        }
        {
            let mut scratch = runtime.scratch.borrow_mut();
            let offset = scratch.step_token_indices.len();
            if !scratch.step_token_offsets.push(offset) {
                scratch.fail(PlannerError::OutputStepsFull);
                return;
            }
        }
        for group in 0..group_count {
            let sequence = group_ids[group];
            let base = offsets[sequence] + used[sequence];
            for position in 0..rows {
                let Some(&index) = sequence_indices.get(base + position) else {
                    fail(runtime, PlannerError::AlgorithmFailed);
                    return;
                };
                if !runtime.scratch.borrow_mut().step_token_indices.push(index) {
                    fail(runtime, PlannerError::OutputIndicesFull);
                    return;
                }
            }
            used[sequence] += rows;
            remaining -= rows;
        }
        if !runtime
            .scratch
            .borrow_mut()
            .step_sizes
            .push(rows * group_count)
        {
            fail(runtime, PlannerError::OutputStepsFull);
            return;
        }
    }
    finalize_offsets(runtime);
}

impl BatchPlannerModesEqualStateMachineContext for BatchPlannerModesEqualContext {
    fn effect_begin_planning(&mut self, event: PlanRuntime) -> Result<(), ()> {
        event.scratch.borrow_mut().reset();
        Ok(())
    }

    fn effect_emit_internal_plan_error_from_state_planning(&mut self) -> Result<(), ()> { Ok(()) }
    fn effect_emit_internal_plan_error_from_state_planning_done(&mut self) -> Result<(), ()> { Ok(()) }
    fn effect_emit_internal_plan_error_from_state_planning_failed(&mut self) -> Result<(), ()> { Ok(()) }
    fn effect_emit_internal_plan_error_from_state_planning_fast_capacity_decision(&mut self) -> Result<(), ()> { Ok(()) }
    fn effect_emit_internal_plan_error_from_state_planning_fast_execute(&mut self) -> Result<(), ()> { Ok(()) }
    fn effect_emit_internal_plan_error_from_state_planning_fast_input_decision(&mut self) -> Result<(), ()> { Ok(()) }
    fn effect_emit_internal_plan_error_from_state_planning_fast_result_decision(&mut self) -> Result<(), ()> { Ok(()) }
    fn effect_emit_internal_plan_error_from_state_planning_general_capacity_decision(&mut self) -> Result<(), ()> { Ok(()) }
    fn effect_emit_internal_plan_error_from_state_planning_general_execute(&mut self) -> Result<(), ()> { Ok(()) }
    fn effect_emit_internal_plan_error_from_state_planning_general_input_decision(&mut self) -> Result<(), ()> { Ok(()) }
    fn effect_emit_internal_plan_error_from_state_planning_general_result_decision(&mut self) -> Result<(), ()> { Ok(()) }
    fn effect_emit_internal_plan_error_from_state_planning_mode_decision(&mut self) -> Result<(), ()> { Ok(()) }
    fn effect_emit_internal_plan_error_from_state_preparing(&mut self) -> Result<(), ()> { Ok(()) }

    fn effect_emit_plan_done_from_state_planning_fast_result_decision(&mut self, _event: PlanRuntime) -> Result<(), ()> { Ok(()) }
    fn effect_emit_plan_done_from_state_planning_general_result_decision(&mut self, _event: PlanRuntime) -> Result<(), ()> { Ok(()) }
    fn effect_plan_equal_batches(&mut self, event: PlanRuntime) -> Result<(), ()> {
        plan_equal_batches(&event);
        Ok(())
    }
    fn effect_plan_equal_primary_batches(&mut self, event: PlanRuntime) -> Result<(), ()> {
        plan_equal_primary_batches(&event);
        Ok(())
    }
    fn effect_reject_invalid_sequence_id_from_state_planning_fast_input_decision(&mut self, event: PlanRuntime) -> Result<(), ()> {
        fail(&event, PlannerError::InvalidSequenceId);
        Ok(())
    }
    fn effect_reject_invalid_step_size_from_state_planning_fast_input_decision(&mut self, event: PlanRuntime) -> Result<(), ()> {
        fail(&event, PlannerError::InvalidStepSize);
        Ok(())
    }
    fn effect_reject_invalid_step_size_from_state_planning_general_input_decision(&mut self, event: PlanRuntime) -> Result<(), ()> {
        fail(&event, PlannerError::InvalidStepSize);
        Ok(())
    }
    fn effect_reject_output_indices_full_from_state_planning_fast_capacity_decision(&mut self, event: PlanRuntime) -> Result<(), ()> {
        fail(&event, PlannerError::OutputIndicesFull);
        Ok(())
    }
    fn effect_reject_output_indices_full_from_state_planning_general_capacity_decision(&mut self, event: PlanRuntime) -> Result<(), ()> {
        fail(&event, PlannerError::OutputIndicesFull);
        Ok(())
    }
    fn effect_reject_output_steps_full_from_state_planning_fast_capacity_decision(&mut self, event: PlanRuntime) -> Result<(), ()> {
        fail(&event, PlannerError::OutputStepsFull);
        Ok(())
    }
    fn effect_reject_output_steps_full_from_state_planning_general_capacity_decision(&mut self, event: PlanRuntime) -> Result<(), ()> {
        fail(&event, PlannerError::OutputStepsFull);
        Ok(())
    }
    fn effect_reject_planning_progress_stalled_from_state_planning_fast_result_decision(&mut self, event: PlanRuntime) -> Result<(), ()> {
        fail(&event, PlannerError::PlanningProgressStalled);
        Ok(())
    }
    fn effect_reject_planning_progress_stalled_from_state_planning_general_result_decision(&mut self, event: PlanRuntime) -> Result<(), ()> {
        fail(&event, PlannerError::PlanningProgressStalled);
        Ok(())
    }

    fn guard_fast_path_input_valid(&self, event: &PlanRuntime) -> Result<bool, ()> {
        Ok(self.guard_has_valid_step_size(event)?
            && self.guard_fast_path_has_primary_ids(event)?
            && self.guard_fast_path_primary_ids_valid(event)?)
    }
    fn guard_fast_path_missing_primary_ids(&self, event: &PlanRuntime) -> Result<bool, ()> {
        Ok(!self.guard_fast_path_has_primary_ids(event)?)
    }
    fn guard_fast_path_primary_ids_invalid(&self, event: &PlanRuntime) -> Result<bool, ()> {
        Ok(self.guard_fast_path_has_primary_ids(event)? && !self.guard_fast_path_primary_ids_valid(event)?)
    }
    fn guard_general_input_valid(&self, event: &PlanRuntime) -> Result<bool, ()> {
        self.guard_has_valid_step_size(event)
    }
    fn guard_has_invalid_step_size(&self, event: &PlanRuntime) -> Result<bool, ()> {
        Ok(!self.guard_has_valid_step_size(event)?)
    }
    fn guard_lacks_index_capacity(&self, event: &PlanRuntime) -> Result<bool, ()> {
        let scratch = event.scratch.borrow();
        Ok(event.request.token_ids.len() > MAX_PLAN_STEPS.saturating_sub(scratch.step_token_indices.len()))
    }
    fn guard_lacks_step_capacity(&self, event: &PlanRuntime) -> Result<bool, ()> {
        Ok(event.scratch.borrow().step_sizes.len() >= MAX_PLAN_STEPS)
    }
    fn guard_mode_is_general_path(&self, event: &PlanRuntime) -> Result<bool, ()> {
        Ok(!self.guard_mode_is_primary_fast_path(event)?)
    }
    fn guard_mode_is_primary_fast_path(&self, event: &PlanRuntime) -> Result<bool, ()> {
        Ok(event.request.seq_masks.is_none() && event.request.seq_primary_ids.is_some())
    }
    fn guard_planning_failed(&self, event: &PlanRuntime) -> Result<bool, ()> {
        Ok(!self.guard_planning_succeeded(event)?)
    }
    fn guard_planning_succeeded(&self, event: &PlanRuntime) -> Result<bool, ()> {
        let scratch = event.scratch.borrow();
        Ok(scratch.error.is_none()
            && !scratch.step_sizes.is_empty()
            && scratch.step_token_indices.len() == event.request.token_ids.len()
            && scratch.step_token_offsets.len() == scratch.step_sizes.len() + 1)
    }
    fn guard_storage_capacity_valid(&self, event: &PlanRuntime) -> Result<bool, ()> {
        Ok(!self.guard_lacks_step_capacity(event)? && !self.guard_lacks_index_capacity(event)?)
    }

    fn guard_fast_path_has_primary_ids(&self, event: &PlanRuntime) -> Result<bool, ()> {
        Ok(event.request.seq_primary_ids.is_some())
    }
    fn guard_fast_path_primary_ids_valid(&self, event: &PlanRuntime) -> Result<bool, ()> {
        let Some(ids) = event.request.seq_primary_ids.as_ref() else {
            return Ok(false);
        };
        let max_seq = event.request.seq_mask_words.saturating_mul(64);
        Ok(max_seq > 0 && max_seq <= MAX_SEQ
            && ids.len() >= event.request.token_ids.len()
            && ids[..event.request.token_ids.len()]
                .iter()
                .all(|id| *id >= 0 && (*id as usize) < max_seq))
    }
}

pub(crate) fn run(runtime: PlanRuntime) {
    let mut machine = BatchPlannerModesEqualStateMachine::new(BatchPlannerModesEqualContext);
    let _ = machine.process_event(BatchPlannerModesEqualEvents::Plan(runtime));
}
