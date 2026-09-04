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
    EqualStrategy, MAX_PLAN_STEPS, MAX_SEQ, ModeResult, ModeScratch, PlanRuntime, PlanScratch,
    PlannerError, SeqMask, mask_equal, mask_overlaps, normalized_seq_mask,
};
use sml::sml;
#[derive(Clone, Debug)]
struct Plan;
sml! { BatchPlannerModesEqual {
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
} }
#[derive(Debug, Default)]
pub struct BatchPlannerModesEqualContext {
    scratch: ModeScratch,
}
fn fail(s: &mut ModeScratch, e: PlannerError) {
    s.fail(e);
}
fn offsets(scratch: &mut ModeScratch) {
    let output_offset = scratch.step_token_indices.len();
    if !scratch.step_token_offsets.push(output_offset) {
        scratch.fail(PlannerError::OutputStepsFull);
    }
}
fn append_equal_group(
    runtime: &PlanRuntime,
    scratch: &mut ModeScratch,
    used: &mut [bool; MAX_PLAN_STEPS],
    token_count: usize,
    group_mask: SeqMask,
    rows_per_group: usize,
    processed_count: &mut usize,
) -> Result<(), PlannerError> {
    let mut remaining_rows = rows_per_group;
    for (token_index, token_used) in used.iter_mut().enumerate().take(token_count) {
        if remaining_rows == 0 {
            break;
        }
        if *token_used
            || !mask_equal(
                normalized_seq_mask(&runtime.request, token_index),
                group_mask,
            )
        {
            continue;
        }
        *token_used = true;
        *processed_count += 1;
        if !scratch.step_token_indices.push(token_index) {
            return Err(PlannerError::OutputIndicesFull);
        }
        remaining_rows -= 1;
    }
    if remaining_rows != 0 {
        return Err(PlannerError::AlgorithmFailed);
    }
    Ok(())
}

fn emit_equal_batches(
    runtime: &PlanRuntime,
    scratch: &mut ModeScratch,
    step_size: usize,
    token_count: usize,
) {
    let mut used = [false; MAX_PLAN_STEPS];
    let mut processed_count = 0;
    while processed_count < token_count {
        #[allow(clippy::large_stack_arrays)]
        let mut groups = [SeqMask::default(); MAX_PLAN_STEPS];
        let mut group_count = 0;
        let mut previous_primary_id = -1;
        for (token_index, token_used) in used.iter().enumerate().take(token_count) {
            if *token_used {
                continue;
            }
            let sequence_mask = normalized_seq_mask(&runtime.request, token_index);
            if (0..group_count).any(|group_index| mask_overlaps(groups[group_index], sequence_mask))
            {
                continue;
            }
            let sequence = runtime.request.config().sequence();
            if matches!(
                runtime.request.config().equal_strategy(),
                EqualStrategy::Sequential
            ) && let Some(primary_ids) = sequence.primary_ids()
            {
                let primary_id = primary_ids.get(token_index).copied().unwrap_or(-1);
                if group_count > 0 && primary_id != previous_primary_id + 1 {
                    continue;
                }
                previous_primary_id = primary_id;
            }
            groups[group_count] = sequence_mask;
            group_count += 1;
            if group_count > step_size {
                break;
            }
        }
        if group_count == 0 {
            fail(scratch, PlannerError::PlanningProgressStalled);
            return;
        }
        let mut minimum_rows = token_count + 1;
        for group_mask in groups.iter().take(group_count) {
            minimum_rows = minimum_rows.min(
                (0..token_count)
                    .filter(|token_index| {
                        !used[*token_index]
                            && mask_equal(
                                normalized_seq_mask(&runtime.request, *token_index),
                                *group_mask,
                            )
                    })
                    .count(),
            );
        }
        let rows_per_group = (step_size / group_count).min(minimum_rows);
        if rows_per_group == 0 {
            fail(scratch, PlannerError::PlanningProgressStalled);
            return;
        }
        let output_offset = scratch.step_token_indices.len();
        if !scratch.step_token_offsets.push(output_offset) {
            fail(scratch, PlannerError::OutputStepsFull);
            return;
        }
        for &group_mask in groups.iter().take(group_count) {
            if let Err(error) = append_equal_group(
                runtime,
                scratch,
                &mut used,
                token_count,
                group_mask,
                rows_per_group,
                &mut processed_count,
            ) {
                fail(scratch, error);
                return;
            }
        }
        if !scratch.step_sizes.push(rows_per_group * group_count) {
            fail(scratch, PlannerError::OutputStepsFull);
            return;
        }
    }
}

fn plan_equal_batches(runtime: &PlanRuntime, scratch: &mut ModeScratch) {
    let step_size = scratch.effective_step_size;
    let token_count = runtime.request.token_ids().len();
    if step_size == 0 {
        fail(scratch, PlannerError::InvalidStepSize);
        return;
    }
    emit_equal_batches(runtime, scratch, step_size, token_count);
    if scratch.error.is_none() {
        offsets(scratch);
    }
}
fn prepare_primary_batches(
    primary_ids: &[i32],
    token_count: usize,
    max_sequence_id: usize,
    occurrence_counts: &mut [usize; MAX_SEQ],
    bucket_offsets: &mut [usize; MAX_SEQ + 1],
    next_bucket_offsets: &mut [usize; MAX_SEQ],
    sorted_indices: &mut [usize; MAX_PLAN_STEPS],
) -> Result<(), PlannerError> {
    for &primary_id in &primary_ids[..token_count] {
        let Ok(sequence_id) = usize::try_from(primary_id) else {
            return Err(PlannerError::InvalidSequenceId);
        };
        if sequence_id >= max_sequence_id {
            return Err(PlannerError::InvalidSequenceId);
        }
        occurrence_counts[sequence_id] += 1;
    }
    for sequence_id in 0..max_sequence_id {
        bucket_offsets[sequence_id + 1] =
            bucket_offsets[sequence_id] + occurrence_counts[sequence_id];
        next_bucket_offsets[sequence_id] = bucket_offsets[sequence_id];
    }
    for (token_index, &primary_id) in primary_ids[..token_count].iter().enumerate() {
        let Ok(sequence_id) = usize::try_from(primary_id) else {
            return Err(PlannerError::InvalidSequenceId);
        };
        sorted_indices[next_bucket_offsets[sequence_id]] = token_index;
        next_bucket_offsets[sequence_id] += 1;
    }
    Ok(())
}

fn append_primary_group(
    scratch: &mut ModeScratch,
    sorted_indices: &[usize; MAX_PLAN_STEPS],
    bucket_offset: usize,
    rows: usize,
) -> Result<(), PlannerError> {
    for row_offset in 0..rows {
        let Some(&token_index) = sorted_indices.get(bucket_offset + row_offset) else {
            return Err(PlannerError::AlgorithmFailed);
        };
        if !scratch.step_token_indices.push(token_index) {
            return Err(PlannerError::OutputIndicesFull);
        }
    }
    Ok(())
}

struct PrimaryBatchInput<'a> {
    runtime: &'a PlanRuntime,
    primary_ids: &'a [i32],
    token_count: usize,
    step_size: usize,
    occurrence_counts: &'a [usize; MAX_SEQ],
    bucket_offsets: &'a [usize; MAX_SEQ + 1],
    sorted_indices: &'a [usize; MAX_PLAN_STEPS],
}

fn emit_primary_batches(input: &PrimaryBatchInput<'_>, scratch: &mut ModeScratch) {
    let &PrimaryBatchInput {
        runtime,
        primary_ids,
        token_count,
        step_size,
        occurrence_counts,
        bucket_offsets,
        sorted_indices,
    } = input;
    let mut consumed_counts = [0; MAX_SEQ];
    let mut remaining_tokens = token_count;
    while remaining_tokens > 0 {
        let mut group_used = [false; MAX_SEQ];
        let mut group_sequence_ids = [0; MAX_SEQ];
        let mut group_count = 0;
        let mut previous_sequence_id = None;
        for &primary_id in &primary_ids[..token_count] {
            let Ok(sequence_id) = usize::try_from(primary_id) else {
                fail(scratch, PlannerError::InvalidSequenceId);
                return;
            };
            if consumed_counts[sequence_id] >= occurrence_counts[sequence_id]
                || group_used[sequence_id]
            {
                continue;
            }
            if matches!(
                runtime.request.config().equal_strategy(),
                EqualStrategy::Sequential
            ) && group_count > 0
                && previous_sequence_id != Some(sequence_id.wrapping_sub(1))
            {
                continue;
            }
            group_used[sequence_id] = true;
            group_sequence_ids[group_count] = sequence_id;
            group_count += 1;
            previous_sequence_id = Some(sequence_id);
            if group_count > step_size {
                break;
            }
        }
        if group_count == 0 {
            fail(scratch, PlannerError::PlanningProgressStalled);
            return;
        }
        let rows_per_group = (step_size / group_count).min(
            (0..group_count)
                .map(|group_index| {
                    let sequence_id = group_sequence_ids[group_index];
                    occurrence_counts[sequence_id] - consumed_counts[sequence_id]
                })
                .min()
                .unwrap_or(0),
        );
        if rows_per_group == 0 {
            fail(scratch, PlannerError::PlanningProgressStalled);
            return;
        }
        let output_offset = scratch.step_token_indices.len();
        if !scratch.step_token_offsets.push(output_offset) {
            fail(scratch, PlannerError::OutputStepsFull);
            return;
        }
        for &sequence_id in group_sequence_ids.iter().take(group_count) {
            let bucket_offset = bucket_offsets[sequence_id] + consumed_counts[sequence_id];
            if let Err(error) =
                append_primary_group(scratch, sorted_indices, bucket_offset, rows_per_group)
            {
                fail(scratch, error);
                return;
            }
            consumed_counts[sequence_id] += rows_per_group;
            remaining_tokens -= rows_per_group;
        }
        if !scratch.step_sizes.push(rows_per_group * group_count) {
            fail(scratch, PlannerError::OutputStepsFull);
            return;
        }
    }
}

fn plan_equal_primary_batches(runtime: &PlanRuntime, scratch: &mut ModeScratch) {
    let step_size = scratch.effective_step_size;
    let token_count = runtime.request.token_ids().len();
    let Some(primary_ids) = runtime.request.config().sequence().primary_ids() else {
        fail(scratch, PlannerError::InvalidSequenceId);
        return;
    };
    let max_sequence_id = runtime
        .request
        .config()
        .sequence()
        .mask_words()
        .saturating_mul(64);
    if step_size == 0
        || max_sequence_id == 0
        || max_sequence_id > MAX_SEQ
        || primary_ids.len() < token_count
    {
        fail(
            scratch,
            if step_size == 0 {
                PlannerError::InvalidStepSize
            } else {
                PlannerError::InvalidSequenceId
            },
        );
        return;
    }
    let mut occurrence_counts = [0; MAX_SEQ];
    let mut bucket_offsets = [0; MAX_SEQ + 1];
    let mut next_bucket_offsets = [0; MAX_SEQ];
    #[allow(clippy::large_stack_arrays)]
    let mut sorted_indices = [0; MAX_PLAN_STEPS];
    if let Err(error) = prepare_primary_batches(
        primary_ids,
        token_count,
        max_sequence_id,
        &mut occurrence_counts,
        &mut bucket_offsets,
        &mut next_bucket_offsets,
        &mut sorted_indices,
    ) {
        fail(scratch, error);
        return;
    }
    emit_primary_batches(
        &PrimaryBatchInput {
            runtime,
            primary_ids,
            token_count,
            step_size,
            occurrence_counts: &occurrence_counts,
            bucket_offsets: &bucket_offsets,
            sorted_indices: &sorted_indices,
        },
        scratch,
    );
    if scratch.error.is_none() {
        offsets(scratch);
    }
}
impl BatchPlannerModesEqualStateMachineContext for BatchPlannerModesEqualContext {
    fn effect_begin_planning(&mut self, e: PlanRuntime) -> Result<(), ()> {
        self.scratch.reset();
        self.scratch.effective_step_size = e.effective_step_size;
        self.scratch.total_outputs = e.total_outputs;
        Ok(())
    }
    fn effect_plan_equal_batches(&mut self, e: PlanRuntime) -> Result<(), ()> {
        plan_equal_batches(&e, &mut self.scratch);
        Ok(())
    }
    fn effect_plan_equal_primary_batches(&mut self, e: PlanRuntime) -> Result<(), ()> {
        plan_equal_primary_batches(&e, &mut self.scratch);
        Ok(())
    }
    fn effect_emit_plan_done_from_state_planning_fast_result_decision(
        &mut self,
        _: PlanRuntime,
    ) -> Result<(), ()> {
        Ok(())
    }
    fn effect_emit_plan_done_from_state_planning_general_result_decision(
        &mut self,
        _: PlanRuntime,
    ) -> Result<(), ()> {
        Ok(())
    }
    fn effect_emit_internal_plan_error_from_state_preparing(&mut self) -> Result<(), ()> {
        self.scratch.fail(PlannerError::Internal);
        Ok(())
    }
    fn effect_emit_internal_plan_error_from_state_planning(&mut self) -> Result<(), ()> {
        self.scratch.fail(PlannerError::Internal);
        Ok(())
    }
    fn effect_emit_internal_plan_error_from_state_planning_mode_decision(
        &mut self,
    ) -> Result<(), ()> {
        self.scratch.fail(PlannerError::Internal);
        Ok(())
    }
    fn effect_emit_internal_plan_error_from_state_planning_fast_input_decision(
        &mut self,
    ) -> Result<(), ()> {
        self.scratch.fail(PlannerError::Internal);
        Ok(())
    }
    fn effect_emit_internal_plan_error_from_state_planning_fast_capacity_decision(
        &mut self,
    ) -> Result<(), ()> {
        self.scratch.fail(PlannerError::Internal);
        Ok(())
    }
    fn effect_emit_internal_plan_error_from_state_planning_fast_execute(
        &mut self,
    ) -> Result<(), ()> {
        self.scratch.fail(PlannerError::Internal);
        Ok(())
    }
    fn effect_emit_internal_plan_error_from_state_planning_fast_result_decision(
        &mut self,
    ) -> Result<(), ()> {
        self.scratch.fail(PlannerError::Internal);
        Ok(())
    }
    fn effect_emit_internal_plan_error_from_state_planning_general_input_decision(
        &mut self,
    ) -> Result<(), ()> {
        self.scratch.fail(PlannerError::Internal);
        Ok(())
    }
    fn effect_emit_internal_plan_error_from_state_planning_general_capacity_decision(
        &mut self,
    ) -> Result<(), ()> {
        self.scratch.fail(PlannerError::Internal);
        Ok(())
    }
    fn effect_emit_internal_plan_error_from_state_planning_general_execute(
        &mut self,
    ) -> Result<(), ()> {
        self.scratch.fail(PlannerError::Internal);
        Ok(())
    }
    fn effect_emit_internal_plan_error_from_state_planning_general_result_decision(
        &mut self,
    ) -> Result<(), ()> {
        self.scratch.fail(PlannerError::Internal);
        Ok(())
    }
    fn effect_emit_internal_plan_error_from_state_planning_done(&mut self) -> Result<(), ()> {
        self.scratch.fail(PlannerError::Internal);
        Ok(())
    }
    fn effect_emit_internal_plan_error_from_state_planning_failed(&mut self) -> Result<(), ()> {
        self.scratch.fail(PlannerError::Internal);
        Ok(())
    }
    fn effect_reject_invalid_sequence_id_from_state_planning_fast_input_decision(
        &mut self,
        _: PlanRuntime,
    ) -> Result<(), ()> {
        self.scratch.fail(PlannerError::InvalidSequenceId);
        Ok(())
    }
    fn effect_reject_invalid_step_size_from_state_planning_fast_input_decision(
        &mut self,
        _: PlanRuntime,
    ) -> Result<(), ()> {
        self.scratch.fail(PlannerError::InvalidStepSize);
        Ok(())
    }
    fn effect_reject_invalid_step_size_from_state_planning_general_input_decision(
        &mut self,
        _: PlanRuntime,
    ) -> Result<(), ()> {
        self.scratch.fail(PlannerError::InvalidStepSize);
        Ok(())
    }
    fn effect_reject_output_indices_full_from_state_planning_fast_capacity_decision(
        &mut self,
        _: PlanRuntime,
    ) -> Result<(), ()> {
        self.scratch.fail(PlannerError::OutputIndicesFull);
        Ok(())
    }
    fn effect_reject_output_indices_full_from_state_planning_general_capacity_decision(
        &mut self,
        _: PlanRuntime,
    ) -> Result<(), ()> {
        self.scratch.fail(PlannerError::OutputIndicesFull);
        Ok(())
    }
    fn effect_reject_output_steps_full_from_state_planning_fast_capacity_decision(
        &mut self,
        _: PlanRuntime,
    ) -> Result<(), ()> {
        self.scratch.fail(PlannerError::OutputStepsFull);
        Ok(())
    }
    fn effect_reject_output_steps_full_from_state_planning_general_capacity_decision(
        &mut self,
        _: PlanRuntime,
    ) -> Result<(), ()> {
        self.scratch.fail(PlannerError::OutputStepsFull);
        Ok(())
    }
    fn effect_reject_planning_progress_stalled_from_state_planning_fast_result_decision(
        &mut self,
        _: PlanRuntime,
    ) -> Result<(), ()> {
        self.scratch.fail(PlannerError::PlanningProgressStalled);
        Ok(())
    }
    fn effect_reject_planning_progress_stalled_from_state_planning_general_result_decision(
        &mut self,
        _: PlanRuntime,
    ) -> Result<(), ()> {
        self.scratch.fail(PlannerError::PlanningProgressStalled);
        Ok(())
    }
    fn guard_fast_path_input_valid(&self, e: &PlanRuntime) -> Result<bool, ()> {
        Ok(self.scratch.effective_step_size > 0
            && e.request.config().sequence().primary_ids().is_some()
            && fast_path_primary_ids_valid(e))
    }
    fn guard_fast_path_missing_primary_ids(&self, e: &PlanRuntime) -> Result<bool, ()> {
        Ok(e.request.config().sequence().primary_ids().is_none())
    }
    fn guard_fast_path_primary_ids_invalid(&self, e: &PlanRuntime) -> Result<bool, ()> {
        Ok(
            e.request.config().sequence().primary_ids().is_some()
                && !fast_path_primary_ids_valid(e),
        )
    }
    fn guard_general_input_valid(&self, _: &PlanRuntime) -> Result<bool, ()> {
        Ok(self.scratch.effective_step_size > 0)
    }
    fn guard_has_invalid_step_size(&self, _: &PlanRuntime) -> Result<bool, ()> {
        Ok(self.scratch.effective_step_size == 0)
    }
    fn guard_lacks_index_capacity(&self, e: &PlanRuntime) -> Result<bool, ()> {
        Ok(e.request.token_ids().len()
            > MAX_PLAN_STEPS.saturating_sub(self.scratch.step_token_indices.len()))
    }
    fn guard_lacks_step_capacity(&self, _: &PlanRuntime) -> Result<bool, ()> {
        Ok(self.scratch.step_sizes.len() >= MAX_PLAN_STEPS)
    }
    fn guard_mode_is_general_path(&self, e: &PlanRuntime) -> Result<bool, ()> {
        Ok(!(e.request.config().sequence().masks().is_none()
            && e.request.config().sequence().primary_ids().is_some()))
    }
    fn guard_mode_is_primary_fast_path(&self, e: &PlanRuntime) -> Result<bool, ()> {
        Ok(e.request.config().sequence().masks().is_none()
            && e.request.config().sequence().primary_ids().is_some())
    }
    fn guard_planning_failed(&self, e: &PlanRuntime) -> Result<bool, ()> {
        Ok(!self.guard_planning_succeeded(e)?)
    }
    fn guard_planning_succeeded(&self, e: &PlanRuntime) -> Result<bool, ()> {
        Ok(self.scratch.error.is_none()
            && !self.scratch.step_sizes.is_empty()
            && self.scratch.step_token_indices.len() == e.request.token_ids().len()
            && self.scratch.step_token_offsets.len() == self.scratch.step_sizes.len() + 1)
    }
    fn guard_storage_capacity_valid(&self, e: &PlanRuntime) -> Result<bool, ()> {
        Ok(!self.guard_lacks_step_capacity(e)? && !self.guard_lacks_index_capacity(e)?)
    }
}
fn fast_path_primary_ids_valid(e: &PlanRuntime) -> bool {
    let Some(ids) = e.request.config().sequence().primary_ids() else {
        return false;
    };
    let max = e
        .request
        .config()
        .sequence()
        .mask_words()
        .saturating_mul(64);
    max > 0
        && max <= MAX_SEQ
        && ids.len() >= e.request.token_ids().len()
        && ids[..e.request.token_ids().len()]
            .iter()
            .all(|id| usize::try_from(*id).is_ok_and(|id| id < max))
}
/// Synchronous actor around the generated equal-planning machine.
#[allow(missing_debug_implementations)]
pub(crate) struct Actor {
    machine: BatchPlannerModesEqualStateMachine<BatchPlannerModesEqualContext>,
}

impl Default for Actor {
    fn default() -> Self {
        Self::new()
    }
}

impl Actor {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            machine: BatchPlannerModesEqualStateMachine::new(
                BatchPlannerModesEqualContext::default(),
            ),
        }
    }

    /// Resets the child machine and its reusable scratch without allocating.
    pub(crate) fn reset(&mut self) {
        self.machine.context_mut().scratch.reset();
        self.machine
            .set_state(BatchPlannerModesEqualStates::StatePreparing);
    }

    /// Dispatches one planner request to the child and copies its bounded output.
    pub(crate) fn process_event(
        &mut self,
        runtime: PlanRuntime,
        target: &mut PlanScratch,
    ) -> ModeResult {
        if self
            .machine
            .process_event(BatchPlannerModesEqualEvents::Plan(runtime))
            .is_err()
        {
            target.fail(PlannerError::Internal);
            return ModeResult::Error(PlannerError::Internal);
        }
        let result = self.machine.context_mut().scratch.result();
        self.machine.context_mut().scratch.copy_into(target);
        result
    }
}
impl core::fmt::Debug for Actor {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("EqualPlannerActor")
            .finish_non_exhaustive()
    }
}
