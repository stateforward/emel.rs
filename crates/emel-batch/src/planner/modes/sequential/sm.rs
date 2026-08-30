//! Sequential mask-subset batch planning algorithm.

#![allow(
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::empty_structs_with_brackets,
    clippy::missing_const_for_fn,
    dead_code,
    missing_docs,
    private_interfaces,
)]

use crate::planner::sm::{
    mask_subset, normalized_seq_mask, PlanRuntime, PlannerError, MAX_PLAN_STEPS,
};
use sml::sml;

#[derive(Clone, Debug)]
struct Plan;

// --- machine BatchPlannerModesSequential from emel.cpp/src/emel/batch/planner/modes/sequential/sm.hpp ---
sml! {
    BatchPlannerModesSequential {
        "state_planning"_s <= *"state_preparing"_s + Plan(PlanRuntime) / effect_begin_planning,
        "state_planning_input_decision"_s <= "state_planning"_s + completion<Plan>(PlanRuntime),
        "state_planning_failed"_s <= "state_planning_input_decision"_s + completion<Plan>(PlanRuntime) [guard_has_invalid_step_size] / effect_reject_invalid_step_size,
        "state_planning_capacity_decision"_s <= "state_planning_input_decision"_s + completion<Plan>(PlanRuntime) [guard_has_valid_step_size],
        "state_planning_failed"_s <= "state_planning_capacity_decision"_s + completion<Plan>(PlanRuntime) [guard_exceeds_step_capacity] / effect_reject_output_steps_full,
        "state_planning_failed"_s <= "state_planning_capacity_decision"_s + completion<Plan>(PlanRuntime) [guard_exceeds_index_capacity] / effect_reject_output_indices_full,
        "state_planning_execute"_s <= "state_planning_capacity_decision"_s + completion<Plan>(PlanRuntime) [guard_sequential_plan_capacity_ok],
        "state_planning_result_decision"_s <= "state_planning_execute"_s + completion<Plan>(PlanRuntime) / effect_plan_sequential_batches,
        "state_planning_done"_s <= "state_planning_result_decision"_s + completion<Plan>(PlanRuntime) [guard_planning_succeeded] / effect_emit_plan_done,
        "state_planning_failed"_s <= "state_planning_result_decision"_s + completion<Plan>(PlanRuntime) [guard_planning_failed] / effect_reject_planning_progress_stalled,
        "state_planning_failed"_s <= "state_planning_done"_s + unexpected_event<_> / effect_emit_internal_plan_error_from_state_planning_done,
        "state_planning_failed"_s <= "state_planning_failed"_s + unexpected_event<_> / effect_emit_internal_plan_error_from_state_planning_failed,
        "state_planning_failed"_s <= "state_preparing"_s + unexpected_event<_> / effect_emit_internal_plan_error_from_state_preparing,
        "state_planning_failed"_s <= "state_planning"_s + unexpected_event<_> / effect_emit_internal_plan_error_from_state_planning,
        "state_planning_failed"_s <= "state_planning_input_decision"_s + unexpected_event<_> / effect_emit_internal_plan_error_from_state_planning_input_decision,
        "state_planning_failed"_s <= "state_planning_capacity_decision"_s + unexpected_event<_> / effect_emit_internal_plan_error_from_state_planning_capacity_decision,
        "state_planning_failed"_s <= "state_planning_execute"_s + unexpected_event<_> / effect_emit_internal_plan_error_from_state_planning_execute,
        "state_planning_failed"_s <= "state_planning_result_decision"_s + unexpected_event<_> / effect_emit_internal_plan_error_from_state_planning_result_decision,
    }
}

#[derive(Debug, Default)]
struct Context {
    scratch: Option<crate::planner::sm::SharedScratch>,
}

impl Context {
    fn fail(&self, error: PlannerError) {
        if let Some(scratch) = &self.scratch {
            scratch.borrow_mut().fail(error);
        }
    }
}

impl BatchPlannerModesSequentialStateMachineContext for Context {
    fn effect_begin_planning(&mut self, event: PlanRuntime) -> Result<(), ()> {
        self.scratch = Some(event.scratch.clone());
        let mut scratch = event.scratch.borrow_mut();
        scratch.error = None;
        scratch.step_sizes.clear();
        scratch.step_token_indices.clear();
        scratch.step_token_offsets.clear();
        Ok(())
    }

    fn effect_emit_internal_plan_error_from_state_planning(&mut self) -> Result<(), ()> {
        self.fail(PlannerError::Internal);
        Ok(())
    }

    fn effect_emit_internal_plan_error_from_state_planning_capacity_decision(
        &mut self,
    ) -> Result<(), ()> {
        self.fail(PlannerError::Internal);
        Ok(())
    }

    fn effect_emit_internal_plan_error_from_state_planning_done(&mut self) -> Result<(), ()> {
        self.fail(PlannerError::Internal);
        Ok(())
    }

    fn effect_emit_internal_plan_error_from_state_planning_execute(&mut self) -> Result<(), ()> {
        self.fail(PlannerError::Internal);
        Ok(())
    }

    fn effect_emit_internal_plan_error_from_state_planning_failed(&mut self) -> Result<(), ()> {
        self.fail(PlannerError::Internal);
        Ok(())
    }

    fn effect_emit_internal_plan_error_from_state_planning_input_decision(
        &mut self,
    ) -> Result<(), ()> {
        self.fail(PlannerError::Internal);
        Ok(())
    }

    fn effect_emit_internal_plan_error_from_state_planning_result_decision(
        &mut self,
    ) -> Result<(), ()> {
        self.fail(PlannerError::Internal);
        Ok(())
    }

    fn effect_emit_internal_plan_error_from_state_preparing(&mut self) -> Result<(), ()> {
        self.fail(PlannerError::Internal);
        Ok(())
    }

    fn effect_emit_plan_done(&mut self, _event: PlanRuntime) -> Result<(), ()> {
        Ok(())
    }

    fn effect_plan_sequential_batches(&mut self, event: PlanRuntime) -> Result<(), ()> {
        let size = event.scratch.borrow().effective_step_size;
        if size == 0 {
            event.scratch.borrow_mut().fail(PlannerError::InvalidStepSize);
            return Ok(());
        }

        let n_tokens = event.request.token_ids.len();
        let mut used = [false; MAX_PLAN_STEPS];
        let mut used_count = 0usize;
        let mut scratch = event.scratch.borrow_mut();

        while used_count < n_tokens {
            let Some(mut current) = (0..n_tokens).find(|&index| !used[index]) else {
                break;
            };
            let mut current_mask = normalized_seq_mask(&event.request, current);
            let mut chunk = 0usize;
            let offset = scratch.step_token_indices.len();
            if !scratch.step_token_offsets.push(offset) {
                scratch.fail(PlannerError::OutputStepsFull);
                return Ok(());
            }

            loop {
                used[current] = true;
                used_count += 1;
                chunk += 1;
                if !scratch.step_token_indices.push(current) {
                    scratch.fail(PlannerError::OutputIndicesFull);
                    return Ok(());
                }
                if chunk >= size {
                    break;
                }

                let next = ((current + 1)..n_tokens).find(|&index| {
                    !used[index]
                        && mask_subset(current_mask, normalized_seq_mask(&event.request, index))
                });
                let Some(next) = next else { break };
                current = next;
                current_mask = normalized_seq_mask(&event.request, current);
            }

            if !scratch.step_sizes.push(chunk) {
                scratch.fail(PlannerError::OutputStepsFull);
                return Ok(());
            }
        }

        let offset = scratch.step_token_indices.len();
        if !scratch.step_token_offsets.push(offset) {
            scratch.fail(PlannerError::OutputStepsFull);
        }
        Ok(())
    }

    fn effect_reject_invalid_step_size(&mut self, event: PlanRuntime) -> Result<(), ()> {
        event.scratch.borrow_mut().fail(PlannerError::InvalidStepSize);
        Ok(())
    }

    fn effect_reject_output_indices_full(&mut self, event: PlanRuntime) -> Result<(), ()> {
        event.scratch.borrow_mut().fail(PlannerError::OutputIndicesFull);
        Ok(())
    }

    fn effect_reject_output_steps_full(&mut self, event: PlanRuntime) -> Result<(), ()> {
        event.scratch.borrow_mut().fail(PlannerError::OutputStepsFull);
        Ok(())
    }

    fn effect_reject_planning_progress_stalled(
        &mut self,
        event: PlanRuntime,
    ) -> Result<(), ()> {
        event
            .scratch
            .borrow_mut()
            .fail(PlannerError::PlanningProgressStalled);
        Ok(())
    }

    fn guard_exceeds_index_capacity(&self, event: &PlanRuntime) -> Result<bool, ()> {
        Ok(event.request.token_ids.len() > MAX_PLAN_STEPS)
    }

    fn guard_exceeds_step_capacity(&self, event: &PlanRuntime) -> Result<bool, ()> {
        let scratch = event.scratch.borrow();
        let size = scratch.effective_step_size;
        Ok(size > 0 && event.request.token_ids.len().div_ceil(size) > MAX_PLAN_STEPS)
    }

    fn guard_has_invalid_step_size(&self, event: &PlanRuntime) -> Result<bool, ()> {
        Ok(event.scratch.borrow().effective_step_size == 0)
    }

    fn guard_has_valid_step_size(&self, event: &PlanRuntime) -> Result<bool, ()> {
        Ok(event.scratch.borrow().effective_step_size > 0)
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

    fn guard_sequential_plan_capacity_ok(&self, event: &PlanRuntime) -> Result<bool, ()> {
        Ok(self.guard_has_valid_step_size(event)?
            && !self.guard_exceeds_step_capacity(event)?
            && !self.guard_exceeds_index_capacity(event)?)
    }
}

pub(crate) fn run(runtime: PlanRuntime) {
    let mut machine = BatchPlannerModesSequentialStateMachine::new(Context::default());
    let _ = machine.process_event(BatchPlannerModesSequentialEvents::Plan(runtime));
}
