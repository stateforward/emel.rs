//! Sequential mask-subset batch planning algorithm.
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
    missing_docs,
    private_interfaces
)]
use crate::planner::sm::{
    MAX_PLAN_STEPS, ModeResult, ModeScratch, PlanRuntime, PlanScratch, PlannerError, mask_subset,
    normalized_seq_mask,
};
use sml::sml;
#[derive(Clone, Debug)]
struct Plan;
sml! { BatchPlannerModesSequential {
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
} }
#[derive(Debug, Default)]
struct Context {
    scratch: ModeScratch,
}
impl BatchPlannerModesSequentialStateMachineContext for Context {
    fn effect_begin_planning(&mut self, e: PlanRuntime) -> Result<(), ()> {
        self.scratch.reset();
        self.scratch.effective_step_size = e.effective_step_size;
        self.scratch.total_outputs = e.total_outputs;
        Ok(())
    }
    fn effect_emit_plan_done(&mut self, _: PlanRuntime) -> Result<(), ()> {
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
    fn effect_emit_internal_plan_error_from_state_planning_capacity_decision(
        &mut self,
    ) -> Result<(), ()> {
        self.scratch.fail(PlannerError::Internal);
        Ok(())
    }
    fn effect_emit_internal_plan_error_from_state_planning_done(&mut self) -> Result<(), ()> {
        self.scratch.fail(PlannerError::Internal);
        Ok(())
    }
    fn effect_emit_internal_plan_error_from_state_planning_execute(&mut self) -> Result<(), ()> {
        self.scratch.fail(PlannerError::Internal);
        Ok(())
    }
    fn effect_emit_internal_plan_error_from_state_planning_failed(&mut self) -> Result<(), ()> {
        self.scratch.fail(PlannerError::Internal);
        Ok(())
    }
    fn effect_emit_internal_plan_error_from_state_planning_input_decision(
        &mut self,
    ) -> Result<(), ()> {
        self.scratch.fail(PlannerError::Internal);
        Ok(())
    }
    fn effect_emit_internal_plan_error_from_state_planning_result_decision(
        &mut self,
    ) -> Result<(), ()> {
        self.scratch.fail(PlannerError::Internal);
        Ok(())
    }
    fn effect_plan_sequential_batches(&mut self, e: PlanRuntime) -> Result<(), ()> {
        let z = self.scratch.effective_step_size;
        if z == 0 {
            self.scratch.fail(PlannerError::InvalidStepSize);
            return Ok(());
        }
        let n = e.request.token_ids().len();
        let mut used = [false; MAX_PLAN_STEPS];
        let mut count = 0;
        while count < n {
            let Some(mut current) = (0..n).find(|&i| !used[i]) else {
                self.scratch.fail(PlannerError::PlanningProgressStalled);
                return Ok(());
            };
            let mut mask = normalized_seq_mask(&e.request, current);
            let mut chunk = 0;
            let o = self.scratch.step_token_indices.len();
            if !self.scratch.step_token_offsets.push(o) {
                self.scratch.fail(PlannerError::OutputStepsFull);
                return Ok(());
            }
            loop {
                used[current] = true;
                count += 1;
                chunk += 1;
                if !self.scratch.step_token_indices.push(current) {
                    self.scratch.fail(PlannerError::OutputIndicesFull);
                    return Ok(());
                }
                if chunk >= z {
                    break;
                }
                let Some(next) = ((current + 1)..n)
                    .find(|&i| !used[i] && mask_subset(mask, normalized_seq_mask(&e.request, i)))
                else {
                    break;
                };
                current = next;
                mask = normalized_seq_mask(&e.request, current);
            }
            if !self.scratch.step_sizes.push(chunk) {
                self.scratch.fail(PlannerError::OutputStepsFull);
                return Ok(());
            }
        }
        let o = self.scratch.step_token_indices.len();
        if !self.scratch.step_token_offsets.push(o) {
            self.scratch.fail(PlannerError::OutputStepsFull);
        }
        Ok(())
    }
    fn effect_reject_invalid_step_size(&mut self, _: PlanRuntime) -> Result<(), ()> {
        self.scratch.fail(PlannerError::InvalidStepSize);
        Ok(())
    }
    fn effect_reject_output_indices_full(&mut self, _: PlanRuntime) -> Result<(), ()> {
        self.scratch.fail(PlannerError::OutputIndicesFull);
        Ok(())
    }
    fn effect_reject_output_steps_full(&mut self, _: PlanRuntime) -> Result<(), ()> {
        self.scratch.fail(PlannerError::OutputStepsFull);
        Ok(())
    }
    fn effect_reject_planning_progress_stalled(&mut self, _: PlanRuntime) -> Result<(), ()> {
        self.scratch.fail(PlannerError::PlanningProgressStalled);
        Ok(())
    }
    fn guard_exceeds_index_capacity(&self, e: &PlanRuntime) -> Result<bool, ()> {
        Ok(e.request.token_ids().len() > MAX_PLAN_STEPS)
    }
    fn guard_exceeds_step_capacity(&self, e: &PlanRuntime) -> Result<bool, ()> {
        Ok(self.scratch.effective_step_size > 0
            && e.request
                .token_ids()
                .len()
                .div_ceil(self.scratch.effective_step_size)
                > MAX_PLAN_STEPS)
    }
    fn guard_has_invalid_step_size(&self, _: &PlanRuntime) -> Result<bool, ()> {
        Ok(self.scratch.effective_step_size == 0)
    }
    fn guard_has_valid_step_size(&self, _: &PlanRuntime) -> Result<bool, ()> {
        Ok(self.scratch.effective_step_size > 0)
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
    fn guard_sequential_plan_capacity_ok(&self, e: &PlanRuntime) -> Result<bool, ()> {
        Ok(self.guard_has_valid_step_size(e)?
            && !self.guard_exceeds_step_capacity(e)?
            && !self.guard_exceeds_index_capacity(e)?)
    }
}
/// Synchronous actor around the generated sequential-planning machine.
#[allow(missing_debug_implementations)]
pub(crate) struct Actor {
    machine: BatchPlannerModesSequentialStateMachine<Context>,
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
            machine: BatchPlannerModesSequentialStateMachine::new(Context::default()),
        }
    }

    /// Resets the child machine and its reusable scratch without allocating.
    pub(crate) fn reset(&mut self) {
        self.machine.context_mut().scratch.reset();
        self.machine
            .set_state(BatchPlannerModesSequentialStates::StatePreparing);
    }

    /// Dispatches one planner request to the child and copies its bounded output.
    pub(crate) fn process_event(
        &mut self,
        runtime: PlanRuntime,
        target: &mut PlanScratch,
    ) -> ModeResult {
        if self
            .machine
            .process_event(BatchPlannerModesSequentialEvents::Plan(runtime))
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
            .debug_struct("SequentialPlannerActor")
            .finish_non_exhaustive()
    }
}
