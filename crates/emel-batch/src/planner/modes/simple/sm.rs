//! Simple fixed-size batch planning algorithm.

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
    MAX_PLAN_STEPS, ModeResult, ModeScratch, PlanRuntime, PlanScratch, PlannerError,
};
use sml::sml;

#[derive(Clone, Debug)]
struct Plan;

// --- machine BatchPlannerModesSimple from emel.cpp/src/emel/batch/planner/modes/simple/sm.hpp ---
sml! {
    BatchPlannerModesSimple {
        "state_planning"_s <= *"state_preparing"_s + Plan(PlanRuntime) / effect_begin_planning,
        "state_planning_input_decision"_s <= "state_planning"_s + completion<Plan>(PlanRuntime),
        "state_planning_failed"_s <= "state_planning_input_decision"_s + completion<Plan>(PlanRuntime) [guard_has_invalid_step_size] / effect_reject_invalid_step_size,
        "state_planning_capacity_decision"_s <= "state_planning_input_decision"_s + completion<Plan>(PlanRuntime) [guard_has_valid_step_size],
        "state_planning_failed"_s <= "state_planning_capacity_decision"_s + completion<Plan>(PlanRuntime) [guard_exceeds_step_capacity] / effect_reject_output_steps_full,
        "state_planning_failed"_s <= "state_planning_capacity_decision"_s + completion<Plan>(PlanRuntime) [guard_exceeds_index_capacity] / effect_reject_output_indices_full,
        "state_planning_decision"_s <= "state_planning_capacity_decision"_s + completion<Plan>(PlanRuntime) [guard_simple_plan_capacity_ok] / effect_plan,
        "state_planning_done"_s <= "state_planning_decision"_s + completion<Plan>(PlanRuntime) [guard_planning_succeeded] / effect_emit_plan_done,
        "state_planning_failed"_s <= "state_planning_decision"_s + completion<Plan>(PlanRuntime) [guard_planning_failed] / effect_reject_planning_progress_stalled,
        "state_planning_failed"_s <= "state_planning_done"_s + unexpected_event<_> / effect_emit_internal_plan_error,
        "state_planning_failed"_s <= "state_planning_failed"_s + unexpected_event<_> / effect_emit_internal_plan_error,
        "state_planning_failed"_s <= "state_preparing"_s + unexpected_event<_> / effect_emit_internal_plan_error,
        "state_planning_failed"_s <= "state_planning"_s + unexpected_event<_> / effect_emit_internal_plan_error,
        "state_planning_failed"_s <= "state_planning_input_decision"_s + unexpected_event<_> / effect_emit_internal_plan_error,
        "state_planning_failed"_s <= "state_planning_capacity_decision"_s + unexpected_event<_> / effect_emit_internal_plan_error,
        "state_planning_failed"_s <= "state_planning_decision"_s + unexpected_event<_> / effect_emit_internal_plan_error,
    }
}

#[derive(Debug, Default)]
struct Context {
    scratch: ModeScratch,
}

impl Context {
    fn fail(&mut self, error: PlannerError) {
        self.scratch.fail(error);
    }
}

impl BatchPlannerModesSimpleStateMachineContext for Context {
    fn effect_begin_planning(&mut self, event: PlanRuntime) -> Result<(), ()> {
        self.scratch.reset();
        self.scratch.effective_step_size = event.effective_step_size;
        self.scratch.total_outputs = event.total_outputs;
        Ok(())
    }

    fn effect_emit_internal_plan_error(&mut self) -> Result<(), ()> {
        self.fail(PlannerError::Internal);
        Ok(())
    }

    fn effect_emit_plan_done(&mut self, _event: PlanRuntime) -> Result<(), ()> {
        Ok(())
    }

    fn effect_plan(&mut self, event: PlanRuntime) -> Result<(), ()> {
        let size = self.scratch.effective_step_size;
        if size == 0 {
            self.scratch.fail(PlannerError::InvalidStepSize);
            return Ok(());
        }
        let n = event.request.token_ids().len();
        let mut start = 0usize;
        while start < n {
            let offset = self.scratch.step_token_indices.len();
            if !self.scratch.step_token_offsets.push(offset) {
                self.scratch.fail(PlannerError::OutputStepsFull);
                return Ok(());
            }
            let end = (start + size).min(n);
            for index in start..end {
                if !self.scratch.step_token_indices.push(index) {
                    self.scratch.fail(PlannerError::OutputIndicesFull);
                    return Ok(());
                }
            }
            if !self.scratch.step_sizes.push(end - start) {
                self.scratch.fail(PlannerError::OutputStepsFull);
                return Ok(());
            }
            start = end;
        }
        let offset = self.scratch.step_token_indices.len();
        if !self.scratch.step_token_offsets.push(offset) {
            self.scratch.fail(PlannerError::OutputStepsFull);
        }
        Ok(())
    }

    fn effect_reject_invalid_step_size(&mut self, _event: PlanRuntime) -> Result<(), ()> {
        self.fail(PlannerError::InvalidStepSize);
        Ok(())
    }
    fn effect_reject_output_indices_full(&mut self, _event: PlanRuntime) -> Result<(), ()> {
        self.fail(PlannerError::OutputIndicesFull);
        Ok(())
    }
    fn effect_reject_output_steps_full(&mut self, _event: PlanRuntime) -> Result<(), ()> {
        self.fail(PlannerError::OutputStepsFull);
        Ok(())
    }
    fn effect_reject_planning_progress_stalled(&mut self, _event: PlanRuntime) -> Result<(), ()> {
        self.fail(PlannerError::PlanningProgressStalled);
        Ok(())
    }

    fn guard_exceeds_index_capacity(&self, event: &PlanRuntime) -> Result<bool, ()> {
        Ok(event.request.token_ids().len() > MAX_PLAN_STEPS)
    }

    fn guard_exceeds_step_capacity(&self, event: &PlanRuntime) -> Result<bool, ()> {
        let size = self.scratch.effective_step_size;
        Ok(size > 0 && event.request.token_ids().len().div_ceil(size) > MAX_PLAN_STEPS)
    }

    fn guard_has_invalid_step_size(&self, _event: &PlanRuntime) -> Result<bool, ()> {
        Ok(self.scratch.effective_step_size == 0)
    }

    fn guard_has_valid_step_size(&self, _event: &PlanRuntime) -> Result<bool, ()> {
        Ok(self.scratch.effective_step_size > 0)
    }

    fn guard_planning_failed(&self, event: &PlanRuntime) -> Result<bool, ()> {
        Ok(!self.guard_planning_succeeded(event)?)
    }

    fn guard_planning_succeeded(&self, event: &PlanRuntime) -> Result<bool, ()> {
        Ok(self.scratch.error.is_none()
            && !self.scratch.step_sizes.is_empty()
            && self.scratch.step_token_indices.len() == event.request.token_ids().len()
            && self.scratch.step_token_offsets.len() == self.scratch.step_sizes.len() + 1)
    }

    fn guard_simple_plan_capacity_ok(&self, event: &PlanRuntime) -> Result<bool, ()> {
        Ok(self.guard_has_valid_step_size(event)?
            && !self.guard_exceeds_step_capacity(event)?
            && !self.guard_exceeds_index_capacity(event)?)
    }
}

/// Synchronous actor around the generated simple-planning machine.
#[allow(missing_debug_implementations)]
pub(crate) struct Actor {
    machine: BatchPlannerModesSimpleStateMachine<Context>,
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
            machine: BatchPlannerModesSimpleStateMachine::new(Context::default()),
        }
    }

    /// Resets the child machine and its reusable scratch without allocating.
    pub(crate) fn reset(&mut self) {
        self.machine.context_mut().scratch.reset();
        self.machine
            .set_state(BatchPlannerModesSimpleStates::StatePreparing);
    }

    /// Dispatches one planner request to the child and copies its bounded output.
    pub(crate) fn process_event(
        &mut self,
        runtime: PlanRuntime,
        target: &mut PlanScratch,
    ) -> ModeResult {
        if self
            .machine
            .process_event(BatchPlannerModesSimpleEvents::Plan(runtime))
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
            .debug_struct("SimplePlannerActor")
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use crate::planner::sm::{PlanConfig, PlanMode, PlanRequest, Planner, StepSize};
    use core::num::NonZeroUsize;

    #[test]
    fn chunks_remainder_and_preserves_offsets() {
        let mut planner = Planner::new();
        let result = planner
            .plan(PlanRequest::new(
                (0..5).collect(),
                PlanConfig::new(PlanMode::Simple)
                    .with_step_size(StepSize::fixed(NonZeroUsize::new(2).unwrap())),
            ))
            .expect("simple planning should succeed");
        assert_eq!(result.step_sizes, [2, 2, 1]);
        assert_eq!(result.step_token_indices, [0, 1, 2, 3, 4]);
        assert_eq!(result.step_token_offsets, [0, 2, 4, 5]);
    }

    #[test]
    fn zero_requested_step_size_defaults_to_all_tokens() {
        let mut planner = Planner::new();
        let result = planner
            .plan(PlanRequest::new(
                (0..3).collect(),
                PlanConfig::new(PlanMode::Simple).with_step_size(StepSize::automatic()),
            ))
            .expect("zero step size uses token count");
        assert_eq!(result.step_sizes, [3]);
        assert_eq!(result.step_token_offsets, [0, 3]);
    }
    #[test]
    fn reuses_scratch_after_completed_plan() {
        let mut planner = Planner::new();
        let request = PlanRequest::new(
            vec![0, 1],
            PlanConfig::new(PlanMode::Simple).with_step_size(StepSize::automatic()),
        );
        let result = planner
            .plan(request)
            .expect("zero requested size normalizes to token count");
        assert_eq!(result.step_sizes, [2]);

        let result = planner
            .plan(PlanRequest::new(
                vec![0, 1],
                PlanConfig::new(PlanMode::Simple)
                    .with_step_size(StepSize::fixed(NonZeroUsize::new(1).unwrap())),
            ))
            .expect("planner should remain reusable");
        assert_eq!(result.step_sizes, [1, 1]);
        assert_eq!(result.step_token_indices, [0, 1]);
        assert_eq!(result.step_token_offsets, [0, 1, 2]);
    }
}
