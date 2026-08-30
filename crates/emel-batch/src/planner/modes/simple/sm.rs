//! Simple fixed-size batch planning algorithm.

#![allow(
    clippy::derive_partial_eq_without_eq,
    clippy::redundant_pub_crate,
    private_interfaces
)]

use crate::planner::sm::{MAX_PLAN_STEPS, PlanRuntime, PlannerError};
use sml::sml;

#[derive(Clone, Debug)]
struct Plan;

sml! {
    BatchPlannerModesSimple {
        "planning_input"_s <= *"preparing"_s + Plan(PlanRuntime) / effect_begin,
        "planning_capacity"_s <= "planning_input"_s + completion<Plan>(PlanRuntime) [guard_valid],
        "planning_failed"_s <= "planning_input"_s + completion<Plan>(PlanRuntime) [guard_invalid] / effect_invalid,
        "planning"_s <= "planning_capacity"_s + completion<Plan>(PlanRuntime) [guard_capacity],
        "planning_failed"_s <= "planning_capacity"_s + completion<Plan>(PlanRuntime) [guard_no_capacity] / effect_capacity,
        "planning_done"_s <= "planning"_s + completion<Plan>(PlanRuntime) / effect_plan,
        "planning_failed"_s <= "preparing"_s + unexpected_event<_> / effect_unexpected,
        "planning_failed"_s <= "planning_input"_s + unexpected_event<_> / effect_unexpected,
        "planning_failed"_s <= "planning_capacity"_s + unexpected_event<_> / effect_unexpected,
        "planning_failed"_s <= "planning"_s + unexpected_event<_> / effect_unexpected,
        "planning_failed"_s <= "planning_done"_s + unexpected_event<_> / effect_unexpected,
        "planning_failed"_s <= "planning_failed"_s + unexpected_event<_> / effect_unexpected,
    }
}

#[derive(Debug, Default)]
struct Context;

impl BatchPlannerModesSimpleStateMachineContext for Context {
    fn effect_begin(&mut self, _event: PlanRuntime) -> Result<(), ()> {
        Ok(())
    }

    fn guard_valid(&self, event: &PlanRuntime) -> Result<bool, ()> {
        Ok(event.scratch.borrow().effective_step_size > 0)
    }

    fn guard_invalid(&self, event: &PlanRuntime) -> Result<bool, ()> {
        Ok(!self.guard_valid(event)?)
    }

    fn guard_capacity(&self, event: &PlanRuntime) -> Result<bool, ()> {
        let scratch = event.scratch.borrow();
        let n = event.request.token_ids.len();
        let size = scratch.effective_step_size;
        Ok(size > 0 && n <= MAX_PLAN_STEPS && n.div_ceil(size) <= MAX_PLAN_STEPS)
    }

    fn guard_no_capacity(&self, event: &PlanRuntime) -> Result<bool, ()> {
        Ok(!self.guard_capacity(event)?)
    }

    fn effect_capacity(&mut self, event: PlanRuntime) -> Result<(), ()> {
        event.scratch.borrow_mut().fail(PlannerError::OutputStepsFull);
        Ok(())
    }

    fn effect_invalid(&mut self, event: PlanRuntime) -> Result<(), ()> {
        event.scratch.borrow_mut().fail(PlannerError::InvalidStepSize);
        Ok(())
    }

    fn effect_plan(&mut self, event: PlanRuntime) -> Result<(), ()> {
        let size = event.scratch.borrow().effective_step_size;
        let n = event.request.token_ids.len();
        let mut scratch = event.scratch.borrow_mut();
        let mut start = 0;
        while start < n {
            let offset = scratch.step_token_indices.len();
            if !scratch.step_token_offsets.push(offset) {
                scratch.fail(PlannerError::OutputStepsFull);
                return Ok(());
            }
            let end = (start + size).min(n);
            for index in start..end {
                if !scratch.step_token_indices.push(index) {
                    scratch.fail(PlannerError::OutputIndicesFull);
                    return Ok(());
                }
            }
            if !scratch.step_sizes.push(end - start) {
                scratch.fail(PlannerError::OutputStepsFull);
                return Ok(());
            }
            start = end;
        }
        let offset = scratch.step_token_indices.len();
        if !scratch.step_token_offsets.push(offset) {
            scratch.fail(PlannerError::OutputStepsFull);
        }
        Ok(())
    }

    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Ok(())
    }
}

pub(crate) fn run(runtime: PlanRuntime) {
    let mut machine = BatchPlannerModesSimpleStateMachine::new(Context);
    let _ = machine.process_event(BatchPlannerModesSimpleEvents::Plan(runtime));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planner::sm::{PlanMode, PlanRequest, Planner};

    #[test]
    fn chunks_remainder_and_preserves_offsets() {
        let mut planner = Planner::new();
        let result = planner.plan(PlanRequest::new((0..5).collect(), 2, PlanMode::Simple));
        let result = result.expect("simple planning should succeed");
        assert_eq!(result.step_sizes, [2, 2, 1]);
        assert_eq!(result.step_token_indices, [0, 1, 2, 3, 4]);
        assert_eq!(result.step_token_offsets, [0, 2, 4, 5]);
    }

    #[test]
    fn zero_requested_step_size_defaults_to_all_tokens() {
        let mut planner = Planner::new();
        let result = planner
            .plan(PlanRequest::new((0..3).collect(), 0, PlanMode::Simple))
            .expect("zero step size uses token count");
        assert_eq!(result.step_sizes, [3]);
        assert_eq!(result.step_token_offsets, [0, 3]);
    }
}
