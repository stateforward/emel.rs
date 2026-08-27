//! State machine scaffold port — not a stable public API.
//! Bodies are stubs (`todo!`) until contexts/guards/actions are ported from C++.

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
    missing_docs
)]

use sml::sml;

// --- machine BatchPlannerModesSimple from emel.cpp/src/emel/batch/planner/modes/simple/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventPlanRuntime;

sml! {
    BatchPlannerModesSimple {
        "state_planning"_s <= *"state_preparing"_s + event<EventPlanRuntime> / effect_begin_planning,
        "state_planning_input_decision"_s <= "state_planning"_s + completion<EventPlanRuntime>,
        "state_planning_failed"_s <= "state_planning_input_decision"_s + completion<EventPlanRuntime> [guard_has_invalid_step_size] / effect_reject_invalid_step_size,
        "state_planning_capacity_decision"_s <= "state_planning_input_decision"_s + completion<EventPlanRuntime> [guard_has_valid_step_size],
        "state_planning_failed"_s <= "state_planning_capacity_decision"_s + completion<EventPlanRuntime> [guard_exceeds_step_capacity] / effect_reject_output_steps_full,
        "state_planning_failed"_s <= "state_planning_capacity_decision"_s + completion<EventPlanRuntime> [guard_exceeds_index_capacity] / effect_reject_output_indices_full,
        "state_planning_decision"_s <= "state_planning_capacity_decision"_s + completion<EventPlanRuntime> [guard_simple_plan_capacity_ok] / effect_plan_simple_batches,
        "state_planning_done"_s <= "state_planning_decision"_s + completion<EventPlanRuntime> [guard_planning_succeeded] / effect_emit_plan_done,
        "state_planning_failed"_s <= "state_planning_decision"_s + completion<EventPlanRuntime> [guard_planning_failed] / effect_reject_planning_progress_stalled,
        "state_planning_failed"_s <= "state_planning_done"_s + unexpected_event<_> / effect_emit_internal_plan_error_from_state_planning_done,
        "state_planning_failed"_s <= "state_planning_failed"_s + unexpected_event<_> / effect_emit_internal_plan_error_from_state_planning_failed,
        "state_planning_failed"_s <= "state_preparing"_s + unexpected_event<_> / effect_emit_internal_plan_error_from_state_preparing,
        "state_planning_failed"_s <= "state_planning"_s + unexpected_event<_> / effect_emit_internal_plan_error_from_state_planning,
        "state_planning_failed"_s <= "state_planning_input_decision"_s + unexpected_event<_> / effect_emit_internal_plan_error_from_state_planning_input_decision,
        "state_planning_failed"_s <= "state_planning_capacity_decision"_s + unexpected_event<_> / effect_emit_internal_plan_error_from_state_planning_capacity_decision,
        "state_planning_failed"_s <= "state_planning_decision"_s + unexpected_event<_> / effect_emit_internal_plan_error_from_state_planning_decision,
    }
}

/// Context for `BatchPlannerModesSimple` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct BatchPlannerModesSimpleContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl BatchPlannerModesSimpleStateMachineContext for BatchPlannerModesSimpleContext {
    fn effect_begin_planning(&mut self, _event: &EventPlanRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/batch/planner/modes/simple/actions.hpp::effect_begin_planning
        todo!(
            "TODO: port action `effect_begin_planning` from emel.cpp/src/emel/batch/planner/modes/simple/actions.hpp"
        )
    }
    fn effect_emit_internal_plan_error_from_state_planning(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/batch/planner/modes/simple/actions.hpp::effect_emit_internal_plan_error
        todo!(
            "TODO: port action `effect_emit_internal_plan_error` from emel.cpp/src/emel/batch/planner/modes/simple/actions.hpp"
        )
    }
    fn effect_emit_internal_plan_error_from_state_planning_capacity_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/batch/planner/modes/simple/actions.hpp::effect_emit_internal_plan_error
        todo!(
            "TODO: port action `effect_emit_internal_plan_error` from emel.cpp/src/emel/batch/planner/modes/simple/actions.hpp"
        )
    }
    fn effect_emit_internal_plan_error_from_state_planning_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/batch/planner/modes/simple/actions.hpp::effect_emit_internal_plan_error
        todo!(
            "TODO: port action `effect_emit_internal_plan_error` from emel.cpp/src/emel/batch/planner/modes/simple/actions.hpp"
        )
    }
    fn effect_emit_internal_plan_error_from_state_planning_done(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/batch/planner/modes/simple/actions.hpp::effect_emit_internal_plan_error
        todo!(
            "TODO: port action `effect_emit_internal_plan_error` from emel.cpp/src/emel/batch/planner/modes/simple/actions.hpp"
        )
    }
    fn effect_emit_internal_plan_error_from_state_planning_failed(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/batch/planner/modes/simple/actions.hpp::effect_emit_internal_plan_error
        todo!(
            "TODO: port action `effect_emit_internal_plan_error` from emel.cpp/src/emel/batch/planner/modes/simple/actions.hpp"
        )
    }
    fn effect_emit_internal_plan_error_from_state_planning_input_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/batch/planner/modes/simple/actions.hpp::effect_emit_internal_plan_error
        todo!(
            "TODO: port action `effect_emit_internal_plan_error` from emel.cpp/src/emel/batch/planner/modes/simple/actions.hpp"
        )
    }
    fn effect_emit_internal_plan_error_from_state_preparing(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/batch/planner/modes/simple/actions.hpp::effect_emit_internal_plan_error
        todo!(
            "TODO: port action `effect_emit_internal_plan_error` from emel.cpp/src/emel/batch/planner/modes/simple/actions.hpp"
        )
    }
    fn effect_emit_plan_done(&mut self, _event: &EventPlanRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/batch/planner/modes/simple/actions.hpp::effect_emit_plan_done
        todo!(
            "TODO: port action `effect_emit_plan_done` from emel.cpp/src/emel/batch/planner/modes/simple/actions.hpp"
        )
    }
    fn effect_plan_simple_batches(&mut self, _event: &EventPlanRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/batch/planner/modes/simple/actions.hpp::effect_plan_simple_batches
        todo!(
            "TODO: port action `effect_plan_simple_batches` from emel.cpp/src/emel/batch/planner/modes/simple/actions.hpp"
        )
    }
    fn effect_reject_invalid_step_size(&mut self, _event: &EventPlanRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/batch/planner/modes/simple/actions.hpp::effect_reject_invalid_step_size
        todo!(
            "TODO: port action `effect_reject_invalid_step_size` from emel.cpp/src/emel/batch/planner/modes/simple/actions.hpp"
        )
    }
    fn effect_reject_output_indices_full(&mut self, _event: &EventPlanRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/batch/planner/modes/simple/actions.hpp::effect_reject_output_indices_full
        todo!(
            "TODO: port action `effect_reject_output_indices_full` from emel.cpp/src/emel/batch/planner/modes/simple/actions.hpp"
        )
    }
    fn effect_reject_output_steps_full(&mut self, _event: &EventPlanRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/batch/planner/modes/simple/actions.hpp::effect_reject_output_steps_full
        todo!(
            "TODO: port action `effect_reject_output_steps_full` from emel.cpp/src/emel/batch/planner/modes/simple/actions.hpp"
        )
    }
    fn effect_reject_planning_progress_stalled(
        &mut self,
        _event: &EventPlanRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/batch/planner/modes/simple/actions.hpp::effect_reject_planning_progress_stalled
        todo!(
            "TODO: port action `effect_reject_planning_progress_stalled` from emel.cpp/src/emel/batch/planner/modes/simple/actions.hpp"
        )
    }
    fn guard_exceeds_index_capacity(&self, _event: &EventPlanRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/batch/planner/modes/simple/guards.hpp::guard_exceeds_index_capacity
        todo!(
            "TODO: port guard `guard_exceeds_index_capacity` from emel.cpp/src/emel/batch/planner/modes/simple/guards.hpp"
        )
    }
    fn guard_exceeds_step_capacity(&self, _event: &EventPlanRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/batch/planner/modes/simple/guards.hpp::guard_exceeds_step_capacity
        todo!(
            "TODO: port guard `guard_exceeds_step_capacity` from emel.cpp/src/emel/batch/planner/modes/simple/guards.hpp"
        )
    }
    fn guard_has_invalid_step_size(&self, _event: &EventPlanRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/batch/planner/modes/simple/guards.hpp::guard_has_invalid_step_size
        todo!(
            "TODO: port guard `guard_has_invalid_step_size` from emel.cpp/src/emel/batch/planner/modes/simple/guards.hpp"
        )
    }
    fn guard_has_valid_step_size(&self, _event: &EventPlanRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/batch/planner/modes/simple/guards.hpp::guard_has_valid_step_size
        todo!(
            "TODO: port guard `guard_has_valid_step_size` from emel.cpp/src/emel/batch/planner/modes/simple/guards.hpp"
        )
    }
    fn guard_planning_failed(&self, _event: &EventPlanRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/batch/planner/modes/simple/guards.hpp::guard_planning_failed
        todo!(
            "TODO: port guard `guard_planning_failed` from emel.cpp/src/emel/batch/planner/modes/simple/guards.hpp"
        )
    }
    fn guard_planning_succeeded(&self, _event: &EventPlanRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/batch/planner/modes/simple/guards.hpp::guard_planning_succeeded
        todo!(
            "TODO: port guard `guard_planning_succeeded` from emel.cpp/src/emel/batch/planner/modes/simple/guards.hpp"
        )
    }
    fn guard_simple_plan_capacity_ok(&self, _event: &EventPlanRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/batch/planner/modes/simple/guards.hpp::guard_simple_plan_capacity_ok
        todo!(
            "TODO: port guard `guard_simple_plan_capacity_ok` from emel.cpp/src/emel/batch/planner/modes/simple/guards.hpp"
        )
    }
}
