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

// --- machine TextGeneratorMatmul from emel.cpp/src/emel/text/generator/matmul/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventConfigureKernelKind;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventExecuteParallel;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventExecuteSerial;

sml! {
    TextGeneratorMatmul {
        "state_ready"_s <= *"state_ready"_s + event<EventConfigureKernelKind> / effect_configure_kernel_kind,
        "state_serial_result_decision"_s <= "state_ready"_s + event<EventExecuteSerial> / effect_execute_serial,
        "state_ready"_s <= "state_serial_result_decision"_s + completion<EventExecuteSerial> [guard_serial_accepted] / effect_accept_serial_execution,
        "state_ready"_s <= "state_serial_result_decision"_s + completion<EventExecuteSerial> [guard_serial_rejected] / effect_reject_serial_execution,
        "state_parallel_result_decision"_s <= "state_ready"_s + event<EventExecuteParallel> [guard_parallel_x8_ready] / effect_execute_parallel_x8,
        "state_parallel_result_decision"_s <= "state_ready"_s + event<EventExecuteParallel> [guard_parallel_x4_ready] / effect_execute_parallel_x4,
        "state_parallel_result_decision"_s <= "state_ready"_s + event<EventExecuteParallel> [guard_parallel_unit_ready] / effect_execute_parallel_unit,
        "state_ready"_s <= "state_ready"_s + event<EventExecuteParallel> [guard_parallel_unavailable] / effect_reject_parallel_execution_from_state_ready,
        "state_ready"_s <= "state_parallel_result_decision"_s + completion<EventExecuteParallel> [guard_parallel_submission_failed] / effect_reject_parallel_execution_from_state_parallel_result_decision,
        "state_ready"_s <= "state_parallel_result_decision"_s + completion<EventExecuteParallel> [guard_parallel_join_failed] / effect_reject_parallel_execution_from_state_parallel_result_decision,
        "state_ready"_s <= "state_parallel_result_decision"_s + completion<EventExecuteParallel> [guard_parallel_lane_rejected] / effect_reject_parallel_execution_from_state_parallel_result_decision,
        "state_ready"_s <= "state_parallel_result_decision"_s + completion<EventExecuteParallel> [guard_parallel_all_lanes_accepted] / effect_accept_parallel_execution,
        "state_ready"_s <= "state_ready"_s + unexpected_event<_> / effect_on_unexpected_from_state_ready,
        "state_ready"_s <= "state_serial_result_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_serial_result_decision,
        "state_ready"_s <= "state_parallel_result_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_parallel_result_decision,
    }
}

/// Context for `TextGeneratorMatmul` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct TextGeneratorMatmulContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl TextGeneratorMatmulStateMachineContext for TextGeneratorMatmulContext {
    fn effect_accept_parallel_execution(
        &mut self,
        _event: &EventExecuteParallel,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/matmul/actions.hpp::effect_accept_parallel_execution
        todo!(
            "TODO: port action `effect_accept_parallel_execution` from emel.cpp/src/emel/text/generator/matmul/actions.hpp"
        )
    }
    fn effect_accept_serial_execution(&mut self, _event: &EventExecuteSerial) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/matmul/actions.hpp::effect_accept_serial_execution
        todo!(
            "TODO: port action `effect_accept_serial_execution` from emel.cpp/src/emel/text/generator/matmul/actions.hpp"
        )
    }
    fn effect_configure_kernel_kind(
        &mut self,
        _event: &EventConfigureKernelKind,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/matmul/actions.hpp::effect_configure_kernel_kind
        todo!(
            "TODO: port action `effect_configure_kernel_kind` from emel.cpp/src/emel/text/generator/matmul/actions.hpp"
        )
    }
    fn effect_execute_parallel_unit(&mut self, _event: &EventExecuteParallel) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/matmul/actions.hpp::effect_execute_parallel_unit
        todo!(
            "TODO: port action `effect_execute_parallel_unit` from emel.cpp/src/emel/text/generator/matmul/actions.hpp"
        )
    }
    fn effect_execute_parallel_x4(&mut self, _event: &EventExecuteParallel) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/matmul/actions.hpp::effect_execute_parallel_x4
        todo!(
            "TODO: port action `effect_execute_parallel_x4` from emel.cpp/src/emel/text/generator/matmul/actions.hpp"
        )
    }
    fn effect_execute_parallel_x8(&mut self, _event: &EventExecuteParallel) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/matmul/actions.hpp::effect_execute_parallel_x8
        todo!(
            "TODO: port action `effect_execute_parallel_x8` from emel.cpp/src/emel/text/generator/matmul/actions.hpp"
        )
    }
    fn effect_execute_serial(&mut self, _event: &EventExecuteSerial) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/matmul/actions.hpp::effect_execute_serial
        todo!(
            "TODO: port action `effect_execute_serial` from emel.cpp/src/emel/text/generator/matmul/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_parallel_result_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/matmul/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/text/generator/matmul/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_ready(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/matmul/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/text/generator/matmul/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_serial_result_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/matmul/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/text/generator/matmul/actions.hpp"
        )
    }
    fn effect_reject_parallel_execution_from_state_parallel_result_decision(
        &mut self,
        _event: &EventExecuteParallel,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/matmul/actions.hpp::effect_reject_parallel_execution
        todo!(
            "TODO: port action `effect_reject_parallel_execution` from emel.cpp/src/emel/text/generator/matmul/actions.hpp"
        )
    }
    fn effect_reject_parallel_execution_from_state_ready(
        &mut self,
        _event: &EventExecuteParallel,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/matmul/actions.hpp::effect_reject_parallel_execution
        todo!(
            "TODO: port action `effect_reject_parallel_execution` from emel.cpp/src/emel/text/generator/matmul/actions.hpp"
        )
    }
    fn effect_reject_serial_execution(&mut self, _event: &EventExecuteSerial) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/matmul/actions.hpp::effect_reject_serial_execution
        todo!(
            "TODO: port action `effect_reject_serial_execution` from emel.cpp/src/emel/text/generator/matmul/actions.hpp"
        )
    }
    fn guard_parallel_all_lanes_accepted(&self, _event: &EventExecuteParallel) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/matmul/guards.hpp::guard_parallel_all_lanes_accepted
        todo!(
            "TODO: port guard `guard_parallel_all_lanes_accepted` from emel.cpp/src/emel/text/generator/matmul/guards.hpp"
        )
    }
    fn guard_parallel_join_failed(&self, _event: &EventExecuteParallel) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/matmul/guards.hpp::guard_parallel_join_failed
        todo!(
            "TODO: port guard `guard_parallel_join_failed` from emel.cpp/src/emel/text/generator/matmul/guards.hpp"
        )
    }
    fn guard_parallel_lane_rejected(&self, _event: &EventExecuteParallel) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/matmul/guards.hpp::guard_parallel_lane_rejected
        todo!(
            "TODO: port guard `guard_parallel_lane_rejected` from emel.cpp/src/emel/text/generator/matmul/guards.hpp"
        )
    }
    fn guard_parallel_submission_failed(&self, _event: &EventExecuteParallel) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/matmul/guards.hpp::guard_parallel_submission_failed
        todo!(
            "TODO: port guard `guard_parallel_submission_failed` from emel.cpp/src/emel/text/generator/matmul/guards.hpp"
        )
    }
    fn guard_parallel_unavailable(&self, _event: &EventExecuteParallel) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/matmul/guards.hpp::guard_parallel_unavailable
        todo!(
            "TODO: port guard `guard_parallel_unavailable` from emel.cpp/src/emel/text/generator/matmul/guards.hpp"
        )
    }
    fn guard_parallel_unit_ready(&self, _event: &EventExecuteParallel) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/matmul/guards.hpp::guard_parallel_unit_ready
        todo!(
            "TODO: port guard `guard_parallel_unit_ready` from emel.cpp/src/emel/text/generator/matmul/guards.hpp"
        )
    }
    fn guard_parallel_x4_ready(&self, _event: &EventExecuteParallel) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/matmul/guards.hpp::guard_parallel_x4_ready
        todo!(
            "TODO: port guard `guard_parallel_x4_ready` from emel.cpp/src/emel/text/generator/matmul/guards.hpp"
        )
    }
    fn guard_parallel_x8_ready(&self, _event: &EventExecuteParallel) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/matmul/guards.hpp::guard_parallel_x8_ready
        todo!(
            "TODO: port guard `guard_parallel_x8_ready` from emel.cpp/src/emel/text/generator/matmul/guards.hpp"
        )
    }
    fn guard_serial_accepted(&self, _event: &EventExecuteSerial) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/matmul/guards.hpp::guard_serial_accepted
        todo!(
            "TODO: port guard `guard_serial_accepted` from emel.cpp/src/emel/text/generator/matmul/guards.hpp"
        )
    }
    fn guard_serial_rejected(&self, _event: &EventExecuteSerial) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/matmul/guards.hpp::guard_serial_rejected
        todo!(
            "TODO: port guard `guard_serial_rejected` from emel.cpp/src/emel/text/generator/matmul/guards.hpp"
        )
    }
}
