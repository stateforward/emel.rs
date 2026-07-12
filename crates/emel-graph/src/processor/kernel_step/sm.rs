//! State machine scaffold port — not a stable public API.
//! Bodies are stubs (`todo!`) until contexts/guards/actions are ported from C++.

#![allow(
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

// --- machine GraphProcessorKernelStep from emel.cpp/src/emel/graph/processor/kernel_step/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct ProcessorEventExecuteStep;

sml! {
    GraphProcessorKernelStep {
        "execute_failed"_s <= *"deciding"_s + completion<ProcessorEventExecuteStep> [phase_prefailed] / mark_failed_existing_error,
        "callback_decision"_s <= "deciding"_s + completion<ProcessorEventExecuteStep> [phase_request_callback] / run_callback,
        "execute_failed"_s <= "deciding"_s + completion<ProcessorEventExecuteStep> [phase_missing_callback] / mark_failed_invalid_request,
        "executed"_s <= "callback_decision"_s + completion<ProcessorEventExecuteStep> [callback_ok] / mark_done,
        "execute_failed"_s <= "callback_decision"_s + completion<ProcessorEventExecuteStep> [callback_error] / mark_failed_callback_error,
        "execute_failed"_s <= "callback_decision"_s + completion<ProcessorEventExecuteStep> [callback_failed_without_error] / mark_failed_callback_without_error,
        "unexpected_event"_s <= "deciding"_s + unexpected_event<_> / on_unexpected_from_deciding,
        "unexpected_event"_s <= "callback_decision"_s + unexpected_event<_> / on_unexpected_from_callback_decision,
        "unexpected_event"_s <= "executed"_s + unexpected_event<_> / on_unexpected_from_executed,
        "unexpected_event"_s <= "execute_failed"_s + unexpected_event<_> / on_unexpected_from_execute_failed,
        "unexpected_event"_s <= "unexpected_event"_s + unexpected_event<_> / on_unexpected_from_unexpected_event,
        "executed"_s = X,
        "execute_failed"_s = X,
    }
}

/// Context for `GraphProcessorKernelStep` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct GraphProcessorKernelStepContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl GraphProcessorKernelStepStateMachineContext for GraphProcessorKernelStepContext {
    fn callback_error(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/processor/kernel_step/guards.hpp::callback_error
        todo!(
            "TODO: port guard `callback_error` from emel.cpp/src/emel/graph/processor/kernel_step/guards.hpp"
        )
    }
    fn callback_failed_without_error(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/processor/kernel_step/guards.hpp::callback_failed_without_error
        todo!(
            "TODO: port guard `callback_failed_without_error` from emel.cpp/src/emel/graph/processor/kernel_step/guards.hpp"
        )
    }
    fn callback_ok(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/processor/kernel_step/guards.hpp::callback_ok
        todo!(
            "TODO: port guard `callback_ok` from emel.cpp/src/emel/graph/processor/kernel_step/guards.hpp"
        )
    }
    fn mark_done(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/processor/kernel_step/actions.hpp::mark_done
        todo!(
            "TODO: port action `mark_done` from emel.cpp/src/emel/graph/processor/kernel_step/actions.hpp"
        )
    }
    fn mark_failed_callback_error(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/processor/kernel_step/actions.hpp::mark_failed_callback_error
        todo!(
            "TODO: port action `mark_failed_callback_error` from emel.cpp/src/emel/graph/processor/kernel_step/actions.hpp"
        )
    }
    fn mark_failed_callback_without_error(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/processor/kernel_step/actions.hpp::mark_failed_callback_without_error
        todo!(
            "TODO: port action `mark_failed_callback_without_error` from emel.cpp/src/emel/graph/processor/kernel_step/actions.hpp"
        )
    }
    fn mark_failed_existing_error(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/processor/kernel_step/actions.hpp::mark_failed_existing_error
        todo!(
            "TODO: port action `mark_failed_existing_error` from emel.cpp/src/emel/graph/processor/kernel_step/actions.hpp"
        )
    }
    fn mark_failed_invalid_request(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/processor/kernel_step/actions.hpp::mark_failed_invalid_request
        todo!(
            "TODO: port action `mark_failed_invalid_request` from emel.cpp/src/emel/graph/processor/kernel_step/actions.hpp"
        )
    }
    fn on_unexpected_from_callback_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/processor/kernel_step/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/processor/kernel_step/actions.hpp"
        )
    }
    fn on_unexpected_from_deciding(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/processor/kernel_step/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/processor/kernel_step/actions.hpp"
        )
    }
    fn on_unexpected_from_execute_failed(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/processor/kernel_step/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/processor/kernel_step/actions.hpp"
        )
    }
    fn on_unexpected_from_executed(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/processor/kernel_step/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/processor/kernel_step/actions.hpp"
        )
    }
    fn on_unexpected_from_unexpected_event(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/processor/kernel_step/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/processor/kernel_step/actions.hpp"
        )
    }
    fn phase_missing_callback(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/processor/kernel_step/guards.hpp::phase_missing_callback
        todo!(
            "TODO: port guard `phase_missing_callback` from emel.cpp/src/emel/graph/processor/kernel_step/guards.hpp"
        )
    }
    fn phase_prefailed(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/processor/kernel_step/guards.hpp::phase_prefailed
        todo!(
            "TODO: port guard `phase_prefailed` from emel.cpp/src/emel/graph/processor/kernel_step/guards.hpp"
        )
    }
    fn phase_request_callback(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/processor/kernel_step/guards.hpp::phase_request_callback
        todo!(
            "TODO: port guard `phase_request_callback` from emel.cpp/src/emel/graph/processor/kernel_step/guards.hpp"
        )
    }
    fn run_callback(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/processor/kernel_step/actions.hpp::run_callback
        todo!(
            "TODO: port action `run_callback` from emel.cpp/src/emel/graph/processor/kernel_step/actions.hpp"
        )
    }
}
