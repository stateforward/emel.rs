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

// --- machine LogitsValidator from emel.cpp/src/emel/logits/validator/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventBuildRuntime;

sml! {
    LogitsValidator {
        "request_decision"_s <= *"ready"_s + event<EventBuildRuntime> / begin_build,
        "done"_s <= "request_decision"_s + completion<EventBuildRuntime> [valid_request] / execute_build,
        "errored"_s <= "request_decision"_s + completion<EventBuildRuntime> [invalid_request] / mark_invalid_request,
        "ready"_s <= "done"_s + completion<EventBuildRuntime> / publish_done,
        "ready"_s <= "errored"_s + completion<EventBuildRuntime> / publish_error,
        "ready"_s <= "ready"_s + unexpected_event<_> / on_unexpected_from_ready,
        "ready"_s <= "request_decision"_s + unexpected_event<_> / on_unexpected_from_request_decision,
        "ready"_s <= "done"_s + unexpected_event<_> / on_unexpected_from_done,
        "ready"_s <= "errored"_s + unexpected_event<_> / on_unexpected_from_errored,
    }
}

/// Context for `LogitsValidator` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct LogitsValidatorContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl LogitsValidatorStateMachineContext for LogitsValidatorContext {
    fn begin_build(&mut self, _event: &EventBuildRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/logits/validator/actions.hpp::begin_build
        todo!("TODO: port action `begin_build` from emel.cpp/src/emel/logits/validator/actions.hpp")
    }
    fn execute_build(&mut self, _event: &EventBuildRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/logits/validator/actions.hpp::execute_build
        todo!(
            "TODO: port action `execute_build` from emel.cpp/src/emel/logits/validator/actions.hpp"
        )
    }
    fn invalid_request(&self, _event: &EventBuildRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/logits/validator/guards.hpp::invalid_request
        todo!(
            "TODO: port guard `invalid_request` from emel.cpp/src/emel/logits/validator/guards.hpp"
        )
    }
    fn mark_invalid_request(&mut self, _event: &EventBuildRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/logits/validator/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/logits/validator/actions.hpp"
        )
    }
    fn on_unexpected_from_done(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/logits/validator/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/logits/validator/actions.hpp"
        )
    }
    fn on_unexpected_from_errored(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/logits/validator/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/logits/validator/actions.hpp"
        )
    }
    fn on_unexpected_from_ready(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/logits/validator/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/logits/validator/actions.hpp"
        )
    }
    fn on_unexpected_from_request_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/logits/validator/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/logits/validator/actions.hpp"
        )
    }
    fn publish_done(&mut self, _event: &EventBuildRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/logits/validator/actions.hpp::publish_done
        todo!(
            "TODO: port action `publish_done` from emel.cpp/src/emel/logits/validator/actions.hpp"
        )
    }
    fn publish_error(&mut self, _event: &EventBuildRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/logits/validator/actions.hpp::publish_error
        todo!(
            "TODO: port action `publish_error` from emel.cpp/src/emel/logits/validator/actions.hpp"
        )
    }
    fn valid_request(&self, _event: &EventBuildRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/logits/validator/guards.hpp::valid_request
        todo!("TODO: port guard `valid_request` from emel.cpp/src/emel/logits/validator/guards.hpp")
    }
}
