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

// --- machine TextJinjaFormatter from emel.cpp/src/emel/text/jinja/formatter/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventRenderRuntime;

sml! {
    TextJinjaFormatter {
        "request_decision"_s <= *"initialized"_s + event<EventRenderRuntime> [valid_render] / begin_render_from_initialized,
        "result_decision"_s <= "initialized"_s + event<EventRenderRuntime> [invalid_render_with_callbacks] / reject_invalid_render_from_initialized,
        "errored"_s <= "initialized"_s + event<EventRenderRuntime> [invalid_render_without_callbacks] / reject_invalid_render_from_initialized,
        "request_decision"_s <= "done"_s + event<EventRenderRuntime> [valid_render] / begin_render_from_done,
        "result_decision"_s <= "done"_s + event<EventRenderRuntime> [invalid_render_with_callbacks] / reject_invalid_render_from_done,
        "errored"_s <= "done"_s + event<EventRenderRuntime> [invalid_render_without_callbacks] / reject_invalid_render_from_done,
        "request_decision"_s <= "errored"_s + event<EventRenderRuntime> [valid_render] / begin_render_from_errored,
        "result_decision"_s <= "errored"_s + event<EventRenderRuntime> [invalid_render_with_callbacks] / reject_invalid_render_from_errored,
        "errored"_s <= "errored"_s + event<EventRenderRuntime> [invalid_render_without_callbacks] / reject_invalid_render_from_errored,
        "request_decision"_s <= "unexpected"_s + event<EventRenderRuntime> [valid_render] / begin_render_from_unexpected,
        "result_decision"_s <= "unexpected"_s + event<EventRenderRuntime> [invalid_render_with_callbacks] / reject_invalid_render_from_unexpected,
        "errored"_s <= "unexpected"_s + event<EventRenderRuntime> [invalid_render_without_callbacks] / reject_invalid_render_from_unexpected,
        "result_decision"_s <= "request_decision"_s + completion<EventRenderRuntime> [source_empty] / mark_empty_output,
        "copy_exec"_s <= "request_decision"_s + completion<EventRenderRuntime> [copy_ready] / copy_source_text,
        "result_decision"_s <= "request_decision"_s + completion<EventRenderRuntime> [source_overflow] / mark_capacity_error,
        "result_decision"_s <= "copy_exec"_s + completion<EventRenderRuntime>,
        "done"_s <= "result_decision"_s + completion<EventRenderRuntime> [request_ok] / dispatch_done,
        "errored"_s <= "result_decision"_s + completion<EventRenderRuntime> [request_failed] / dispatch_error,
        "unexpected"_s <= "initialized"_s + unexpected_event<_> / on_unexpected_from_initialized,
        "unexpected"_s <= "request_decision"_s + unexpected_event<_> / on_unexpected_from_request_decision,
        "unexpected"_s <= "copy_exec"_s + unexpected_event<_> / on_unexpected_from_copy_exec,
        "unexpected"_s <= "result_decision"_s + unexpected_event<_> / on_unexpected_from_result_decision,
        "unexpected"_s <= "done"_s + unexpected_event<_> / on_unexpected_from_done,
        "unexpected"_s <= "errored"_s + unexpected_event<_> / on_unexpected_from_errored,
        "unexpected"_s <= "unexpected"_s + unexpected_event<_> / on_unexpected_from_unexpected,
    }
}

/// Context for `TextJinjaFormatter` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct TextJinjaFormatterContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl TextJinjaFormatterStateMachineContext for TextJinjaFormatterContext {
    fn begin_render_from_done(&mut self, _event: &EventRenderRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/formatter/actions.hpp::begin_render
        todo!(
            "TODO: port action `begin_render` from emel.cpp/src/emel/text/jinja/formatter/actions.hpp"
        )
    }
    fn begin_render_from_errored(&mut self, _event: &EventRenderRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/formatter/actions.hpp::begin_render
        todo!(
            "TODO: port action `begin_render` from emel.cpp/src/emel/text/jinja/formatter/actions.hpp"
        )
    }
    fn begin_render_from_initialized(&mut self, _event: &EventRenderRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/formatter/actions.hpp::begin_render
        todo!(
            "TODO: port action `begin_render` from emel.cpp/src/emel/text/jinja/formatter/actions.hpp"
        )
    }
    fn begin_render_from_unexpected(&mut self, _event: &EventRenderRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/formatter/actions.hpp::begin_render
        todo!(
            "TODO: port action `begin_render` from emel.cpp/src/emel/text/jinja/formatter/actions.hpp"
        )
    }
    fn copy_ready(&self, _event: &EventRenderRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/formatter/guards.hpp::copy_ready
        todo!(
            "TODO: port guard `copy_ready` from emel.cpp/src/emel/text/jinja/formatter/guards.hpp"
        )
    }
    fn copy_source_text(&mut self, _event: &EventRenderRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/formatter/actions.hpp::copy_source_text
        todo!(
            "TODO: port action `copy_source_text` from emel.cpp/src/emel/text/jinja/formatter/actions.hpp"
        )
    }
    fn dispatch_done(&mut self, _event: &EventRenderRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/formatter/actions.hpp::dispatch_done
        todo!(
            "TODO: port action `dispatch_done` from emel.cpp/src/emel/text/jinja/formatter/actions.hpp"
        )
    }
    fn dispatch_error(&mut self, _event: &EventRenderRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/formatter/actions.hpp::dispatch_error
        todo!(
            "TODO: port action `dispatch_error` from emel.cpp/src/emel/text/jinja/formatter/actions.hpp"
        )
    }
    fn invalid_render_with_callbacks(&self, _event: &EventRenderRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/formatter/guards.hpp::invalid_render_with_callbacks
        todo!(
            "TODO: port guard `invalid_render_with_callbacks` from emel.cpp/src/emel/text/jinja/formatter/guards.hpp"
        )
    }
    fn invalid_render_without_callbacks(&self, _event: &EventRenderRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/formatter/guards.hpp::invalid_render_without_callbacks
        todo!(
            "TODO: port guard `invalid_render_without_callbacks` from emel.cpp/src/emel/text/jinja/formatter/guards.hpp"
        )
    }
    fn mark_capacity_error(&mut self, _event: &EventRenderRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/formatter/actions.hpp::mark_capacity_error
        todo!(
            "TODO: port action `mark_capacity_error` from emel.cpp/src/emel/text/jinja/formatter/actions.hpp"
        )
    }
    fn mark_empty_output(&mut self, _event: &EventRenderRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/formatter/actions.hpp::mark_empty_output
        todo!(
            "TODO: port action `mark_empty_output` from emel.cpp/src/emel/text/jinja/formatter/actions.hpp"
        )
    }
    fn on_unexpected_from_copy_exec(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/formatter/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/formatter/actions.hpp"
        )
    }
    fn on_unexpected_from_done(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/formatter/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/formatter/actions.hpp"
        )
    }
    fn on_unexpected_from_errored(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/formatter/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/formatter/actions.hpp"
        )
    }
    fn on_unexpected_from_initialized(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/formatter/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/formatter/actions.hpp"
        )
    }
    fn on_unexpected_from_request_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/formatter/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/formatter/actions.hpp"
        )
    }
    fn on_unexpected_from_result_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/formatter/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/formatter/actions.hpp"
        )
    }
    fn on_unexpected_from_unexpected(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/formatter/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/formatter/actions.hpp"
        )
    }
    fn reject_invalid_render_from_done(&mut self, _event: &EventRenderRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/formatter/actions.hpp::reject_invalid_render
        todo!(
            "TODO: port action `reject_invalid_render` from emel.cpp/src/emel/text/jinja/formatter/actions.hpp"
        )
    }
    fn reject_invalid_render_from_errored(
        &mut self,
        _event: &EventRenderRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/formatter/actions.hpp::reject_invalid_render
        todo!(
            "TODO: port action `reject_invalid_render` from emel.cpp/src/emel/text/jinja/formatter/actions.hpp"
        )
    }
    fn reject_invalid_render_from_initialized(
        &mut self,
        _event: &EventRenderRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/formatter/actions.hpp::reject_invalid_render
        todo!(
            "TODO: port action `reject_invalid_render` from emel.cpp/src/emel/text/jinja/formatter/actions.hpp"
        )
    }
    fn reject_invalid_render_from_unexpected(
        &mut self,
        _event: &EventRenderRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/formatter/actions.hpp::reject_invalid_render
        todo!(
            "TODO: port action `reject_invalid_render` from emel.cpp/src/emel/text/jinja/formatter/actions.hpp"
        )
    }
    fn request_failed(&self, _event: &EventRenderRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/formatter/guards.hpp::request_failed
        todo!(
            "TODO: port guard `request_failed` from emel.cpp/src/emel/text/jinja/formatter/guards.hpp"
        )
    }
    fn request_ok(&self, _event: &EventRenderRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/formatter/guards.hpp::request_ok
        todo!(
            "TODO: port guard `request_ok` from emel.cpp/src/emel/text/jinja/formatter/guards.hpp"
        )
    }
    fn source_empty(&self, _event: &EventRenderRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/formatter/guards.hpp::source_empty
        todo!(
            "TODO: port guard `source_empty` from emel.cpp/src/emel/text/jinja/formatter/guards.hpp"
        )
    }
    fn source_overflow(&self, _event: &EventRenderRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/formatter/guards.hpp::source_overflow
        todo!(
            "TODO: port guard `source_overflow` from emel.cpp/src/emel/text/jinja/formatter/guards.hpp"
        )
    }
    fn valid_render(&self, _event: &EventRenderRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/formatter/guards.hpp::valid_render
        todo!(
            "TODO: port guard `valid_render` from emel.cpp/src/emel/text/jinja/formatter/guards.hpp"
        )
    }
}
