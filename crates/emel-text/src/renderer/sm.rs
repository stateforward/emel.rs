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

// --- machine TextRenderer from emel.cpp/src/emel/text/renderer/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventFlushRuntime;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventInitializeRuntime;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventRenderRuntime;

sml! {
    TextRenderer {
        "initializing"_s <= *"uninitialized"_s + event<EventInitializeRuntime> [valid_initialize] / begin_initialize_from_uninitialized,
        "initialize_publish_error"_s <= "uninitialized"_s + event<EventInitializeRuntime> [invalid_initialize] / reject_initialize_from_uninitialized,
        "render_publish_error"_s <= "uninitialized"_s + event<EventRenderRuntime> / reject_render_from_uninitialized,
        "flush_publish_error"_s <= "uninitialized"_s + event<EventFlushRuntime> / reject_flush_from_uninitialized,
        "initializing"_s <= "initialized"_s + event<EventInitializeRuntime> [valid_initialize] / begin_initialize_from_initialized,
        "initialize_publish_error"_s <= "initialized"_s + event<EventInitializeRuntime> [invalid_initialize] / reject_initialize_from_initialized,
        "rendering"_s <= "initialized"_s + event<EventRenderRuntime> [valid_render] / begin_render_from_initialized,
        "render_publish_error"_s <= "initialized"_s + event<EventRenderRuntime> [invalid_render] / reject_render_from_initialized,
        "flushing"_s <= "initialized"_s + event<EventFlushRuntime> [valid_flush] / begin_flush_from_initialized,
        "flush_publish_error"_s <= "initialized"_s + event<EventFlushRuntime> [invalid_flush] / reject_flush_from_initialized,
        "initializing"_s <= "done"_s + event<EventInitializeRuntime> [valid_initialize] / begin_initialize_from_done,
        "initialize_publish_error"_s <= "done"_s + event<EventInitializeRuntime> [invalid_initialize] / reject_initialize_from_done,
        "rendering"_s <= "done"_s + event<EventRenderRuntime> [valid_render] / begin_render_from_done,
        "render_publish_error"_s <= "done"_s + event<EventRenderRuntime> [invalid_render] / reject_render_from_done,
        "flushing"_s <= "done"_s + event<EventFlushRuntime> [valid_flush] / begin_flush_from_done,
        "flush_publish_error"_s <= "done"_s + event<EventFlushRuntime> [invalid_flush] / reject_flush_from_done,
        "initializing"_s <= "errored"_s + event<EventInitializeRuntime> [valid_initialize] / begin_initialize_from_errored,
        "initialize_publish_error"_s <= "errored"_s + event<EventInitializeRuntime> [invalid_initialize] / reject_initialize_from_errored,
        "rendering"_s <= "errored"_s + event<EventRenderRuntime> [valid_render] / begin_render_from_errored,
        "render_publish_error"_s <= "errored"_s + event<EventRenderRuntime> [invalid_render] / reject_render_from_errored,
        "flushing"_s <= "errored"_s + event<EventFlushRuntime> [valid_flush] / begin_flush_from_errored,
        "flush_publish_error"_s <= "errored"_s + event<EventFlushRuntime> [invalid_flush] / reject_flush_from_errored,
        "initializing"_s <= "unexpected"_s + event<EventInitializeRuntime> [valid_initialize] / begin_initialize_from_unexpected,
        "unexpected"_s <= "unexpected"_s + event<EventInitializeRuntime> [invalid_initialize] / reject_initialize_from_unexpected,
        "rendering"_s <= "unexpected"_s + event<EventRenderRuntime> [valid_render] / begin_render_from_unexpected,
        "unexpected"_s <= "unexpected"_s + event<EventRenderRuntime> [invalid_render] / reject_render_from_unexpected,
        "flushing"_s <= "unexpected"_s + event<EventFlushRuntime> [valid_flush] / begin_flush_from_unexpected,
        "unexpected"_s <= "unexpected"_s + event<EventFlushRuntime> [invalid_flush] / reject_flush_from_unexpected,
        "initialize_publish_success"_s <= "initialization_decision"_s + completion<EventInitializeRuntime> [initialize_dispatch_ok] / commit_initialize_success,
        "initialize_publish_error"_s <= "initialization_decision"_s + completion<EventInitializeRuntime> [initialize_dispatch_backend_failure] / set_backend_error_event_initialize_runtime,
        "initialize_publish_error"_s <= "initialization_decision"_s + completion<EventInitializeRuntime> [initialize_dispatch_reported_error] / set_error_from_detokenizer_event_initialize_runtime,
        "initialize_publish_error"_s <= "initialization_decision"_s + completion<EventInitializeRuntime> / set_error_from_detokenizer_event_initialize_runtime,
        "initialized"_s <= "initialize_publish_success"_s + completion<EventInitializeRuntime> / publish_initialize_done,
        "errored"_s <= "initialize_publish_error"_s + completion<EventInitializeRuntime> / publish_initialize_error,
        "initialization_decision"_s <= "initializing"_s + completion<EventInitializeRuntime> / dispatch_initialize_detokenizer,
        "render_publish_success"_s <= "rendering"_s + completion<EventRenderRuntime> [sequence_stop_matched] / render_sequence_already_stopped,
        "render_dispatch_decision"_s <= "rendering"_s + completion<EventRenderRuntime> [sequence_running] / dispatch_render_detokenizer,
        "render_result_decision"_s <= "render_dispatch_decision"_s + completion<EventRenderRuntime> [render_dispatch_ok],
        "render_publish_error"_s <= "render_dispatch_decision"_s + completion<EventRenderRuntime> [render_dispatch_backend_failure] / set_backend_error_event_render_runtime,
        "render_publish_error"_s <= "render_dispatch_decision"_s + completion<EventRenderRuntime> [render_dispatch_reported_error] / set_error_from_detokenizer_event_render_runtime,
        "render_publish_error"_s <= "render_dispatch_decision"_s + completion<EventRenderRuntime> [render_dispatch_lengths_invalid] / set_invalid_request_event_render_runtime,
        "render_publish_error"_s <= "render_dispatch_decision"_s + completion<EventRenderRuntime> / ensure_last_error_from_render_dispatch_decision,
        "render_commit_output_exec"_s <= "render_result_decision"_s + completion<EventRenderRuntime>,
        "render_strip_decision"_s <= "render_commit_output_exec"_s + completion<EventRenderRuntime> / commit_render_detokenizer_output,
        "render_strip_prefix_scan_exec"_s <= "render_strip_decision"_s + completion<EventRenderRuntime> [strip_needed],
        "render_strip_state_exec"_s <= "render_strip_decision"_s + completion<EventRenderRuntime> [strip_not_needed],
        "render_publish_error"_s <= "render_strip_decision"_s + completion<EventRenderRuntime> / ensure_last_error_from_render_strip_decision,
        "render_strip_prefix_decision"_s <= "render_strip_prefix_scan_exec"_s + completion<EventRenderRuntime> / compute_render_leading_space_prefix,
        "render_strip_apply_exec"_s <= "render_strip_prefix_decision"_s + completion<EventRenderRuntime> [strip_prefix_nonzero] / apply_render_leading_space_strip,
        "render_strip_state_exec"_s <= "render_strip_prefix_decision"_s + completion<EventRenderRuntime> [strip_prefix_zero],
        "render_publish_error"_s <= "render_strip_prefix_decision"_s + completion<EventRenderRuntime> / ensure_last_error_from_render_strip_prefix_decision,
        "render_strip_state_exec"_s <= "render_strip_apply_exec"_s + completion<EventRenderRuntime>,
        "render_stop_match_exec"_s <= "render_strip_state_exec"_s + completion<EventRenderRuntime> / update_render_strip_state,
        "render_finalize_decision"_s <= "render_stop_match_exec"_s + completion<EventRenderRuntime> / apply_render_stop_matching,
        "render_publish_success"_s <= "render_finalize_decision"_s + completion<EventRenderRuntime> [request_ok] / mark_done,
        "render_publish_error"_s <= "render_finalize_decision"_s + completion<EventRenderRuntime> [request_failed] / ensure_last_error_from_render_finalize_decision,
        "render_publish_error"_s <= "render_finalize_decision"_s + completion<EventRenderRuntime> / ensure_last_error_from_render_finalize_decision,
        "done"_s <= "render_publish_success"_s + completion<EventRenderRuntime> / publish_render_done,
        "errored"_s <= "render_publish_error"_s + completion<EventRenderRuntime> / publish_render_error,
        "flush_publish_success"_s <= "flushing"_s + completion<EventFlushRuntime> [flush_output_fits] / flush_copy_sequence_buffers,
        "flush_publish_error"_s <= "flushing"_s + completion<EventFlushRuntime> [flush_output_too_large] / set_invalid_request_event_flush_runtime,
        "done"_s <= "flush_publish_success"_s + completion<EventFlushRuntime> / publish_flush_done,
        "errored"_s <= "flush_publish_error"_s + completion<EventFlushRuntime> / publish_flush_error,
        "unexpected"_s <= "uninitialized"_s + unexpected_event<_> / on_unexpected_from_uninitialized,
        "unexpected"_s <= "initializing"_s + unexpected_event<_> / on_unexpected_from_initializing,
        "unexpected"_s <= "initialization_decision"_s + unexpected_event<_> / on_unexpected_from_initialization_decision,
        "unexpected"_s <= "initialize_publish_success"_s + unexpected_event<_> / on_unexpected_from_initialize_publish_success,
        "unexpected"_s <= "initialize_publish_error"_s + unexpected_event<_> / on_unexpected_from_initialize_publish_error,
        "unexpected"_s <= "initialized"_s + unexpected_event<_> / on_unexpected_from_initialized,
        "unexpected"_s <= "rendering"_s + unexpected_event<_> / on_unexpected_from_rendering,
        "unexpected"_s <= "render_dispatch_decision"_s + unexpected_event<_> / on_unexpected_from_render_dispatch_decision,
        "unexpected"_s <= "render_result_decision"_s + unexpected_event<_> / on_unexpected_from_render_result_decision,
        "unexpected"_s <= "render_commit_output_exec"_s + unexpected_event<_> / on_unexpected_from_render_commit_output_exec,
        "unexpected"_s <= "render_strip_decision"_s + unexpected_event<_> / on_unexpected_from_render_strip_decision,
        "unexpected"_s <= "render_strip_prefix_scan_exec"_s + unexpected_event<_> / on_unexpected_from_render_strip_prefix_scan_exec,
        "unexpected"_s <= "render_strip_prefix_decision"_s + unexpected_event<_> / on_unexpected_from_render_strip_prefix_decision,
        "unexpected"_s <= "render_strip_apply_exec"_s + unexpected_event<_> / on_unexpected_from_render_strip_apply_exec,
        "unexpected"_s <= "render_strip_state_exec"_s + unexpected_event<_> / on_unexpected_from_render_strip_state_exec,
        "unexpected"_s <= "render_stop_match_exec"_s + unexpected_event<_> / on_unexpected_from_render_stop_match_exec,
        "unexpected"_s <= "render_finalize_decision"_s + unexpected_event<_> / on_unexpected_from_render_finalize_decision,
        "unexpected"_s <= "render_publish_success"_s + unexpected_event<_> / on_unexpected_from_render_publish_success,
        "unexpected"_s <= "render_publish_error"_s + unexpected_event<_> / on_unexpected_from_render_publish_error,
        "unexpected"_s <= "flushing"_s + unexpected_event<_> / on_unexpected_from_flushing,
        "unexpected"_s <= "flush_publish_success"_s + unexpected_event<_> / on_unexpected_from_flush_publish_success,
        "unexpected"_s <= "flush_publish_error"_s + unexpected_event<_> / on_unexpected_from_flush_publish_error,
        "unexpected"_s <= "done"_s + unexpected_event<_> / on_unexpected_from_done,
        "unexpected"_s <= "errored"_s + unexpected_event<_> / on_unexpected_from_errored,
        "unexpected"_s <= "unexpected"_s + unexpected_event<_> / on_unexpected_from_unexpected,
    }
}

/// Context for `TextRenderer` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct TextRendererContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl TextRendererStateMachineContext for TextRendererContext {
    fn apply_render_leading_space_strip(&mut self, _event: &EventRenderRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::apply_render_leading_space_strip
        todo!(
            "TODO: port action `apply_render_leading_space_strip` from emel.cpp/src/emel/text/renderer/actions.hpp"
        )
    }
    fn apply_render_stop_matching(&mut self, _event: &EventRenderRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::apply_render_stop_matching
        todo!(
            "TODO: port action `apply_render_stop_matching` from emel.cpp/src/emel/text/renderer/actions.hpp"
        )
    }
    fn begin_flush_from_done(&mut self, _event: &EventFlushRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::begin_flush
        todo!("TODO: port action `begin_flush` from emel.cpp/src/emel/text/renderer/actions.hpp")
    }
    fn begin_flush_from_errored(&mut self, _event: &EventFlushRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::begin_flush
        todo!("TODO: port action `begin_flush` from emel.cpp/src/emel/text/renderer/actions.hpp")
    }
    fn begin_flush_from_initialized(&mut self, _event: &EventFlushRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::begin_flush
        todo!("TODO: port action `begin_flush` from emel.cpp/src/emel/text/renderer/actions.hpp")
    }
    fn begin_flush_from_unexpected(&mut self, _event: &EventFlushRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::begin_flush
        todo!("TODO: port action `begin_flush` from emel.cpp/src/emel/text/renderer/actions.hpp")
    }
    fn begin_initialize_from_done(&mut self, _event: &EventInitializeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::begin_initialize
        todo!(
            "TODO: port action `begin_initialize` from emel.cpp/src/emel/text/renderer/actions.hpp"
        )
    }
    fn begin_initialize_from_errored(&mut self, _event: &EventInitializeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::begin_initialize
        todo!(
            "TODO: port action `begin_initialize` from emel.cpp/src/emel/text/renderer/actions.hpp"
        )
    }
    fn begin_initialize_from_initialized(
        &mut self,
        _event: &EventInitializeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::begin_initialize
        todo!(
            "TODO: port action `begin_initialize` from emel.cpp/src/emel/text/renderer/actions.hpp"
        )
    }
    fn begin_initialize_from_unexpected(
        &mut self,
        _event: &EventInitializeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::begin_initialize
        todo!(
            "TODO: port action `begin_initialize` from emel.cpp/src/emel/text/renderer/actions.hpp"
        )
    }
    fn begin_initialize_from_uninitialized(
        &mut self,
        _event: &EventInitializeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::begin_initialize
        todo!(
            "TODO: port action `begin_initialize` from emel.cpp/src/emel/text/renderer/actions.hpp"
        )
    }
    fn begin_render_from_done(&mut self, _event: &EventRenderRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::begin_render
        todo!("TODO: port action `begin_render` from emel.cpp/src/emel/text/renderer/actions.hpp")
    }
    fn begin_render_from_errored(&mut self, _event: &EventRenderRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::begin_render
        todo!("TODO: port action `begin_render` from emel.cpp/src/emel/text/renderer/actions.hpp")
    }
    fn begin_render_from_initialized(&mut self, _event: &EventRenderRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::begin_render
        todo!("TODO: port action `begin_render` from emel.cpp/src/emel/text/renderer/actions.hpp")
    }
    fn begin_render_from_unexpected(&mut self, _event: &EventRenderRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::begin_render
        todo!("TODO: port action `begin_render` from emel.cpp/src/emel/text/renderer/actions.hpp")
    }
    fn commit_initialize_success(&mut self, _event: &EventInitializeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::commit_initialize_success
        todo!(
            "TODO: port action `commit_initialize_success` from emel.cpp/src/emel/text/renderer/actions.hpp"
        )
    }
    fn commit_render_detokenizer_output(&mut self, _event: &EventRenderRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::commit_render_detokenizer_output
        todo!(
            "TODO: port action `commit_render_detokenizer_output` from emel.cpp/src/emel/text/renderer/actions.hpp"
        )
    }
    fn compute_render_leading_space_prefix(
        &mut self,
        _event: &EventRenderRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::compute_render_leading_space_prefix
        todo!(
            "TODO: port action `compute_render_leading_space_prefix` from emel.cpp/src/emel/text/renderer/actions.hpp"
        )
    }
    fn dispatch_initialize_detokenizer(
        &mut self,
        _event: &EventInitializeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::dispatch_initialize_detokenizer
        todo!(
            "TODO: port action `dispatch_initialize_detokenizer` from emel.cpp/src/emel/text/renderer/actions.hpp"
        )
    }
    fn dispatch_render_detokenizer(&mut self, _event: &EventRenderRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::dispatch_render_detokenizer
        todo!(
            "TODO: port action `dispatch_render_detokenizer` from emel.cpp/src/emel/text/renderer/actions.hpp"
        )
    }
    fn ensure_last_error_from_render_dispatch_decision(
        &mut self,
        _event: &EventRenderRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::ensure_last_error
        todo!(
            "TODO: port action `ensure_last_error` from emel.cpp/src/emel/text/renderer/actions.hpp"
        )
    }
    fn ensure_last_error_from_render_finalize_decision(
        &mut self,
        _event: &EventRenderRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::ensure_last_error
        todo!(
            "TODO: port action `ensure_last_error` from emel.cpp/src/emel/text/renderer/actions.hpp"
        )
    }
    fn ensure_last_error_from_render_strip_decision(
        &mut self,
        _event: &EventRenderRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::ensure_last_error
        todo!(
            "TODO: port action `ensure_last_error` from emel.cpp/src/emel/text/renderer/actions.hpp"
        )
    }
    fn ensure_last_error_from_render_strip_prefix_decision(
        &mut self,
        _event: &EventRenderRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::ensure_last_error
        todo!(
            "TODO: port action `ensure_last_error` from emel.cpp/src/emel/text/renderer/actions.hpp"
        )
    }
    fn flush_copy_sequence_buffers(&mut self, _event: &EventFlushRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::flush_copy_sequence_buffers
        todo!(
            "TODO: port action `flush_copy_sequence_buffers` from emel.cpp/src/emel/text/renderer/actions.hpp"
        )
    }
    fn flush_output_fits(&self, _event: &EventFlushRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/guards.hpp::flush_output_fits
        todo!(
            "TODO: port guard `flush_output_fits` from emel.cpp/src/emel/text/renderer/guards.hpp"
        )
    }
    fn flush_output_too_large(&self, _event: &EventFlushRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/guards.hpp::flush_output_too_large
        todo!(
            "TODO: port guard `flush_output_too_large` from emel.cpp/src/emel/text/renderer/guards.hpp"
        )
    }
    fn initialize_dispatch_backend_failure(
        &self,
        _event: &EventInitializeRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/guards.hpp::initialize_dispatch_backend_failure
        todo!(
            "TODO: port guard `initialize_dispatch_backend_failure` from emel.cpp/src/emel/text/renderer/guards.hpp"
        )
    }
    fn initialize_dispatch_ok(&self, _event: &EventInitializeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/guards.hpp::initialize_dispatch_ok
        todo!(
            "TODO: port guard `initialize_dispatch_ok` from emel.cpp/src/emel/text/renderer/guards.hpp"
        )
    }
    fn initialize_dispatch_reported_error(
        &self,
        _event: &EventInitializeRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/guards.hpp::initialize_dispatch_reported_error
        todo!(
            "TODO: port guard `initialize_dispatch_reported_error` from emel.cpp/src/emel/text/renderer/guards.hpp"
        )
    }
    fn invalid_flush(&self, _event: &EventFlushRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/guards.hpp::invalid_flush
        todo!("TODO: port guard `invalid_flush` from emel.cpp/src/emel/text/renderer/guards.hpp")
    }
    fn invalid_initialize(&self, _event: &EventInitializeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/guards.hpp::invalid_initialize
        todo!(
            "TODO: port guard `invalid_initialize` from emel.cpp/src/emel/text/renderer/guards.hpp"
        )
    }
    fn invalid_render(&self, _event: &EventRenderRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/guards.hpp::invalid_render
        todo!("TODO: port guard `invalid_render` from emel.cpp/src/emel/text/renderer/guards.hpp")
    }
    fn mark_done(&mut self, _event: &EventRenderRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::mark_done
        todo!("TODO: port action `mark_done` from emel.cpp/src/emel/text/renderer/actions.hpp")
    }
    fn on_unexpected_from_done(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/renderer/actions.hpp")
    }
    fn on_unexpected_from_errored(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/renderer/actions.hpp")
    }
    fn on_unexpected_from_flush_publish_error(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/renderer/actions.hpp")
    }
    fn on_unexpected_from_flush_publish_success(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/renderer/actions.hpp")
    }
    fn on_unexpected_from_flushing(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/renderer/actions.hpp")
    }
    fn on_unexpected_from_initialization_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/renderer/actions.hpp")
    }
    fn on_unexpected_from_initialize_publish_error(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/renderer/actions.hpp")
    }
    fn on_unexpected_from_initialize_publish_success(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/renderer/actions.hpp")
    }
    fn on_unexpected_from_initialized(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/renderer/actions.hpp")
    }
    fn on_unexpected_from_initializing(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/renderer/actions.hpp")
    }
    fn on_unexpected_from_render_commit_output_exec(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/renderer/actions.hpp")
    }
    fn on_unexpected_from_render_dispatch_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/renderer/actions.hpp")
    }
    fn on_unexpected_from_render_finalize_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/renderer/actions.hpp")
    }
    fn on_unexpected_from_render_publish_error(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/renderer/actions.hpp")
    }
    fn on_unexpected_from_render_publish_success(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/renderer/actions.hpp")
    }
    fn on_unexpected_from_render_result_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/renderer/actions.hpp")
    }
    fn on_unexpected_from_render_stop_match_exec(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/renderer/actions.hpp")
    }
    fn on_unexpected_from_render_strip_apply_exec(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/renderer/actions.hpp")
    }
    fn on_unexpected_from_render_strip_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/renderer/actions.hpp")
    }
    fn on_unexpected_from_render_strip_prefix_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/renderer/actions.hpp")
    }
    fn on_unexpected_from_render_strip_prefix_scan_exec(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/renderer/actions.hpp")
    }
    fn on_unexpected_from_render_strip_state_exec(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/renderer/actions.hpp")
    }
    fn on_unexpected_from_rendering(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/renderer/actions.hpp")
    }
    fn on_unexpected_from_unexpected(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/renderer/actions.hpp")
    }
    fn on_unexpected_from_uninitialized(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/renderer/actions.hpp")
    }
    fn publish_flush_done(&mut self, _event: &EventFlushRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::publish_flush_done
        todo!(
            "TODO: port action `publish_flush_done` from emel.cpp/src/emel/text/renderer/actions.hpp"
        )
    }
    fn publish_flush_error(&mut self, _event: &EventFlushRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::publish_flush_error
        todo!(
            "TODO: port action `publish_flush_error` from emel.cpp/src/emel/text/renderer/actions.hpp"
        )
    }
    fn publish_initialize_done(&mut self, _event: &EventInitializeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::publish_initialize_done
        todo!(
            "TODO: port action `publish_initialize_done` from emel.cpp/src/emel/text/renderer/actions.hpp"
        )
    }
    fn publish_initialize_error(&mut self, _event: &EventInitializeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::publish_initialize_error
        todo!(
            "TODO: port action `publish_initialize_error` from emel.cpp/src/emel/text/renderer/actions.hpp"
        )
    }
    fn publish_render_done(&mut self, _event: &EventRenderRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::publish_render_done
        todo!(
            "TODO: port action `publish_render_done` from emel.cpp/src/emel/text/renderer/actions.hpp"
        )
    }
    fn publish_render_error(&mut self, _event: &EventRenderRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::publish_render_error
        todo!(
            "TODO: port action `publish_render_error` from emel.cpp/src/emel/text/renderer/actions.hpp"
        )
    }
    fn reject_flush_from_done(&mut self, _event: &EventFlushRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::reject_flush
        todo!("TODO: port action `reject_flush` from emel.cpp/src/emel/text/renderer/actions.hpp")
    }
    fn reject_flush_from_errored(&mut self, _event: &EventFlushRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::reject_flush
        todo!("TODO: port action `reject_flush` from emel.cpp/src/emel/text/renderer/actions.hpp")
    }
    fn reject_flush_from_initialized(&mut self, _event: &EventFlushRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::reject_flush
        todo!("TODO: port action `reject_flush` from emel.cpp/src/emel/text/renderer/actions.hpp")
    }
    fn reject_flush_from_unexpected(&mut self, _event: &EventFlushRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::reject_flush
        todo!("TODO: port action `reject_flush` from emel.cpp/src/emel/text/renderer/actions.hpp")
    }
    fn reject_flush_from_uninitialized(&mut self, _event: &EventFlushRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::reject_flush
        todo!("TODO: port action `reject_flush` from emel.cpp/src/emel/text/renderer/actions.hpp")
    }
    fn reject_initialize_from_done(&mut self, _event: &EventInitializeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::reject_initialize
        todo!(
            "TODO: port action `reject_initialize` from emel.cpp/src/emel/text/renderer/actions.hpp"
        )
    }
    fn reject_initialize_from_errored(
        &mut self,
        _event: &EventInitializeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::reject_initialize
        todo!(
            "TODO: port action `reject_initialize` from emel.cpp/src/emel/text/renderer/actions.hpp"
        )
    }
    fn reject_initialize_from_initialized(
        &mut self,
        _event: &EventInitializeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::reject_initialize
        todo!(
            "TODO: port action `reject_initialize` from emel.cpp/src/emel/text/renderer/actions.hpp"
        )
    }
    fn reject_initialize_from_unexpected(
        &mut self,
        _event: &EventInitializeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::reject_initialize
        todo!(
            "TODO: port action `reject_initialize` from emel.cpp/src/emel/text/renderer/actions.hpp"
        )
    }
    fn reject_initialize_from_uninitialized(
        &mut self,
        _event: &EventInitializeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::reject_initialize
        todo!(
            "TODO: port action `reject_initialize` from emel.cpp/src/emel/text/renderer/actions.hpp"
        )
    }
    fn reject_render_from_done(&mut self, _event: &EventRenderRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::reject_render
        todo!("TODO: port action `reject_render` from emel.cpp/src/emel/text/renderer/actions.hpp")
    }
    fn reject_render_from_errored(&mut self, _event: &EventRenderRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::reject_render
        todo!("TODO: port action `reject_render` from emel.cpp/src/emel/text/renderer/actions.hpp")
    }
    fn reject_render_from_initialized(&mut self, _event: &EventRenderRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::reject_render
        todo!("TODO: port action `reject_render` from emel.cpp/src/emel/text/renderer/actions.hpp")
    }
    fn reject_render_from_unexpected(&mut self, _event: &EventRenderRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::reject_render
        todo!("TODO: port action `reject_render` from emel.cpp/src/emel/text/renderer/actions.hpp")
    }
    fn reject_render_from_uninitialized(&mut self, _event: &EventRenderRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::reject_render
        todo!("TODO: port action `reject_render` from emel.cpp/src/emel/text/renderer/actions.hpp")
    }
    fn render_dispatch_backend_failure(&self, _event: &EventRenderRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/guards.hpp::render_dispatch_backend_failure
        todo!(
            "TODO: port guard `render_dispatch_backend_failure` from emel.cpp/src/emel/text/renderer/guards.hpp"
        )
    }
    fn render_dispatch_lengths_invalid(&self, _event: &EventRenderRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/guards.hpp::render_dispatch_lengths_invalid
        todo!(
            "TODO: port guard `render_dispatch_lengths_invalid` from emel.cpp/src/emel/text/renderer/guards.hpp"
        )
    }
    fn render_dispatch_ok(&self, _event: &EventRenderRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/guards.hpp::render_dispatch_ok
        todo!(
            "TODO: port guard `render_dispatch_ok` from emel.cpp/src/emel/text/renderer/guards.hpp"
        )
    }
    fn render_dispatch_reported_error(&self, _event: &EventRenderRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/guards.hpp::render_dispatch_reported_error
        todo!(
            "TODO: port guard `render_dispatch_reported_error` from emel.cpp/src/emel/text/renderer/guards.hpp"
        )
    }
    fn render_sequence_already_stopped(&mut self, _event: &EventRenderRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::render_sequence_already_stopped
        todo!(
            "TODO: port action `render_sequence_already_stopped` from emel.cpp/src/emel/text/renderer/actions.hpp"
        )
    }
    fn request_failed(&self, _event: &EventRenderRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/guards.hpp::request_failed
        todo!("TODO: port guard `request_failed` from emel.cpp/src/emel/text/renderer/guards.hpp")
    }
    fn request_ok(&self, _event: &EventRenderRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/guards.hpp::request_ok
        todo!("TODO: port guard `request_ok` from emel.cpp/src/emel/text/renderer/guards.hpp")
    }
    fn sequence_running(&self, _event: &EventRenderRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/guards.hpp::sequence_running
        todo!("TODO: port guard `sequence_running` from emel.cpp/src/emel/text/renderer/guards.hpp")
    }
    fn sequence_stop_matched(&self, _event: &EventRenderRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/guards.hpp::sequence_stop_matched
        todo!(
            "TODO: port guard `sequence_stop_matched` from emel.cpp/src/emel/text/renderer/guards.hpp"
        )
    }
    fn set_backend_error_event_initialize_runtime(
        &mut self,
        _event: &EventInitializeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::set_backend_error
        todo!(
            "TODO: port action `set_backend_error` from emel.cpp/src/emel/text/renderer/actions.hpp"
        )
    }
    fn set_backend_error_event_render_runtime(
        &mut self,
        _event: &EventRenderRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::set_backend_error
        todo!(
            "TODO: port action `set_backend_error` from emel.cpp/src/emel/text/renderer/actions.hpp"
        )
    }
    fn set_error_from_detokenizer_event_initialize_runtime(
        &mut self,
        _event: &EventInitializeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::set_error_from_detokenizer
        todo!(
            "TODO: port action `set_error_from_detokenizer` from emel.cpp/src/emel/text/renderer/actions.hpp"
        )
    }
    fn set_error_from_detokenizer_event_render_runtime(
        &mut self,
        _event: &EventRenderRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::set_error_from_detokenizer
        todo!(
            "TODO: port action `set_error_from_detokenizer` from emel.cpp/src/emel/text/renderer/actions.hpp"
        )
    }
    fn set_invalid_request_event_flush_runtime(
        &mut self,
        _event: &EventFlushRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::set_invalid_request
        todo!(
            "TODO: port action `set_invalid_request` from emel.cpp/src/emel/text/renderer/actions.hpp"
        )
    }
    fn set_invalid_request_event_render_runtime(
        &mut self,
        _event: &EventRenderRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::set_invalid_request
        todo!(
            "TODO: port action `set_invalid_request` from emel.cpp/src/emel/text/renderer/actions.hpp"
        )
    }
    fn strip_needed(&self, _event: &EventRenderRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/guards.hpp::strip_needed
        todo!("TODO: port guard `strip_needed` from emel.cpp/src/emel/text/renderer/guards.hpp")
    }
    fn strip_not_needed(&self, _event: &EventRenderRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/guards.hpp::strip_not_needed
        todo!("TODO: port guard `strip_not_needed` from emel.cpp/src/emel/text/renderer/guards.hpp")
    }
    fn strip_prefix_nonzero(&self, _event: &EventRenderRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/guards.hpp::strip_prefix_nonzero
        todo!(
            "TODO: port guard `strip_prefix_nonzero` from emel.cpp/src/emel/text/renderer/guards.hpp"
        )
    }
    fn strip_prefix_zero(&self, _event: &EventRenderRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/guards.hpp::strip_prefix_zero
        todo!(
            "TODO: port guard `strip_prefix_zero` from emel.cpp/src/emel/text/renderer/guards.hpp"
        )
    }
    fn update_render_strip_state(&mut self, _event: &EventRenderRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/actions.hpp::update_render_strip_state
        todo!(
            "TODO: port action `update_render_strip_state` from emel.cpp/src/emel/text/renderer/actions.hpp"
        )
    }
    fn valid_flush(&self, _event: &EventFlushRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/guards.hpp::valid_flush
        todo!("TODO: port guard `valid_flush` from emel.cpp/src/emel/text/renderer/guards.hpp")
    }
    fn valid_initialize(&self, _event: &EventInitializeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/guards.hpp::valid_initialize
        todo!("TODO: port guard `valid_initialize` from emel.cpp/src/emel/text/renderer/guards.hpp")
    }
    fn valid_render(&self, _event: &EventRenderRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/renderer/guards.hpp::valid_render
        todo!("TODO: port guard `valid_render` from emel.cpp/src/emel/text/renderer/guards.hpp")
    }
}
