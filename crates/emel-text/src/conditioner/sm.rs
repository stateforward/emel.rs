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

// --- machine TextConditioner from emel.cpp/src/emel/text/conditioner/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventBindRuntime;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventPrepareRuntime;

sml! {
    TextConditioner {
        "binding"_s <= *"uninitialized"_s + event<EventBindRuntime> [valid_bind] / begin_bind_from_uninitialized,
        "bind_error"_s <= "uninitialized"_s + event<EventBindRuntime> [invalid_bind] / reject_bind_from_uninitialized,
        "prepare_error"_s <= "uninitialized"_s + event<EventPrepareRuntime> / reject_prepare_from_uninitialized,
        "binding"_s <= "idle"_s + event<EventBindRuntime> [valid_bind] / begin_bind_from_idle,
        "bind_error"_s <= "idle"_s + event<EventBindRuntime> [invalid_bind] / reject_bind_from_idle,
        "preparing"_s <= "idle"_s + event<EventPrepareRuntime> [valid_prepare_with_bind_defaults] / begin_prepare_bind_defaults_from_idle,
        "preparing"_s <= "idle"_s + event<EventPrepareRuntime> [valid_prepare_with_request_overrides] / begin_prepare_from_request_from_idle,
        "prepare_error"_s <= "idle"_s + event<EventPrepareRuntime> [invalid_prepare] / reject_prepare_from_idle,
        "binding"_s <= "done"_s + event<EventBindRuntime> [valid_bind] / begin_bind_from_done,
        "bind_error"_s <= "done"_s + event<EventBindRuntime> [invalid_bind] / reject_bind_from_done,
        "preparing"_s <= "done"_s + event<EventPrepareRuntime> [valid_prepare_with_bind_defaults] / begin_prepare_bind_defaults_from_done,
        "preparing"_s <= "done"_s + event<EventPrepareRuntime> [valid_prepare_with_request_overrides] / begin_prepare_from_request_from_done,
        "prepare_error"_s <= "done"_s + event<EventPrepareRuntime> [invalid_prepare] / reject_prepare_from_done,
        "binding"_s <= "errored"_s + event<EventBindRuntime> [valid_bind] / begin_bind_from_errored,
        "bind_error"_s <= "errored"_s + event<EventBindRuntime> [invalid_bind] / reject_bind_from_errored,
        "preparing"_s <= "errored"_s + event<EventPrepareRuntime> [valid_prepare_with_bind_defaults] / begin_prepare_bind_defaults_from_errored,
        "preparing"_s <= "errored"_s + event<EventPrepareRuntime> [valid_prepare_with_request_overrides] / begin_prepare_from_request_from_errored,
        "prepare_error"_s <= "errored"_s + event<EventPrepareRuntime> [invalid_prepare] / reject_prepare_from_errored,
        "binding"_s <= "unexpected"_s + event<EventBindRuntime> [valid_bind] / begin_bind_from_unexpected,
        "bind_error"_s <= "unexpected"_s + event<EventBindRuntime> [invalid_bind] / reject_bind_from_unexpected,
        "preparing"_s <= "unexpected"_s + event<EventPrepareRuntime> [valid_prepare_with_bind_defaults] / begin_prepare_bind_defaults_from_unexpected,
        "preparing"_s <= "unexpected"_s + event<EventPrepareRuntime> [valid_prepare_with_request_overrides] / begin_prepare_from_request_from_unexpected,
        "prepare_error"_s <= "unexpected"_s + event<EventPrepareRuntime> [invalid_prepare] / reject_prepare_from_unexpected,
        "bind_decision"_s <= "binding"_s + completion<EventBindRuntime> / dispatch_bind_tokenizer,
        "bind_error"_s <= "bind_decision"_s + completion<EventBindRuntime> [bind_rejected_no_error] / bind_error_backend,
        "bind_error"_s <= "bind_decision"_s + completion<EventBindRuntime> [bind_error_invalid_argument_code] / set_error_invalid_argument_event_bind_runtime,
        "bind_error"_s <= "bind_decision"_s + completion<EventBindRuntime> [bind_error_model_invalid_code] / set_error_model_invalid_event_bind_runtime,
        "bind_error"_s <= "bind_decision"_s + completion<EventBindRuntime> [bind_error_capacity_code] / set_error_capacity_event_bind_runtime,
        "bind_error"_s <= "bind_decision"_s + completion<EventBindRuntime> [bind_error_backend_code] / set_error_backend_event_bind_runtime,
        "bind_error"_s <= "bind_decision"_s + completion<EventBindRuntime> [bind_error_untracked_code] / set_error_untracked_event_bind_runtime,
        "bind_success"_s <= "bind_decision"_s + completion<EventBindRuntime> [bind_successful] / bind_success,
        "bind_publish_success"_s <= "bind_success"_s + completion<EventBindRuntime> [has_bind_error_out] / write_bind_error_out_from_bind_success,
        "bind_publish_success"_s <= "bind_success"_s + completion<EventBindRuntime> [no_bind_error_out],
        "idle"_s <= "bind_publish_success"_s + completion<EventBindRuntime> [has_bind_done_callback] / emit_bind_done,
        "idle"_s <= "bind_publish_success"_s + completion<EventBindRuntime> [no_bind_done_callback],
        "bind_publish_error"_s <= "bind_error"_s + completion<EventBindRuntime> [has_bind_error_out] / write_bind_error_out_from_bind_error,
        "bind_publish_error"_s <= "bind_error"_s + completion<EventBindRuntime> [no_bind_error_out],
        "errored"_s <= "bind_publish_error"_s + completion<EventBindRuntime> [has_bind_error_callback] / emit_bind_error,
        "errored"_s <= "bind_publish_error"_s + completion<EventBindRuntime> [no_bind_error_callback],
        "format_decision"_s <= "preparing"_s + completion<EventPrepareRuntime> / dispatch_format,
        "prepare_error"_s <= "format_decision"_s + completion<EventPrepareRuntime> [format_rejected_no_error] / format_error_backend,
        "prepare_error"_s <= "format_decision"_s + completion<EventPrepareRuntime> [format_error_invalid_argument_code] / set_error_invalid_argument_event_prepare_runtime,
        "prepare_error"_s <= "format_decision"_s + completion<EventPrepareRuntime> [format_error_model_invalid_code] / set_error_model_invalid_event_prepare_runtime,
        "prepare_error"_s <= "format_decision"_s + completion<EventPrepareRuntime> [format_error_capacity_code] / set_error_capacity_event_prepare_runtime,
        "prepare_error"_s <= "format_decision"_s + completion<EventPrepareRuntime> [format_error_backend_code] / set_error_backend_event_prepare_runtime,
        "prepare_error"_s <= "format_decision"_s + completion<EventPrepareRuntime> [format_error_untracked_code] / set_error_untracked_event_prepare_runtime,
        "prepare_error"_s <= "format_decision"_s + completion<EventPrepareRuntime> [format_length_overflow] / format_error_invalid_argument,
        "tokenizing"_s <= "format_decision"_s + completion<EventPrepareRuntime> [format_successful],
        "tokenize_decision"_s <= "tokenizing"_s + completion<EventPrepareRuntime> / dispatch_tokenize,
        "prepare_error"_s <= "tokenize_decision"_s + completion<EventPrepareRuntime> [tokenize_rejected_no_error] / tokenize_error_backend_from_tokenize_decision,
        "prepare_error"_s <= "tokenize_decision"_s + completion<EventPrepareRuntime> [tokenize_error_invalid_argument_code] / set_error_invalid_argument_event_prepare_runtime,
        "prepare_error"_s <= "tokenize_decision"_s + completion<EventPrepareRuntime> [tokenize_error_model_invalid_code] / set_error_model_invalid_event_prepare_runtime,
        "prepare_error"_s <= "tokenize_decision"_s + completion<EventPrepareRuntime> [tokenize_error_capacity_code] / set_error_capacity_event_prepare_runtime,
        "prepare_error"_s <= "tokenize_decision"_s + completion<EventPrepareRuntime> [tokenize_error_backend_code] / set_error_backend_event_prepare_runtime,
        "prepare_error"_s <= "tokenize_decision"_s + completion<EventPrepareRuntime> [tokenize_error_untracked_code] / set_error_untracked_event_prepare_runtime,
        "prepare_error"_s <= "tokenize_decision"_s + completion<EventPrepareRuntime> [tokenize_count_invalid] / tokenize_error_backend_from_tokenize_decision,
        "prepare_success"_s <= "tokenize_decision"_s + completion<EventPrepareRuntime> [tokenize_successful] / prepare_success,
        "prepare_publish_success_count"_s <= "prepare_success"_s + completion<EventPrepareRuntime> / write_prepare_token_count_from_prepare_success,
        "prepare_publish_success_error"_s <= "prepare_publish_success_count"_s + completion<EventPrepareRuntime> / write_prepare_error_out_from_prepare_publish_success_count,
        "done"_s <= "prepare_publish_success_error"_s + completion<EventPrepareRuntime> [has_prepare_done_callback] / emit_prepare_done,
        "done"_s <= "prepare_publish_success_error"_s + completion<EventPrepareRuntime> [no_prepare_done_callback],
        "prepare_publish_error_count"_s <= "prepare_error"_s + completion<EventPrepareRuntime> / write_prepare_token_count_from_prepare_error,
        "prepare_publish_error"_s <= "prepare_publish_error_count"_s + completion<EventPrepareRuntime> / write_prepare_error_out_from_prepare_publish_error_count,
        "errored"_s <= "prepare_publish_error"_s + completion<EventPrepareRuntime> [has_prepare_error_callback] / emit_prepare_error,
        "errored"_s <= "prepare_publish_error"_s + completion<EventPrepareRuntime> [no_prepare_error_callback],
        "unexpected"_s <= "uninitialized"_s + unexpected_event<_> / on_unexpected_from_uninitialized,
        "unexpected"_s <= "binding"_s + unexpected_event<_> / on_unexpected_from_binding,
        "unexpected"_s <= "bind_decision"_s + unexpected_event<_> / on_unexpected_from_bind_decision,
        "unexpected"_s <= "bind_success"_s + unexpected_event<_> / on_unexpected_from_bind_success,
        "unexpected"_s <= "bind_error"_s + unexpected_event<_> / on_unexpected_from_bind_error,
        "unexpected"_s <= "bind_publish_success"_s + unexpected_event<_> / on_unexpected_from_bind_publish_success,
        "unexpected"_s <= "bind_publish_error"_s + unexpected_event<_> / on_unexpected_from_bind_publish_error,
        "unexpected"_s <= "preparing"_s + unexpected_event<_> / on_unexpected_from_preparing,
        "unexpected"_s <= "format_decision"_s + unexpected_event<_> / on_unexpected_from_format_decision,
        "unexpected"_s <= "tokenizing"_s + unexpected_event<_> / on_unexpected_from_tokenizing,
        "unexpected"_s <= "tokenize_decision"_s + unexpected_event<_> / on_unexpected_from_tokenize_decision,
        "unexpected"_s <= "prepare_success"_s + unexpected_event<_> / on_unexpected_from_prepare_success,
        "unexpected"_s <= "prepare_error"_s + unexpected_event<_> / on_unexpected_from_prepare_error,
        "unexpected"_s <= "prepare_publish_success_count"_s + unexpected_event<_> / on_unexpected_from_prepare_publish_success_count,
        "unexpected"_s <= "prepare_publish_success_error"_s + unexpected_event<_> / on_unexpected_from_prepare_publish_success_error,
        "unexpected"_s <= "prepare_publish_error_count"_s + unexpected_event<_> / on_unexpected_from_prepare_publish_error_count,
        "unexpected"_s <= "prepare_publish_error"_s + unexpected_event<_> / on_unexpected_from_prepare_publish_error,
        "unexpected"_s <= "done"_s + unexpected_event<_> / on_unexpected_from_done,
        "unexpected"_s <= "errored"_s + unexpected_event<_> / on_unexpected_from_errored,
        "unexpected"_s <= "idle"_s + unexpected_event<_> / on_unexpected_from_idle,
        "unexpected"_s <= "unexpected"_s + unexpected_event<_> / on_unexpected_from_unexpected,
    }
}

/// Context for `TextConditioner` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct TextConditionerContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl TextConditionerStateMachineContext for TextConditionerContext {
    fn begin_bind_from_done(&mut self, _event: &EventBindRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::begin_bind
        todo!("TODO: port action `begin_bind` from emel.cpp/src/emel/text/conditioner/actions.hpp")
    }
    fn begin_bind_from_errored(&mut self, _event: &EventBindRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::begin_bind
        todo!("TODO: port action `begin_bind` from emel.cpp/src/emel/text/conditioner/actions.hpp")
    }
    fn begin_bind_from_idle(&mut self, _event: &EventBindRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::begin_bind
        todo!("TODO: port action `begin_bind` from emel.cpp/src/emel/text/conditioner/actions.hpp")
    }
    fn begin_bind_from_unexpected(&mut self, _event: &EventBindRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::begin_bind
        todo!("TODO: port action `begin_bind` from emel.cpp/src/emel/text/conditioner/actions.hpp")
    }
    fn begin_bind_from_uninitialized(&mut self, _event: &EventBindRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::begin_bind
        todo!("TODO: port action `begin_bind` from emel.cpp/src/emel/text/conditioner/actions.hpp")
    }
    fn begin_prepare_bind_defaults_from_done(
        &mut self,
        _event: &EventPrepareRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::begin_prepare_bind_defaults
        todo!(
            "TODO: port action `begin_prepare_bind_defaults` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn begin_prepare_bind_defaults_from_errored(
        &mut self,
        _event: &EventPrepareRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::begin_prepare_bind_defaults
        todo!(
            "TODO: port action `begin_prepare_bind_defaults` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn begin_prepare_bind_defaults_from_idle(
        &mut self,
        _event: &EventPrepareRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::begin_prepare_bind_defaults
        todo!(
            "TODO: port action `begin_prepare_bind_defaults` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn begin_prepare_bind_defaults_from_unexpected(
        &mut self,
        _event: &EventPrepareRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::begin_prepare_bind_defaults
        todo!(
            "TODO: port action `begin_prepare_bind_defaults` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn begin_prepare_from_request_from_done(
        &mut self,
        _event: &EventPrepareRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::begin_prepare_from_request
        todo!(
            "TODO: port action `begin_prepare_from_request` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn begin_prepare_from_request_from_errored(
        &mut self,
        _event: &EventPrepareRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::begin_prepare_from_request
        todo!(
            "TODO: port action `begin_prepare_from_request` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn begin_prepare_from_request_from_idle(
        &mut self,
        _event: &EventPrepareRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::begin_prepare_from_request
        todo!(
            "TODO: port action `begin_prepare_from_request` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn begin_prepare_from_request_from_unexpected(
        &mut self,
        _event: &EventPrepareRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::begin_prepare_from_request
        todo!(
            "TODO: port action `begin_prepare_from_request` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn bind_error_backend(&mut self, _event: &EventBindRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::bind_error_backend
        todo!(
            "TODO: port action `bind_error_backend` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn bind_error_backend_code(&self, _event: &EventBindRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/guards.hpp::bind_error_backend_code
        todo!(
            "TODO: port guard `bind_error_backend_code` from emel.cpp/src/emel/text/conditioner/guards.hpp"
        )
    }
    fn bind_error_capacity_code(&self, _event: &EventBindRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/guards.hpp::bind_error_capacity_code
        todo!(
            "TODO: port guard `bind_error_capacity_code` from emel.cpp/src/emel/text/conditioner/guards.hpp"
        )
    }
    fn bind_error_invalid_argument_code(&self, _event: &EventBindRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/guards.hpp::bind_error_invalid_argument_code
        todo!(
            "TODO: port guard `bind_error_invalid_argument_code` from emel.cpp/src/emel/text/conditioner/guards.hpp"
        )
    }
    fn bind_error_model_invalid_code(&self, _event: &EventBindRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/guards.hpp::bind_error_model_invalid_code
        todo!(
            "TODO: port guard `bind_error_model_invalid_code` from emel.cpp/src/emel/text/conditioner/guards.hpp"
        )
    }
    fn bind_error_untracked_code(&self, _event: &EventBindRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/guards.hpp::bind_error_untracked_code
        todo!(
            "TODO: port guard `bind_error_untracked_code` from emel.cpp/src/emel/text/conditioner/guards.hpp"
        )
    }
    fn bind_rejected_no_error(&self, _event: &EventBindRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/guards.hpp::bind_rejected_no_error
        todo!(
            "TODO: port guard `bind_rejected_no_error` from emel.cpp/src/emel/text/conditioner/guards.hpp"
        )
    }
    fn bind_success(&mut self, _event: &EventBindRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::bind_success
        todo!(
            "TODO: port action `bind_success` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn bind_successful(&self, _event: &EventBindRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/guards.hpp::bind_successful
        todo!(
            "TODO: port guard `bind_successful` from emel.cpp/src/emel/text/conditioner/guards.hpp"
        )
    }
    fn dispatch_bind_tokenizer(&mut self, _event: &EventBindRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::dispatch_bind_tokenizer
        todo!(
            "TODO: port action `dispatch_bind_tokenizer` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn dispatch_format(&mut self, _event: &EventPrepareRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::dispatch_format
        todo!(
            "TODO: port action `dispatch_format` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn dispatch_tokenize(&mut self, _event: &EventPrepareRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::dispatch_tokenize
        todo!(
            "TODO: port action `dispatch_tokenize` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn emit_bind_done(&mut self, _event: &EventBindRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::emit_bind_done
        todo!(
            "TODO: port action `emit_bind_done` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn emit_bind_error(&mut self, _event: &EventBindRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::emit_bind_error
        todo!(
            "TODO: port action `emit_bind_error` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn emit_prepare_done(&mut self, _event: &EventPrepareRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::emit_prepare_done
        todo!(
            "TODO: port action `emit_prepare_done` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn emit_prepare_error(&mut self, _event: &EventPrepareRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::emit_prepare_error
        todo!(
            "TODO: port action `emit_prepare_error` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn format_error_backend(&mut self, _event: &EventPrepareRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::format_error_backend
        todo!(
            "TODO: port action `format_error_backend` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn format_error_backend_code(&self, _event: &EventPrepareRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/guards.hpp::format_error_backend_code
        todo!(
            "TODO: port guard `format_error_backend_code` from emel.cpp/src/emel/text/conditioner/guards.hpp"
        )
    }
    fn format_error_capacity_code(&self, _event: &EventPrepareRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/guards.hpp::format_error_capacity_code
        todo!(
            "TODO: port guard `format_error_capacity_code` from emel.cpp/src/emel/text/conditioner/guards.hpp"
        )
    }
    fn format_error_invalid_argument(&mut self, _event: &EventPrepareRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::format_error_invalid_argument
        todo!(
            "TODO: port action `format_error_invalid_argument` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn format_error_invalid_argument_code(&self, _event: &EventPrepareRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/guards.hpp::format_error_invalid_argument_code
        todo!(
            "TODO: port guard `format_error_invalid_argument_code` from emel.cpp/src/emel/text/conditioner/guards.hpp"
        )
    }
    fn format_error_model_invalid_code(&self, _event: &EventPrepareRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/guards.hpp::format_error_model_invalid_code
        todo!(
            "TODO: port guard `format_error_model_invalid_code` from emel.cpp/src/emel/text/conditioner/guards.hpp"
        )
    }
    fn format_error_untracked_code(&self, _event: &EventPrepareRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/guards.hpp::format_error_untracked_code
        todo!(
            "TODO: port guard `format_error_untracked_code` from emel.cpp/src/emel/text/conditioner/guards.hpp"
        )
    }
    fn format_length_overflow(&self, _event: &EventPrepareRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/guards.hpp::format_length_overflow
        todo!(
            "TODO: port guard `format_length_overflow` from emel.cpp/src/emel/text/conditioner/guards.hpp"
        )
    }
    fn format_rejected_no_error(&self, _event: &EventPrepareRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/guards.hpp::format_rejected_no_error
        todo!(
            "TODO: port guard `format_rejected_no_error` from emel.cpp/src/emel/text/conditioner/guards.hpp"
        )
    }
    fn format_successful(&self, _event: &EventPrepareRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/guards.hpp::format_successful
        todo!(
            "TODO: port guard `format_successful` from emel.cpp/src/emel/text/conditioner/guards.hpp"
        )
    }
    fn has_bind_done_callback(&self, _event: &EventBindRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/guards.hpp::has_bind_done_callback
        todo!(
            "TODO: port guard `has_bind_done_callback` from emel.cpp/src/emel/text/conditioner/guards.hpp"
        )
    }
    fn has_bind_error_callback(&self, _event: &EventBindRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/guards.hpp::has_bind_error_callback
        todo!(
            "TODO: port guard `has_bind_error_callback` from emel.cpp/src/emel/text/conditioner/guards.hpp"
        )
    }
    fn has_bind_error_out(&self, _event: &EventBindRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/guards.hpp::has_bind_error_out
        todo!(
            "TODO: port guard `has_bind_error_out` from emel.cpp/src/emel/text/conditioner/guards.hpp"
        )
    }
    fn has_prepare_done_callback(&self, _event: &EventPrepareRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/guards.hpp::has_prepare_done_callback
        todo!(
            "TODO: port guard `has_prepare_done_callback` from emel.cpp/src/emel/text/conditioner/guards.hpp"
        )
    }
    fn has_prepare_error_callback(&self, _event: &EventPrepareRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/guards.hpp::has_prepare_error_callback
        todo!(
            "TODO: port guard `has_prepare_error_callback` from emel.cpp/src/emel/text/conditioner/guards.hpp"
        )
    }
    fn invalid_bind(&self, _event: &EventBindRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/guards.hpp::invalid_bind
        todo!("TODO: port guard `invalid_bind` from emel.cpp/src/emel/text/conditioner/guards.hpp")
    }
    fn invalid_prepare(&self, _event: &EventPrepareRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/guards.hpp::invalid_prepare
        todo!(
            "TODO: port guard `invalid_prepare` from emel.cpp/src/emel/text/conditioner/guards.hpp"
        )
    }
    fn no_bind_done_callback(&self, _event: &EventBindRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/guards.hpp::no_bind_done_callback
        todo!(
            "TODO: port guard `no_bind_done_callback` from emel.cpp/src/emel/text/conditioner/guards.hpp"
        )
    }
    fn no_bind_error_callback(&self, _event: &EventBindRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/guards.hpp::no_bind_error_callback
        todo!(
            "TODO: port guard `no_bind_error_callback` from emel.cpp/src/emel/text/conditioner/guards.hpp"
        )
    }
    fn no_bind_error_out(&self, _event: &EventBindRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/guards.hpp::no_bind_error_out
        todo!(
            "TODO: port guard `no_bind_error_out` from emel.cpp/src/emel/text/conditioner/guards.hpp"
        )
    }
    fn no_prepare_done_callback(&self, _event: &EventPrepareRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/guards.hpp::no_prepare_done_callback
        todo!(
            "TODO: port guard `no_prepare_done_callback` from emel.cpp/src/emel/text/conditioner/guards.hpp"
        )
    }
    fn no_prepare_error_callback(&self, _event: &EventPrepareRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/guards.hpp::no_prepare_error_callback
        todo!(
            "TODO: port guard `no_prepare_error_callback` from emel.cpp/src/emel/text/conditioner/guards.hpp"
        )
    }
    fn on_unexpected_from_bind_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn on_unexpected_from_bind_error(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn on_unexpected_from_bind_publish_error(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn on_unexpected_from_bind_publish_success(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn on_unexpected_from_bind_success(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn on_unexpected_from_binding(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn on_unexpected_from_done(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn on_unexpected_from_errored(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn on_unexpected_from_format_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn on_unexpected_from_idle(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn on_unexpected_from_prepare_error(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn on_unexpected_from_prepare_publish_error(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn on_unexpected_from_prepare_publish_error_count(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn on_unexpected_from_prepare_publish_success_count(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn on_unexpected_from_prepare_publish_success_error(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn on_unexpected_from_prepare_success(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn on_unexpected_from_preparing(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn on_unexpected_from_tokenize_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn on_unexpected_from_tokenizing(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn on_unexpected_from_unexpected(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn on_unexpected_from_uninitialized(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn prepare_success(&mut self, _event: &EventPrepareRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::prepare_success
        todo!(
            "TODO: port action `prepare_success` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn reject_bind_from_done(&mut self, _event: &EventBindRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::reject_bind
        todo!("TODO: port action `reject_bind` from emel.cpp/src/emel/text/conditioner/actions.hpp")
    }
    fn reject_bind_from_errored(&mut self, _event: &EventBindRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::reject_bind
        todo!("TODO: port action `reject_bind` from emel.cpp/src/emel/text/conditioner/actions.hpp")
    }
    fn reject_bind_from_idle(&mut self, _event: &EventBindRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::reject_bind
        todo!("TODO: port action `reject_bind` from emel.cpp/src/emel/text/conditioner/actions.hpp")
    }
    fn reject_bind_from_unexpected(&mut self, _event: &EventBindRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::reject_bind
        todo!("TODO: port action `reject_bind` from emel.cpp/src/emel/text/conditioner/actions.hpp")
    }
    fn reject_bind_from_uninitialized(&mut self, _event: &EventBindRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::reject_bind
        todo!("TODO: port action `reject_bind` from emel.cpp/src/emel/text/conditioner/actions.hpp")
    }
    fn reject_prepare_from_done(&mut self, _event: &EventPrepareRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::reject_prepare
        todo!(
            "TODO: port action `reject_prepare` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn reject_prepare_from_errored(&mut self, _event: &EventPrepareRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::reject_prepare
        todo!(
            "TODO: port action `reject_prepare` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn reject_prepare_from_idle(&mut self, _event: &EventPrepareRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::reject_prepare
        todo!(
            "TODO: port action `reject_prepare` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn reject_prepare_from_unexpected(&mut self, _event: &EventPrepareRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::reject_prepare
        todo!(
            "TODO: port action `reject_prepare` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn reject_prepare_from_uninitialized(
        &mut self,
        _event: &EventPrepareRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::reject_prepare
        todo!(
            "TODO: port action `reject_prepare` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn set_error_backend_event_bind_runtime(
        &mut self,
        _event: &EventBindRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::set_error_backend
        todo!(
            "TODO: port action `set_error_backend` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn set_error_backend_event_prepare_runtime(
        &mut self,
        _event: &EventPrepareRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::set_error_backend
        todo!(
            "TODO: port action `set_error_backend` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn set_error_capacity_event_bind_runtime(
        &mut self,
        _event: &EventBindRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::set_error_capacity
        todo!(
            "TODO: port action `set_error_capacity` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn set_error_capacity_event_prepare_runtime(
        &mut self,
        _event: &EventPrepareRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::set_error_capacity
        todo!(
            "TODO: port action `set_error_capacity` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn set_error_invalid_argument_event_bind_runtime(
        &mut self,
        _event: &EventBindRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::set_error_invalid_argument
        todo!(
            "TODO: port action `set_error_invalid_argument` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn set_error_invalid_argument_event_prepare_runtime(
        &mut self,
        _event: &EventPrepareRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::set_error_invalid_argument
        todo!(
            "TODO: port action `set_error_invalid_argument` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn set_error_model_invalid_event_bind_runtime(
        &mut self,
        _event: &EventBindRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::set_error_model_invalid
        todo!(
            "TODO: port action `set_error_model_invalid` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn set_error_model_invalid_event_prepare_runtime(
        &mut self,
        _event: &EventPrepareRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::set_error_model_invalid
        todo!(
            "TODO: port action `set_error_model_invalid` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn set_error_untracked_event_bind_runtime(
        &mut self,
        _event: &EventBindRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::set_error_untracked
        todo!(
            "TODO: port action `set_error_untracked` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn set_error_untracked_event_prepare_runtime(
        &mut self,
        _event: &EventPrepareRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::set_error_untracked
        todo!(
            "TODO: port action `set_error_untracked` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn tokenize_count_invalid(&self, _event: &EventPrepareRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/guards.hpp::tokenize_count_invalid
        todo!(
            "TODO: port guard `tokenize_count_invalid` from emel.cpp/src/emel/text/conditioner/guards.hpp"
        )
    }
    fn tokenize_error_backend_code(&self, _event: &EventPrepareRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/guards.hpp::tokenize_error_backend_code
        todo!(
            "TODO: port guard `tokenize_error_backend_code` from emel.cpp/src/emel/text/conditioner/guards.hpp"
        )
    }
    fn tokenize_error_backend_from_tokenize_decision(
        &mut self,
        _event: &EventPrepareRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::tokenize_error_backend
        todo!(
            "TODO: port action `tokenize_error_backend` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn tokenize_error_capacity_code(&self, _event: &EventPrepareRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/guards.hpp::tokenize_error_capacity_code
        todo!(
            "TODO: port guard `tokenize_error_capacity_code` from emel.cpp/src/emel/text/conditioner/guards.hpp"
        )
    }
    fn tokenize_error_invalid_argument_code(
        &self,
        _event: &EventPrepareRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/guards.hpp::tokenize_error_invalid_argument_code
        todo!(
            "TODO: port guard `tokenize_error_invalid_argument_code` from emel.cpp/src/emel/text/conditioner/guards.hpp"
        )
    }
    fn tokenize_error_model_invalid_code(&self, _event: &EventPrepareRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/guards.hpp::tokenize_error_model_invalid_code
        todo!(
            "TODO: port guard `tokenize_error_model_invalid_code` from emel.cpp/src/emel/text/conditioner/guards.hpp"
        )
    }
    fn tokenize_error_untracked_code(&self, _event: &EventPrepareRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/guards.hpp::tokenize_error_untracked_code
        todo!(
            "TODO: port guard `tokenize_error_untracked_code` from emel.cpp/src/emel/text/conditioner/guards.hpp"
        )
    }
    fn tokenize_rejected_no_error(&self, _event: &EventPrepareRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/guards.hpp::tokenize_rejected_no_error
        todo!(
            "TODO: port guard `tokenize_rejected_no_error` from emel.cpp/src/emel/text/conditioner/guards.hpp"
        )
    }
    fn tokenize_successful(&self, _event: &EventPrepareRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/guards.hpp::tokenize_successful
        todo!(
            "TODO: port guard `tokenize_successful` from emel.cpp/src/emel/text/conditioner/guards.hpp"
        )
    }
    fn valid_bind(&self, _event: &EventBindRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/guards.hpp::valid_bind
        todo!("TODO: port guard `valid_bind` from emel.cpp/src/emel/text/conditioner/guards.hpp")
    }
    fn valid_prepare_with_bind_defaults(&self, _event: &EventPrepareRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/guards.hpp::valid_prepare_with_bind_defaults
        todo!(
            "TODO: port guard `valid_prepare_with_bind_defaults` from emel.cpp/src/emel/text/conditioner/guards.hpp"
        )
    }
    fn valid_prepare_with_request_overrides(
        &self,
        _event: &EventPrepareRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/guards.hpp::valid_prepare_with_request_overrides
        todo!(
            "TODO: port guard `valid_prepare_with_request_overrides` from emel.cpp/src/emel/text/conditioner/guards.hpp"
        )
    }
    fn write_bind_error_out_from_bind_error(
        &mut self,
        _event: &EventBindRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::write_bind_error_out
        todo!(
            "TODO: port action `write_bind_error_out` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn write_bind_error_out_from_bind_success(
        &mut self,
        _event: &EventBindRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::write_bind_error_out
        todo!(
            "TODO: port action `write_bind_error_out` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn write_prepare_error_out_from_prepare_publish_error_count(
        &mut self,
        _event: &EventPrepareRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::write_prepare_error_out
        todo!(
            "TODO: port action `write_prepare_error_out` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn write_prepare_error_out_from_prepare_publish_success_count(
        &mut self,
        _event: &EventPrepareRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::write_prepare_error_out
        todo!(
            "TODO: port action `write_prepare_error_out` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn write_prepare_token_count_from_prepare_error(
        &mut self,
        _event: &EventPrepareRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::write_prepare_token_count
        todo!(
            "TODO: port action `write_prepare_token_count` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
    fn write_prepare_token_count_from_prepare_success(
        &mut self,
        _event: &EventPrepareRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/conditioner/actions.hpp::write_prepare_token_count
        todo!(
            "TODO: port action `write_prepare_token_count` from emel.cpp/src/emel/text/conditioner/actions.hpp"
        )
    }
}
