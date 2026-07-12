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

// --- machine TextEncodersFallback from emel.cpp/src/emel/text/encoders/fallback/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventsEncodingDone;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventsEncodingError;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct RuntimeEncodeRuntime;

sml! {
    TextEncodersFallback {
        "encode_validity_decision"_s <= *"initialized"_s + event<RuntimeEncodeRuntime>,
        "encode_validity_decision"_s <= "done"_s + event<RuntimeEncodeRuntime>,
        "encode_validity_decision"_s <= "errored"_s + event<RuntimeEncodeRuntime>,
        "encode_validity_decision"_s <= "unexpected"_s + event<RuntimeEncodeRuntime>,
        "encode_vocab_sync_decision"_s <= "encode_validity_decision"_s + completion<RuntimeEncodeRuntime> [valid_encode],
        "errored"_s <= "encode_validity_decision"_s + completion<RuntimeEncodeRuntime> [invalid_encode] / reject_invalid_encode_from_encode_validity_decision,
        "errored"_s <= "encode_validity_decision"_s + completion<RuntimeEncodeRuntime> / reject_invalid_encode_from_encode_validity_decision,
        "encode_precheck_decision"_s <= "encode_vocab_sync_decision"_s + completion<RuntimeEncodeRuntime> [vocab_changed] / begin_encode_sync_vocab,
        "encode_precheck_decision"_s <= "encode_vocab_sync_decision"_s + completion<RuntimeEncodeRuntime> [vocab_unchanged] / begin_encode,
        "errored"_s <= "encode_vocab_sync_decision"_s + completion<RuntimeEncodeRuntime> / reject_invalid_encode_from_encode_vocab_sync_decision,
        "done"_s <= "encode_precheck_decision"_s + completion<RuntimeEncodeRuntime> [text_empty] / mark_done_from_encode_precheck_decision,
        "encode_table_prepare"_s <= "encode_precheck_decision"_s + completion<RuntimeEncodeRuntime> [text_non_empty] / prepare_tables,
        "encode_exec"_s <= "encode_table_prepare"_s + completion<RuntimeEncodeRuntime> [table_prepare_ok],
        "errored"_s <= "encode_table_prepare"_s + completion<RuntimeEncodeRuntime> [table_prepare_invalid_argument_error] / ensure_last_error_from_encode_table_prepare,
        "errored"_s <= "encode_table_prepare"_s + completion<RuntimeEncodeRuntime> [table_prepare_backend_error] / ensure_last_error_from_encode_table_prepare,
        "errored"_s <= "encode_table_prepare"_s + completion<RuntimeEncodeRuntime> [table_prepare_model_invalid_error] / ensure_last_error_from_encode_table_prepare,
        "errored"_s <= "encode_table_prepare"_s + completion<RuntimeEncodeRuntime> [table_prepare_unclassified_error_code] / ensure_last_error_from_encode_table_prepare,
        "emit_result_decision"_s <= "encode_exec"_s + completion<RuntimeEncodeRuntime> / run_encode_exec,
        "encode_result_decision"_s <= "emit_result_decision"_s + completion<RuntimeEncodeRuntime> [emit_result_ok] / apply_emit_result_ok,
        "encode_result_decision"_s <= "emit_result_decision"_s + completion<RuntimeEncodeRuntime> [emit_result_failed] / apply_emit_result_failed,
        "errored"_s <= "emit_result_decision"_s + completion<RuntimeEncodeRuntime> / ensure_last_error_from_emit_result_decision,
        "done"_s <= "encode_result_decision"_s + completion<RuntimeEncodeRuntime> [encode_result_ok] / mark_done_from_encode_result_decision,
        "errored"_s <= "encode_result_decision"_s + completion<RuntimeEncodeRuntime> [encode_result_invalid_argument_error] / ensure_last_error_from_encode_result_decision,
        "errored"_s <= "encode_result_decision"_s + completion<RuntimeEncodeRuntime> [encode_result_backend_error] / ensure_last_error_from_encode_result_decision,
        "errored"_s <= "encode_result_decision"_s + completion<RuntimeEncodeRuntime> [encode_result_model_invalid_error] / ensure_last_error_from_encode_result_decision,
        "errored"_s <= "encode_result_decision"_s + completion<RuntimeEncodeRuntime> [encode_result_unclassified_error_code] / ensure_last_error_from_encode_result_decision,
        "unexpected"_s <= "encode_validity_decision"_s + event<RuntimeEncodeRuntime> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_vocab_sync_decision"_s + event<RuntimeEncodeRuntime> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_precheck_decision"_s + event<RuntimeEncodeRuntime> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_table_prepare"_s + event<RuntimeEncodeRuntime> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_exec"_s + event<RuntimeEncodeRuntime> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "emit_result_decision"_s + event<RuntimeEncodeRuntime> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_result_decision"_s + event<RuntimeEncodeRuntime> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "initialized"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "initialized"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "encode_validity_decision"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "encode_validity_decision"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "encode_vocab_sync_decision"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "encode_vocab_sync_decision"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "encode_precheck_decision"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "encode_precheck_decision"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "encode_table_prepare"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "encode_table_prepare"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "encode_exec"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "encode_exec"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "emit_result_decision"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "emit_result_decision"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "encode_result_decision"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "encode_result_decision"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "done"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "done"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "errored"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "errored"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "unexpected"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "unexpected"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "initialized"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_validity_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_vocab_sync_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_precheck_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_table_prepare"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_exec"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "emit_result_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_result_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "done"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "errored"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "unexpected"_s + unexpected_event<_> / on_unexpected_unexp_wild,
    }
}

/// Context for `TextEncodersFallback` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct TextEncodersFallbackContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl TextEncodersFallbackStateMachineContext for TextEncodersFallbackContext {
    fn apply_emit_result_failed(&mut self, _event: &RuntimeEncodeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/fallback/actions.hpp::apply_emit_result_failed
        todo!(
            "TODO: port action `apply_emit_result_failed` from emel.cpp/src/emel/text/encoders/fallback/actions.hpp"
        )
    }
    fn apply_emit_result_ok(&mut self, _event: &RuntimeEncodeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/fallback/actions.hpp::apply_emit_result_ok
        todo!(
            "TODO: port action `apply_emit_result_ok` from emel.cpp/src/emel/text/encoders/fallback/actions.hpp"
        )
    }
    fn begin_encode(&mut self, _event: &RuntimeEncodeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/fallback/actions.hpp::begin_encode
        todo!(
            "TODO: port action `begin_encode` from emel.cpp/src/emel/text/encoders/fallback/actions.hpp"
        )
    }
    fn begin_encode_sync_vocab(&mut self, _event: &RuntimeEncodeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/fallback/actions.hpp::begin_encode_sync_vocab
        todo!(
            "TODO: port action `begin_encode_sync_vocab` from emel.cpp/src/emel/text/encoders/fallback/actions.hpp"
        )
    }
    fn emit_result_failed(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/fallback/guards.hpp::emit_result_failed
        todo!(
            "TODO: port guard `emit_result_failed` from emel.cpp/src/emel/text/encoders/fallback/guards.hpp"
        )
    }
    fn emit_result_ok(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/fallback/guards.hpp::emit_result_ok
        todo!(
            "TODO: port guard `emit_result_ok` from emel.cpp/src/emel/text/encoders/fallback/guards.hpp"
        )
    }
    fn encode_result_backend_error(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/fallback/guards.hpp::encode_result_backend_error
        todo!(
            "TODO: port guard `encode_result_backend_error` from emel.cpp/src/emel/text/encoders/fallback/guards.hpp"
        )
    }
    fn encode_result_invalid_argument_error(
        &self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/fallback/guards.hpp::encode_result_invalid_argument_error
        todo!(
            "TODO: port guard `encode_result_invalid_argument_error` from emel.cpp/src/emel/text/encoders/fallback/guards.hpp"
        )
    }
    fn encode_result_model_invalid_error(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/fallback/guards.hpp::encode_result_model_invalid_error
        todo!(
            "TODO: port guard `encode_result_model_invalid_error` from emel.cpp/src/emel/text/encoders/fallback/guards.hpp"
        )
    }
    fn encode_result_ok(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/fallback/guards.hpp::encode_result_ok
        todo!(
            "TODO: port guard `encode_result_ok` from emel.cpp/src/emel/text/encoders/fallback/guards.hpp"
        )
    }
    fn encode_result_unclassified_error_code(
        &self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/fallback/guards.hpp::encode_result_unclassified_error_code
        todo!(
            "TODO: port guard `encode_result_unclassified_error_code` from emel.cpp/src/emel/text/encoders/fallback/guards.hpp"
        )
    }
    fn ensure_last_error_from_emit_result_decision(
        &mut self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/fallback/actions.hpp::ensure_last_error
        todo!(
            "TODO: port action `ensure_last_error` from emel.cpp/src/emel/text/encoders/fallback/actions.hpp"
        )
    }
    fn ensure_last_error_from_encode_result_decision(
        &mut self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/fallback/actions.hpp::ensure_last_error
        todo!(
            "TODO: port action `ensure_last_error` from emel.cpp/src/emel/text/encoders/fallback/actions.hpp"
        )
    }
    fn ensure_last_error_from_encode_table_prepare(
        &mut self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/fallback/actions.hpp::ensure_last_error
        todo!(
            "TODO: port action `ensure_last_error` from emel.cpp/src/emel/text/encoders/fallback/actions.hpp"
        )
    }
    fn invalid_encode(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/fallback/guards.hpp::invalid_encode
        todo!(
            "TODO: port guard `invalid_encode` from emel.cpp/src/emel/text/encoders/fallback/guards.hpp"
        )
    }
    fn mark_done_from_encode_precheck_decision(
        &mut self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/fallback/actions.hpp::mark_done
        todo!(
            "TODO: port action `mark_done` from emel.cpp/src/emel/text/encoders/fallback/actions.hpp"
        )
    }
    fn mark_done_from_encode_result_decision(
        &mut self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/fallback/actions.hpp::mark_done
        todo!(
            "TODO: port action `mark_done` from emel.cpp/src/emel/text/encoders/fallback/actions.hpp"
        )
    }
    fn on_unexpected_events_encoding_done(
        &mut self,
        _event: &EventsEncodingDone,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/fallback/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/encoders/fallback/actions.hpp"
        )
    }
    fn on_unexpected_events_encoding_error(
        &mut self,
        _event: &EventsEncodingError,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/fallback/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/encoders/fallback/actions.hpp"
        )
    }
    fn on_unexpected_runtime_encode_runtime(
        &mut self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/fallback/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/encoders/fallback/actions.hpp"
        )
    }
    fn on_unexpected_unexp_wild(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/fallback/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/encoders/fallback/actions.hpp"
        )
    }
    fn prepare_tables(&mut self, _event: &RuntimeEncodeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/fallback/actions.hpp::prepare_tables
        todo!(
            "TODO: port action `prepare_tables` from emel.cpp/src/emel/text/encoders/fallback/actions.hpp"
        )
    }
    fn reject_invalid_encode_from_encode_validity_decision(
        &mut self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/fallback/actions.hpp::reject_invalid_encode
        todo!(
            "TODO: port action `reject_invalid_encode` from emel.cpp/src/emel/text/encoders/fallback/actions.hpp"
        )
    }
    fn reject_invalid_encode_from_encode_vocab_sync_decision(
        &mut self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/fallback/actions.hpp::reject_invalid_encode
        todo!(
            "TODO: port action `reject_invalid_encode` from emel.cpp/src/emel/text/encoders/fallback/actions.hpp"
        )
    }
    fn run_encode_exec(&mut self, _event: &RuntimeEncodeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/fallback/actions.hpp::run_encode_exec
        todo!(
            "TODO: port action `run_encode_exec` from emel.cpp/src/emel/text/encoders/fallback/actions.hpp"
        )
    }
    fn table_prepare_backend_error(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/fallback/guards.hpp::table_prepare_backend_error
        todo!(
            "TODO: port guard `table_prepare_backend_error` from emel.cpp/src/emel/text/encoders/fallback/guards.hpp"
        )
    }
    fn table_prepare_invalid_argument_error(
        &self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/fallback/guards.hpp::table_prepare_invalid_argument_error
        todo!(
            "TODO: port guard `table_prepare_invalid_argument_error` from emel.cpp/src/emel/text/encoders/fallback/guards.hpp"
        )
    }
    fn table_prepare_model_invalid_error(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/fallback/guards.hpp::table_prepare_model_invalid_error
        todo!(
            "TODO: port guard `table_prepare_model_invalid_error` from emel.cpp/src/emel/text/encoders/fallback/guards.hpp"
        )
    }
    fn table_prepare_ok(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/fallback/guards.hpp::table_prepare_ok
        todo!(
            "TODO: port guard `table_prepare_ok` from emel.cpp/src/emel/text/encoders/fallback/guards.hpp"
        )
    }
    fn table_prepare_unclassified_error_code(
        &self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/fallback/guards.hpp::table_prepare_unclassified_error_code
        todo!(
            "TODO: port guard `table_prepare_unclassified_error_code` from emel.cpp/src/emel/text/encoders/fallback/guards.hpp"
        )
    }
    fn text_empty(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/fallback/guards.hpp::text_empty
        todo!(
            "TODO: port guard `text_empty` from emel.cpp/src/emel/text/encoders/fallback/guards.hpp"
        )
    }
    fn text_non_empty(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/fallback/guards.hpp::text_non_empty
        todo!(
            "TODO: port guard `text_non_empty` from emel.cpp/src/emel/text/encoders/fallback/guards.hpp"
        )
    }
    fn valid_encode(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/fallback/guards.hpp::valid_encode
        todo!(
            "TODO: port guard `valid_encode` from emel.cpp/src/emel/text/encoders/fallback/guards.hpp"
        )
    }
    fn vocab_changed(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/fallback/guards.hpp::vocab_changed
        todo!(
            "TODO: port guard `vocab_changed` from emel.cpp/src/emel/text/encoders/fallback/guards.hpp"
        )
    }
    fn vocab_unchanged(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/fallback/guards.hpp::vocab_unchanged
        todo!(
            "TODO: port guard `vocab_unchanged` from emel.cpp/src/emel/text/encoders/fallback/guards.hpp"
        )
    }
}
