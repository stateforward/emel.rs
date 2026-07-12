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

// --- machine TextEncodersUgm from emel.cpp/src/emel/text/encoders/ugm/sm.hpp ---
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
    TextEncodersUgm {
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
        "table_policy_decision"_s <= "encode_precheck_decision"_s + completion<RuntimeEncodeRuntime> [text_non_empty],
        "errored"_s <= "encode_precheck_decision"_s + completion<RuntimeEncodeRuntime> / ensure_last_error_from_encode_precheck_decision,
        "table_sync_exec"_s <= "table_policy_decision"_s + completion<RuntimeEncodeRuntime> [tables_missing],
        "unk_resolution_decision"_s <= "table_policy_decision"_s + completion<RuntimeEncodeRuntime> [tables_ready],
        "errored"_s <= "table_policy_decision"_s + completion<RuntimeEncodeRuntime> / ensure_last_error_from_table_policy_decision,
        "table_sync_result_decision"_s <= "table_sync_exec"_s + completion<RuntimeEncodeRuntime> / sync_tables,
        "unk_resolution_decision"_s <= "table_sync_result_decision"_s + completion<RuntimeEncodeRuntime> [table_sync_ok],
        "errored"_s <= "table_sync_result_decision"_s + completion<RuntimeEncodeRuntime> [table_sync_invalid_argument_error] / ensure_last_error_from_table_sync_result_decision,
        "errored"_s <= "table_sync_result_decision"_s + completion<RuntimeEncodeRuntime> [table_sync_backend_error] / ensure_last_error_from_table_sync_result_decision,
        "errored"_s <= "table_sync_result_decision"_s + completion<RuntimeEncodeRuntime> [table_sync_model_invalid_error] / ensure_last_error_from_table_sync_result_decision,
        "errored"_s <= "table_sync_result_decision"_s + completion<RuntimeEncodeRuntime> [table_sync_unclassified_error_code] / ensure_last_error_from_table_sync_result_decision,
        "normalize_exec"_s <= "unk_resolution_decision"_s + completion<RuntimeEncodeRuntime> [vocab_unk_present] / resolve_vocab_unk,
        "unk_lookup_exec"_s <= "unk_resolution_decision"_s + completion<RuntimeEncodeRuntime> [vocab_unk_missing],
        "normalize_exec"_s <= "unk_lookup_exec"_s + completion<RuntimeEncodeRuntime> / lookup_unk_id,
        "normalize_result_decision"_s <= "normalize_exec"_s + completion<RuntimeEncodeRuntime> / normalize_input,
        "input_prepare_exec"_s <= "normalize_result_decision"_s + completion<RuntimeEncodeRuntime> [normalize_result_ok],
        "errored"_s <= "normalize_result_decision"_s + completion<RuntimeEncodeRuntime> [normalize_result_invalid_argument_error] / ensure_last_error_from_normalize_result_decision,
        "errored"_s <= "normalize_result_decision"_s + completion<RuntimeEncodeRuntime> [normalize_result_backend_error] / ensure_last_error_from_normalize_result_decision,
        "errored"_s <= "normalize_result_decision"_s + completion<RuntimeEncodeRuntime> [normalize_result_model_invalid_error] / ensure_last_error_from_normalize_result_decision,
        "errored"_s <= "normalize_result_decision"_s + completion<RuntimeEncodeRuntime> [normalize_result_unclassified_error_code] / ensure_last_error_from_normalize_result_decision,
        "input_prepare_result_decision"_s <= "input_prepare_exec"_s + completion<RuntimeEncodeRuntime> / prepare_dp_input,
        "dp_forward_exec"_s <= "input_prepare_result_decision"_s + completion<RuntimeEncodeRuntime> [input_prepare_result_non_empty_ok],
        "done"_s <= "input_prepare_result_decision"_s + completion<RuntimeEncodeRuntime> [input_prepare_result_empty_ok] / mark_done_from_input_prepare_result_decision,
        "errored"_s <= "input_prepare_result_decision"_s + completion<RuntimeEncodeRuntime> [input_prepare_result_invalid_argument_error] / ensure_last_error_from_input_prepare_result_decision,
        "errored"_s <= "input_prepare_result_decision"_s + completion<RuntimeEncodeRuntime> [input_prepare_result_backend_error] / ensure_last_error_from_input_prepare_result_decision,
        "errored"_s <= "input_prepare_result_decision"_s + completion<RuntimeEncodeRuntime> [input_prepare_result_model_invalid_error] / ensure_last_error_from_input_prepare_result_decision,
        "errored"_s <= "input_prepare_result_decision"_s + completion<RuntimeEncodeRuntime> [input_prepare_result_unclassified_error_code] / ensure_last_error_from_input_prepare_result_decision,
        "dp_forward_result_decision"_s <= "dp_forward_exec"_s + completion<RuntimeEncodeRuntime> / run_dp_forward,
        "dp_backtrace_exec"_s <= "dp_forward_result_decision"_s + completion<RuntimeEncodeRuntime> [dp_forward_result_ok],
        "errored"_s <= "dp_forward_result_decision"_s + completion<RuntimeEncodeRuntime> [dp_forward_result_invalid_argument_error] / ensure_last_error_from_dp_forward_result_decision,
        "errored"_s <= "dp_forward_result_decision"_s + completion<RuntimeEncodeRuntime> [dp_forward_result_backend_error] / ensure_last_error_from_dp_forward_result_decision,
        "errored"_s <= "dp_forward_result_decision"_s + completion<RuntimeEncodeRuntime> [dp_forward_result_model_invalid_error] / ensure_last_error_from_dp_forward_result_decision,
        "errored"_s <= "dp_forward_result_decision"_s + completion<RuntimeEncodeRuntime> [dp_forward_result_unclassified_error_code] / ensure_last_error_from_dp_forward_result_decision,
        "dp_backtrace_result_decision"_s <= "dp_backtrace_exec"_s + completion<RuntimeEncodeRuntime> / run_dp_backtrace,
        "emit_exec"_s <= "dp_backtrace_result_decision"_s + completion<RuntimeEncodeRuntime> [backtrace_ok],
        "errored"_s <= "dp_backtrace_result_decision"_s + completion<RuntimeEncodeRuntime> [backtrace_failed] / mark_backtrace_failed,
        "errored"_s <= "dp_backtrace_result_decision"_s + completion<RuntimeEncodeRuntime> / ensure_last_error_from_dp_backtrace_result_decision,
        "encode_result_decision"_s <= "emit_exec"_s + completion<RuntimeEncodeRuntime> / emit_tokens,
        "done"_s <= "encode_result_decision"_s + completion<RuntimeEncodeRuntime> [emit_ok] / mark_done_from_encode_result_decision,
        "errored"_s <= "encode_result_decision"_s + completion<RuntimeEncodeRuntime> [emit_failed] / mark_emit_failed,
        "errored"_s <= "encode_result_decision"_s + completion<RuntimeEncodeRuntime> / ensure_last_error_from_encode_result_decision,
        "unexpected"_s <= "encode_validity_decision"_s + event<RuntimeEncodeRuntime> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_vocab_sync_decision"_s + event<RuntimeEncodeRuntime> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_precheck_decision"_s + event<RuntimeEncodeRuntime> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "table_policy_decision"_s + event<RuntimeEncodeRuntime> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "table_sync_exec"_s + event<RuntimeEncodeRuntime> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "table_sync_result_decision"_s + event<RuntimeEncodeRuntime> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "unk_resolution_decision"_s + event<RuntimeEncodeRuntime> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "unk_lookup_exec"_s + event<RuntimeEncodeRuntime> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "normalize_exec"_s + event<RuntimeEncodeRuntime> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "normalize_result_decision"_s + event<RuntimeEncodeRuntime> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "input_prepare_exec"_s + event<RuntimeEncodeRuntime> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "input_prepare_result_decision"_s + event<RuntimeEncodeRuntime> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "dp_forward_exec"_s + event<RuntimeEncodeRuntime> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "dp_forward_result_decision"_s + event<RuntimeEncodeRuntime> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "dp_backtrace_exec"_s + event<RuntimeEncodeRuntime> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "dp_backtrace_result_decision"_s + event<RuntimeEncodeRuntime> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "emit_exec"_s + event<RuntimeEncodeRuntime> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_result_decision"_s + event<RuntimeEncodeRuntime> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "initialized"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "initialized"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "encode_validity_decision"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "encode_validity_decision"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "encode_vocab_sync_decision"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "encode_vocab_sync_decision"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "encode_precheck_decision"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "encode_precheck_decision"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "table_policy_decision"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "table_policy_decision"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "table_sync_exec"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "table_sync_exec"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "table_sync_result_decision"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "table_sync_result_decision"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "unk_resolution_decision"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "unk_resolution_decision"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "unk_lookup_exec"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "unk_lookup_exec"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "normalize_exec"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "normalize_exec"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "normalize_result_decision"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "normalize_result_decision"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "input_prepare_exec"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "input_prepare_exec"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "input_prepare_result_decision"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "input_prepare_result_decision"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "dp_forward_exec"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "dp_forward_exec"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "dp_forward_result_decision"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "dp_forward_result_decision"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "dp_backtrace_exec"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "dp_backtrace_exec"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "dp_backtrace_result_decision"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "dp_backtrace_result_decision"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "emit_exec"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "emit_exec"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
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
        "unexpected"_s <= "table_policy_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "table_sync_exec"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "table_sync_result_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "unk_resolution_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "unk_lookup_exec"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "normalize_exec"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "normalize_result_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "input_prepare_exec"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "input_prepare_result_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "dp_forward_exec"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "dp_forward_result_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "dp_backtrace_exec"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "dp_backtrace_result_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "emit_exec"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_result_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "done"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "errored"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "unexpected"_s + unexpected_event<_> / on_unexpected_unexp_wild,
    }
}

/// Context for `TextEncodersUgm` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct TextEncodersUgmContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl TextEncodersUgmStateMachineContext for TextEncodersUgmContext {
    fn backtrace_failed(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/guards.hpp::backtrace_failed
        todo!(
            "TODO: port guard `backtrace_failed` from emel.cpp/src/emel/text/encoders/ugm/guards.hpp"
        )
    }
    fn backtrace_ok(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/guards.hpp::backtrace_ok
        todo!("TODO: port guard `backtrace_ok` from emel.cpp/src/emel/text/encoders/ugm/guards.hpp")
    }
    fn begin_encode(&mut self, _event: &RuntimeEncodeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/actions.hpp::begin_encode
        todo!(
            "TODO: port action `begin_encode` from emel.cpp/src/emel/text/encoders/ugm/actions.hpp"
        )
    }
    fn begin_encode_sync_vocab(&mut self, _event: &RuntimeEncodeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/actions.hpp::begin_encode_sync_vocab
        todo!(
            "TODO: port action `begin_encode_sync_vocab` from emel.cpp/src/emel/text/encoders/ugm/actions.hpp"
        )
    }
    fn dp_forward_result_backend_error(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/guards.hpp::dp_forward_result_backend_error
        todo!(
            "TODO: port guard `dp_forward_result_backend_error` from emel.cpp/src/emel/text/encoders/ugm/guards.hpp"
        )
    }
    fn dp_forward_result_invalid_argument_error(
        &self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/guards.hpp::dp_forward_result_invalid_argument_error
        todo!(
            "TODO: port guard `dp_forward_result_invalid_argument_error` from emel.cpp/src/emel/text/encoders/ugm/guards.hpp"
        )
    }
    fn dp_forward_result_model_invalid_error(
        &self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/guards.hpp::dp_forward_result_model_invalid_error
        todo!(
            "TODO: port guard `dp_forward_result_model_invalid_error` from emel.cpp/src/emel/text/encoders/ugm/guards.hpp"
        )
    }
    fn dp_forward_result_ok(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/guards.hpp::dp_forward_result_ok
        todo!(
            "TODO: port guard `dp_forward_result_ok` from emel.cpp/src/emel/text/encoders/ugm/guards.hpp"
        )
    }
    fn dp_forward_result_unclassified_error_code(
        &self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/guards.hpp::dp_forward_result_unclassified_error_code
        todo!(
            "TODO: port guard `dp_forward_result_unclassified_error_code` from emel.cpp/src/emel/text/encoders/ugm/guards.hpp"
        )
    }
    fn emit_failed(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/guards.hpp::emit_failed
        todo!("TODO: port guard `emit_failed` from emel.cpp/src/emel/text/encoders/ugm/guards.hpp")
    }
    fn emit_ok(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/guards.hpp::emit_ok
        todo!("TODO: port guard `emit_ok` from emel.cpp/src/emel/text/encoders/ugm/guards.hpp")
    }
    fn emit_tokens(&mut self, _event: &RuntimeEncodeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/actions.hpp::emit_tokens
        todo!(
            "TODO: port action `emit_tokens` from emel.cpp/src/emel/text/encoders/ugm/actions.hpp"
        )
    }
    fn ensure_last_error_from_dp_backtrace_result_decision(
        &mut self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/actions.hpp::ensure_last_error
        todo!(
            "TODO: port action `ensure_last_error` from emel.cpp/src/emel/text/encoders/ugm/actions.hpp"
        )
    }
    fn ensure_last_error_from_dp_forward_result_decision(
        &mut self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/actions.hpp::ensure_last_error
        todo!(
            "TODO: port action `ensure_last_error` from emel.cpp/src/emel/text/encoders/ugm/actions.hpp"
        )
    }
    fn ensure_last_error_from_encode_precheck_decision(
        &mut self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/actions.hpp::ensure_last_error
        todo!(
            "TODO: port action `ensure_last_error` from emel.cpp/src/emel/text/encoders/ugm/actions.hpp"
        )
    }
    fn ensure_last_error_from_encode_result_decision(
        &mut self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/actions.hpp::ensure_last_error
        todo!(
            "TODO: port action `ensure_last_error` from emel.cpp/src/emel/text/encoders/ugm/actions.hpp"
        )
    }
    fn ensure_last_error_from_input_prepare_result_decision(
        &mut self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/actions.hpp::ensure_last_error
        todo!(
            "TODO: port action `ensure_last_error` from emel.cpp/src/emel/text/encoders/ugm/actions.hpp"
        )
    }
    fn ensure_last_error_from_normalize_result_decision(
        &mut self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/actions.hpp::ensure_last_error
        todo!(
            "TODO: port action `ensure_last_error` from emel.cpp/src/emel/text/encoders/ugm/actions.hpp"
        )
    }
    fn ensure_last_error_from_table_policy_decision(
        &mut self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/actions.hpp::ensure_last_error
        todo!(
            "TODO: port action `ensure_last_error` from emel.cpp/src/emel/text/encoders/ugm/actions.hpp"
        )
    }
    fn ensure_last_error_from_table_sync_result_decision(
        &mut self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/actions.hpp::ensure_last_error
        todo!(
            "TODO: port action `ensure_last_error` from emel.cpp/src/emel/text/encoders/ugm/actions.hpp"
        )
    }
    fn input_prepare_result_backend_error(
        &self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/guards.hpp::input_prepare_result_backend_error
        todo!(
            "TODO: port guard `input_prepare_result_backend_error` from emel.cpp/src/emel/text/encoders/ugm/guards.hpp"
        )
    }
    fn input_prepare_result_empty_ok(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/guards.hpp::input_prepare_result_empty_ok
        todo!(
            "TODO: port guard `input_prepare_result_empty_ok` from emel.cpp/src/emel/text/encoders/ugm/guards.hpp"
        )
    }
    fn input_prepare_result_invalid_argument_error(
        &self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/guards.hpp::input_prepare_result_invalid_argument_error
        todo!(
            "TODO: port guard `input_prepare_result_invalid_argument_error` from emel.cpp/src/emel/text/encoders/ugm/guards.hpp"
        )
    }
    fn input_prepare_result_model_invalid_error(
        &self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/guards.hpp::input_prepare_result_model_invalid_error
        todo!(
            "TODO: port guard `input_prepare_result_model_invalid_error` from emel.cpp/src/emel/text/encoders/ugm/guards.hpp"
        )
    }
    fn input_prepare_result_non_empty_ok(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/guards.hpp::input_prepare_result_non_empty_ok
        todo!(
            "TODO: port guard `input_prepare_result_non_empty_ok` from emel.cpp/src/emel/text/encoders/ugm/guards.hpp"
        )
    }
    fn input_prepare_result_unclassified_error_code(
        &self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/guards.hpp::input_prepare_result_unclassified_error_code
        todo!(
            "TODO: port guard `input_prepare_result_unclassified_error_code` from emel.cpp/src/emel/text/encoders/ugm/guards.hpp"
        )
    }
    fn invalid_encode(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/guards.hpp::invalid_encode
        todo!(
            "TODO: port guard `invalid_encode` from emel.cpp/src/emel/text/encoders/ugm/guards.hpp"
        )
    }
    fn lookup_unk_id(&mut self, _event: &RuntimeEncodeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/actions.hpp::lookup_unk_id
        todo!(
            "TODO: port action `lookup_unk_id` from emel.cpp/src/emel/text/encoders/ugm/actions.hpp"
        )
    }
    fn mark_backtrace_failed(&mut self, _event: &RuntimeEncodeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/actions.hpp::mark_backtrace_failed
        todo!(
            "TODO: port action `mark_backtrace_failed` from emel.cpp/src/emel/text/encoders/ugm/actions.hpp"
        )
    }
    fn mark_done_from_encode_precheck_decision(
        &mut self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/actions.hpp::mark_done
        todo!("TODO: port action `mark_done` from emel.cpp/src/emel/text/encoders/ugm/actions.hpp")
    }
    fn mark_done_from_encode_result_decision(
        &mut self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/actions.hpp::mark_done
        todo!("TODO: port action `mark_done` from emel.cpp/src/emel/text/encoders/ugm/actions.hpp")
    }
    fn mark_done_from_input_prepare_result_decision(
        &mut self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/actions.hpp::mark_done
        todo!("TODO: port action `mark_done` from emel.cpp/src/emel/text/encoders/ugm/actions.hpp")
    }
    fn mark_emit_failed(&mut self, _event: &RuntimeEncodeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/actions.hpp::mark_emit_failed
        todo!(
            "TODO: port action `mark_emit_failed` from emel.cpp/src/emel/text/encoders/ugm/actions.hpp"
        )
    }
    fn normalize_input(&mut self, _event: &RuntimeEncodeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/actions.hpp::normalize_input
        todo!(
            "TODO: port action `normalize_input` from emel.cpp/src/emel/text/encoders/ugm/actions.hpp"
        )
    }
    fn normalize_result_backend_error(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/guards.hpp::normalize_result_backend_error
        todo!(
            "TODO: port guard `normalize_result_backend_error` from emel.cpp/src/emel/text/encoders/ugm/guards.hpp"
        )
    }
    fn normalize_result_invalid_argument_error(
        &self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/guards.hpp::normalize_result_invalid_argument_error
        todo!(
            "TODO: port guard `normalize_result_invalid_argument_error` from emel.cpp/src/emel/text/encoders/ugm/guards.hpp"
        )
    }
    fn normalize_result_model_invalid_error(
        &self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/guards.hpp::normalize_result_model_invalid_error
        todo!(
            "TODO: port guard `normalize_result_model_invalid_error` from emel.cpp/src/emel/text/encoders/ugm/guards.hpp"
        )
    }
    fn normalize_result_ok(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/guards.hpp::normalize_result_ok
        todo!(
            "TODO: port guard `normalize_result_ok` from emel.cpp/src/emel/text/encoders/ugm/guards.hpp"
        )
    }
    fn normalize_result_unclassified_error_code(
        &self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/guards.hpp::normalize_result_unclassified_error_code
        todo!(
            "TODO: port guard `normalize_result_unclassified_error_code` from emel.cpp/src/emel/text/encoders/ugm/guards.hpp"
        )
    }
    fn on_unexpected_events_encoding_done(
        &mut self,
        _event: &EventsEncodingDone,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/encoders/ugm/actions.hpp"
        )
    }
    fn on_unexpected_events_encoding_error(
        &mut self,
        _event: &EventsEncodingError,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/encoders/ugm/actions.hpp"
        )
    }
    fn on_unexpected_runtime_encode_runtime(
        &mut self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/encoders/ugm/actions.hpp"
        )
    }
    fn on_unexpected_unexp_wild(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/encoders/ugm/actions.hpp"
        )
    }
    fn prepare_dp_input(&mut self, _event: &RuntimeEncodeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/actions.hpp::prepare_dp_input
        todo!(
            "TODO: port action `prepare_dp_input` from emel.cpp/src/emel/text/encoders/ugm/actions.hpp"
        )
    }
    fn reject_invalid_encode_from_encode_validity_decision(
        &mut self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/actions.hpp::reject_invalid_encode
        todo!(
            "TODO: port action `reject_invalid_encode` from emel.cpp/src/emel/text/encoders/ugm/actions.hpp"
        )
    }
    fn reject_invalid_encode_from_encode_vocab_sync_decision(
        &mut self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/actions.hpp::reject_invalid_encode
        todo!(
            "TODO: port action `reject_invalid_encode` from emel.cpp/src/emel/text/encoders/ugm/actions.hpp"
        )
    }
    fn resolve_vocab_unk(&mut self, _event: &RuntimeEncodeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/actions.hpp::resolve_vocab_unk
        todo!(
            "TODO: port action `resolve_vocab_unk` from emel.cpp/src/emel/text/encoders/ugm/actions.hpp"
        )
    }
    fn run_dp_backtrace(&mut self, _event: &RuntimeEncodeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/actions.hpp::run_dp_backtrace
        todo!(
            "TODO: port action `run_dp_backtrace` from emel.cpp/src/emel/text/encoders/ugm/actions.hpp"
        )
    }
    fn run_dp_forward(&mut self, _event: &RuntimeEncodeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/actions.hpp::run_dp_forward
        todo!(
            "TODO: port action `run_dp_forward` from emel.cpp/src/emel/text/encoders/ugm/actions.hpp"
        )
    }
    fn sync_tables(&mut self, _event: &RuntimeEncodeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/actions.hpp::sync_tables
        todo!(
            "TODO: port action `sync_tables` from emel.cpp/src/emel/text/encoders/ugm/actions.hpp"
        )
    }
    fn table_sync_backend_error(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/guards.hpp::table_sync_backend_error
        todo!(
            "TODO: port guard `table_sync_backend_error` from emel.cpp/src/emel/text/encoders/ugm/guards.hpp"
        )
    }
    fn table_sync_invalid_argument_error(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/guards.hpp::table_sync_invalid_argument_error
        todo!(
            "TODO: port guard `table_sync_invalid_argument_error` from emel.cpp/src/emel/text/encoders/ugm/guards.hpp"
        )
    }
    fn table_sync_model_invalid_error(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/guards.hpp::table_sync_model_invalid_error
        todo!(
            "TODO: port guard `table_sync_model_invalid_error` from emel.cpp/src/emel/text/encoders/ugm/guards.hpp"
        )
    }
    fn table_sync_ok(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/guards.hpp::table_sync_ok
        todo!(
            "TODO: port guard `table_sync_ok` from emel.cpp/src/emel/text/encoders/ugm/guards.hpp"
        )
    }
    fn table_sync_unclassified_error_code(
        &self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/guards.hpp::table_sync_unclassified_error_code
        todo!(
            "TODO: port guard `table_sync_unclassified_error_code` from emel.cpp/src/emel/text/encoders/ugm/guards.hpp"
        )
    }
    fn tables_missing(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/guards.hpp::tables_missing
        todo!(
            "TODO: port guard `tables_missing` from emel.cpp/src/emel/text/encoders/ugm/guards.hpp"
        )
    }
    fn tables_ready(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/guards.hpp::tables_ready
        todo!("TODO: port guard `tables_ready` from emel.cpp/src/emel/text/encoders/ugm/guards.hpp")
    }
    fn text_empty(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/guards.hpp::text_empty
        todo!("TODO: port guard `text_empty` from emel.cpp/src/emel/text/encoders/ugm/guards.hpp")
    }
    fn text_non_empty(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/guards.hpp::text_non_empty
        todo!(
            "TODO: port guard `text_non_empty` from emel.cpp/src/emel/text/encoders/ugm/guards.hpp"
        )
    }
    fn valid_encode(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/guards.hpp::valid_encode
        todo!("TODO: port guard `valid_encode` from emel.cpp/src/emel/text/encoders/ugm/guards.hpp")
    }
    fn vocab_changed(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/guards.hpp::vocab_changed
        todo!(
            "TODO: port guard `vocab_changed` from emel.cpp/src/emel/text/encoders/ugm/guards.hpp"
        )
    }
    fn vocab_unchanged(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/guards.hpp::vocab_unchanged
        todo!(
            "TODO: port guard `vocab_unchanged` from emel.cpp/src/emel/text/encoders/ugm/guards.hpp"
        )
    }
    fn vocab_unk_missing(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/guards.hpp::vocab_unk_missing
        todo!(
            "TODO: port guard `vocab_unk_missing` from emel.cpp/src/emel/text/encoders/ugm/guards.hpp"
        )
    }
    fn vocab_unk_present(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/ugm/guards.hpp::vocab_unk_present
        todo!(
            "TODO: port guard `vocab_unk_present` from emel.cpp/src/emel/text/encoders/ugm/guards.hpp"
        )
    }
}
