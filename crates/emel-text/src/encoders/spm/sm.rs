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

// --- machine TextEncodersSpm from emel.cpp/src/emel/text/encoders/spm/sm.hpp ---
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
    TextEncodersSpm {
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
        "encode_prepare_exec"_s <= "table_policy_decision"_s + completion<RuntimeEncodeRuntime> [tables_ready],
        "errored"_s <= "table_policy_decision"_s + completion<RuntimeEncodeRuntime> / ensure_last_error_from_table_policy_decision,
        "table_sync_result_decision"_s <= "table_sync_exec"_s + completion<RuntimeEncodeRuntime> / sync_tables,
        "encode_prepare_exec"_s <= "table_sync_result_decision"_s + completion<RuntimeEncodeRuntime> [table_sync_ok],
        "errored"_s <= "table_sync_result_decision"_s + completion<RuntimeEncodeRuntime> [table_sync_invalid_argument_error] / ensure_last_error_from_table_sync_result_decision,
        "errored"_s <= "table_sync_result_decision"_s + completion<RuntimeEncodeRuntime> [table_sync_backend_error] / ensure_last_error_from_table_sync_result_decision,
        "errored"_s <= "table_sync_result_decision"_s + completion<RuntimeEncodeRuntime> [table_sync_model_invalid_error] / ensure_last_error_from_table_sync_result_decision,
        "errored"_s <= "table_sync_result_decision"_s + completion<RuntimeEncodeRuntime> [table_sync_unclassified_error_code] / ensure_last_error_from_table_sync_result_decision,
        "encode_prepare_result_decision"_s <= "encode_prepare_exec"_s + completion<RuntimeEncodeRuntime> / run_prepare,
        "encode_merge_input_capacity_decision"_s <= "encode_prepare_result_decision"_s + completion<RuntimeEncodeRuntime> [prepare_result_ok],
        "errored"_s <= "encode_prepare_result_decision"_s + completion<RuntimeEncodeRuntime> [prepare_result_invalid_argument_error] / ensure_last_error_from_encode_prepare_result_decision,
        "errored"_s <= "encode_prepare_result_decision"_s + completion<RuntimeEncodeRuntime> [prepare_result_backend_error] / ensure_last_error_from_encode_prepare_result_decision,
        "errored"_s <= "encode_prepare_result_decision"_s + completion<RuntimeEncodeRuntime> [prepare_result_model_invalid_error] / ensure_last_error_from_encode_prepare_result_decision,
        "errored"_s <= "encode_prepare_result_decision"_s + completion<RuntimeEncodeRuntime> [prepare_result_unclassified_error_code] / ensure_last_error_from_encode_prepare_result_decision,
        "encode_merge_exec"_s <= "encode_merge_input_capacity_decision"_s + completion<RuntimeEncodeRuntime> [merge_symbol_capacity_within_limit],
        "errored"_s <= "encode_merge_input_capacity_decision"_s + completion<RuntimeEncodeRuntime> [merge_symbol_capacity_exceeded] / reject_invalid_encode_from_encode_merge_input_capacity_decision,
        "errored"_s <= "encode_merge_input_capacity_decision"_s + completion<RuntimeEncodeRuntime> / reject_invalid_encode_from_encode_merge_input_capacity_decision,
        "encode_merge_result_decision"_s <= "encode_merge_exec"_s + completion<RuntimeEncodeRuntime> / run_merge,
        "encode_emit_input_decision"_s <= "encode_merge_result_decision"_s + completion<RuntimeEncodeRuntime> [merge_result_ok],
        "errored"_s <= "encode_merge_result_decision"_s + completion<RuntimeEncodeRuntime> [merge_result_invalid_argument_error] / ensure_last_error_from_encode_merge_result_decision,
        "errored"_s <= "encode_merge_result_decision"_s + completion<RuntimeEncodeRuntime> [merge_result_backend_error] / ensure_last_error_from_encode_merge_result_decision,
        "errored"_s <= "encode_merge_result_decision"_s + completion<RuntimeEncodeRuntime> [merge_result_model_invalid_error] / ensure_last_error_from_encode_merge_result_decision,
        "errored"_s <= "encode_merge_result_decision"_s + completion<RuntimeEncodeRuntime> [merge_result_unclassified_error_code] / ensure_last_error_from_encode_merge_result_decision,
        "encode_exec"_s <= "encode_emit_input_decision"_s + completion<RuntimeEncodeRuntime> [symbols_present],
        "encode_result_decision"_s <= "encode_emit_input_decision"_s + completion<RuntimeEncodeRuntime> [symbols_absent] / set_emit_result_empty,
        "errored"_s <= "encode_emit_input_decision"_s + completion<RuntimeEncodeRuntime> / ensure_last_error_from_encode_emit_input_decision,
        "emit_result_decision"_s <= "encode_exec"_s + completion<RuntimeEncodeRuntime> / run_encode,
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
        "unexpected"_s <= "table_policy_decision"_s + event<RuntimeEncodeRuntime> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "table_sync_exec"_s + event<RuntimeEncodeRuntime> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "table_sync_result_decision"_s + event<RuntimeEncodeRuntime> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_prepare_exec"_s + event<RuntimeEncodeRuntime> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_prepare_result_decision"_s + event<RuntimeEncodeRuntime> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_merge_input_capacity_decision"_s + event<RuntimeEncodeRuntime> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_merge_exec"_s + event<RuntimeEncodeRuntime> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_merge_result_decision"_s + event<RuntimeEncodeRuntime> / on_unexpected_runtime_encode_runtime,
        "unexpected"_s <= "encode_emit_input_decision"_s + event<RuntimeEncodeRuntime> / on_unexpected_runtime_encode_runtime,
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
        "unexpected"_s <= "table_policy_decision"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "table_policy_decision"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "table_sync_exec"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "table_sync_exec"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "table_sync_result_decision"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "table_sync_result_decision"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "encode_prepare_exec"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "encode_prepare_exec"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "encode_prepare_result_decision"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "encode_prepare_result_decision"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "encode_merge_input_capacity_decision"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "encode_merge_input_capacity_decision"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "encode_merge_exec"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "encode_merge_exec"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "encode_merge_result_decision"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "encode_merge_result_decision"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "encode_emit_input_decision"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "encode_emit_input_decision"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
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
        "unexpected"_s <= "table_policy_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "table_sync_exec"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "table_sync_result_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_prepare_exec"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_prepare_result_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_merge_input_capacity_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_merge_exec"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_merge_result_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_emit_input_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_exec"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "emit_result_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_result_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "done"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "errored"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "unexpected"_s + unexpected_event<_> / on_unexpected_unexp_wild,
    }
}

/// Context for `TextEncodersSpm` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct TextEncodersSpmContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl TextEncodersSpmStateMachineContext for TextEncodersSpmContext {
    fn apply_emit_result_failed(&mut self, _event: &RuntimeEncodeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/actions.hpp::apply_emit_result_failed
        todo!(
            "TODO: port action `apply_emit_result_failed` from emel.cpp/src/emel/text/encoders/spm/actions.hpp"
        )
    }
    fn apply_emit_result_ok(&mut self, _event: &RuntimeEncodeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/actions.hpp::apply_emit_result_ok
        todo!(
            "TODO: port action `apply_emit_result_ok` from emel.cpp/src/emel/text/encoders/spm/actions.hpp"
        )
    }
    fn begin_encode(&mut self, _event: &RuntimeEncodeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/actions.hpp::begin_encode
        todo!(
            "TODO: port action `begin_encode` from emel.cpp/src/emel/text/encoders/spm/actions.hpp"
        )
    }
    fn begin_encode_sync_vocab(&mut self, _event: &RuntimeEncodeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/actions.hpp::begin_encode_sync_vocab
        todo!(
            "TODO: port action `begin_encode_sync_vocab` from emel.cpp/src/emel/text/encoders/spm/actions.hpp"
        )
    }
    fn emit_result_failed(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/guards.hpp::emit_result_failed
        todo!(
            "TODO: port guard `emit_result_failed` from emel.cpp/src/emel/text/encoders/spm/guards.hpp"
        )
    }
    fn emit_result_ok(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/guards.hpp::emit_result_ok
        todo!(
            "TODO: port guard `emit_result_ok` from emel.cpp/src/emel/text/encoders/spm/guards.hpp"
        )
    }
    fn encode_result_backend_error(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/guards.hpp::encode_result_backend_error
        todo!(
            "TODO: port guard `encode_result_backend_error` from emel.cpp/src/emel/text/encoders/spm/guards.hpp"
        )
    }
    fn encode_result_invalid_argument_error(
        &self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/guards.hpp::encode_result_invalid_argument_error
        todo!(
            "TODO: port guard `encode_result_invalid_argument_error` from emel.cpp/src/emel/text/encoders/spm/guards.hpp"
        )
    }
    fn encode_result_model_invalid_error(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/guards.hpp::encode_result_model_invalid_error
        todo!(
            "TODO: port guard `encode_result_model_invalid_error` from emel.cpp/src/emel/text/encoders/spm/guards.hpp"
        )
    }
    fn encode_result_ok(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/guards.hpp::encode_result_ok
        todo!(
            "TODO: port guard `encode_result_ok` from emel.cpp/src/emel/text/encoders/spm/guards.hpp"
        )
    }
    fn encode_result_unclassified_error_code(
        &self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/guards.hpp::encode_result_unclassified_error_code
        todo!(
            "TODO: port guard `encode_result_unclassified_error_code` from emel.cpp/src/emel/text/encoders/spm/guards.hpp"
        )
    }
    fn ensure_last_error_from_emit_result_decision(
        &mut self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/actions.hpp::ensure_last_error
        todo!(
            "TODO: port action `ensure_last_error` from emel.cpp/src/emel/text/encoders/spm/actions.hpp"
        )
    }
    fn ensure_last_error_from_encode_emit_input_decision(
        &mut self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/actions.hpp::ensure_last_error
        todo!(
            "TODO: port action `ensure_last_error` from emel.cpp/src/emel/text/encoders/spm/actions.hpp"
        )
    }
    fn ensure_last_error_from_encode_merge_result_decision(
        &mut self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/actions.hpp::ensure_last_error
        todo!(
            "TODO: port action `ensure_last_error` from emel.cpp/src/emel/text/encoders/spm/actions.hpp"
        )
    }
    fn ensure_last_error_from_encode_precheck_decision(
        &mut self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/actions.hpp::ensure_last_error
        todo!(
            "TODO: port action `ensure_last_error` from emel.cpp/src/emel/text/encoders/spm/actions.hpp"
        )
    }
    fn ensure_last_error_from_encode_prepare_result_decision(
        &mut self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/actions.hpp::ensure_last_error
        todo!(
            "TODO: port action `ensure_last_error` from emel.cpp/src/emel/text/encoders/spm/actions.hpp"
        )
    }
    fn ensure_last_error_from_encode_result_decision(
        &mut self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/actions.hpp::ensure_last_error
        todo!(
            "TODO: port action `ensure_last_error` from emel.cpp/src/emel/text/encoders/spm/actions.hpp"
        )
    }
    fn ensure_last_error_from_table_policy_decision(
        &mut self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/actions.hpp::ensure_last_error
        todo!(
            "TODO: port action `ensure_last_error` from emel.cpp/src/emel/text/encoders/spm/actions.hpp"
        )
    }
    fn ensure_last_error_from_table_sync_result_decision(
        &mut self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/actions.hpp::ensure_last_error
        todo!(
            "TODO: port action `ensure_last_error` from emel.cpp/src/emel/text/encoders/spm/actions.hpp"
        )
    }
    fn invalid_encode(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/guards.hpp::invalid_encode
        todo!(
            "TODO: port guard `invalid_encode` from emel.cpp/src/emel/text/encoders/spm/guards.hpp"
        )
    }
    fn mark_done_from_encode_precheck_decision(
        &mut self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/actions.hpp::mark_done
        todo!("TODO: port action `mark_done` from emel.cpp/src/emel/text/encoders/spm/actions.hpp")
    }
    fn mark_done_from_encode_result_decision(
        &mut self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/actions.hpp::mark_done
        todo!("TODO: port action `mark_done` from emel.cpp/src/emel/text/encoders/spm/actions.hpp")
    }
    fn merge_result_backend_error(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/guards.hpp::merge_result_backend_error
        todo!(
            "TODO: port guard `merge_result_backend_error` from emel.cpp/src/emel/text/encoders/spm/guards.hpp"
        )
    }
    fn merge_result_invalid_argument_error(
        &self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/guards.hpp::merge_result_invalid_argument_error
        todo!(
            "TODO: port guard `merge_result_invalid_argument_error` from emel.cpp/src/emel/text/encoders/spm/guards.hpp"
        )
    }
    fn merge_result_model_invalid_error(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/guards.hpp::merge_result_model_invalid_error
        todo!(
            "TODO: port guard `merge_result_model_invalid_error` from emel.cpp/src/emel/text/encoders/spm/guards.hpp"
        )
    }
    fn merge_result_ok(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/guards.hpp::merge_result_ok
        todo!(
            "TODO: port guard `merge_result_ok` from emel.cpp/src/emel/text/encoders/spm/guards.hpp"
        )
    }
    fn merge_result_unclassified_error_code(
        &self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/guards.hpp::merge_result_unclassified_error_code
        todo!(
            "TODO: port guard `merge_result_unclassified_error_code` from emel.cpp/src/emel/text/encoders/spm/guards.hpp"
        )
    }
    fn merge_symbol_capacity_exceeded(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/guards.hpp::merge_symbol_capacity_exceeded
        todo!(
            "TODO: port guard `merge_symbol_capacity_exceeded` from emel.cpp/src/emel/text/encoders/spm/guards.hpp"
        )
    }
    fn merge_symbol_capacity_within_limit(
        &self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/guards.hpp::merge_symbol_capacity_within_limit
        todo!(
            "TODO: port guard `merge_symbol_capacity_within_limit` from emel.cpp/src/emel/text/encoders/spm/guards.hpp"
        )
    }
    fn on_unexpected_events_encoding_done(
        &mut self,
        _event: &EventsEncodingDone,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/encoders/spm/actions.hpp"
        )
    }
    fn on_unexpected_events_encoding_error(
        &mut self,
        _event: &EventsEncodingError,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/encoders/spm/actions.hpp"
        )
    }
    fn on_unexpected_runtime_encode_runtime(
        &mut self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/encoders/spm/actions.hpp"
        )
    }
    fn on_unexpected_unexp_wild(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/encoders/spm/actions.hpp"
        )
    }
    fn prepare_result_backend_error(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/guards.hpp::prepare_result_backend_error
        todo!(
            "TODO: port guard `prepare_result_backend_error` from emel.cpp/src/emel/text/encoders/spm/guards.hpp"
        )
    }
    fn prepare_result_invalid_argument_error(
        &self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/guards.hpp::prepare_result_invalid_argument_error
        todo!(
            "TODO: port guard `prepare_result_invalid_argument_error` from emel.cpp/src/emel/text/encoders/spm/guards.hpp"
        )
    }
    fn prepare_result_model_invalid_error(
        &self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/guards.hpp::prepare_result_model_invalid_error
        todo!(
            "TODO: port guard `prepare_result_model_invalid_error` from emel.cpp/src/emel/text/encoders/spm/guards.hpp"
        )
    }
    fn prepare_result_ok(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/guards.hpp::prepare_result_ok
        todo!(
            "TODO: port guard `prepare_result_ok` from emel.cpp/src/emel/text/encoders/spm/guards.hpp"
        )
    }
    fn prepare_result_unclassified_error_code(
        &self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/guards.hpp::prepare_result_unclassified_error_code
        todo!(
            "TODO: port guard `prepare_result_unclassified_error_code` from emel.cpp/src/emel/text/encoders/spm/guards.hpp"
        )
    }
    fn reject_invalid_encode_from_encode_merge_input_capacity_decision(
        &mut self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/actions.hpp::reject_invalid_encode
        todo!(
            "TODO: port action `reject_invalid_encode` from emel.cpp/src/emel/text/encoders/spm/actions.hpp"
        )
    }
    fn reject_invalid_encode_from_encode_validity_decision(
        &mut self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/actions.hpp::reject_invalid_encode
        todo!(
            "TODO: port action `reject_invalid_encode` from emel.cpp/src/emel/text/encoders/spm/actions.hpp"
        )
    }
    fn reject_invalid_encode_from_encode_vocab_sync_decision(
        &mut self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/actions.hpp::reject_invalid_encode
        todo!(
            "TODO: port action `reject_invalid_encode` from emel.cpp/src/emel/text/encoders/spm/actions.hpp"
        )
    }
    fn run_encode(&mut self, _event: &RuntimeEncodeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/actions.hpp::run_encode
        todo!("TODO: port action `run_encode` from emel.cpp/src/emel/text/encoders/spm/actions.hpp")
    }
    fn run_merge(&mut self, _event: &RuntimeEncodeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/actions.hpp::run_merge
        todo!("TODO: port action `run_merge` from emel.cpp/src/emel/text/encoders/spm/actions.hpp")
    }
    fn run_prepare(&mut self, _event: &RuntimeEncodeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/actions.hpp::run_prepare
        todo!(
            "TODO: port action `run_prepare` from emel.cpp/src/emel/text/encoders/spm/actions.hpp"
        )
    }
    fn set_emit_result_empty(&mut self, _event: &RuntimeEncodeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/actions.hpp::set_emit_result_empty
        todo!(
            "TODO: port action `set_emit_result_empty` from emel.cpp/src/emel/text/encoders/spm/actions.hpp"
        )
    }
    fn symbols_absent(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/guards.hpp::symbols_absent
        todo!(
            "TODO: port guard `symbols_absent` from emel.cpp/src/emel/text/encoders/spm/guards.hpp"
        )
    }
    fn symbols_present(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/guards.hpp::symbols_present
        todo!(
            "TODO: port guard `symbols_present` from emel.cpp/src/emel/text/encoders/spm/guards.hpp"
        )
    }
    fn sync_tables(&mut self, _event: &RuntimeEncodeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/actions.hpp::sync_tables
        todo!(
            "TODO: port action `sync_tables` from emel.cpp/src/emel/text/encoders/spm/actions.hpp"
        )
    }
    fn table_sync_backend_error(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/guards.hpp::table_sync_backend_error
        todo!(
            "TODO: port guard `table_sync_backend_error` from emel.cpp/src/emel/text/encoders/spm/guards.hpp"
        )
    }
    fn table_sync_invalid_argument_error(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/guards.hpp::table_sync_invalid_argument_error
        todo!(
            "TODO: port guard `table_sync_invalid_argument_error` from emel.cpp/src/emel/text/encoders/spm/guards.hpp"
        )
    }
    fn table_sync_model_invalid_error(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/guards.hpp::table_sync_model_invalid_error
        todo!(
            "TODO: port guard `table_sync_model_invalid_error` from emel.cpp/src/emel/text/encoders/spm/guards.hpp"
        )
    }
    fn table_sync_ok(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/guards.hpp::table_sync_ok
        todo!(
            "TODO: port guard `table_sync_ok` from emel.cpp/src/emel/text/encoders/spm/guards.hpp"
        )
    }
    fn table_sync_unclassified_error_code(
        &self,
        _event: &RuntimeEncodeRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/guards.hpp::table_sync_unclassified_error_code
        todo!(
            "TODO: port guard `table_sync_unclassified_error_code` from emel.cpp/src/emel/text/encoders/spm/guards.hpp"
        )
    }
    fn tables_missing(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/guards.hpp::tables_missing
        todo!(
            "TODO: port guard `tables_missing` from emel.cpp/src/emel/text/encoders/spm/guards.hpp"
        )
    }
    fn tables_ready(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/guards.hpp::tables_ready
        todo!("TODO: port guard `tables_ready` from emel.cpp/src/emel/text/encoders/spm/guards.hpp")
    }
    fn text_empty(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/guards.hpp::text_empty
        todo!("TODO: port guard `text_empty` from emel.cpp/src/emel/text/encoders/spm/guards.hpp")
    }
    fn text_non_empty(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/guards.hpp::text_non_empty
        todo!(
            "TODO: port guard `text_non_empty` from emel.cpp/src/emel/text/encoders/spm/guards.hpp"
        )
    }
    fn valid_encode(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/guards.hpp::valid_encode
        todo!("TODO: port guard `valid_encode` from emel.cpp/src/emel/text/encoders/spm/guards.hpp")
    }
    fn vocab_changed(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/guards.hpp::vocab_changed
        todo!(
            "TODO: port guard `vocab_changed` from emel.cpp/src/emel/text/encoders/spm/guards.hpp"
        )
    }
    fn vocab_unchanged(&self, _event: &RuntimeEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/spm/guards.hpp::vocab_unchanged
        todo!(
            "TODO: port guard `vocab_unchanged` from emel.cpp/src/emel/text/encoders/spm/guards.hpp"
        )
    }
}
