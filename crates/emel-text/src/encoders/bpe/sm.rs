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

// --- machine TextEncodersBpe from emel.cpp/src/emel/text/encoders/bpe/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventEncodeRuntime;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventsEncodingDone;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventsEncodingError;

sml! {
    TextEncodersBpe {
        "encode_validity_decision"_s <= *"initialized"_s + event<EventEncodeRuntime>,
        "encode_validity_decision"_s <= "done"_s + event<EventEncodeRuntime>,
        "encode_validity_decision"_s <= "errored"_s + event<EventEncodeRuntime>,
        "encode_validity_decision"_s <= "unexpected"_s + event<EventEncodeRuntime>,
        "encode_vocab_sync_decision"_s <= "encode_validity_decision"_s + completion<EventEncodeRuntime> [valid_encode],
        "errored"_s <= "encode_validity_decision"_s + completion<EventEncodeRuntime> [invalid_encode] / reject_invalid_encode_from_encode_validity_decision,
        "errored"_s <= "encode_validity_decision"_s + completion<EventEncodeRuntime> / reject_invalid_encode_from_encode_validity_decision,
        "encode_precheck_decision"_s <= "encode_vocab_sync_decision"_s + completion<EventEncodeRuntime> [vocab_changed] / begin_encode_sync_vocab,
        "encode_precheck_decision"_s <= "encode_vocab_sync_decision"_s + completion<EventEncodeRuntime> [vocab_unchanged] / begin_encode,
        "errored"_s <= "encode_vocab_sync_decision"_s + completion<EventEncodeRuntime> / reject_invalid_encode_from_encode_vocab_sync_decision,
        "done"_s <= "encode_precheck_decision"_s + completion<EventEncodeRuntime> [text_empty] / mark_done_from_encode_precheck_decision,
        "encode_input_policy_decision"_s <= "encode_precheck_decision"_s + completion<EventEncodeRuntime> [text_non_empty],
        "errored"_s <= "encode_precheck_decision"_s + completion<EventEncodeRuntime> / ensure_last_error_from_encode_precheck_decision,
        "encode_table_prepare"_s <= "encode_input_policy_decision"_s + completion<EventEncodeRuntime> [preprocessed] / prepare_tables,
        "errored"_s <= "encode_input_policy_decision"_s + completion<EventEncodeRuntime> [not_preprocessed] / reject_invalid_encode_from_encode_input_policy_decision,
        "errored"_s <= "encode_input_policy_decision"_s + completion<EventEncodeRuntime> / reject_invalid_encode_from_encode_input_policy_decision,
        "encode_path_decision"_s <= "encode_table_prepare"_s + completion<EventEncodeRuntime> [table_prepare_ok],
        "errored"_s <= "encode_table_prepare"_s + completion<EventEncodeRuntime> [table_prepare_backend_error] / ensure_last_error_from_encode_table_prepare,
        "errored"_s <= "encode_table_prepare"_s + completion<EventEncodeRuntime> [table_prepare_invalid_argument_error] / ensure_last_error_from_encode_table_prepare,
        "errored"_s <= "encode_table_prepare"_s + completion<EventEncodeRuntime> [table_prepare_model_invalid_error] / ensure_last_error_from_encode_table_prepare,
        "errored"_s <= "encode_table_prepare"_s + completion<EventEncodeRuntime> [table_prepare_unclassified_error_code] / ensure_last_error_from_encode_table_prepare,
        "encode_direct_word_policy_decision"_s <= "encode_path_decision"_s + completion<EventEncodeRuntime> [ignore_merges_enabled],
        "encode_exec"_s <= "encode_path_decision"_s + completion<EventEncodeRuntime>,
        "encode_result_decision"_s <= "encode_direct_word_policy_decision"_s + completion<EventEncodeRuntime> [direct_word_token_available] / run_encode_ignore_merges,
        "encode_merge_input_capacity_decision"_s <= "encode_direct_word_policy_decision"_s + completion<EventEncodeRuntime>,
        "encode_exec"_s <= "encode_merge_input_capacity_decision"_s + completion<EventEncodeRuntime> [merge_symbol_capacity_within_limit],
        "errored"_s <= "encode_merge_input_capacity_decision"_s + completion<EventEncodeRuntime> [merge_symbol_capacity_exceeded] / reject_invalid_encode_from_encode_merge_input_capacity_decision,
        "errored"_s <= "encode_merge_input_capacity_decision"_s + completion<EventEncodeRuntime> / reject_invalid_encode_from_encode_merge_input_capacity_decision,
        "encode_result_decision"_s <= "encode_exec"_s + completion<EventEncodeRuntime> / run_encode_merge_path,
        "done"_s <= "encode_result_decision"_s + completion<EventEncodeRuntime> [encode_result_ok] / mark_done_from_encode_result_decision,
        "errored"_s <= "encode_result_decision"_s + completion<EventEncodeRuntime> [encode_result_invalid_argument_error] / ensure_last_error_from_encode_result_decision,
        "errored"_s <= "encode_result_decision"_s + completion<EventEncodeRuntime> [encode_result_backend_error] / ensure_last_error_from_encode_result_decision,
        "errored"_s <= "encode_result_decision"_s + completion<EventEncodeRuntime> [encode_result_model_invalid_error] / ensure_last_error_from_encode_result_decision,
        "errored"_s <= "encode_result_decision"_s + completion<EventEncodeRuntime> [encode_result_unclassified_error_code] / ensure_last_error_from_encode_result_decision,
        "unexpected"_s <= "encode_validity_decision"_s + event<EventEncodeRuntime> / on_unexpected_event_encode_runtime,
        "unexpected"_s <= "encode_vocab_sync_decision"_s + event<EventEncodeRuntime> / on_unexpected_event_encode_runtime,
        "unexpected"_s <= "encode_precheck_decision"_s + event<EventEncodeRuntime> / on_unexpected_event_encode_runtime,
        "unexpected"_s <= "encode_input_policy_decision"_s + event<EventEncodeRuntime> / on_unexpected_event_encode_runtime,
        "unexpected"_s <= "encode_table_prepare"_s + event<EventEncodeRuntime> / on_unexpected_event_encode_runtime,
        "unexpected"_s <= "encode_path_decision"_s + event<EventEncodeRuntime> / on_unexpected_event_encode_runtime,
        "unexpected"_s <= "encode_direct_word_policy_decision"_s + event<EventEncodeRuntime> / on_unexpected_event_encode_runtime,
        "unexpected"_s <= "encode_merge_input_capacity_decision"_s + event<EventEncodeRuntime> / on_unexpected_event_encode_runtime,
        "unexpected"_s <= "encode_exec"_s + event<EventEncodeRuntime> / on_unexpected_event_encode_runtime,
        "unexpected"_s <= "encode_result_decision"_s + event<EventEncodeRuntime> / on_unexpected_event_encode_runtime,
        "unexpected"_s <= "initialized"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "initialized"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "encode_validity_decision"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "encode_validity_decision"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "encode_vocab_sync_decision"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "encode_vocab_sync_decision"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "encode_precheck_decision"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "encode_precheck_decision"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "encode_input_policy_decision"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "encode_input_policy_decision"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "encode_table_prepare"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "encode_table_prepare"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "encode_path_decision"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "encode_path_decision"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "encode_direct_word_policy_decision"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "encode_direct_word_policy_decision"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "encode_merge_input_capacity_decision"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "encode_merge_input_capacity_decision"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
        "unexpected"_s <= "encode_exec"_s + event<EventsEncodingDone> / on_unexpected_events_encoding_done,
        "unexpected"_s <= "encode_exec"_s + event<EventsEncodingError> / on_unexpected_events_encoding_error,
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
        "unexpected"_s <= "encode_input_policy_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_table_prepare"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_path_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_direct_word_policy_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_merge_input_capacity_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_exec"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "encode_result_decision"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "done"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "errored"_s + unexpected_event<_> / on_unexpected_unexp_wild,
        "unexpected"_s <= "unexpected"_s + unexpected_event<_> / on_unexpected_unexp_wild,
    }
}

/// Context for `TextEncodersBpe` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct TextEncodersBpeContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl TextEncodersBpeStateMachineContext for TextEncodersBpeContext {
    fn begin_encode(&mut self, _event: &EventEncodeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/bpe/actions.hpp::begin_encode
        todo!(
            "TODO: port action `begin_encode` from emel.cpp/src/emel/text/encoders/bpe/actions.hpp"
        )
    }
    fn begin_encode_sync_vocab(&mut self, _event: &EventEncodeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/bpe/actions.hpp::begin_encode_sync_vocab
        todo!(
            "TODO: port action `begin_encode_sync_vocab` from emel.cpp/src/emel/text/encoders/bpe/actions.hpp"
        )
    }
    fn direct_word_token_available(&self, _event: &EventEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/bpe/guards.hpp::direct_word_token_available
        todo!(
            "TODO: port guard `direct_word_token_available` from emel.cpp/src/emel/text/encoders/bpe/guards.hpp"
        )
    }
    fn encode_result_backend_error(&self, _event: &EventEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/bpe/guards.hpp::encode_result_backend_error
        todo!(
            "TODO: port guard `encode_result_backend_error` from emel.cpp/src/emel/text/encoders/bpe/guards.hpp"
        )
    }
    fn encode_result_invalid_argument_error(
        &self,
        _event: &EventEncodeRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/bpe/guards.hpp::encode_result_invalid_argument_error
        todo!(
            "TODO: port guard `encode_result_invalid_argument_error` from emel.cpp/src/emel/text/encoders/bpe/guards.hpp"
        )
    }
    fn encode_result_model_invalid_error(&self, _event: &EventEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/bpe/guards.hpp::encode_result_model_invalid_error
        todo!(
            "TODO: port guard `encode_result_model_invalid_error` from emel.cpp/src/emel/text/encoders/bpe/guards.hpp"
        )
    }
    fn encode_result_ok(&self, _event: &EventEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/bpe/guards.hpp::encode_result_ok
        todo!(
            "TODO: port guard `encode_result_ok` from emel.cpp/src/emel/text/encoders/bpe/guards.hpp"
        )
    }
    fn encode_result_unclassified_error_code(
        &self,
        _event: &EventEncodeRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/bpe/guards.hpp::encode_result_unclassified_error_code
        todo!(
            "TODO: port guard `encode_result_unclassified_error_code` from emel.cpp/src/emel/text/encoders/bpe/guards.hpp"
        )
    }
    fn ensure_last_error_from_encode_precheck_decision(
        &mut self,
        _event: &EventEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/bpe/actions.hpp::ensure_last_error
        todo!(
            "TODO: port action `ensure_last_error` from emel.cpp/src/emel/text/encoders/bpe/actions.hpp"
        )
    }
    fn ensure_last_error_from_encode_result_decision(
        &mut self,
        _event: &EventEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/bpe/actions.hpp::ensure_last_error
        todo!(
            "TODO: port action `ensure_last_error` from emel.cpp/src/emel/text/encoders/bpe/actions.hpp"
        )
    }
    fn ensure_last_error_from_encode_table_prepare(
        &mut self,
        _event: &EventEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/bpe/actions.hpp::ensure_last_error
        todo!(
            "TODO: port action `ensure_last_error` from emel.cpp/src/emel/text/encoders/bpe/actions.hpp"
        )
    }
    fn ignore_merges_enabled(&self, _event: &EventEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/bpe/guards.hpp::ignore_merges_enabled
        todo!(
            "TODO: port guard `ignore_merges_enabled` from emel.cpp/src/emel/text/encoders/bpe/guards.hpp"
        )
    }
    fn invalid_encode(&self, _event: &EventEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/bpe/guards.hpp::invalid_encode
        todo!(
            "TODO: port guard `invalid_encode` from emel.cpp/src/emel/text/encoders/bpe/guards.hpp"
        )
    }
    fn mark_done_from_encode_precheck_decision(
        &mut self,
        _event: &EventEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/bpe/actions.hpp::mark_done
        todo!("TODO: port action `mark_done` from emel.cpp/src/emel/text/encoders/bpe/actions.hpp")
    }
    fn mark_done_from_encode_result_decision(
        &mut self,
        _event: &EventEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/bpe/actions.hpp::mark_done
        todo!("TODO: port action `mark_done` from emel.cpp/src/emel/text/encoders/bpe/actions.hpp")
    }
    fn merge_symbol_capacity_exceeded(&self, _event: &EventEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/bpe/guards.hpp::merge_symbol_capacity_exceeded
        todo!(
            "TODO: port guard `merge_symbol_capacity_exceeded` from emel.cpp/src/emel/text/encoders/bpe/guards.hpp"
        )
    }
    fn merge_symbol_capacity_within_limit(&self, _event: &EventEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/bpe/guards.hpp::merge_symbol_capacity_within_limit
        todo!(
            "TODO: port guard `merge_symbol_capacity_within_limit` from emel.cpp/src/emel/text/encoders/bpe/guards.hpp"
        )
    }
    fn not_preprocessed(&self, _event: &EventEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/bpe/guards.hpp::not_preprocessed
        todo!(
            "TODO: port guard `not_preprocessed` from emel.cpp/src/emel/text/encoders/bpe/guards.hpp"
        )
    }
    fn on_unexpected_event_encode_runtime(
        &mut self,
        _event: &EventEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/bpe/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/encoders/bpe/actions.hpp"
        )
    }
    fn on_unexpected_events_encoding_done(
        &mut self,
        _event: &EventsEncodingDone,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/bpe/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/encoders/bpe/actions.hpp"
        )
    }
    fn on_unexpected_events_encoding_error(
        &mut self,
        _event: &EventsEncodingError,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/bpe/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/encoders/bpe/actions.hpp"
        )
    }
    fn on_unexpected_unexp_wild(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/bpe/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/encoders/bpe/actions.hpp"
        )
    }
    fn prepare_tables(&mut self, _event: &EventEncodeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/bpe/actions.hpp::prepare_tables
        todo!(
            "TODO: port action `prepare_tables` from emel.cpp/src/emel/text/encoders/bpe/actions.hpp"
        )
    }
    fn preprocessed(&self, _event: &EventEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/bpe/guards.hpp::preprocessed
        todo!("TODO: port guard `preprocessed` from emel.cpp/src/emel/text/encoders/bpe/guards.hpp")
    }
    fn reject_invalid_encode_from_encode_input_policy_decision(
        &mut self,
        _event: &EventEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/bpe/actions.hpp::reject_invalid_encode
        todo!(
            "TODO: port action `reject_invalid_encode` from emel.cpp/src/emel/text/encoders/bpe/actions.hpp"
        )
    }
    fn reject_invalid_encode_from_encode_merge_input_capacity_decision(
        &mut self,
        _event: &EventEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/bpe/actions.hpp::reject_invalid_encode
        todo!(
            "TODO: port action `reject_invalid_encode` from emel.cpp/src/emel/text/encoders/bpe/actions.hpp"
        )
    }
    fn reject_invalid_encode_from_encode_validity_decision(
        &mut self,
        _event: &EventEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/bpe/actions.hpp::reject_invalid_encode
        todo!(
            "TODO: port action `reject_invalid_encode` from emel.cpp/src/emel/text/encoders/bpe/actions.hpp"
        )
    }
    fn reject_invalid_encode_from_encode_vocab_sync_decision(
        &mut self,
        _event: &EventEncodeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/bpe/actions.hpp::reject_invalid_encode
        todo!(
            "TODO: port action `reject_invalid_encode` from emel.cpp/src/emel/text/encoders/bpe/actions.hpp"
        )
    }
    fn run_encode_ignore_merges(&mut self, _event: &EventEncodeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/bpe/actions.hpp::run_encode_ignore_merges
        todo!(
            "TODO: port action `run_encode_ignore_merges` from emel.cpp/src/emel/text/encoders/bpe/actions.hpp"
        )
    }
    fn run_encode_merge_path(&mut self, _event: &EventEncodeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/bpe/actions.hpp::run_encode_merge_path
        todo!(
            "TODO: port action `run_encode_merge_path` from emel.cpp/src/emel/text/encoders/bpe/actions.hpp"
        )
    }
    fn table_prepare_backend_error(&self, _event: &EventEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/bpe/guards.hpp::table_prepare_backend_error
        todo!(
            "TODO: port guard `table_prepare_backend_error` from emel.cpp/src/emel/text/encoders/bpe/guards.hpp"
        )
    }
    fn table_prepare_invalid_argument_error(
        &self,
        _event: &EventEncodeRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/bpe/guards.hpp::table_prepare_invalid_argument_error
        todo!(
            "TODO: port guard `table_prepare_invalid_argument_error` from emel.cpp/src/emel/text/encoders/bpe/guards.hpp"
        )
    }
    fn table_prepare_model_invalid_error(&self, _event: &EventEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/bpe/guards.hpp::table_prepare_model_invalid_error
        todo!(
            "TODO: port guard `table_prepare_model_invalid_error` from emel.cpp/src/emel/text/encoders/bpe/guards.hpp"
        )
    }
    fn table_prepare_ok(&self, _event: &EventEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/bpe/guards.hpp::table_prepare_ok
        todo!(
            "TODO: port guard `table_prepare_ok` from emel.cpp/src/emel/text/encoders/bpe/guards.hpp"
        )
    }
    fn table_prepare_unclassified_error_code(
        &self,
        _event: &EventEncodeRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/bpe/guards.hpp::table_prepare_unclassified_error_code
        todo!(
            "TODO: port guard `table_prepare_unclassified_error_code` from emel.cpp/src/emel/text/encoders/bpe/guards.hpp"
        )
    }
    fn text_empty(&self, _event: &EventEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/bpe/guards.hpp::text_empty
        todo!("TODO: port guard `text_empty` from emel.cpp/src/emel/text/encoders/bpe/guards.hpp")
    }
    fn text_non_empty(&self, _event: &EventEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/bpe/guards.hpp::text_non_empty
        todo!(
            "TODO: port guard `text_non_empty` from emel.cpp/src/emel/text/encoders/bpe/guards.hpp"
        )
    }
    fn valid_encode(&self, _event: &EventEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/bpe/guards.hpp::valid_encode
        todo!("TODO: port guard `valid_encode` from emel.cpp/src/emel/text/encoders/bpe/guards.hpp")
    }
    fn vocab_changed(&self, _event: &EventEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/bpe/guards.hpp::vocab_changed
        todo!(
            "TODO: port guard `vocab_changed` from emel.cpp/src/emel/text/encoders/bpe/guards.hpp"
        )
    }
    fn vocab_unchanged(&self, _event: &EventEncodeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/encoders/bpe/guards.hpp::vocab_unchanged
        todo!(
            "TODO: port guard `vocab_unchanged` from emel.cpp/src/emel/text/encoders/bpe/guards.hpp"
        )
    }
}
