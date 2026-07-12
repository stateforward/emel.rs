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

// --- machine TextTokenizerPreprocessorPlamo2 from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventPreprocessRuntime;

sml! {
    TextTokenizerPreprocessorPlamo2 {
        "request_buffer_decision"_s <= *"idle"_s + event<EventPreprocessRuntime>,
        "request_buffer_decision"_s <= "done"_s + event<EventPreprocessRuntime>,
        "request_buffer_decision"_s <= "errored"_s + event<EventPreprocessRuntime>,
        "request_buffer_decision"_s <= "unexpected"_s + event<EventPreprocessRuntime>,
        "request_capacity_nonzero_decision"_s <= "request_buffer_decision"_s + completion<EventPreprocessRuntime> [fragments_buffer_present],
        "errored"_s <= "request_buffer_decision"_s + completion<EventPreprocessRuntime> [fragments_buffer_missing] / reject_invalid_from_request_buffer_decision,
        "errored"_s <= "request_buffer_decision"_s + completion<EventPreprocessRuntime> / reject_invalid_from_request_buffer_decision,
        "request_capacity_limit_decision"_s <= "request_capacity_nonzero_decision"_s + completion<EventPreprocessRuntime> [fragments_capacity_nonzero],
        "errored"_s <= "request_capacity_nonzero_decision"_s + completion<EventPreprocessRuntime> [fragments_capacity_zero] / reject_invalid_from_request_capacity_nonzero_decision,
        "errored"_s <= "request_capacity_nonzero_decision"_s + completion<EventPreprocessRuntime> / reject_invalid_from_request_capacity_nonzero_decision,
        "preparing"_s <= "request_capacity_limit_decision"_s + completion<EventPreprocessRuntime> [fragments_capacity_within_limit] / begin_preprocess,
        "errored"_s <= "request_capacity_limit_decision"_s + completion<EventPreprocessRuntime> [fragments_capacity_exceeds_limit] / reject_invalid_from_request_capacity_limit_decision,
        "errored"_s <= "request_capacity_limit_decision"_s + completion<EventPreprocessRuntime> / reject_invalid_from_request_capacity_limit_decision,
        "build_specials_decision"_s <= "preparing"_s + completion<EventPreprocessRuntime> / build_specials,
        "partition_specials_decision"_s <= "build_specials_decision"_s + completion<EventPreprocessRuntime> [build_specials_ok],
        "errored"_s <= "build_specials_decision"_s + completion<EventPreprocessRuntime> [build_specials_invalid_request_error] / ensure_last_error_from_build_specials_decision,
        "errored"_s <= "build_specials_decision"_s + completion<EventPreprocessRuntime> [build_specials_backend_error] / ensure_last_error_from_build_specials_decision,
        "errored"_s <= "build_specials_decision"_s + completion<EventPreprocessRuntime> [build_specials_unknown_error] / ensure_last_error_from_build_specials_decision,
        "partitioning_no_specials_input_decision"_s <= "partition_specials_decision"_s + completion<EventPreprocessRuntime> [no_specials],
        "partition_parse_special_decision"_s <= "partition_specials_decision"_s + completion<EventPreprocessRuntime> [has_specials],
        "errored"_s <= "partition_specials_decision"_s + completion<EventPreprocessRuntime> / ensure_last_error_from_partition_specials_decision,
        "partitioning_non_bpe_parse_input_decision"_s <= "partition_parse_special_decision"_s + completion<EventPreprocessRuntime> [parse_special_enabled],
        "partitioning_non_bpe_skip_input_decision"_s <= "partition_parse_special_decision"_s + completion<EventPreprocessRuntime> [parse_special_disabled],
        "errored"_s <= "partition_parse_special_decision"_s + completion<EventPreprocessRuntime> / ensure_last_error_from_partition_parse_special_decision,
        "partition_decision"_s <= "partitioning_no_specials_input_decision"_s + completion<EventPreprocessRuntime> [request_text_empty] / set_empty_partition_result_from_partitioning_no_specials_input_decision,
        "partitioning_no_specials"_s <= "partitioning_no_specials_input_decision"_s + completion<EventPreprocessRuntime> [request_text_nonempty],
        "errored"_s <= "partitioning_no_specials_input_decision"_s + completion<EventPreprocessRuntime> / ensure_last_error_from_partitioning_no_specials_input_decision,
        "partition_decision"_s <= "partitioning_non_bpe_parse_input_decision"_s + completion<EventPreprocessRuntime> [request_text_empty] / set_empty_partition_result_from_partitioning_non_bpe_parse_input_decision,
        "partitioning_non_bpe_parse_special"_s <= "partitioning_non_bpe_parse_input_decision"_s + completion<EventPreprocessRuntime> [request_text_nonempty],
        "errored"_s <= "partitioning_non_bpe_parse_input_decision"_s + completion<EventPreprocessRuntime> / ensure_last_error_from_partitioning_non_bpe_parse_input_decision,
        "partition_decision"_s <= "partitioning_non_bpe_skip_input_decision"_s + completion<EventPreprocessRuntime> [request_text_empty] / set_empty_partition_result_from_partitioning_non_bpe_skip_input_decision,
        "partitioning_non_bpe_skip_special"_s <= "partitioning_non_bpe_skip_input_decision"_s + completion<EventPreprocessRuntime> [request_text_nonempty],
        "errored"_s <= "partitioning_non_bpe_skip_input_decision"_s + completion<EventPreprocessRuntime> / ensure_last_error_from_partitioning_non_bpe_skip_input_decision,
        "partition_decision"_s <= "partitioning_no_specials"_s + completion<EventPreprocessRuntime> / partition_no_specials,
        "partition_decision"_s <= "partitioning_non_bpe_parse_special"_s + completion<EventPreprocessRuntime> / partition_non_bpe_parse_special,
        "partition_decision"_s <= "partitioning_non_bpe_skip_special"_s + completion<EventPreprocessRuntime> / partition_non_bpe_skip_special,
        "done"_s <= "partition_decision"_s + completion<EventPreprocessRuntime> [partition_ok] / mark_done,
        "errored"_s <= "partition_decision"_s + completion<EventPreprocessRuntime> [partition_invalid_request_error] / ensure_last_error_from_partition_decision,
        "errored"_s <= "partition_decision"_s + completion<EventPreprocessRuntime> [partition_backend_error] / ensure_last_error_from_partition_decision,
        "errored"_s <= "partition_decision"_s + completion<EventPreprocessRuntime> [partition_unknown_error] / ensure_last_error_from_partition_decision,
        "unexpected"_s <= "idle"_s + unexpected_event<_> / on_unexpected_from_idle,
        "unexpected"_s <= "request_buffer_decision"_s + unexpected_event<_> / on_unexpected_from_request_buffer_decision,
        "unexpected"_s <= "request_capacity_nonzero_decision"_s + unexpected_event<_> / on_unexpected_from_request_capacity_nonzero_decision,
        "unexpected"_s <= "request_capacity_limit_decision"_s + unexpected_event<_> / on_unexpected_from_request_capacity_limit_decision,
        "unexpected"_s <= "preparing"_s + unexpected_event<_> / on_unexpected_from_preparing,
        "unexpected"_s <= "build_specials_decision"_s + unexpected_event<_> / on_unexpected_from_build_specials_decision,
        "unexpected"_s <= "partition_specials_decision"_s + unexpected_event<_> / on_unexpected_from_partition_specials_decision,
        "unexpected"_s <= "partition_parse_special_decision"_s + unexpected_event<_> / on_unexpected_from_partition_parse_special_decision,
        "unexpected"_s <= "partitioning_no_specials_input_decision"_s + unexpected_event<_> / on_unexpected_from_partitioning_no_specials_input_decision,
        "unexpected"_s <= "partitioning_non_bpe_parse_input_decision"_s + unexpected_event<_> / on_unexpected_from_partitioning_non_bpe_parse_input_decision,
        "unexpected"_s <= "partitioning_non_bpe_skip_input_decision"_s + unexpected_event<_> / on_unexpected_from_partitioning_non_bpe_skip_input_decision,
        "unexpected"_s <= "partitioning_no_specials"_s + unexpected_event<_> / on_unexpected_from_partitioning_no_specials,
        "unexpected"_s <= "partitioning_non_bpe_parse_special"_s + unexpected_event<_> / on_unexpected_from_partitioning_non_bpe_parse_special,
        "unexpected"_s <= "partitioning_non_bpe_skip_special"_s + unexpected_event<_> / on_unexpected_from_partitioning_non_bpe_skip_special,
        "unexpected"_s <= "partition_decision"_s + unexpected_event<_> / on_unexpected_from_partition_decision,
        "unexpected"_s <= "done"_s + unexpected_event<_> / on_unexpected_from_done,
        "unexpected"_s <= "errored"_s + unexpected_event<_> / on_unexpected_from_errored,
        "unexpected"_s <= "unexpected"_s + unexpected_event<_> / on_unexpected_from_unexpected,
    }
}

/// Context for `TextTokenizerPreprocessorPlamo2` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct TextTokenizerPreprocessorPlamo2Context {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl TextTokenizerPreprocessorPlamo2StateMachineContext for TextTokenizerPreprocessorPlamo2Context {
    fn begin_preprocess(&mut self, _event: &EventPreprocessRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp::begin_preprocess
        todo!(
            "TODO: port action `begin_preprocess` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp"
        )
    }
    fn build_specials(&mut self, _event: &EventPreprocessRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp::build_specials
        todo!(
            "TODO: port action `build_specials` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp"
        )
    }
    fn build_specials_backend_error(&self, _event: &EventPreprocessRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/guards.hpp::build_specials_backend_error
        todo!(
            "TODO: port guard `build_specials_backend_error` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/guards.hpp"
        )
    }
    fn build_specials_invalid_request_error(
        &self,
        _event: &EventPreprocessRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/guards.hpp::build_specials_invalid_request_error
        todo!(
            "TODO: port guard `build_specials_invalid_request_error` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/guards.hpp"
        )
    }
    fn build_specials_ok(&self, _event: &EventPreprocessRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/guards.hpp::build_specials_ok
        todo!(
            "TODO: port guard `build_specials_ok` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/guards.hpp"
        )
    }
    fn build_specials_unknown_error(&self, _event: &EventPreprocessRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/guards.hpp::build_specials_unknown_error
        todo!(
            "TODO: port guard `build_specials_unknown_error` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/guards.hpp"
        )
    }
    fn ensure_last_error_from_build_specials_decision(
        &mut self,
        _event: &EventPreprocessRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp::ensure_last_error
        todo!(
            "TODO: port action `ensure_last_error` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp"
        )
    }
    fn ensure_last_error_from_partition_decision(
        &mut self,
        _event: &EventPreprocessRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp::ensure_last_error
        todo!(
            "TODO: port action `ensure_last_error` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp"
        )
    }
    fn ensure_last_error_from_partition_parse_special_decision(
        &mut self,
        _event: &EventPreprocessRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp::ensure_last_error
        todo!(
            "TODO: port action `ensure_last_error` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp"
        )
    }
    fn ensure_last_error_from_partition_specials_decision(
        &mut self,
        _event: &EventPreprocessRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp::ensure_last_error
        todo!(
            "TODO: port action `ensure_last_error` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp"
        )
    }
    fn ensure_last_error_from_partitioning_no_specials_input_decision(
        &mut self,
        _event: &EventPreprocessRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp::ensure_last_error
        todo!(
            "TODO: port action `ensure_last_error` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp"
        )
    }
    fn ensure_last_error_from_partitioning_non_bpe_parse_input_decision(
        &mut self,
        _event: &EventPreprocessRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp::ensure_last_error
        todo!(
            "TODO: port action `ensure_last_error` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp"
        )
    }
    fn ensure_last_error_from_partitioning_non_bpe_skip_input_decision(
        &mut self,
        _event: &EventPreprocessRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp::ensure_last_error
        todo!(
            "TODO: port action `ensure_last_error` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp"
        )
    }
    fn fragments_buffer_missing(&self, _event: &EventPreprocessRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/guards.hpp::fragments_buffer_missing
        todo!(
            "TODO: port guard `fragments_buffer_missing` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/guards.hpp"
        )
    }
    fn fragments_buffer_present(&self, _event: &EventPreprocessRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/guards.hpp::fragments_buffer_present
        todo!(
            "TODO: port guard `fragments_buffer_present` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/guards.hpp"
        )
    }
    fn fragments_capacity_exceeds_limit(
        &self,
        _event: &EventPreprocessRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/guards.hpp::fragments_capacity_exceeds_limit
        todo!(
            "TODO: port guard `fragments_capacity_exceeds_limit` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/guards.hpp"
        )
    }
    fn fragments_capacity_nonzero(&self, _event: &EventPreprocessRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/guards.hpp::fragments_capacity_nonzero
        todo!(
            "TODO: port guard `fragments_capacity_nonzero` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/guards.hpp"
        )
    }
    fn fragments_capacity_within_limit(&self, _event: &EventPreprocessRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/guards.hpp::fragments_capacity_within_limit
        todo!(
            "TODO: port guard `fragments_capacity_within_limit` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/guards.hpp"
        )
    }
    fn fragments_capacity_zero(&self, _event: &EventPreprocessRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/guards.hpp::fragments_capacity_zero
        todo!(
            "TODO: port guard `fragments_capacity_zero` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/guards.hpp"
        )
    }
    fn has_specials(&self, _event: &EventPreprocessRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/guards.hpp::has_specials
        todo!(
            "TODO: port guard `has_specials` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/guards.hpp"
        )
    }
    fn mark_done(&mut self, _event: &EventPreprocessRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp::mark_done
        todo!(
            "TODO: port action `mark_done` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp"
        )
    }
    fn no_specials(&self, _event: &EventPreprocessRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/guards.hpp::no_specials
        todo!(
            "TODO: port guard `no_specials` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/guards.hpp"
        )
    }
    fn on_unexpected_from_build_specials_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp"
        )
    }
    fn on_unexpected_from_done(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp"
        )
    }
    fn on_unexpected_from_errored(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp"
        )
    }
    fn on_unexpected_from_idle(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp"
        )
    }
    fn on_unexpected_from_partition_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp"
        )
    }
    fn on_unexpected_from_partition_parse_special_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp"
        )
    }
    fn on_unexpected_from_partition_specials_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp"
        )
    }
    fn on_unexpected_from_partitioning_no_specials(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp"
        )
    }
    fn on_unexpected_from_partitioning_no_specials_input_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp"
        )
    }
    fn on_unexpected_from_partitioning_non_bpe_parse_input_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp"
        )
    }
    fn on_unexpected_from_partitioning_non_bpe_parse_special(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp"
        )
    }
    fn on_unexpected_from_partitioning_non_bpe_skip_input_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp"
        )
    }
    fn on_unexpected_from_partitioning_non_bpe_skip_special(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp"
        )
    }
    fn on_unexpected_from_preparing(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp"
        )
    }
    fn on_unexpected_from_request_buffer_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp"
        )
    }
    fn on_unexpected_from_request_capacity_limit_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp"
        )
    }
    fn on_unexpected_from_request_capacity_nonzero_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp"
        )
    }
    fn on_unexpected_from_unexpected(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp"
        )
    }
    fn parse_special_disabled(&self, _event: &EventPreprocessRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/guards.hpp::parse_special_disabled
        todo!(
            "TODO: port guard `parse_special_disabled` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/guards.hpp"
        )
    }
    fn parse_special_enabled(&self, _event: &EventPreprocessRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/guards.hpp::parse_special_enabled
        todo!(
            "TODO: port guard `parse_special_enabled` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/guards.hpp"
        )
    }
    fn partition_backend_error(&self, _event: &EventPreprocessRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/guards.hpp::partition_backend_error
        todo!(
            "TODO: port guard `partition_backend_error` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/guards.hpp"
        )
    }
    fn partition_invalid_request_error(&self, _event: &EventPreprocessRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/guards.hpp::partition_invalid_request_error
        todo!(
            "TODO: port guard `partition_invalid_request_error` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/guards.hpp"
        )
    }
    fn partition_no_specials(&mut self, _event: &EventPreprocessRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp::partition_no_specials
        todo!(
            "TODO: port action `partition_no_specials` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp"
        )
    }
    fn partition_non_bpe_parse_special(
        &mut self,
        _event: &EventPreprocessRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp::partition_non_bpe_parse_special
        todo!(
            "TODO: port action `partition_non_bpe_parse_special` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp"
        )
    }
    fn partition_non_bpe_skip_special(
        &mut self,
        _event: &EventPreprocessRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp::partition_non_bpe_skip_special
        todo!(
            "TODO: port action `partition_non_bpe_skip_special` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp"
        )
    }
    fn partition_ok(&self, _event: &EventPreprocessRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/guards.hpp::partition_ok
        todo!(
            "TODO: port guard `partition_ok` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/guards.hpp"
        )
    }
    fn partition_unknown_error(&self, _event: &EventPreprocessRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/guards.hpp::partition_unknown_error
        todo!(
            "TODO: port guard `partition_unknown_error` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/guards.hpp"
        )
    }
    fn reject_invalid_from_request_buffer_decision(
        &mut self,
        _event: &EventPreprocessRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp::reject_invalid
        todo!(
            "TODO: port action `reject_invalid` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp"
        )
    }
    fn reject_invalid_from_request_capacity_limit_decision(
        &mut self,
        _event: &EventPreprocessRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp::reject_invalid
        todo!(
            "TODO: port action `reject_invalid` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp"
        )
    }
    fn reject_invalid_from_request_capacity_nonzero_decision(
        &mut self,
        _event: &EventPreprocessRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp::reject_invalid
        todo!(
            "TODO: port action `reject_invalid` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp"
        )
    }
    fn request_text_empty(&self, _event: &EventPreprocessRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/guards.hpp::request_text_empty
        todo!(
            "TODO: port guard `request_text_empty` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/guards.hpp"
        )
    }
    fn request_text_nonempty(&self, _event: &EventPreprocessRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/guards.hpp::request_text_nonempty
        todo!(
            "TODO: port guard `request_text_nonempty` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/guards.hpp"
        )
    }
    fn set_empty_partition_result_from_partitioning_no_specials_input_decision(
        &mut self,
        _event: &EventPreprocessRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp::set_empty_partition_result
        todo!(
            "TODO: port action `set_empty_partition_result` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp"
        )
    }
    fn set_empty_partition_result_from_partitioning_non_bpe_parse_input_decision(
        &mut self,
        _event: &EventPreprocessRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp::set_empty_partition_result
        todo!(
            "TODO: port action `set_empty_partition_result` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp"
        )
    }
    fn set_empty_partition_result_from_partitioning_non_bpe_skip_input_decision(
        &mut self,
        _event: &EventPreprocessRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp::set_empty_partition_result
        todo!(
            "TODO: port action `set_empty_partition_result` from emel.cpp/src/emel/text/tokenizer/preprocessor/plamo2/actions.hpp"
        )
    }
}
