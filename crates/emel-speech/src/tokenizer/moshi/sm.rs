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

// --- machine SpeechTokenizerMoshi from emel.cpp/src/emel/speech/tokenizer/moshi/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventAdvance;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventDetokenizeRun;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventInitialize;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventReset;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventRestoreCache;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventTokenize;

sml! {
    SpeechTokenizerMoshi {
        "state_ready"_s <= *"state_uninitialized"_s + event<EventInitialize> [guard_configuration_valid] / effect_initialize_from_state_uninitialized,
        "state_uninitialized"_s <= "state_uninitialized"_s + event<EventInitialize> [guard_configuration_invalid] / effect_reject_error_invalid_configuration_from_state_uninitialized,
        "state_ready"_s <= "state_ready"_s + event<EventInitialize> / effect_reject_error_already_initialized_from_state_ready,
        "state_prepared_full"_s <= "state_prepared_full"_s + event<EventInitialize> / effect_reject_error_already_initialized_from_state_prepared_full,
        "state_prepared_generated"_s <= "state_prepared_generated"_s + event<EventInitialize> / effect_reject_error_already_initialized_from_state_prepared_generated,
        "state_prepared_tail"_s <= "state_prepared_tail"_s + event<EventInitialize> / effect_reject_error_already_initialized_from_state_prepared_tail,
        "state_ready"_s <= "state_errored"_s + event<EventInitialize> [guard_configuration_valid] / effect_initialize_from_state_errored,
        "state_errored"_s <= "state_errored"_s + event<EventInitialize> [guard_configuration_invalid] / effect_reject_error_invalid_configuration_from_state_errored,
        "state_prepared_full"_s <= "state_ready"_s + event<EventTokenize> [guard_tokenize_full] / effect_tokenize_full,
        "state_prepared_tail"_s <= "state_ready"_s + event<EventTokenize> [guard_tokenize_tail] / effect_tokenize_tail,
        "state_prepared_generated"_s <= "state_ready"_s + event<EventTokenize> [guard_tokenize_empty] / effect_tokenize_empty,
        "state_ready"_s <= "state_ready"_s + event<EventTokenize> [guard_tokenize_invalid] / effect_reject_error_request_shape_event_tokenize,
        "state_ready"_s <= "state_ready"_s + event<EventTokenize> [guard_tokenize_position_overflow] / effect_reject_error_position_overflow_event_tokenize,
        "state_prepared_full"_s <= "state_prepared_full"_s + event<EventTokenize> / effect_reject_error_phase_order_event_tokenize,
        "state_prepared_generated"_s <= "state_prepared_generated"_s + event<EventTokenize> / effect_reject_error_phase_order_event_tokenize,
        "state_prepared_tail"_s <= "state_prepared_tail"_s + event<EventTokenize> / effect_reject_error_phase_order_event_tokenize,
        "state_commit_full_zero"_s <= "state_prepared_full"_s + event<EventDetokenizeRun> [guard_detokenize_valid_replace] / effect_begin_detokenize_from_state_prepared_full,
        "state_commit_full_generated"_s <= "state_prepared_full"_s + event<EventDetokenizeRun> [guard_detokenize_valid_generated] / effect_begin_detokenize_from_state_prepared_full,
        "state_prepared_full"_s <= "state_prepared_full"_s + event<EventDetokenizeRun> [guard_detokenize_request_invalid] / effect_reject_detokenize_error_request_shape_from_state_prepared_full,
        "state_prepared_full"_s <= "state_prepared_full"_s + event<EventDetokenizeRun> [guard_position_overflow] / effect_reject_detokenize_error_position_overflow_from_state_prepared_full,
        "state_commit_full_zero"_s <= "state_prepared_tail"_s + event<EventDetokenizeRun> [guard_detokenize_valid_replace] / effect_begin_detokenize_from_state_prepared_tail,
        "state_commit_full_generated"_s <= "state_prepared_tail"_s + event<EventDetokenizeRun> [guard_detokenize_valid_generated] / effect_begin_detokenize_from_state_prepared_tail,
        "state_prepared_tail"_s <= "state_prepared_tail"_s + event<EventDetokenizeRun> [guard_detokenize_request_invalid] / effect_reject_detokenize_error_request_shape_from_state_prepared_tail,
        "state_prepared_tail"_s <= "state_prepared_tail"_s + event<EventDetokenizeRun> [guard_position_overflow] / effect_reject_detokenize_error_position_overflow_from_state_prepared_tail,
        "state_commit_generated_zero"_s <= "state_prepared_generated"_s + event<EventDetokenizeRun> [guard_detokenize_valid_replace] / effect_begin_detokenize_from_state_prepared_generated,
        "state_commit_generated"_s <= "state_prepared_generated"_s + event<EventDetokenizeRun> [guard_detokenize_valid_generated] / effect_begin_detokenize_from_state_prepared_generated,
        "state_prepared_generated"_s <= "state_prepared_generated"_s + event<EventDetokenizeRun> [guard_detokenize_request_invalid] / effect_reject_detokenize_error_request_shape_from_state_prepared_generated,
        "state_prepared_generated"_s <= "state_prepared_generated"_s + event<EventDetokenizeRun> [guard_position_overflow] / effect_reject_detokenize_error_position_overflow_from_state_prepared_generated,
        "state_output_decision"_s <= "state_commit_full_zero"_s + completion<EventDetokenizeRun>,
        "state_output_decision"_s <= "state_commit_full_generated"_s + completion<EventDetokenizeRun>,
        "state_output_decision"_s <= "state_commit_generated_zero"_s + completion<EventDetokenizeRun>,
        "state_output_decision"_s <= "state_commit_generated"_s + completion<EventDetokenizeRun>,
        "state_ready"_s <= "state_output_decision"_s + completion<EventDetokenizeRun> [guard_before_output_delay] / effect_publish_no_output_from_state_output_decision,
        "state_output_validation"_s <= "state_output_decision"_s + completion<EventDetokenizeRun> [guard_past_output_delay] / effect_collect_output,
        "state_ready"_s <= "state_output_validation"_s + completion<EventDetokenizeRun> [guard_output_complete] / effect_publish_output,
        "state_ready"_s <= "state_output_validation"_s + completion<EventDetokenizeRun> [guard_output_incomplete] / effect_publish_no_output_from_state_output_validation,
        "state_ready"_s <= "state_ready"_s + event<EventRestoreCache> [guard_restore_valid] / effect_restore_column_major_cache,
        "state_ready"_s <= "state_ready"_s + event<EventRestoreCache> [guard_restore_invalid] / effect_reject_error_request_shape_event_restore_cache,
        "state_prepared_full"_s <= "state_prepared_full"_s + event<EventRestoreCache> / effect_reject_error_phase_order_event_restore_cache,
        "state_prepared_generated"_s <= "state_prepared_generated"_s + event<EventRestoreCache> / effect_reject_error_phase_order_event_restore_cache,
        "state_prepared_tail"_s <= "state_prepared_tail"_s + event<EventRestoreCache> / effect_reject_error_phase_order_event_restore_cache,
        "state_ready"_s <= "state_prepared_full"_s + event<EventAdvance> [guard_advance_position_available] / effect_advance,
        "state_prepared_full"_s <= "state_prepared_full"_s + event<EventAdvance> [guard_advance_position_overflow] / effect_reject_error_position_overflow_event_advance,
        "state_prepared_generated"_s <= "state_prepared_generated"_s + event<EventAdvance> / effect_reject_error_phase_order_event_advance,
        "state_prepared_tail"_s <= "state_prepared_tail"_s + event<EventAdvance> / effect_reject_error_phase_order_event_advance,
        "state_ready"_s <= "state_ready"_s + event<EventAdvance> / effect_reject_error_phase_order_event_advance,
        "state_ready"_s <= "state_ready"_s + event<EventReset> / effect_reset_from_state_ready,
        "state_ready"_s <= "state_prepared_full"_s + event<EventReset> / effect_reset_from_state_prepared_full,
        "state_ready"_s <= "state_prepared_generated"_s + event<EventReset> / effect_reset_from_state_prepared_generated,
        "state_ready"_s <= "state_prepared_tail"_s + event<EventReset> / effect_reset_from_state_prepared_tail,
        "state_ready"_s <= "state_errored"_s + event<EventReset> / effect_reset_from_state_errored,
        "state_uninitialized"_s <= "state_uninitialized"_s + event<EventTokenize> / effect_reject_error_uninitialized_event_tokenize,
        "state_uninitialized"_s <= "state_uninitialized"_s + event<EventDetokenizeRun> / effect_reject_detokenize_error_uninitialized,
        "state_uninitialized"_s <= "state_uninitialized"_s + event<EventRestoreCache> / effect_reject_error_uninitialized_event_restore_cache,
        "state_uninitialized"_s <= "state_uninitialized"_s + event<EventAdvance> / effect_reject_error_uninitialized_event_advance,
        "state_uninitialized"_s <= "state_uninitialized"_s + event<EventReset> / effect_reject_error_uninitialized_event_reset,
        "state_ready"_s <= "state_ready"_s + event<EventDetokenizeRun> / effect_reject_detokenize_error_phase_order,
        "state_errored"_s <= "state_errored"_s + event<EventTokenize> / effect_reject_error_internal_error_event_tokenize,
        "state_errored"_s <= "state_errored"_s + event<EventDetokenizeRun> / effect_reject_detokenize_error_internal_error,
        "state_errored"_s <= "state_errored"_s + event<EventRestoreCache> / effect_reject_error_internal_error_event_restore_cache,
        "state_errored"_s <= "state_errored"_s + event<EventAdvance> / effect_reject_error_internal_error_event_advance,
        "state_errored"_s <= "state_uninitialized"_s + unexpected_event<_> / effect_unexpected_from_state_uninitialized,
        "state_errored"_s <= "state_ready"_s + unexpected_event<_> / effect_unexpected_from_state_ready,
        "state_errored"_s <= "state_prepared_full"_s + unexpected_event<_> / effect_unexpected_from_state_prepared_full,
        "state_errored"_s <= "state_prepared_generated"_s + unexpected_event<_> / effect_unexpected_from_state_prepared_generated,
        "state_errored"_s <= "state_prepared_tail"_s + unexpected_event<_> / effect_unexpected_from_state_prepared_tail,
        "state_errored"_s <= "state_errored"_s + unexpected_event<_> / effect_unexpected_from_state_errored,
    }
}

/// Context for `SpeechTokenizerMoshi` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct SpeechTokenizerMoshiContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl SpeechTokenizerMoshiStateMachineContext for SpeechTokenizerMoshiContext {
    fn effect_advance(&mut self, _event: &EventAdvance) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_advance
        todo!(
            "TODO: port action `effect_advance` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_begin_detokenize_from_state_prepared_full(
        &mut self,
        _event: &EventDetokenizeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_begin_detokenize
        todo!(
            "TODO: port action `effect_begin_detokenize` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_begin_detokenize_from_state_prepared_generated(
        &mut self,
        _event: &EventDetokenizeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_begin_detokenize
        todo!(
            "TODO: port action `effect_begin_detokenize` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_begin_detokenize_from_state_prepared_tail(
        &mut self,
        _event: &EventDetokenizeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_begin_detokenize
        todo!(
            "TODO: port action `effect_begin_detokenize` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_collect_output(&mut self, _event: &EventDetokenizeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_collect_output
        todo!(
            "TODO: port action `effect_collect_output` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_initialize_from_state_errored(&mut self, _event: &EventInitialize) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_initialize
        todo!(
            "TODO: port action `effect_initialize` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_initialize_from_state_uninitialized(
        &mut self,
        _event: &EventInitialize,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_initialize
        todo!(
            "TODO: port action `effect_initialize` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_publish_no_output_from_state_output_decision(
        &mut self,
        _event: &EventDetokenizeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_publish_no_output
        todo!(
            "TODO: port action `effect_publish_no_output` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_publish_no_output_from_state_output_validation(
        &mut self,
        _event: &EventDetokenizeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_publish_no_output
        todo!(
            "TODO: port action `effect_publish_no_output` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_publish_output(&mut self, _event: &EventDetokenizeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_publish_output
        todo!(
            "TODO: port action `effect_publish_output` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_reject_detokenize_error_internal_error(
        &mut self,
        _event: &EventDetokenizeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_reject_detokenize
        todo!(
            "TODO: port action `effect_reject_detokenize` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_reject_detokenize_error_phase_order(
        &mut self,
        _event: &EventDetokenizeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_reject_detokenize
        todo!(
            "TODO: port action `effect_reject_detokenize` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_reject_detokenize_error_position_overflow_from_state_prepared_full(
        &mut self,
        _event: &EventDetokenizeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_reject_detokenize
        todo!(
            "TODO: port action `effect_reject_detokenize` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_reject_detokenize_error_position_overflow_from_state_prepared_generated(
        &mut self,
        _event: &EventDetokenizeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_reject_detokenize
        todo!(
            "TODO: port action `effect_reject_detokenize` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_reject_detokenize_error_position_overflow_from_state_prepared_tail(
        &mut self,
        _event: &EventDetokenizeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_reject_detokenize
        todo!(
            "TODO: port action `effect_reject_detokenize` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_reject_detokenize_error_request_shape_from_state_prepared_full(
        &mut self,
        _event: &EventDetokenizeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_reject_detokenize
        todo!(
            "TODO: port action `effect_reject_detokenize` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_reject_detokenize_error_request_shape_from_state_prepared_generated(
        &mut self,
        _event: &EventDetokenizeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_reject_detokenize
        todo!(
            "TODO: port action `effect_reject_detokenize` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_reject_detokenize_error_request_shape_from_state_prepared_tail(
        &mut self,
        _event: &EventDetokenizeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_reject_detokenize
        todo!(
            "TODO: port action `effect_reject_detokenize` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_reject_detokenize_error_uninitialized(
        &mut self,
        _event: &EventDetokenizeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_reject_detokenize
        todo!(
            "TODO: port action `effect_reject_detokenize` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_reject_error_already_initialized_from_state_prepared_full(
        &mut self,
        _event: &EventInitialize,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_reject
        todo!(
            "TODO: port action `effect_reject` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_reject_error_already_initialized_from_state_prepared_generated(
        &mut self,
        _event: &EventInitialize,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_reject
        todo!(
            "TODO: port action `effect_reject` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_reject_error_already_initialized_from_state_prepared_tail(
        &mut self,
        _event: &EventInitialize,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_reject
        todo!(
            "TODO: port action `effect_reject` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_reject_error_already_initialized_from_state_ready(
        &mut self,
        _event: &EventInitialize,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_reject
        todo!(
            "TODO: port action `effect_reject` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_reject_error_internal_error_event_advance(
        &mut self,
        _event: &EventAdvance,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_reject
        todo!(
            "TODO: port action `effect_reject` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_reject_error_internal_error_event_restore_cache(
        &mut self,
        _event: &EventRestoreCache,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_reject
        todo!(
            "TODO: port action `effect_reject` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_reject_error_internal_error_event_tokenize(
        &mut self,
        _event: &EventTokenize,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_reject
        todo!(
            "TODO: port action `effect_reject` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_reject_error_invalid_configuration_from_state_errored(
        &mut self,
        _event: &EventInitialize,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_reject
        todo!(
            "TODO: port action `effect_reject` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_reject_error_invalid_configuration_from_state_uninitialized(
        &mut self,
        _event: &EventInitialize,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_reject
        todo!(
            "TODO: port action `effect_reject` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_reject_error_phase_order_event_advance(
        &mut self,
        _event: &EventAdvance,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_reject
        todo!(
            "TODO: port action `effect_reject` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_reject_error_phase_order_event_restore_cache(
        &mut self,
        _event: &EventRestoreCache,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_reject
        todo!(
            "TODO: port action `effect_reject` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_reject_error_phase_order_event_tokenize(
        &mut self,
        _event: &EventTokenize,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_reject
        todo!(
            "TODO: port action `effect_reject` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_reject_error_position_overflow_event_advance(
        &mut self,
        _event: &EventAdvance,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_reject
        todo!(
            "TODO: port action `effect_reject` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_reject_error_position_overflow_event_tokenize(
        &mut self,
        _event: &EventTokenize,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_reject
        todo!(
            "TODO: port action `effect_reject` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_reject_error_request_shape_event_restore_cache(
        &mut self,
        _event: &EventRestoreCache,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_reject
        todo!(
            "TODO: port action `effect_reject` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_reject_error_request_shape_event_tokenize(
        &mut self,
        _event: &EventTokenize,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_reject
        todo!(
            "TODO: port action `effect_reject` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_reject_error_uninitialized_event_advance(
        &mut self,
        _event: &EventAdvance,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_reject
        todo!(
            "TODO: port action `effect_reject` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_reject_error_uninitialized_event_reset(
        &mut self,
        _event: &EventReset,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_reject
        todo!(
            "TODO: port action `effect_reject` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_reject_error_uninitialized_event_restore_cache(
        &mut self,
        _event: &EventRestoreCache,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_reject
        todo!(
            "TODO: port action `effect_reject` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_reject_error_uninitialized_event_tokenize(
        &mut self,
        _event: &EventTokenize,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_reject
        todo!(
            "TODO: port action `effect_reject` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_reset_from_state_errored(&mut self, _event: &EventReset) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_reset
        todo!(
            "TODO: port action `effect_reset` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_reset_from_state_prepared_full(&mut self, _event: &EventReset) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_reset
        todo!(
            "TODO: port action `effect_reset` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_reset_from_state_prepared_generated(
        &mut self,
        _event: &EventReset,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_reset
        todo!(
            "TODO: port action `effect_reset` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_reset_from_state_prepared_tail(&mut self, _event: &EventReset) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_reset
        todo!(
            "TODO: port action `effect_reset` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_reset_from_state_ready(&mut self, _event: &EventReset) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_reset
        todo!(
            "TODO: port action `effect_reset` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_restore_column_major_cache(&mut self, _event: &EventRestoreCache) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_restore_column_major_cache
        todo!(
            "TODO: port action `effect_restore_column_major_cache` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_tokenize_empty(&mut self, _event: &EventTokenize) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_tokenize_empty
        todo!(
            "TODO: port action `effect_tokenize_empty` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_tokenize_full(&mut self, _event: &EventTokenize) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_tokenize_full
        todo!(
            "TODO: port action `effect_tokenize_full` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_tokenize_tail(&mut self, _event: &EventTokenize) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_tokenize_tail
        todo!(
            "TODO: port action `effect_tokenize_tail` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_unexpected_from_state_errored(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_unexpected
        todo!(
            "TODO: port action `effect_unexpected` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_unexpected_from_state_prepared_full(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_unexpected
        todo!(
            "TODO: port action `effect_unexpected` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_unexpected_from_state_prepared_generated(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_unexpected
        todo!(
            "TODO: port action `effect_unexpected` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_unexpected_from_state_prepared_tail(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_unexpected
        todo!(
            "TODO: port action `effect_unexpected` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_unexpected_from_state_ready(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_unexpected
        todo!(
            "TODO: port action `effect_unexpected` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn effect_unexpected_from_state_uninitialized(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp::effect_unexpected
        todo!(
            "TODO: port action `effect_unexpected` from emel.cpp/src/emel/speech/tokenizer/moshi/actions.hpp"
        )
    }
    fn guard_advance_position_available(&self, _event: &EventAdvance) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/guards.hpp::guard_advance_position_available
        todo!(
            "TODO: port guard `guard_advance_position_available` from emel.cpp/src/emel/speech/tokenizer/moshi/guards.hpp"
        )
    }
    fn guard_advance_position_overflow(&self, _event: &EventAdvance) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/guards.hpp::guard_advance_position_overflow
        todo!(
            "TODO: port guard `guard_advance_position_overflow` from emel.cpp/src/emel/speech/tokenizer/moshi/guards.hpp"
        )
    }
    fn guard_before_output_delay(&self, _event: &EventDetokenizeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/guards.hpp::guard_before_output_delay
        todo!(
            "TODO: port guard `guard_before_output_delay` from emel.cpp/src/emel/speech/tokenizer/moshi/guards.hpp"
        )
    }
    fn guard_configuration_invalid(&self, _event: &EventInitialize) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/guards.hpp::guard_configuration_invalid
        todo!(
            "TODO: port guard `guard_configuration_invalid` from emel.cpp/src/emel/speech/tokenizer/moshi/guards.hpp"
        )
    }
    fn guard_configuration_valid(&self, _event: &EventInitialize) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/guards.hpp::guard_configuration_valid
        todo!(
            "TODO: port guard `guard_configuration_valid` from emel.cpp/src/emel/speech/tokenizer/moshi/guards.hpp"
        )
    }
    fn guard_detokenize_request_invalid(&self, _event: &EventDetokenizeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/guards.hpp::guard_detokenize_request_invalid
        todo!(
            "TODO: port guard `guard_detokenize_request_invalid` from emel.cpp/src/emel/speech/tokenizer/moshi/guards.hpp"
        )
    }
    fn guard_detokenize_valid_generated(&self, _event: &EventDetokenizeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/guards.hpp::guard_detokenize_valid_generated
        todo!(
            "TODO: port guard `guard_detokenize_valid_generated` from emel.cpp/src/emel/speech/tokenizer/moshi/guards.hpp"
        )
    }
    fn guard_detokenize_valid_replace(&self, _event: &EventDetokenizeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/guards.hpp::guard_detokenize_valid_replace
        todo!(
            "TODO: port guard `guard_detokenize_valid_replace` from emel.cpp/src/emel/speech/tokenizer/moshi/guards.hpp"
        )
    }
    fn guard_output_complete(&self, _event: &EventDetokenizeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/guards.hpp::guard_output_complete
        todo!(
            "TODO: port guard `guard_output_complete` from emel.cpp/src/emel/speech/tokenizer/moshi/guards.hpp"
        )
    }
    fn guard_output_incomplete(&self, _event: &EventDetokenizeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/guards.hpp::guard_output_incomplete
        todo!(
            "TODO: port guard `guard_output_incomplete` from emel.cpp/src/emel/speech/tokenizer/moshi/guards.hpp"
        )
    }
    fn guard_past_output_delay(&self, _event: &EventDetokenizeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/guards.hpp::guard_past_output_delay
        todo!(
            "TODO: port guard `guard_past_output_delay` from emel.cpp/src/emel/speech/tokenizer/moshi/guards.hpp"
        )
    }
    fn guard_position_overflow(&self, _event: &EventDetokenizeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/guards.hpp::guard_position_overflow
        todo!(
            "TODO: port guard `guard_position_overflow` from emel.cpp/src/emel/speech/tokenizer/moshi/guards.hpp"
        )
    }
    fn guard_restore_invalid(&self, _event: &EventRestoreCache) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/guards.hpp::guard_restore_invalid
        todo!(
            "TODO: port guard `guard_restore_invalid` from emel.cpp/src/emel/speech/tokenizer/moshi/guards.hpp"
        )
    }
    fn guard_restore_valid(&self, _event: &EventRestoreCache) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/guards.hpp::guard_restore_valid
        todo!(
            "TODO: port guard `guard_restore_valid` from emel.cpp/src/emel/speech/tokenizer/moshi/guards.hpp"
        )
    }
    fn guard_tokenize_empty(&self, _event: &EventTokenize) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/guards.hpp::guard_tokenize_empty
        todo!(
            "TODO: port guard `guard_tokenize_empty` from emel.cpp/src/emel/speech/tokenizer/moshi/guards.hpp"
        )
    }
    fn guard_tokenize_full(&self, _event: &EventTokenize) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/guards.hpp::guard_tokenize_full
        todo!(
            "TODO: port guard `guard_tokenize_full` from emel.cpp/src/emel/speech/tokenizer/moshi/guards.hpp"
        )
    }
    fn guard_tokenize_invalid(&self, _event: &EventTokenize) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/guards.hpp::guard_tokenize_invalid
        todo!(
            "TODO: port guard `guard_tokenize_invalid` from emel.cpp/src/emel/speech/tokenizer/moshi/guards.hpp"
        )
    }
    fn guard_tokenize_position_overflow(&self, _event: &EventTokenize) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/guards.hpp::guard_tokenize_position_overflow
        todo!(
            "TODO: port guard `guard_tokenize_position_overflow` from emel.cpp/src/emel/speech/tokenizer/moshi/guards.hpp"
        )
    }
    fn guard_tokenize_tail(&self, _event: &EventTokenize) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/speech/tokenizer/moshi/guards.hpp::guard_tokenize_tail
        todo!(
            "TODO: port guard `guard_tokenize_tail` from emel.cpp/src/emel/speech/tokenizer/moshi/guards.hpp"
        )
    }
}
