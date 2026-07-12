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

// --- machine IoStagedRead from emel.cpp/src/emel/io/staged_read/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct DetailStagedWindowBatchRuntime;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct DetailStagedWindowRuntime;

sml! {
    IoStagedRead {
        "state_guard_staged_callbacks_decision"_s <= *"state_ready"_s + event<DetailStagedWindowRuntime> / effect_begin_staged_window,
        "state_guard_source_contract_decision"_s <= "state_guard_staged_callbacks_decision"_s + completion<DetailStagedWindowRuntime> [guard_staged_window_callbacks_present],
        "state_invalid_callbacks_error_decision"_s <= "state_guard_staged_callbacks_decision"_s + completion<DetailStagedWindowRuntime> [guard_staged_window_callbacks_missing] / effect_mark_invalid_callbacks,
        "state_guard_target_window_decision"_s <= "state_guard_source_contract_decision"_s + completion<DetailStagedWindowRuntime> [guard_stg_source_contract_valid],
        "state_invalid_staging_contract_error_decision"_s <= "state_guard_source_contract_decision"_s + completion<DetailStagedWindowRuntime> [guard_stg_source_contract_invalid] / effect_mark_invalid_staging_contract,
        "state_guard_platform_decision"_s <= "state_guard_target_window_decision"_s + completion<DetailStagedWindowRuntime> [guard_stg_target_window_valid],
        "state_invalid_target_window_error_decision"_s <= "state_guard_target_window_decision"_s + completion<DetailStagedWindowRuntime> [guard_stg_target_window_invalid] / effect_mark_invalid_target_window,
        "state_guard_copy_source_decision"_s <= "state_guard_platform_decision"_s + completion<DetailStagedWindowRuntime> [guard_platform_staged_read_supported],
        "state_unsupported_platform_error_decision"_s <= "state_guard_platform_decision"_s + completion<DetailStagedWindowRuntime> [guard_platform_staged_read_unsupported] / effect_mark_unsupported_platform,
        "state_staged_pre_ready"_s <= "state_guard_source_span_decision"_s + completion<DetailStagedWindowRuntime> [guard_stg_copy_span_valid] / effect_mark_staged_validation_accepted,
        "state_insufficient_source_span_error_decision"_s <= "state_guard_source_span_decision"_s + completion<DetailStagedWindowRuntime> [guard_stg_source_span_insufficient] / effect_mark_insufficient_source_span,
        "state_source_span_size_mismatch_error_decision"_s <= "state_guard_source_span_decision"_s + completion<DetailStagedWindowRuntime> [guard_stg_source_span_size_mismatch] / effect_mark_source_span_size_mismatch,
        "state_guard_source_span_decision"_s <= "state_guard_copy_source_decision"_s + completion<DetailStagedWindowRuntime> [guard_stg_source_span_present],
        "state_null_source_span_error_decision"_s <= "state_guard_copy_source_decision"_s + completion<DetailStagedWindowRuntime> [guard_stg_source_span_missing] / effect_mark_null_source_span,
        "state_ready"_s <= "state_staged_pre_ready"_s + completion<DetailStagedWindowRuntime> [guard_stg_logical_chunk_aligned] / effect_publish_staged_window_done_aligned,
        "state_ready"_s <= "state_staged_pre_ready"_s + completion<DetailStagedWindowRuntime> [guard_stg_logical_chunk_remainder] / effect_publish_staged_window_done_remainder,
        "state_staged_window_error_callback"_s <= "state_invalid_callbacks_error_decision"_s + completion<DetailStagedWindowRuntime> [error_callback_present] / effect_publish_staged_window_error_from_state_invalid_callbacks_error_decision,
        "state_ready"_s <= "state_invalid_callbacks_error_decision"_s + completion<DetailStagedWindowRuntime> [error_callback_absent] / effect_record_staged_window_error_from_state_invalid_callbacks_error_decision,
        "state_staged_window_error_callback"_s <= "state_invalid_staging_contract_error_decision"_s + completion<DetailStagedWindowRuntime> [error_callback_present] / effect_publish_staged_window_error_from_state_invalid_staging_contract_error_decision,
        "state_ready"_s <= "state_invalid_staging_contract_error_decision"_s + completion<DetailStagedWindowRuntime> [error_callback_absent] / effect_record_staged_window_error_from_state_invalid_staging_contract_error_decision,
        "state_staged_window_error_callback"_s <= "state_invalid_target_window_error_decision"_s + completion<DetailStagedWindowRuntime> [error_callback_present] / effect_publish_staged_window_error_from_state_invalid_target_window_error_decision,
        "state_ready"_s <= "state_invalid_target_window_error_decision"_s + completion<DetailStagedWindowRuntime> [error_callback_absent] / effect_record_staged_window_error_from_state_invalid_target_window_error_decision,
        "state_staged_window_error_callback"_s <= "state_unsupported_platform_error_decision"_s + completion<DetailStagedWindowRuntime> [error_callback_present] / effect_publish_staged_window_error_from_state_unsupported_platform_error_decision,
        "state_ready"_s <= "state_unsupported_platform_error_decision"_s + completion<DetailStagedWindowRuntime> [error_callback_absent] / effect_record_staged_window_error_from_state_unsupported_platform_error_decision,
        "state_staged_window_error_callback"_s <= "state_null_source_span_error_decision"_s + completion<DetailStagedWindowRuntime> [error_callback_present] / effect_publish_staged_window_error_from_state_null_source_span_error_decision,
        "state_ready"_s <= "state_null_source_span_error_decision"_s + completion<DetailStagedWindowRuntime> [error_callback_absent] / effect_record_staged_window_error_from_state_null_source_span_error_decision,
        "state_staged_window_error_callback"_s <= "state_source_span_size_mismatch_error_decision"_s + completion<DetailStagedWindowRuntime> [error_callback_present] / effect_publish_staged_window_error_from_state_source_span_size_mismatch_error_decision,
        "state_ready"_s <= "state_source_span_size_mismatch_error_decision"_s + completion<DetailStagedWindowRuntime> [error_callback_absent] / effect_record_staged_window_error_from_state_source_span_size_mismatch_error_decision,
        "state_staged_window_error_callback"_s <= "state_insufficient_source_span_error_decision"_s + completion<DetailStagedWindowRuntime> [error_callback_present] / effect_publish_staged_window_error_from_state_insufficient_source_span_error_decision,
        "state_ready"_s <= "state_insufficient_source_span_error_decision"_s + completion<DetailStagedWindowRuntime> [error_callback_absent] / effect_record_staged_window_error_from_state_insufficient_source_span_error_decision,
        "state_ready"_s <= "state_staged_window_error_callback"_s + completion<DetailStagedWindowRuntime> / effect_record_staged_window_error_from_state_staged_window_error_callback,
        "state_batch_guard_callbacks_decision"_s <= "state_ready"_s + event<DetailStagedWindowBatchRuntime> / effect_begin_staged_window_batch,
        "state_batch_guard_requests_decision"_s <= "state_batch_guard_callbacks_decision"_s + completion<DetailStagedWindowBatchRuntime> [guard_staged_window_batch_callbacks_present],
        "state_batch_invalid_callbacks_error_decision"_s <= "state_batch_guard_callbacks_decision"_s + completion<DetailStagedWindowBatchRuntime> [guard_staged_window_batch_callbacks_missing] / effect_mark_batch_invalid_callbacks,
        "state_batch_guard_platform_decision"_s <= "state_batch_guard_requests_decision"_s + completion<DetailStagedWindowBatchRuntime> [guard_stg_batch_requests_valid],
        "state_batch_invalid_staging_contract_error_decision"_s <= "state_batch_guard_requests_decision"_s + completion<DetailStagedWindowBatchRuntime> [guard_stg_batch_requests_invalid] / effect_mark_batch_invalid_staging_contract,
        "state_batch_done_callback"_s <= "state_batch_guard_platform_decision"_s + completion<DetailStagedWindowBatchRuntime> [guard_platform_staged_read_batch_supported] / effect_publish_staged_window_batch_done,
        "state_batch_unsupported_platform_error_decision"_s <= "state_batch_guard_platform_decision"_s + completion<DetailStagedWindowBatchRuntime> [guard_platform_staged_read_batch_unsupported] / effect_mark_batch_unsupported_platform,
        "state_ready"_s <= "state_batch_done_callback"_s + completion<DetailStagedWindowBatchRuntime> / effect_record_staged_window_batch_done,
        "state_staged_window_batch_error_callback"_s <= "state_batch_invalid_callbacks_error_decision"_s + completion<DetailStagedWindowBatchRuntime> [batch_error_callback_present] / effect_publish_staged_window_batch_error_from_state_batch_invalid_callbacks_error_decision,
        "state_ready"_s <= "state_batch_invalid_callbacks_error_decision"_s + completion<DetailStagedWindowBatchRuntime> [batch_error_callback_absent] / effect_record_staged_window_batch_error_from_state_batch_invalid_callbacks_error_decision,
        "state_staged_window_batch_error_callback"_s <= "state_batch_invalid_staging_contract_error_decision"_s + completion<DetailStagedWindowBatchRuntime> [batch_error_callback_present] / effect_publish_staged_window_batch_error_from_state_batch_invalid_staging_contract_error_decision,
        "state_ready"_s <= "state_batch_invalid_staging_contract_error_decision"_s + completion<DetailStagedWindowBatchRuntime> [batch_error_callback_absent] / effect_record_staged_window_batch_error_from_state_batch_invalid_staging_contract_error_decision,
        "state_staged_window_batch_error_callback"_s <= "state_batch_unsupported_platform_error_decision"_s + completion<DetailStagedWindowBatchRuntime> [batch_error_callback_present] / effect_publish_staged_window_batch_error_from_state_batch_unsupported_platform_error_decision,
        "state_ready"_s <= "state_batch_unsupported_platform_error_decision"_s + completion<DetailStagedWindowBatchRuntime> [batch_error_callback_absent] / effect_record_staged_window_batch_error_from_state_batch_unsupported_platform_error_decision,
        "state_ready"_s <= "state_staged_window_batch_error_callback"_s + completion<DetailStagedWindowBatchRuntime> / effect_record_staged_window_batch_error_from_state_staged_window_batch_error_callback,
        "state_ready"_s <= "state_ready"_s + unexpected_event<_> / effect_on_unexpected_from_state_ready,
        "state_ready"_s <= "state_guard_staged_callbacks_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_guard_staged_callbacks_decision,
        "state_ready"_s <= "state_guard_source_contract_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_guard_source_contract_decision,
        "state_ready"_s <= "state_guard_target_window_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_guard_target_window_decision,
        "state_ready"_s <= "state_guard_platform_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_guard_platform_decision,
        "state_ready"_s <= "state_guard_copy_source_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_guard_copy_source_decision,
        "state_ready"_s <= "state_guard_source_span_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_guard_source_span_decision,
        "state_ready"_s <= "state_staged_pre_ready"_s + unexpected_event<_> / effect_on_unexpected_from_state_staged_pre_ready,
        "state_ready"_s <= "state_invalid_callbacks_error_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_invalid_callbacks_error_decision,
        "state_ready"_s <= "state_invalid_staging_contract_error_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_invalid_staging_contract_error_decision,
        "state_ready"_s <= "state_invalid_target_window_error_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_invalid_target_window_error_decision,
        "state_ready"_s <= "state_unsupported_platform_error_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_unsupported_platform_error_decision,
        "state_ready"_s <= "state_null_source_span_error_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_null_source_span_error_decision,
        "state_ready"_s <= "state_source_span_size_mismatch_error_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_source_span_size_mismatch_error_decision,
        "state_ready"_s <= "state_insufficient_source_span_error_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_insufficient_source_span_error_decision,
        "state_ready"_s <= "state_staged_window_error_callback"_s + unexpected_event<_> / effect_on_unexpected_from_state_staged_window_error_callback,
        "state_ready"_s <= "state_batch_guard_callbacks_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_batch_guard_callbacks_decision,
        "state_ready"_s <= "state_batch_guard_requests_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_batch_guard_requests_decision,
        "state_ready"_s <= "state_batch_guard_platform_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_batch_guard_platform_decision,
        "state_ready"_s <= "state_batch_done_callback"_s + unexpected_event<_> / effect_on_unexpected_from_state_batch_done_callback,
        "state_ready"_s <= "state_batch_invalid_callbacks_error_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_batch_invalid_callbacks_error_decision,
        "state_ready"_s <= "state_batch_invalid_staging_contract_error_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_batch_invalid_staging_contract_error_decision,
        "state_ready"_s <= "state_batch_unsupported_platform_error_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_batch_unsupported_platform_error_decision,
        "state_ready"_s <= "state_staged_window_batch_error_callback"_s + unexpected_event<_> / effect_on_unexpected_from_state_staged_window_batch_error_callback,
    }
}

/// Context for `IoStagedRead` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct IoStagedReadContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl IoStagedReadStateMachineContext for IoStagedReadContext {
    fn batch_error_callback_absent(
        &self,
        _event: &DetailStagedWindowBatchRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/guards.hpp::batch_error_callback_absent
        todo!(
            "TODO: port guard `batch_error_callback_absent` from emel.cpp/src/emel/io/staged_read/guards.hpp"
        )
    }
    fn batch_error_callback_present(
        &self,
        _event: &DetailStagedWindowBatchRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/guards.hpp::batch_error_callback_present
        todo!(
            "TODO: port guard `batch_error_callback_present` from emel.cpp/src/emel/io/staged_read/guards.hpp"
        )
    }
    fn effect_begin_staged_window(&mut self, _event: &DetailStagedWindowRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_begin_staged_window
        todo!(
            "TODO: port action `effect_begin_staged_window` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_begin_staged_window_batch(
        &mut self,
        _event: &DetailStagedWindowBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_begin_staged_window_batch
        todo!(
            "TODO: port action `effect_begin_staged_window_batch` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_mark_batch_invalid_callbacks(
        &mut self,
        _event: &DetailStagedWindowBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_mark_batch_invalid_callbacks
        todo!(
            "TODO: port action `effect_mark_batch_invalid_callbacks` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_mark_batch_invalid_staging_contract(
        &mut self,
        _event: &DetailStagedWindowBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_mark_batch_invalid_staging_contract
        todo!(
            "TODO: port action `effect_mark_batch_invalid_staging_contract` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_mark_batch_unsupported_platform(
        &mut self,
        _event: &DetailStagedWindowBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_mark_batch_unsupported_platform
        todo!(
            "TODO: port action `effect_mark_batch_unsupported_platform` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_mark_insufficient_source_span(
        &mut self,
        _event: &DetailStagedWindowRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_mark_insufficient_source_span
        todo!(
            "TODO: port action `effect_mark_insufficient_source_span` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_mark_invalid_callbacks(
        &mut self,
        _event: &DetailStagedWindowRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_mark_invalid_callbacks
        todo!(
            "TODO: port action `effect_mark_invalid_callbacks` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_mark_invalid_staging_contract(
        &mut self,
        _event: &DetailStagedWindowRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_mark_invalid_staging_contract
        todo!(
            "TODO: port action `effect_mark_invalid_staging_contract` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_mark_invalid_target_window(
        &mut self,
        _event: &DetailStagedWindowRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_mark_invalid_target_window
        todo!(
            "TODO: port action `effect_mark_invalid_target_window` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_mark_null_source_span(
        &mut self,
        _event: &DetailStagedWindowRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_mark_null_source_span
        todo!(
            "TODO: port action `effect_mark_null_source_span` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_mark_source_span_size_mismatch(
        &mut self,
        _event: &DetailStagedWindowRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_mark_source_span_size_mismatch
        todo!(
            "TODO: port action `effect_mark_source_span_size_mismatch` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_mark_staged_validation_accepted(
        &mut self,
        _event: &DetailStagedWindowRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_mark_staged_validation_accepted
        todo!(
            "TODO: port action `effect_mark_staged_validation_accepted` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_mark_unsupported_platform(
        &mut self,
        _event: &DetailStagedWindowRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_mark_unsupported_platform
        todo!(
            "TODO: port action `effect_mark_unsupported_platform` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_batch_done_callback(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_batch_guard_callbacks_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_batch_guard_platform_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_batch_guard_requests_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_batch_invalid_callbacks_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_batch_invalid_staging_contract_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_batch_unsupported_platform_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_guard_copy_source_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_guard_platform_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_guard_source_contract_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_guard_source_span_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_guard_staged_callbacks_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_guard_target_window_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_insufficient_source_span_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_invalid_callbacks_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_invalid_staging_contract_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_invalid_target_window_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_null_source_span_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_ready(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_source_span_size_mismatch_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_staged_pre_ready(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_staged_window_batch_error_callback(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_staged_window_error_callback(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_unsupported_platform_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_publish_staged_window_batch_done(
        &mut self,
        _event: &DetailStagedWindowBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_publish_staged_window_batch_done
        todo!(
            "TODO: port action `effect_publish_staged_window_batch_done` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_publish_staged_window_batch_error_from_state_batch_invalid_callbacks_error_decision(
        &mut self,
        _event: &DetailStagedWindowBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_publish_staged_window_batch_error
        todo!(
            "TODO: port action `effect_publish_staged_window_batch_error` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_publish_staged_window_batch_error_from_state_batch_invalid_staging_contract_error_decision(
        &mut self,
        _event: &DetailStagedWindowBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_publish_staged_window_batch_error
        todo!(
            "TODO: port action `effect_publish_staged_window_batch_error` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_publish_staged_window_batch_error_from_state_batch_unsupported_platform_error_decision(
        &mut self,
        _event: &DetailStagedWindowBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_publish_staged_window_batch_error
        todo!(
            "TODO: port action `effect_publish_staged_window_batch_error` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_publish_staged_window_done_aligned(
        &mut self,
        _event: &DetailStagedWindowRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_publish_staged_window_done_aligned
        todo!(
            "TODO: port action `effect_publish_staged_window_done_aligned` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_publish_staged_window_done_remainder(
        &mut self,
        _event: &DetailStagedWindowRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_publish_staged_window_done_remainder
        todo!(
            "TODO: port action `effect_publish_staged_window_done_remainder` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_publish_staged_window_error_from_state_insufficient_source_span_error_decision(
        &mut self,
        _event: &DetailStagedWindowRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_publish_staged_window_error
        todo!(
            "TODO: port action `effect_publish_staged_window_error` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_publish_staged_window_error_from_state_invalid_callbacks_error_decision(
        &mut self,
        _event: &DetailStagedWindowRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_publish_staged_window_error
        todo!(
            "TODO: port action `effect_publish_staged_window_error` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_publish_staged_window_error_from_state_invalid_staging_contract_error_decision(
        &mut self,
        _event: &DetailStagedWindowRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_publish_staged_window_error
        todo!(
            "TODO: port action `effect_publish_staged_window_error` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_publish_staged_window_error_from_state_invalid_target_window_error_decision(
        &mut self,
        _event: &DetailStagedWindowRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_publish_staged_window_error
        todo!(
            "TODO: port action `effect_publish_staged_window_error` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_publish_staged_window_error_from_state_null_source_span_error_decision(
        &mut self,
        _event: &DetailStagedWindowRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_publish_staged_window_error
        todo!(
            "TODO: port action `effect_publish_staged_window_error` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_publish_staged_window_error_from_state_source_span_size_mismatch_error_decision(
        &mut self,
        _event: &DetailStagedWindowRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_publish_staged_window_error
        todo!(
            "TODO: port action `effect_publish_staged_window_error` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_publish_staged_window_error_from_state_unsupported_platform_error_decision(
        &mut self,
        _event: &DetailStagedWindowRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_publish_staged_window_error
        todo!(
            "TODO: port action `effect_publish_staged_window_error` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_record_staged_window_batch_done(
        &mut self,
        _event: &DetailStagedWindowBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_record_staged_window_batch_done
        todo!(
            "TODO: port action `effect_record_staged_window_batch_done` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_record_staged_window_batch_error_from_state_batch_invalid_callbacks_error_decision(
        &mut self,
        _event: &DetailStagedWindowBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_record_staged_window_batch_error
        todo!(
            "TODO: port action `effect_record_staged_window_batch_error` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_record_staged_window_batch_error_from_state_batch_invalid_staging_contract_error_decision(
        &mut self,
        _event: &DetailStagedWindowBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_record_staged_window_batch_error
        todo!(
            "TODO: port action `effect_record_staged_window_batch_error` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_record_staged_window_batch_error_from_state_batch_unsupported_platform_error_decision(
        &mut self,
        _event: &DetailStagedWindowBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_record_staged_window_batch_error
        todo!(
            "TODO: port action `effect_record_staged_window_batch_error` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_record_staged_window_batch_error_from_state_staged_window_batch_error_callback(
        &mut self,
        _event: &DetailStagedWindowBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_record_staged_window_batch_error
        todo!(
            "TODO: port action `effect_record_staged_window_batch_error` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_record_staged_window_error_from_state_insufficient_source_span_error_decision(
        &mut self,
        _event: &DetailStagedWindowRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_record_staged_window_error
        todo!(
            "TODO: port action `effect_record_staged_window_error` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_record_staged_window_error_from_state_invalid_callbacks_error_decision(
        &mut self,
        _event: &DetailStagedWindowRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_record_staged_window_error
        todo!(
            "TODO: port action `effect_record_staged_window_error` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_record_staged_window_error_from_state_invalid_staging_contract_error_decision(
        &mut self,
        _event: &DetailStagedWindowRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_record_staged_window_error
        todo!(
            "TODO: port action `effect_record_staged_window_error` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_record_staged_window_error_from_state_invalid_target_window_error_decision(
        &mut self,
        _event: &DetailStagedWindowRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_record_staged_window_error
        todo!(
            "TODO: port action `effect_record_staged_window_error` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_record_staged_window_error_from_state_null_source_span_error_decision(
        &mut self,
        _event: &DetailStagedWindowRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_record_staged_window_error
        todo!(
            "TODO: port action `effect_record_staged_window_error` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_record_staged_window_error_from_state_source_span_size_mismatch_error_decision(
        &mut self,
        _event: &DetailStagedWindowRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_record_staged_window_error
        todo!(
            "TODO: port action `effect_record_staged_window_error` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_record_staged_window_error_from_state_staged_window_error_callback(
        &mut self,
        _event: &DetailStagedWindowRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_record_staged_window_error
        todo!(
            "TODO: port action `effect_record_staged_window_error` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn effect_record_staged_window_error_from_state_unsupported_platform_error_decision(
        &mut self,
        _event: &DetailStagedWindowRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/actions.hpp::effect_record_staged_window_error
        todo!(
            "TODO: port action `effect_record_staged_window_error` from emel.cpp/src/emel/io/staged_read/actions.hpp"
        )
    }
    fn error_callback_absent(&self, _event: &DetailStagedWindowRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/guards.hpp::error_callback_absent
        todo!(
            "TODO: port guard `error_callback_absent` from emel.cpp/src/emel/io/staged_read/guards.hpp"
        )
    }
    fn error_callback_present(&self, _event: &DetailStagedWindowRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/guards.hpp::error_callback_present
        todo!(
            "TODO: port guard `error_callback_present` from emel.cpp/src/emel/io/staged_read/guards.hpp"
        )
    }
    fn guard_platform_staged_read_batch_supported(
        &self,
        _event: &DetailStagedWindowBatchRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/guards.hpp::guard_platform_staged_read_batch_supported
        todo!(
            "TODO: port guard `guard_platform_staged_read_batch_supported` from emel.cpp/src/emel/io/staged_read/guards.hpp"
        )
    }
    fn guard_platform_staged_read_batch_unsupported(
        &self,
        _event: &DetailStagedWindowBatchRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/guards.hpp::guard_platform_staged_read_batch_unsupported
        todo!(
            "TODO: port guard `guard_platform_staged_read_batch_unsupported` from emel.cpp/src/emel/io/staged_read/guards.hpp"
        )
    }
    fn guard_platform_staged_read_supported(
        &self,
        _event: &DetailStagedWindowRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/guards.hpp::guard_platform_staged_read_supported
        todo!(
            "TODO: port guard `guard_platform_staged_read_supported` from emel.cpp/src/emel/io/staged_read/guards.hpp"
        )
    }
    fn guard_platform_staged_read_unsupported(
        &self,
        _event: &DetailStagedWindowRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/guards.hpp::guard_platform_staged_read_unsupported
        todo!(
            "TODO: port guard `guard_platform_staged_read_unsupported` from emel.cpp/src/emel/io/staged_read/guards.hpp"
        )
    }
    fn guard_staged_window_batch_callbacks_missing(
        &self,
        _event: &DetailStagedWindowBatchRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/guards.hpp::guard_staged_window_batch_callbacks_missing
        todo!(
            "TODO: port guard `guard_staged_window_batch_callbacks_missing` from emel.cpp/src/emel/io/staged_read/guards.hpp"
        )
    }
    fn guard_staged_window_batch_callbacks_present(
        &self,
        _event: &DetailStagedWindowBatchRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/guards.hpp::guard_staged_window_batch_callbacks_present
        todo!(
            "TODO: port guard `guard_staged_window_batch_callbacks_present` from emel.cpp/src/emel/io/staged_read/guards.hpp"
        )
    }
    fn guard_staged_window_callbacks_missing(
        &self,
        _event: &DetailStagedWindowRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/guards.hpp::guard_staged_window_callbacks_missing
        todo!(
            "TODO: port guard `guard_staged_window_callbacks_missing` from emel.cpp/src/emel/io/staged_read/guards.hpp"
        )
    }
    fn guard_staged_window_callbacks_present(
        &self,
        _event: &DetailStagedWindowRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/guards.hpp::guard_staged_window_callbacks_present
        todo!(
            "TODO: port guard `guard_staged_window_callbacks_present` from emel.cpp/src/emel/io/staged_read/guards.hpp"
        )
    }
    fn guard_stg_batch_requests_invalid(
        &self,
        _event: &DetailStagedWindowBatchRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/guards.hpp::guard_stg_batch_requests_invalid
        todo!(
            "TODO: port guard `guard_stg_batch_requests_invalid` from emel.cpp/src/emel/io/staged_read/guards.hpp"
        )
    }
    fn guard_stg_batch_requests_valid(
        &self,
        _event: &DetailStagedWindowBatchRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/guards.hpp::guard_stg_batch_requests_valid
        todo!(
            "TODO: port guard `guard_stg_batch_requests_valid` from emel.cpp/src/emel/io/staged_read/guards.hpp"
        )
    }
    fn guard_stg_copy_span_valid(&self, _event: &DetailStagedWindowRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/guards.hpp::guard_stg_copy_span_valid
        todo!(
            "TODO: port guard `guard_stg_copy_span_valid` from emel.cpp/src/emel/io/staged_read/guards.hpp"
        )
    }
    fn guard_stg_logical_chunk_aligned(
        &self,
        _event: &DetailStagedWindowRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/guards.hpp::guard_stg_logical_chunk_aligned
        todo!(
            "TODO: port guard `guard_stg_logical_chunk_aligned` from emel.cpp/src/emel/io/staged_read/guards.hpp"
        )
    }
    fn guard_stg_logical_chunk_remainder(
        &self,
        _event: &DetailStagedWindowRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/guards.hpp::guard_stg_logical_chunk_remainder
        todo!(
            "TODO: port guard `guard_stg_logical_chunk_remainder` from emel.cpp/src/emel/io/staged_read/guards.hpp"
        )
    }
    fn guard_stg_source_contract_invalid(
        &self,
        _event: &DetailStagedWindowRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/guards.hpp::guard_stg_source_contract_invalid
        todo!(
            "TODO: port guard `guard_stg_source_contract_invalid` from emel.cpp/src/emel/io/staged_read/guards.hpp"
        )
    }
    fn guard_stg_source_contract_valid(
        &self,
        _event: &DetailStagedWindowRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/guards.hpp::guard_stg_source_contract_valid
        todo!(
            "TODO: port guard `guard_stg_source_contract_valid` from emel.cpp/src/emel/io/staged_read/guards.hpp"
        )
    }
    fn guard_stg_source_span_insufficient(
        &self,
        _event: &DetailStagedWindowRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/guards.hpp::guard_stg_source_span_insufficient
        todo!(
            "TODO: port guard `guard_stg_source_span_insufficient` from emel.cpp/src/emel/io/staged_read/guards.hpp"
        )
    }
    fn guard_stg_source_span_missing(
        &self,
        _event: &DetailStagedWindowRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/guards.hpp::guard_stg_source_span_missing
        todo!(
            "TODO: port guard `guard_stg_source_span_missing` from emel.cpp/src/emel/io/staged_read/guards.hpp"
        )
    }
    fn guard_stg_source_span_present(
        &self,
        _event: &DetailStagedWindowRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/guards.hpp::guard_stg_source_span_present
        todo!(
            "TODO: port guard `guard_stg_source_span_present` from emel.cpp/src/emel/io/staged_read/guards.hpp"
        )
    }
    fn guard_stg_source_span_size_mismatch(
        &self,
        _event: &DetailStagedWindowRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/guards.hpp::guard_stg_source_span_size_mismatch
        todo!(
            "TODO: port guard `guard_stg_source_span_size_mismatch` from emel.cpp/src/emel/io/staged_read/guards.hpp"
        )
    }
    fn guard_stg_target_window_invalid(
        &self,
        _event: &DetailStagedWindowRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/guards.hpp::guard_stg_target_window_invalid
        todo!(
            "TODO: port guard `guard_stg_target_window_invalid` from emel.cpp/src/emel/io/staged_read/guards.hpp"
        )
    }
    fn guard_stg_target_window_valid(
        &self,
        _event: &DetailStagedWindowRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/staged_read/guards.hpp::guard_stg_target_window_valid
        todo!(
            "TODO: port guard `guard_stg_target_window_valid` from emel.cpp/src/emel/io/staged_read/guards.hpp"
        )
    }
}
