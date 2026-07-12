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

// --- machine IoRead from emel.cpp/src/emel/io/read/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct DetailReadTensorBatchRuntime;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct DetailReadTensorRuntime;

sml! {
    IoRead {
        "state_request_decision"_s <= *"state_ready"_s + event<DetailReadTensorRuntime> / effect_begin_read_tensor,
        "state_file_path_decision"_s <= "state_request_decision"_s + completion<DetailReadTensorRuntime> [request_span_valid],
        "state_invalid_request_error_decision"_s <= "state_request_decision"_s + completion<DetailReadTensorRuntime> [request_span_invalid] / effect_mark_invalid_request_from_state_request_decision,
        "state_file_decision"_s <= "state_file_path_decision"_s + completion<DetailReadTensorRuntime> [file_path_valid],
        "state_invalid_request_error_decision"_s <= "state_file_path_decision"_s + completion<DetailReadTensorRuntime> [file_path_invalid] / effect_mark_invalid_request_from_state_file_path_decision,
        "state_length_decision"_s <= "state_file_decision"_s + completion<DetailReadTensorRuntime> [file_index_valid],
        "state_unsupported_resource_error_decision"_s <= "state_file_decision"_s + completion<DetailReadTensorRuntime> [file_index_invalid] / effect_mark_unsupported_resource_from_state_file_decision,
        "state_layout_decision"_s <= "state_length_decision"_s + completion<DetailReadTensorRuntime> [length_within_bounds],
        "state_unsupported_resource_error_decision"_s <= "state_length_decision"_s + completion<DetailReadTensorRuntime> [length_overflow] / effect_mark_unsupported_resource_from_state_length_decision,
        "state_target_buffer_decision"_s <= "state_layout_decision"_s + completion<DetailReadTensorRuntime> [layout_supported],
        "state_unsupported_resource_error_decision"_s <= "state_layout_decision"_s + completion<DetailReadTensorRuntime> [layout_unsupported] / effect_mark_unsupported_resource_from_state_layout_decision,
        "state_platform_decision"_s <= "state_target_buffer_decision"_s + completion<DetailReadTensorRuntime> [target_buffer_valid],
        "state_invalid_request_error_decision"_s <= "state_target_buffer_decision"_s + completion<DetailReadTensorRuntime> [target_buffer_invalid] / effect_mark_invalid_request_from_state_target_buffer_decision,
        "state_read_attempt_decision"_s <= "state_platform_decision"_s + completion<DetailReadTensorRuntime> [platform_read_supported] / effect_prepare_read_attempt,
        "state_unsupported_platform_error_decision"_s <= "state_platform_decision"_s + completion<DetailReadTensorRuntime> [platform_read_unsupported] / effect_mark_unsupported_platform,
        "state_file_open_decision"_s <= "state_read_attempt_decision"_s + completion<DetailReadTensorRuntime> [file_open_succeeded] / effect_prepare_read_copy,
        "state_file_open_failed_error_decision"_s <= "state_read_attempt_decision"_s + completion<DetailReadTensorRuntime> [file_open_failed] / effect_mark_file_open_failed,
        "state_file_read_decision"_s <= "state_file_open_decision"_s + completion<DetailReadTensorRuntime> [file_seek_succeeded],
        "state_file_seek_failed_error_decision"_s <= "state_file_open_decision"_s + completion<DetailReadTensorRuntime> [file_seek_failed] / effect_mark_file_seek_failed,
        "state_file_read_failed_error_decision"_s <= "state_file_read_decision"_s + completion<DetailReadTensorRuntime> [file_read_failed] / effect_mark_file_read_failed,
        "state_short_read_error_decision"_s <= "state_file_read_decision"_s + completion<DetailReadTensorRuntime> [file_read_short] / effect_mark_short_read,
        "state_done_callback"_s <= "state_file_read_decision"_s + completion<DetailReadTensorRuntime> [file_read_succeeded] / effect_mark_read_tensor_done,
        "state_ready"_s <= "state_done_callback"_s + completion<DetailReadTensorRuntime> / effect_publish_read_tensor_done,
        "state_batch_count_decision"_s <= "state_ready"_s + event<DetailReadTensorBatchRuntime> / effect_begin_read_tensor_batch,
        "state_batch_request_decision"_s <= "state_batch_count_decision"_s + completion<DetailReadTensorBatchRuntime> [batch_count_valid],
        "state_batch_invalid_request_error_decision"_s <= "state_batch_count_decision"_s + completion<DetailReadTensorBatchRuntime> [batch_count_invalid] / effect_mark_read_tensor_batch_count_invalid,
        "state_batch_resource_decision"_s <= "state_batch_request_decision"_s + completion<DetailReadTensorBatchRuntime> [batch_request_valid],
        "state_batch_invalid_request_error_decision"_s <= "state_batch_request_decision"_s + completion<DetailReadTensorBatchRuntime> [batch_request_invalid] / effect_mark_read_tensor_batch_invalid_request,
        "state_batch_source_open_decision"_s <= "state_batch_resource_decision"_s + completion<DetailReadTensorBatchRuntime> [batch_resource_supported],
        "state_batch_unsupported_resource_error_decision"_s <= "state_batch_resource_decision"_s + completion<DetailReadTensorBatchRuntime> [batch_resource_unsupported] / effect_mark_read_tensor_batch_unsupported_resource,
        "state_batch_source_seek_decision"_s <= "state_batch_source_open_decision"_s + completion<DetailReadTensorBatchRuntime> [batch_source_open_succeeded],
        "state_batch_file_open_failed_error_decision"_s <= "state_batch_source_open_decision"_s + completion<DetailReadTensorBatchRuntime> [batch_source_open_failed] / effect_mark_read_tensor_batch_file_open_failed,
        "state_batch_file_read_decision"_s <= "state_batch_source_seek_decision"_s + completion<DetailReadTensorBatchRuntime> [batch_source_seek_succeeded],
        "state_batch_file_seek_failed_error_decision"_s <= "state_batch_source_seek_decision"_s + completion<DetailReadTensorBatchRuntime> [batch_source_seek_failed] / effect_mark_read_tensor_batch_file_seek_failed,
        "state_batch_file_read_failed_error_decision"_s <= "state_batch_file_read_decision"_s + completion<DetailReadTensorBatchRuntime> [batch_file_read_failed] / effect_mark_read_tensor_batch_file_read_failed,
        "state_batch_short_read_error_decision"_s <= "state_batch_file_read_decision"_s + completion<DetailReadTensorBatchRuntime> [batch_file_read_short] / effect_mark_read_tensor_batch_short_read,
        "state_batch_done_callback"_s <= "state_batch_file_read_decision"_s + completion<DetailReadTensorBatchRuntime> [batch_file_read_succeeded] / effect_mark_read_tensor_batch_done,
        "state_ready"_s <= "state_batch_done_callback"_s + completion<DetailReadTensorBatchRuntime> / effect_publish_read_tensor_batch_done,
        "state_error_callback"_s <= "state_invalid_request_error_decision"_s + completion<DetailReadTensorRuntime> [error_callback_present] / effect_publish_read_tensor_error_from_state_invalid_request_error_decision,
        "state_ready"_s <= "state_invalid_request_error_decision"_s + completion<DetailReadTensorRuntime> [error_callback_absent] / effect_record_read_tensor_error_from_state_invalid_request_error_decision,
        "state_error_callback"_s <= "state_unsupported_resource_error_decision"_s + completion<DetailReadTensorRuntime> [error_callback_present] / effect_publish_read_tensor_error_from_state_unsupported_resource_error_decision,
        "state_ready"_s <= "state_unsupported_resource_error_decision"_s + completion<DetailReadTensorRuntime> [error_callback_absent] / effect_record_read_tensor_error_from_state_unsupported_resource_error_decision,
        "state_error_callback"_s <= "state_unsupported_platform_error_decision"_s + completion<DetailReadTensorRuntime> [error_callback_present] / effect_publish_read_tensor_error_from_state_unsupported_platform_error_decision,
        "state_ready"_s <= "state_unsupported_platform_error_decision"_s + completion<DetailReadTensorRuntime> [error_callback_absent] / effect_record_read_tensor_error_from_state_unsupported_platform_error_decision,
        "state_error_callback"_s <= "state_file_open_failed_error_decision"_s + completion<DetailReadTensorRuntime> [error_callback_present] / effect_publish_read_tensor_error_from_state_file_open_failed_error_decision,
        "state_ready"_s <= "state_file_open_failed_error_decision"_s + completion<DetailReadTensorRuntime> [error_callback_absent] / effect_record_read_tensor_error_from_state_file_open_failed_error_decision,
        "state_error_callback"_s <= "state_file_seek_failed_error_decision"_s + completion<DetailReadTensorRuntime> [error_callback_present] / effect_publish_read_tensor_error_from_state_file_seek_failed_error_decision,
        "state_ready"_s <= "state_file_seek_failed_error_decision"_s + completion<DetailReadTensorRuntime> [error_callback_absent] / effect_record_read_tensor_error_from_state_file_seek_failed_error_decision,
        "state_error_callback"_s <= "state_file_read_failed_error_decision"_s + completion<DetailReadTensorRuntime> [error_callback_present] / effect_publish_read_tensor_error_from_state_file_read_failed_error_decision,
        "state_ready"_s <= "state_file_read_failed_error_decision"_s + completion<DetailReadTensorRuntime> [error_callback_absent] / effect_record_read_tensor_error_from_state_file_read_failed_error_decision,
        "state_error_callback"_s <= "state_short_read_error_decision"_s + completion<DetailReadTensorRuntime> [error_callback_present] / effect_publish_read_tensor_error_from_state_short_read_error_decision,
        "state_ready"_s <= "state_short_read_error_decision"_s + completion<DetailReadTensorRuntime> [error_callback_absent] / effect_record_read_tensor_error_from_state_short_read_error_decision,
        "state_ready"_s <= "state_error_callback"_s + completion<DetailReadTensorRuntime> / effect_record_read_tensor_error_from_state_error_callback,
        "state_batch_error_callback"_s <= "state_batch_invalid_request_error_decision"_s + completion<DetailReadTensorBatchRuntime> [batch_error_callback_present] / effect_publish_read_tensor_batch_error_from_state_batch_invalid_request_error_decision,
        "state_ready"_s <= "state_batch_invalid_request_error_decision"_s + completion<DetailReadTensorBatchRuntime> [batch_error_callback_absent] / effect_record_read_tensor_batch_error_from_state_batch_invalid_request_error_decision,
        "state_batch_error_callback"_s <= "state_batch_unsupported_resource_error_decision"_s + completion<DetailReadTensorBatchRuntime> [batch_error_callback_present] / effect_publish_read_tensor_batch_error_from_state_batch_unsupported_resource_error_decision,
        "state_ready"_s <= "state_batch_unsupported_resource_error_decision"_s + completion<DetailReadTensorBatchRuntime> [batch_error_callback_absent] / effect_record_read_tensor_batch_error_from_state_batch_unsupported_resource_error_decision,
        "state_batch_error_callback"_s <= "state_batch_file_open_failed_error_decision"_s + completion<DetailReadTensorBatchRuntime> [batch_error_callback_present] / effect_publish_read_tensor_batch_error_from_state_batch_file_open_failed_error_decision,
        "state_ready"_s <= "state_batch_file_open_failed_error_decision"_s + completion<DetailReadTensorBatchRuntime> [batch_error_callback_absent] / effect_record_read_tensor_batch_error_from_state_batch_file_open_failed_error_decision,
        "state_batch_error_callback"_s <= "state_batch_file_seek_failed_error_decision"_s + completion<DetailReadTensorBatchRuntime> [batch_error_callback_present] / effect_publish_read_tensor_batch_error_from_state_batch_file_seek_failed_error_decision,
        "state_ready"_s <= "state_batch_file_seek_failed_error_decision"_s + completion<DetailReadTensorBatchRuntime> [batch_error_callback_absent] / effect_record_read_tensor_batch_error_from_state_batch_file_seek_failed_error_decision,
        "state_batch_error_callback"_s <= "state_batch_file_read_failed_error_decision"_s + completion<DetailReadTensorBatchRuntime> [batch_error_callback_present] / effect_publish_read_tensor_batch_error_from_state_batch_file_read_failed_error_decision,
        "state_ready"_s <= "state_batch_file_read_failed_error_decision"_s + completion<DetailReadTensorBatchRuntime> [batch_error_callback_absent] / effect_record_read_tensor_batch_error_from_state_batch_file_read_failed_error_decision,
        "state_batch_error_callback"_s <= "state_batch_short_read_error_decision"_s + completion<DetailReadTensorBatchRuntime> [batch_error_callback_present] / effect_publish_read_tensor_batch_error_from_state_batch_short_read_error_decision,
        "state_ready"_s <= "state_batch_short_read_error_decision"_s + completion<DetailReadTensorBatchRuntime> [batch_error_callback_absent] / effect_record_read_tensor_batch_error_from_state_batch_short_read_error_decision,
        "state_ready"_s <= "state_batch_error_callback"_s + completion<DetailReadTensorBatchRuntime> / effect_record_read_tensor_batch_error_from_state_batch_error_callback,
        "state_ready"_s <= "state_ready"_s + unexpected_event<_> / effect_on_unexpected_from_state_ready,
        "state_ready"_s <= "state_request_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_request_decision,
        "state_ready"_s <= "state_file_path_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_file_path_decision,
        "state_ready"_s <= "state_file_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_file_decision,
        "state_ready"_s <= "state_length_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_length_decision,
        "state_ready"_s <= "state_layout_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_layout_decision,
        "state_ready"_s <= "state_target_buffer_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_target_buffer_decision,
        "state_ready"_s <= "state_platform_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_platform_decision,
        "state_ready"_s <= "state_read_attempt_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_read_attempt_decision,
        "state_ready"_s <= "state_file_open_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_file_open_decision,
        "state_ready"_s <= "state_file_read_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_file_read_decision,
        "state_ready"_s <= "state_done_callback"_s + unexpected_event<_> / effect_on_unexpected_from_state_done_callback,
        "state_ready"_s <= "state_invalid_request_error_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_invalid_request_error_decision,
        "state_ready"_s <= "state_unsupported_resource_error_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_unsupported_resource_error_decision,
        "state_ready"_s <= "state_unsupported_platform_error_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_unsupported_platform_error_decision,
        "state_ready"_s <= "state_file_open_failed_error_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_file_open_failed_error_decision,
        "state_ready"_s <= "state_file_seek_failed_error_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_file_seek_failed_error_decision,
        "state_ready"_s <= "state_file_read_failed_error_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_file_read_failed_error_decision,
        "state_ready"_s <= "state_short_read_error_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_short_read_error_decision,
        "state_ready"_s <= "state_error_callback"_s + unexpected_event<_> / effect_on_unexpected_from_state_error_callback,
        "state_ready"_s <= "state_batch_count_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_batch_count_decision,
        "state_ready"_s <= "state_batch_request_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_batch_request_decision,
        "state_ready"_s <= "state_batch_resource_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_batch_resource_decision,
        "state_ready"_s <= "state_batch_source_open_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_batch_source_open_decision,
        "state_ready"_s <= "state_batch_source_seek_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_batch_source_seek_decision,
        "state_ready"_s <= "state_batch_file_read_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_batch_file_read_decision,
        "state_ready"_s <= "state_batch_done_callback"_s + unexpected_event<_> / effect_on_unexpected_from_state_batch_done_callback,
        "state_ready"_s <= "state_batch_invalid_request_error_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_batch_invalid_request_error_decision,
        "state_ready"_s <= "state_batch_unsupported_resource_error_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_batch_unsupported_resource_error_decision,
        "state_ready"_s <= "state_batch_file_open_failed_error_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_batch_file_open_failed_error_decision,
        "state_ready"_s <= "state_batch_file_seek_failed_error_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_batch_file_seek_failed_error_decision,
        "state_ready"_s <= "state_batch_file_read_failed_error_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_batch_file_read_failed_error_decision,
        "state_ready"_s <= "state_batch_short_read_error_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_batch_short_read_error_decision,
        "state_ready"_s <= "state_batch_error_callback"_s + unexpected_event<_> / effect_on_unexpected_from_state_batch_error_callback,
    }
}

/// Context for `IoRead` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct IoReadContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl IoReadStateMachineContext for IoReadContext {
    fn batch_count_invalid(&self, _event: &DetailReadTensorBatchRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/guards.hpp::batch_count_invalid
        todo!("TODO: port guard `batch_count_invalid` from emel.cpp/src/emel/io/read/guards.hpp")
    }
    fn batch_count_valid(&self, _event: &DetailReadTensorBatchRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/guards.hpp::batch_count_valid
        todo!("TODO: port guard `batch_count_valid` from emel.cpp/src/emel/io/read/guards.hpp")
    }
    fn batch_error_callback_absent(
        &self,
        _event: &DetailReadTensorBatchRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/guards.hpp::batch_error_callback_absent
        todo!(
            "TODO: port guard `batch_error_callback_absent` from emel.cpp/src/emel/io/read/guards.hpp"
        )
    }
    fn batch_error_callback_present(
        &self,
        _event: &DetailReadTensorBatchRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/guards.hpp::batch_error_callback_present
        todo!(
            "TODO: port guard `batch_error_callback_present` from emel.cpp/src/emel/io/read/guards.hpp"
        )
    }
    fn batch_file_read_failed(&self, _event: &DetailReadTensorBatchRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/guards.hpp::batch_file_read_failed
        todo!("TODO: port guard `batch_file_read_failed` from emel.cpp/src/emel/io/read/guards.hpp")
    }
    fn batch_file_read_short(&self, _event: &DetailReadTensorBatchRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/guards.hpp::batch_file_read_short
        todo!("TODO: port guard `batch_file_read_short` from emel.cpp/src/emel/io/read/guards.hpp")
    }
    fn batch_file_read_succeeded(&self, _event: &DetailReadTensorBatchRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/guards.hpp::batch_file_read_succeeded
        todo!(
            "TODO: port guard `batch_file_read_succeeded` from emel.cpp/src/emel/io/read/guards.hpp"
        )
    }
    fn batch_request_invalid(&self, _event: &DetailReadTensorBatchRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/guards.hpp::batch_request_invalid
        todo!("TODO: port guard `batch_request_invalid` from emel.cpp/src/emel/io/read/guards.hpp")
    }
    fn batch_request_valid(&self, _event: &DetailReadTensorBatchRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/guards.hpp::batch_request_valid
        todo!("TODO: port guard `batch_request_valid` from emel.cpp/src/emel/io/read/guards.hpp")
    }
    fn batch_resource_supported(&self, _event: &DetailReadTensorBatchRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/guards.hpp::batch_resource_supported
        todo!(
            "TODO: port guard `batch_resource_supported` from emel.cpp/src/emel/io/read/guards.hpp"
        )
    }
    fn batch_resource_unsupported(
        &self,
        _event: &DetailReadTensorBatchRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/guards.hpp::batch_resource_unsupported
        todo!(
            "TODO: port guard `batch_resource_unsupported` from emel.cpp/src/emel/io/read/guards.hpp"
        )
    }
    fn batch_source_open_failed(&self, _event: &DetailReadTensorBatchRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/guards.hpp::batch_source_open_failed
        todo!(
            "TODO: port guard `batch_source_open_failed` from emel.cpp/src/emel/io/read/guards.hpp"
        )
    }
    fn batch_source_open_succeeded(
        &self,
        _event: &DetailReadTensorBatchRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/guards.hpp::batch_source_open_succeeded
        todo!(
            "TODO: port guard `batch_source_open_succeeded` from emel.cpp/src/emel/io/read/guards.hpp"
        )
    }
    fn batch_source_seek_failed(&self, _event: &DetailReadTensorBatchRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/guards.hpp::batch_source_seek_failed
        todo!(
            "TODO: port guard `batch_source_seek_failed` from emel.cpp/src/emel/io/read/guards.hpp"
        )
    }
    fn batch_source_seek_succeeded(
        &self,
        _event: &DetailReadTensorBatchRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/guards.hpp::batch_source_seek_succeeded
        todo!(
            "TODO: port guard `batch_source_seek_succeeded` from emel.cpp/src/emel/io/read/guards.hpp"
        )
    }
    fn effect_begin_read_tensor(&mut self, _event: &DetailReadTensorRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_begin_read_tensor
        todo!(
            "TODO: port action `effect_begin_read_tensor` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_begin_read_tensor_batch(
        &mut self,
        _event: &DetailReadTensorBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_begin_read_tensor_batch
        todo!(
            "TODO: port action `effect_begin_read_tensor_batch` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_mark_file_open_failed(&mut self, _event: &DetailReadTensorRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_mark_file_open_failed
        todo!(
            "TODO: port action `effect_mark_file_open_failed` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_mark_file_read_failed(&mut self, _event: &DetailReadTensorRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_mark_file_read_failed
        todo!(
            "TODO: port action `effect_mark_file_read_failed` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_mark_file_seek_failed(&mut self, _event: &DetailReadTensorRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_mark_file_seek_failed
        todo!(
            "TODO: port action `effect_mark_file_seek_failed` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_mark_invalid_request_from_state_file_path_decision(
        &mut self,
        _event: &DetailReadTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_mark_invalid_request
        todo!(
            "TODO: port action `effect_mark_invalid_request` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_mark_invalid_request_from_state_request_decision(
        &mut self,
        _event: &DetailReadTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_mark_invalid_request
        todo!(
            "TODO: port action `effect_mark_invalid_request` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_mark_invalid_request_from_state_target_buffer_decision(
        &mut self,
        _event: &DetailReadTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_mark_invalid_request
        todo!(
            "TODO: port action `effect_mark_invalid_request` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_mark_read_tensor_batch_count_invalid(
        &mut self,
        _event: &DetailReadTensorBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_mark_read_tensor_batch_count_invalid
        todo!(
            "TODO: port action `effect_mark_read_tensor_batch_count_invalid` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_mark_read_tensor_batch_done(
        &mut self,
        _event: &DetailReadTensorBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_mark_read_tensor_batch_done
        todo!(
            "TODO: port action `effect_mark_read_tensor_batch_done` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_mark_read_tensor_batch_file_open_failed(
        &mut self,
        _event: &DetailReadTensorBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_mark_read_tensor_batch_file_open_failed
        todo!(
            "TODO: port action `effect_mark_read_tensor_batch_file_open_failed` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_mark_read_tensor_batch_file_read_failed(
        &mut self,
        _event: &DetailReadTensorBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_mark_read_tensor_batch_file_read_failed
        todo!(
            "TODO: port action `effect_mark_read_tensor_batch_file_read_failed` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_mark_read_tensor_batch_file_seek_failed(
        &mut self,
        _event: &DetailReadTensorBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_mark_read_tensor_batch_file_seek_failed
        todo!(
            "TODO: port action `effect_mark_read_tensor_batch_file_seek_failed` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_mark_read_tensor_batch_invalid_request(
        &mut self,
        _event: &DetailReadTensorBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_mark_read_tensor_batch_invalid_request
        todo!(
            "TODO: port action `effect_mark_read_tensor_batch_invalid_request` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_mark_read_tensor_batch_short_read(
        &mut self,
        _event: &DetailReadTensorBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_mark_read_tensor_batch_short_read
        todo!(
            "TODO: port action `effect_mark_read_tensor_batch_short_read` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_mark_read_tensor_batch_unsupported_resource(
        &mut self,
        _event: &DetailReadTensorBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_mark_read_tensor_batch_unsupported_resource
        todo!(
            "TODO: port action `effect_mark_read_tensor_batch_unsupported_resource` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_mark_read_tensor_done(&mut self, _event: &DetailReadTensorRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_mark_read_tensor_done
        todo!(
            "TODO: port action `effect_mark_read_tensor_done` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_mark_short_read(&mut self, _event: &DetailReadTensorRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_mark_short_read
        todo!(
            "TODO: port action `effect_mark_short_read` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_mark_unsupported_platform(
        &mut self,
        _event: &DetailReadTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_mark_unsupported_platform
        todo!(
            "TODO: port action `effect_mark_unsupported_platform` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_mark_unsupported_resource_from_state_file_decision(
        &mut self,
        _event: &DetailReadTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_mark_unsupported_resource
        todo!(
            "TODO: port action `effect_mark_unsupported_resource` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_mark_unsupported_resource_from_state_layout_decision(
        &mut self,
        _event: &DetailReadTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_mark_unsupported_resource
        todo!(
            "TODO: port action `effect_mark_unsupported_resource` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_mark_unsupported_resource_from_state_length_decision(
        &mut self,
        _event: &DetailReadTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_mark_unsupported_resource
        todo!(
            "TODO: port action `effect_mark_unsupported_resource` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_batch_count_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/read/actions.hpp")
    }
    fn effect_on_unexpected_from_state_batch_done_callback(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/read/actions.hpp")
    }
    fn effect_on_unexpected_from_state_batch_error_callback(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/read/actions.hpp")
    }
    fn effect_on_unexpected_from_state_batch_file_open_failed_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/read/actions.hpp")
    }
    fn effect_on_unexpected_from_state_batch_file_read_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/read/actions.hpp")
    }
    fn effect_on_unexpected_from_state_batch_file_read_failed_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/read/actions.hpp")
    }
    fn effect_on_unexpected_from_state_batch_file_seek_failed_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/read/actions.hpp")
    }
    fn effect_on_unexpected_from_state_batch_invalid_request_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/read/actions.hpp")
    }
    fn effect_on_unexpected_from_state_batch_request_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/read/actions.hpp")
    }
    fn effect_on_unexpected_from_state_batch_resource_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/read/actions.hpp")
    }
    fn effect_on_unexpected_from_state_batch_short_read_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/read/actions.hpp")
    }
    fn effect_on_unexpected_from_state_batch_source_open_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/read/actions.hpp")
    }
    fn effect_on_unexpected_from_state_batch_source_seek_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/read/actions.hpp")
    }
    fn effect_on_unexpected_from_state_batch_unsupported_resource_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/read/actions.hpp")
    }
    fn effect_on_unexpected_from_state_done_callback(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/read/actions.hpp")
    }
    fn effect_on_unexpected_from_state_error_callback(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/read/actions.hpp")
    }
    fn effect_on_unexpected_from_state_file_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/read/actions.hpp")
    }
    fn effect_on_unexpected_from_state_file_open_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/read/actions.hpp")
    }
    fn effect_on_unexpected_from_state_file_open_failed_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/read/actions.hpp")
    }
    fn effect_on_unexpected_from_state_file_path_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/read/actions.hpp")
    }
    fn effect_on_unexpected_from_state_file_read_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/read/actions.hpp")
    }
    fn effect_on_unexpected_from_state_file_read_failed_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/read/actions.hpp")
    }
    fn effect_on_unexpected_from_state_file_seek_failed_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/read/actions.hpp")
    }
    fn effect_on_unexpected_from_state_invalid_request_error_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/read/actions.hpp")
    }
    fn effect_on_unexpected_from_state_layout_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/read/actions.hpp")
    }
    fn effect_on_unexpected_from_state_length_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/read/actions.hpp")
    }
    fn effect_on_unexpected_from_state_platform_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/read/actions.hpp")
    }
    fn effect_on_unexpected_from_state_read_attempt_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/read/actions.hpp")
    }
    fn effect_on_unexpected_from_state_ready(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/read/actions.hpp")
    }
    fn effect_on_unexpected_from_state_request_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/read/actions.hpp")
    }
    fn effect_on_unexpected_from_state_short_read_error_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/read/actions.hpp")
    }
    fn effect_on_unexpected_from_state_target_buffer_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/read/actions.hpp")
    }
    fn effect_on_unexpected_from_state_unsupported_platform_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/read/actions.hpp")
    }
    fn effect_on_unexpected_from_state_unsupported_resource_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/read/actions.hpp")
    }
    fn effect_prepare_read_attempt(&mut self, _event: &DetailReadTensorRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_prepare_read_attempt
        todo!(
            "TODO: port action `effect_prepare_read_attempt` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_prepare_read_copy(&mut self, _event: &DetailReadTensorRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_prepare_read_copy
        todo!(
            "TODO: port action `effect_prepare_read_copy` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_publish_read_tensor_batch_done(
        &mut self,
        _event: &DetailReadTensorBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_publish_read_tensor_batch_done
        todo!(
            "TODO: port action `effect_publish_read_tensor_batch_done` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_publish_read_tensor_batch_error_from_state_batch_file_open_failed_error_decision(
        &mut self,
        _event: &DetailReadTensorBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_publish_read_tensor_batch_error
        todo!(
            "TODO: port action `effect_publish_read_tensor_batch_error` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_publish_read_tensor_batch_error_from_state_batch_file_read_failed_error_decision(
        &mut self,
        _event: &DetailReadTensorBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_publish_read_tensor_batch_error
        todo!(
            "TODO: port action `effect_publish_read_tensor_batch_error` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_publish_read_tensor_batch_error_from_state_batch_file_seek_failed_error_decision(
        &mut self,
        _event: &DetailReadTensorBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_publish_read_tensor_batch_error
        todo!(
            "TODO: port action `effect_publish_read_tensor_batch_error` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_publish_read_tensor_batch_error_from_state_batch_invalid_request_error_decision(
        &mut self,
        _event: &DetailReadTensorBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_publish_read_tensor_batch_error
        todo!(
            "TODO: port action `effect_publish_read_tensor_batch_error` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_publish_read_tensor_batch_error_from_state_batch_short_read_error_decision(
        &mut self,
        _event: &DetailReadTensorBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_publish_read_tensor_batch_error
        todo!(
            "TODO: port action `effect_publish_read_tensor_batch_error` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_publish_read_tensor_batch_error_from_state_batch_unsupported_resource_error_decision(
        &mut self,
        _event: &DetailReadTensorBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_publish_read_tensor_batch_error
        todo!(
            "TODO: port action `effect_publish_read_tensor_batch_error` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_publish_read_tensor_done(
        &mut self,
        _event: &DetailReadTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_publish_read_tensor_done
        todo!(
            "TODO: port action `effect_publish_read_tensor_done` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_publish_read_tensor_error_from_state_file_open_failed_error_decision(
        &mut self,
        _event: &DetailReadTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_publish_read_tensor_error
        todo!(
            "TODO: port action `effect_publish_read_tensor_error` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_publish_read_tensor_error_from_state_file_read_failed_error_decision(
        &mut self,
        _event: &DetailReadTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_publish_read_tensor_error
        todo!(
            "TODO: port action `effect_publish_read_tensor_error` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_publish_read_tensor_error_from_state_file_seek_failed_error_decision(
        &mut self,
        _event: &DetailReadTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_publish_read_tensor_error
        todo!(
            "TODO: port action `effect_publish_read_tensor_error` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_publish_read_tensor_error_from_state_invalid_request_error_decision(
        &mut self,
        _event: &DetailReadTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_publish_read_tensor_error
        todo!(
            "TODO: port action `effect_publish_read_tensor_error` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_publish_read_tensor_error_from_state_short_read_error_decision(
        &mut self,
        _event: &DetailReadTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_publish_read_tensor_error
        todo!(
            "TODO: port action `effect_publish_read_tensor_error` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_publish_read_tensor_error_from_state_unsupported_platform_error_decision(
        &mut self,
        _event: &DetailReadTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_publish_read_tensor_error
        todo!(
            "TODO: port action `effect_publish_read_tensor_error` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_publish_read_tensor_error_from_state_unsupported_resource_error_decision(
        &mut self,
        _event: &DetailReadTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_publish_read_tensor_error
        todo!(
            "TODO: port action `effect_publish_read_tensor_error` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_record_read_tensor_batch_error_from_state_batch_error_callback(
        &mut self,
        _event: &DetailReadTensorBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_record_read_tensor_batch_error
        todo!(
            "TODO: port action `effect_record_read_tensor_batch_error` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_record_read_tensor_batch_error_from_state_batch_file_open_failed_error_decision(
        &mut self,
        _event: &DetailReadTensorBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_record_read_tensor_batch_error
        todo!(
            "TODO: port action `effect_record_read_tensor_batch_error` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_record_read_tensor_batch_error_from_state_batch_file_read_failed_error_decision(
        &mut self,
        _event: &DetailReadTensorBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_record_read_tensor_batch_error
        todo!(
            "TODO: port action `effect_record_read_tensor_batch_error` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_record_read_tensor_batch_error_from_state_batch_file_seek_failed_error_decision(
        &mut self,
        _event: &DetailReadTensorBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_record_read_tensor_batch_error
        todo!(
            "TODO: port action `effect_record_read_tensor_batch_error` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_record_read_tensor_batch_error_from_state_batch_invalid_request_error_decision(
        &mut self,
        _event: &DetailReadTensorBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_record_read_tensor_batch_error
        todo!(
            "TODO: port action `effect_record_read_tensor_batch_error` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_record_read_tensor_batch_error_from_state_batch_short_read_error_decision(
        &mut self,
        _event: &DetailReadTensorBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_record_read_tensor_batch_error
        todo!(
            "TODO: port action `effect_record_read_tensor_batch_error` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_record_read_tensor_batch_error_from_state_batch_unsupported_resource_error_decision(
        &mut self,
        _event: &DetailReadTensorBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_record_read_tensor_batch_error
        todo!(
            "TODO: port action `effect_record_read_tensor_batch_error` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_record_read_tensor_error_from_state_error_callback(
        &mut self,
        _event: &DetailReadTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_record_read_tensor_error
        todo!(
            "TODO: port action `effect_record_read_tensor_error` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_record_read_tensor_error_from_state_file_open_failed_error_decision(
        &mut self,
        _event: &DetailReadTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_record_read_tensor_error
        todo!(
            "TODO: port action `effect_record_read_tensor_error` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_record_read_tensor_error_from_state_file_read_failed_error_decision(
        &mut self,
        _event: &DetailReadTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_record_read_tensor_error
        todo!(
            "TODO: port action `effect_record_read_tensor_error` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_record_read_tensor_error_from_state_file_seek_failed_error_decision(
        &mut self,
        _event: &DetailReadTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_record_read_tensor_error
        todo!(
            "TODO: port action `effect_record_read_tensor_error` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_record_read_tensor_error_from_state_invalid_request_error_decision(
        &mut self,
        _event: &DetailReadTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_record_read_tensor_error
        todo!(
            "TODO: port action `effect_record_read_tensor_error` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_record_read_tensor_error_from_state_short_read_error_decision(
        &mut self,
        _event: &DetailReadTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_record_read_tensor_error
        todo!(
            "TODO: port action `effect_record_read_tensor_error` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_record_read_tensor_error_from_state_unsupported_platform_error_decision(
        &mut self,
        _event: &DetailReadTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_record_read_tensor_error
        todo!(
            "TODO: port action `effect_record_read_tensor_error` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn effect_record_read_tensor_error_from_state_unsupported_resource_error_decision(
        &mut self,
        _event: &DetailReadTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/actions.hpp::effect_record_read_tensor_error
        todo!(
            "TODO: port action `effect_record_read_tensor_error` from emel.cpp/src/emel/io/read/actions.hpp"
        )
    }
    fn error_callback_absent(&self, _event: &DetailReadTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/guards.hpp::error_callback_absent
        todo!("TODO: port guard `error_callback_absent` from emel.cpp/src/emel/io/read/guards.hpp")
    }
    fn error_callback_present(&self, _event: &DetailReadTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/guards.hpp::error_callback_present
        todo!("TODO: port guard `error_callback_present` from emel.cpp/src/emel/io/read/guards.hpp")
    }
    fn file_index_invalid(&self, _event: &DetailReadTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/guards.hpp::file_index_invalid
        todo!("TODO: port guard `file_index_invalid` from emel.cpp/src/emel/io/read/guards.hpp")
    }
    fn file_index_valid(&self, _event: &DetailReadTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/guards.hpp::file_index_valid
        todo!("TODO: port guard `file_index_valid` from emel.cpp/src/emel/io/read/guards.hpp")
    }
    fn file_open_failed(&self, _event: &DetailReadTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/guards.hpp::file_open_failed
        todo!("TODO: port guard `file_open_failed` from emel.cpp/src/emel/io/read/guards.hpp")
    }
    fn file_open_succeeded(&self, _event: &DetailReadTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/guards.hpp::file_open_succeeded
        todo!("TODO: port guard `file_open_succeeded` from emel.cpp/src/emel/io/read/guards.hpp")
    }
    fn file_path_invalid(&self, _event: &DetailReadTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/guards.hpp::file_path_invalid
        todo!("TODO: port guard `file_path_invalid` from emel.cpp/src/emel/io/read/guards.hpp")
    }
    fn file_path_valid(&self, _event: &DetailReadTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/guards.hpp::file_path_valid
        todo!("TODO: port guard `file_path_valid` from emel.cpp/src/emel/io/read/guards.hpp")
    }
    fn file_read_failed(&self, _event: &DetailReadTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/guards.hpp::file_read_failed
        todo!("TODO: port guard `file_read_failed` from emel.cpp/src/emel/io/read/guards.hpp")
    }
    fn file_read_short(&self, _event: &DetailReadTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/guards.hpp::file_read_short
        todo!("TODO: port guard `file_read_short` from emel.cpp/src/emel/io/read/guards.hpp")
    }
    fn file_read_succeeded(&self, _event: &DetailReadTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/guards.hpp::file_read_succeeded
        todo!("TODO: port guard `file_read_succeeded` from emel.cpp/src/emel/io/read/guards.hpp")
    }
    fn file_seek_failed(&self, _event: &DetailReadTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/guards.hpp::file_seek_failed
        todo!("TODO: port guard `file_seek_failed` from emel.cpp/src/emel/io/read/guards.hpp")
    }
    fn file_seek_succeeded(&self, _event: &DetailReadTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/guards.hpp::file_seek_succeeded
        todo!("TODO: port guard `file_seek_succeeded` from emel.cpp/src/emel/io/read/guards.hpp")
    }
    fn layout_supported(&self, _event: &DetailReadTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/guards.hpp::layout_supported
        todo!("TODO: port guard `layout_supported` from emel.cpp/src/emel/io/read/guards.hpp")
    }
    fn layout_unsupported(&self, _event: &DetailReadTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/guards.hpp::layout_unsupported
        todo!("TODO: port guard `layout_unsupported` from emel.cpp/src/emel/io/read/guards.hpp")
    }
    fn length_overflow(&self, _event: &DetailReadTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/guards.hpp::length_overflow
        todo!("TODO: port guard `length_overflow` from emel.cpp/src/emel/io/read/guards.hpp")
    }
    fn length_within_bounds(&self, _event: &DetailReadTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/guards.hpp::length_within_bounds
        todo!("TODO: port guard `length_within_bounds` from emel.cpp/src/emel/io/read/guards.hpp")
    }
    fn platform_read_supported(&self, _event: &DetailReadTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/guards.hpp::platform_read_supported
        todo!(
            "TODO: port guard `platform_read_supported` from emel.cpp/src/emel/io/read/guards.hpp"
        )
    }
    fn platform_read_unsupported(&self, _event: &DetailReadTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/guards.hpp::platform_read_unsupported
        todo!(
            "TODO: port guard `platform_read_unsupported` from emel.cpp/src/emel/io/read/guards.hpp"
        )
    }
    fn request_span_invalid(&self, _event: &DetailReadTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/guards.hpp::request_span_invalid
        todo!("TODO: port guard `request_span_invalid` from emel.cpp/src/emel/io/read/guards.hpp")
    }
    fn request_span_valid(&self, _event: &DetailReadTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/guards.hpp::request_span_valid
        todo!("TODO: port guard `request_span_valid` from emel.cpp/src/emel/io/read/guards.hpp")
    }
    fn target_buffer_invalid(&self, _event: &DetailReadTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/guards.hpp::target_buffer_invalid
        todo!("TODO: port guard `target_buffer_invalid` from emel.cpp/src/emel/io/read/guards.hpp")
    }
    fn target_buffer_valid(&self, _event: &DetailReadTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/read/guards.hpp::target_buffer_valid
        todo!("TODO: port guard `target_buffer_valid` from emel.cpp/src/emel/io/read/guards.hpp")
    }
}
