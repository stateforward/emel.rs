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

// --- machine IoMmap from emel.cpp/src/emel/io/mmap/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct DetailAdviseMappingRuntime;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct DetailMapTensorRuntime;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct DetailReleaseMappingRuntime;

sml! {
    IoMmap {
        "state_request_decision"_s <= *"state_ready"_s + event<DetailMapTensorRuntime> / effect_begin_map_tensor,
        "state_file_path_decision"_s <= "state_request_decision"_s + completion<DetailMapTensorRuntime> [request_span_valid],
        "state_invalid_request_error_decision"_s <= "state_request_decision"_s + completion<DetailMapTensorRuntime> [request_span_invalid] / effect_mark_invalid_request_from_state_request_decision,
        "state_file_decision"_s <= "state_file_path_decision"_s + completion<DetailMapTensorRuntime> [file_path_valid],
        "state_invalid_request_error_decision"_s <= "state_file_path_decision"_s + completion<DetailMapTensorRuntime> [file_path_invalid] / effect_mark_invalid_request_from_state_file_path_decision,
        "state_offset_decision"_s <= "state_file_decision"_s + completion<DetailMapTensorRuntime> [file_index_valid],
        "state_unsupported_resource_error_decision"_s <= "state_file_decision"_s + completion<DetailMapTensorRuntime> [file_index_invalid] / effect_mark_unsupported_file,
        "state_length_decision"_s <= "state_offset_decision"_s + completion<DetailMapTensorRuntime> [offset_aligned],
        "state_unsupported_resource_error_decision"_s <= "state_offset_decision"_s + completion<DetailMapTensorRuntime> [offset_unaligned] / effect_mark_unsupported_offset,
        "state_layout_decision"_s <= "state_length_decision"_s + completion<DetailMapTensorRuntime> [length_within_bounds],
        "state_unsupported_resource_error_decision"_s <= "state_length_decision"_s + completion<DetailMapTensorRuntime> [length_overflow] / effect_mark_unsupported_length,
        "state_platform_decision"_s <= "state_layout_decision"_s + completion<DetailMapTensorRuntime> [layout_supported],
        "state_unsupported_resource_error_decision"_s <= "state_layout_decision"_s + completion<DetailMapTensorRuntime> [layout_unsupported] / effect_mark_unsupported_layout,
        "state_slot_reservation_decision"_s <= "state_platform_decision"_s + completion<DetailMapTensorRuntime> [platform_mmap_supported],
        "state_unsupported_platform_error_decision"_s <= "state_platform_decision"_s + completion<DetailMapTensorRuntime> [platform_mmap_unsupported] / effect_mark_unsupported_platform,
        "state_file_open_decision"_s <= "state_slot_reservation_decision"_s + completion<DetailMapTensorRuntime> [slot_capacity_available] / effect_reserve_top_free_slot_then_attempt_open,
        "state_resource_exhausted_error_decision"_s <= "state_slot_reservation_decision"_s + completion<DetailMapTensorRuntime> [slot_pool_exhausted] / effect_mark_resource_exhausted,
        "state_file_size_decision"_s <= "state_file_open_decision"_s + completion<DetailMapTensorRuntime> [file_open_succeeded] / effect_measure_open_file_size,
        "state_file_open_failed_error_decision"_s <= "state_file_open_decision"_s + completion<DetailMapTensorRuntime> [file_open_failed] / effect_release_reserved_slot_on_open_failure,
        "state_mapping_decision"_s <= "state_file_size_decision"_s + completion<DetailMapTensorRuntime> [file_span_within_file] / effect_attempt_mapping,
        "state_unsupported_resource_error_decision"_s <= "state_file_size_decision"_s + completion<DetailMapTensorRuntime> [file_span_exceeds_file] / effect_close_open_resource_and_release_slot_on_file_span_failure,
        "state_done_callback"_s <= "state_mapping_decision"_s + completion<DetailMapTensorRuntime> [mapping_succeeded] / effect_commit_mapping,
        "state_mapping_failed_error_decision"_s <= "state_mapping_decision"_s + completion<DetailMapTensorRuntime> [mapping_failed] / effect_close_open_resource_and_release_slot_on_mapping_failure,
        "state_ready"_s <= "state_done_callback"_s + completion<DetailMapTensorRuntime> / effect_publish_map_tensor_done,
        "state_error_callback"_s <= "state_invalid_request_error_decision"_s + completion<DetailMapTensorRuntime> [error_callback_present] / effect_publish_map_tensor_error_from_state_invalid_request_error_decision,
        "state_ready"_s <= "state_invalid_request_error_decision"_s + completion<DetailMapTensorRuntime> [error_callback_absent] / effect_record_map_tensor_error_from_state_invalid_request_error_decision,
        "state_error_callback"_s <= "state_unsupported_resource_error_decision"_s + completion<DetailMapTensorRuntime> [error_callback_present] / effect_publish_map_tensor_error_from_state_unsupported_resource_error_decision,
        "state_ready"_s <= "state_unsupported_resource_error_decision"_s + completion<DetailMapTensorRuntime> [error_callback_absent] / effect_record_map_tensor_error_from_state_unsupported_resource_error_decision,
        "state_error_callback"_s <= "state_unsupported_platform_error_decision"_s + completion<DetailMapTensorRuntime> [error_callback_present] / effect_publish_map_tensor_error_from_state_unsupported_platform_error_decision,
        "state_ready"_s <= "state_unsupported_platform_error_decision"_s + completion<DetailMapTensorRuntime> [error_callback_absent] / effect_record_map_tensor_error_from_state_unsupported_platform_error_decision,
        "state_error_callback"_s <= "state_resource_exhausted_error_decision"_s + completion<DetailMapTensorRuntime> [error_callback_present] / effect_publish_map_tensor_error_from_state_resource_exhausted_error_decision,
        "state_ready"_s <= "state_resource_exhausted_error_decision"_s + completion<DetailMapTensorRuntime> [error_callback_absent] / effect_record_map_tensor_error_from_state_resource_exhausted_error_decision,
        "state_error_callback"_s <= "state_file_open_failed_error_decision"_s + completion<DetailMapTensorRuntime> [error_callback_present] / effect_publish_map_tensor_error_from_state_file_open_failed_error_decision,
        "state_ready"_s <= "state_file_open_failed_error_decision"_s + completion<DetailMapTensorRuntime> [error_callback_absent] / effect_record_map_tensor_error_from_state_file_open_failed_error_decision,
        "state_error_callback"_s <= "state_mapping_failed_error_decision"_s + completion<DetailMapTensorRuntime> [error_callback_present] / effect_publish_map_tensor_error_from_state_mapping_failed_error_decision,
        "state_ready"_s <= "state_mapping_failed_error_decision"_s + completion<DetailMapTensorRuntime> [error_callback_absent] / effect_record_map_tensor_error_from_state_mapping_failed_error_decision,
        "state_ready"_s <= "state_error_callback"_s + completion<DetailMapTensorRuntime> / effect_record_map_tensor_error_from_state_error_callback,
        "state_release_decision"_s <= "state_ready"_s + event<DetailReleaseMappingRuntime> / effect_begin_release,
        "state_release_in_use_decision"_s <= "state_release_decision"_s + completion<DetailReleaseMappingRuntime> [release_handle_in_range],
        "state_release_invalid_handle_error_decision"_s <= "state_release_decision"_s + completion<DetailReleaseMappingRuntime> [release_handle_out_of_range] / effect_mark_release_invalid_handle_from_state_release_decision,
        "state_unmap_decision"_s <= "state_release_in_use_decision"_s + completion<DetailReleaseMappingRuntime> [release_slot_in_use_owned_by_tensor] / effect_attempt_unmap,
        "state_release_invalid_handle_error_decision"_s <= "state_release_in_use_decision"_s + completion<DetailReleaseMappingRuntime> [release_slot_not_in_use] / effect_mark_release_invalid_handle_from_state_release_in_use_decision,
        "state_release_invalid_handle_error_decision"_s <= "state_release_in_use_decision"_s + completion<DetailReleaseMappingRuntime> [release_slot_in_use_not_owned_by_tensor] / effect_mark_release_invalid_handle_from_state_release_in_use_decision,
        "state_release_publish_done_decision"_s <= "state_unmap_decision"_s + completion<DetailReleaseMappingRuntime> [unmap_succeeded] / effect_release_slot_after_unmap,
        "state_unmap_failed_error_decision"_s <= "state_unmap_decision"_s + completion<DetailReleaseMappingRuntime> [unmap_failed] / effect_mark_unmap_failed_and_release_slot,
        "state_release_done_callback"_s <= "state_release_publish_done_decision"_s + completion<DetailReleaseMappingRuntime> [release_done_callback_present] / effect_publish_release_mapping_done,
        "state_ready"_s <= "state_release_publish_done_decision"_s + completion<DetailReleaseMappingRuntime> [release_done_callback_absent] / effect_record_release_mapping_done_from_state_release_publish_done_decision,
        "state_ready"_s <= "state_release_done_callback"_s + completion<DetailReleaseMappingRuntime> / effect_record_release_mapping_done_from_state_release_done_callback,
        "state_release_error_callback"_s <= "state_release_invalid_handle_error_decision"_s + completion<DetailReleaseMappingRuntime> [release_error_callback_present] / effect_publish_release_mapping_error_from_state_release_invalid_handle_error_decision,
        "state_ready"_s <= "state_release_invalid_handle_error_decision"_s + completion<DetailReleaseMappingRuntime> [release_error_callback_absent] / effect_record_release_mapping_error_from_state_release_invalid_handle_error_decision,
        "state_release_error_callback"_s <= "state_unmap_failed_error_decision"_s + completion<DetailReleaseMappingRuntime> [release_error_callback_present] / effect_publish_release_mapping_error_from_state_unmap_failed_error_decision,
        "state_ready"_s <= "state_unmap_failed_error_decision"_s + completion<DetailReleaseMappingRuntime> [release_error_callback_absent] / effect_record_release_mapping_error_from_state_unmap_failed_error_decision,
        "state_ready"_s <= "state_release_error_callback"_s + completion<DetailReleaseMappingRuntime> / effect_record_release_mapping_error_from_state_release_error_callback,
        "state_advise_decision"_s <= "state_ready"_s + event<DetailAdviseMappingRuntime> / effect_begin_advise,
        "state_advise_owned_decision"_s <= "state_advise_decision"_s + completion<DetailAdviseMappingRuntime> [guard_advise_handle_in_range],
        "state_advise_invalid_handle_error_decision"_s <= "state_advise_decision"_s + completion<DetailAdviseMappingRuntime> [guard_advise_handle_out_of_range] / effect_mark_advise_invalid_handle_from_state_advise_decision,
        "state_advise_range_decision"_s <= "state_advise_owned_decision"_s + completion<DetailAdviseMappingRuntime> [guard_advise_slot_in_use_owned_by_tensor],
        "state_advise_invalid_handle_error_decision"_s <= "state_advise_owned_decision"_s + completion<DetailAdviseMappingRuntime> [guard_advise_slot_unavailable] / effect_mark_advise_invalid_handle_from_state_advise_owned_decision,
        "state_advise_platform_decision"_s <= "state_advise_range_decision"_s + completion<DetailAdviseMappingRuntime> [guard_advise_range_within_mapping],
        "state_advise_invalid_range_error_decision"_s <= "state_advise_range_decision"_s + completion<DetailAdviseMappingRuntime> [guard_advise_range_outside_mapping] / effect_mark_advise_invalid_range,
        "state_advise_kind_decision"_s <= "state_advise_platform_decision"_s + completion<DetailAdviseMappingRuntime> [guard_platform_advise_supported],
        "state_advise_unsupported_platform_error_decision"_s <= "state_advise_platform_decision"_s + completion<DetailAdviseMappingRuntime> [guard_platform_advise_unsupported] / effect_mark_advise_unsupported_platform,
        "state_advise_attempt_decision"_s <= "state_advise_kind_decision"_s + completion<DetailAdviseMappingRuntime> [guard_advise_kind_sequential] / effect_attempt_advise_sequential,
        "state_advise_attempt_decision"_s <= "state_advise_kind_decision"_s + completion<DetailAdviseMappingRuntime> [guard_advise_kind_willneed] / effect_attempt_advise_willneed,
        "state_advise_attempt_decision"_s <= "state_advise_kind_decision"_s + completion<DetailAdviseMappingRuntime> [guard_advise_kind_dontneed] / effect_attempt_advise_dontneed,
        "state_advise_invalid_kind_error_decision"_s <= "state_advise_kind_decision"_s + completion<DetailAdviseMappingRuntime> [guard_advise_kind_invalid] / effect_mark_advise_invalid_kind,
        "state_advise_publish_done_decision"_s <= "state_advise_attempt_decision"_s + completion<DetailAdviseMappingRuntime> [guard_advise_succeeded] / effect_commit_advise,
        "state_advise_failed_error_decision"_s <= "state_advise_attempt_decision"_s + completion<DetailAdviseMappingRuntime> [guard_advise_failed] / effect_mark_advise_failed,
        "state_advise_done_callback"_s <= "state_advise_publish_done_decision"_s + completion<DetailAdviseMappingRuntime> [guard_advise_done_callback_present] / effect_publish_advise_mapping_done,
        "state_ready"_s <= "state_advise_publish_done_decision"_s + completion<DetailAdviseMappingRuntime> [guard_advise_done_callback_absent] / effect_record_advise_mapping_done_from_state_advise_publish_done_decision,
        "state_ready"_s <= "state_advise_done_callback"_s + completion<DetailAdviseMappingRuntime> / effect_record_advise_mapping_done_from_state_advise_done_callback,
        "state_advise_error_callback"_s <= "state_advise_invalid_handle_error_decision"_s + completion<DetailAdviseMappingRuntime> [guard_advise_error_callback_present] / effect_publish_advise_mapping_error_from_state_advise_invalid_handle_error_decision,
        "state_ready"_s <= "state_advise_invalid_handle_error_decision"_s + completion<DetailAdviseMappingRuntime> [guard_advise_error_callback_absent] / effect_record_advise_mapping_error_from_state_advise_invalid_handle_error_decision,
        "state_advise_error_callback"_s <= "state_advise_invalid_range_error_decision"_s + completion<DetailAdviseMappingRuntime> [guard_advise_error_callback_present] / effect_publish_advise_mapping_error_from_state_advise_invalid_range_error_decision,
        "state_ready"_s <= "state_advise_invalid_range_error_decision"_s + completion<DetailAdviseMappingRuntime> [guard_advise_error_callback_absent] / effect_record_advise_mapping_error_from_state_advise_invalid_range_error_decision,
        "state_advise_error_callback"_s <= "state_advise_unsupported_platform_error_decision"_s + completion<DetailAdviseMappingRuntime> [guard_advise_error_callback_present] / effect_publish_advise_mapping_error_from_state_advise_unsupported_platform_error_decision,
        "state_ready"_s <= "state_advise_unsupported_platform_error_decision"_s + completion<DetailAdviseMappingRuntime> [guard_advise_error_callback_absent] / effect_record_advise_mapping_error_from_state_advise_unsupported_platform_error_decision,
        "state_advise_error_callback"_s <= "state_advise_failed_error_decision"_s + completion<DetailAdviseMappingRuntime> [guard_advise_error_callback_present] / effect_publish_advise_mapping_error_from_state_advise_failed_error_decision,
        "state_ready"_s <= "state_advise_failed_error_decision"_s + completion<DetailAdviseMappingRuntime> [guard_advise_error_callback_absent] / effect_record_advise_mapping_error_from_state_advise_failed_error_decision,
        "state_advise_error_callback"_s <= "state_advise_invalid_kind_error_decision"_s + completion<DetailAdviseMappingRuntime> [guard_advise_error_callback_present] / effect_publish_advise_mapping_error_from_state_advise_invalid_kind_error_decision,
        "state_ready"_s <= "state_advise_invalid_kind_error_decision"_s + completion<DetailAdviseMappingRuntime> [guard_advise_error_callback_absent] / effect_record_advise_mapping_error_from_state_advise_invalid_kind_error_decision,
        "state_ready"_s <= "state_advise_error_callback"_s + completion<DetailAdviseMappingRuntime> / effect_record_advise_mapping_error_from_state_advise_error_callback,
        "state_ready"_s <= "state_ready"_s + unexpected_event<_> / effect_on_unexpected_from_state_ready,
        "state_ready"_s <= "state_request_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_request_decision,
        "state_ready"_s <= "state_file_path_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_file_path_decision,
        "state_ready"_s <= "state_file_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_file_decision,
        "state_ready"_s <= "state_offset_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_offset_decision,
        "state_ready"_s <= "state_length_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_length_decision,
        "state_ready"_s <= "state_layout_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_layout_decision,
        "state_ready"_s <= "state_platform_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_platform_decision,
        "state_ready"_s <= "state_slot_reservation_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_slot_reservation_decision,
        "state_ready"_s <= "state_file_open_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_file_open_decision,
        "state_ready"_s <= "state_file_size_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_file_size_decision,
        "state_ready"_s <= "state_mapping_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_mapping_decision,
        "state_ready"_s <= "state_done_callback"_s + unexpected_event<_> / effect_on_unexpected_from_state_done_callback,
        "state_ready"_s <= "state_invalid_request_error_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_invalid_request_error_decision,
        "state_ready"_s <= "state_unsupported_resource_error_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_unsupported_resource_error_decision,
        "state_ready"_s <= "state_unsupported_platform_error_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_unsupported_platform_error_decision,
        "state_ready"_s <= "state_resource_exhausted_error_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_resource_exhausted_error_decision,
        "state_ready"_s <= "state_file_open_failed_error_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_file_open_failed_error_decision,
        "state_ready"_s <= "state_mapping_failed_error_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_mapping_failed_error_decision,
        "state_ready"_s <= "state_error_callback"_s + unexpected_event<_> / effect_on_unexpected_from_state_error_callback,
        "state_ready"_s <= "state_release_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_release_decision,
        "state_ready"_s <= "state_release_in_use_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_release_in_use_decision,
        "state_ready"_s <= "state_unmap_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_unmap_decision,
        "state_ready"_s <= "state_release_publish_done_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_release_publish_done_decision,
        "state_ready"_s <= "state_release_done_callback"_s + unexpected_event<_> / effect_on_unexpected_from_state_release_done_callback,
        "state_ready"_s <= "state_release_invalid_handle_error_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_release_invalid_handle_error_decision,
        "state_ready"_s <= "state_unmap_failed_error_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_unmap_failed_error_decision,
        "state_ready"_s <= "state_release_error_callback"_s + unexpected_event<_> / effect_on_unexpected_from_state_release_error_callback,
        "state_ready"_s <= "state_advise_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_advise_decision,
        "state_ready"_s <= "state_advise_owned_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_advise_owned_decision,
        "state_ready"_s <= "state_advise_range_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_advise_range_decision,
        "state_ready"_s <= "state_advise_platform_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_advise_platform_decision,
        "state_ready"_s <= "state_advise_kind_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_advise_kind_decision,
        "state_ready"_s <= "state_advise_attempt_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_advise_attempt_decision,
        "state_ready"_s <= "state_advise_publish_done_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_advise_publish_done_decision,
        "state_ready"_s <= "state_advise_done_callback"_s + unexpected_event<_> / effect_on_unexpected_from_state_advise_done_callback,
        "state_ready"_s <= "state_advise_invalid_handle_error_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_advise_invalid_handle_error_decision,
        "state_ready"_s <= "state_advise_invalid_range_error_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_advise_invalid_range_error_decision,
        "state_ready"_s <= "state_advise_invalid_kind_error_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_advise_invalid_kind_error_decision,
        "state_ready"_s <= "state_advise_unsupported_platform_error_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_advise_unsupported_platform_error_decision,
        "state_ready"_s <= "state_advise_failed_error_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_advise_failed_error_decision,
        "state_ready"_s <= "state_advise_error_callback"_s + unexpected_event<_> / effect_on_unexpected_from_state_advise_error_callback,
    }
}

/// Context for `IoMmap` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct IoMmapContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl IoMmapStateMachineContext for IoMmapContext {
    fn effect_attempt_advise_dontneed(
        &mut self,
        _event: &DetailAdviseMappingRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_attempt_advise_dontneed
        todo!(
            "TODO: port action `effect_attempt_advise_dontneed` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_attempt_advise_sequential(
        &mut self,
        _event: &DetailAdviseMappingRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_attempt_advise_sequential
        todo!(
            "TODO: port action `effect_attempt_advise_sequential` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_attempt_advise_willneed(
        &mut self,
        _event: &DetailAdviseMappingRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_attempt_advise_willneed
        todo!(
            "TODO: port action `effect_attempt_advise_willneed` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_attempt_mapping(&mut self, _event: &DetailMapTensorRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_attempt_mapping
        todo!(
            "TODO: port action `effect_attempt_mapping` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_attempt_unmap(&mut self, _event: &DetailReleaseMappingRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_attempt_unmap
        todo!("TODO: port action `effect_attempt_unmap` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_begin_advise(&mut self, _event: &DetailAdviseMappingRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_begin_advise
        todo!("TODO: port action `effect_begin_advise` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_begin_map_tensor(&mut self, _event: &DetailMapTensorRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_begin_map_tensor
        todo!(
            "TODO: port action `effect_begin_map_tensor` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_begin_release(&mut self, _event: &DetailReleaseMappingRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_begin_release
        todo!("TODO: port action `effect_begin_release` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_close_open_resource_and_release_slot_on_file_span_failure(
        &mut self,
        _event: &DetailMapTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_close_open_resource_and_release_slot_on_file_span_failure
        todo!(
            "TODO: port action `effect_close_open_resource_and_release_slot_on_file_span_failure` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_close_open_resource_and_release_slot_on_mapping_failure(
        &mut self,
        _event: &DetailMapTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_close_open_resource_and_release_slot_on_mapping_failure
        todo!(
            "TODO: port action `effect_close_open_resource_and_release_slot_on_mapping_failure` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_commit_advise(&mut self, _event: &DetailAdviseMappingRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_commit_advise
        todo!("TODO: port action `effect_commit_advise` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_commit_mapping(&mut self, _event: &DetailMapTensorRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_commit_mapping
        todo!(
            "TODO: port action `effect_commit_mapping` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_mark_advise_failed(&mut self, _event: &DetailAdviseMappingRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_mark_advise_failed
        todo!(
            "TODO: port action `effect_mark_advise_failed` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_mark_advise_invalid_handle_from_state_advise_decision(
        &mut self,
        _event: &DetailAdviseMappingRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_mark_advise_invalid_handle
        todo!(
            "TODO: port action `effect_mark_advise_invalid_handle` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_mark_advise_invalid_handle_from_state_advise_owned_decision(
        &mut self,
        _event: &DetailAdviseMappingRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_mark_advise_invalid_handle
        todo!(
            "TODO: port action `effect_mark_advise_invalid_handle` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_mark_advise_invalid_kind(
        &mut self,
        _event: &DetailAdviseMappingRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_mark_advise_invalid_kind
        todo!(
            "TODO: port action `effect_mark_advise_invalid_kind` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_mark_advise_invalid_range(
        &mut self,
        _event: &DetailAdviseMappingRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_mark_advise_invalid_range
        todo!(
            "TODO: port action `effect_mark_advise_invalid_range` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_mark_advise_unsupported_platform(
        &mut self,
        _event: &DetailAdviseMappingRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_mark_advise_unsupported_platform
        todo!(
            "TODO: port action `effect_mark_advise_unsupported_platform` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_mark_invalid_request_from_state_file_path_decision(
        &mut self,
        _event: &DetailMapTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_mark_invalid_request
        todo!(
            "TODO: port action `effect_mark_invalid_request` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_mark_invalid_request_from_state_request_decision(
        &mut self,
        _event: &DetailMapTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_mark_invalid_request
        todo!(
            "TODO: port action `effect_mark_invalid_request` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_mark_release_invalid_handle_from_state_release_decision(
        &mut self,
        _event: &DetailReleaseMappingRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_mark_release_invalid_handle
        todo!(
            "TODO: port action `effect_mark_release_invalid_handle` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_mark_release_invalid_handle_from_state_release_in_use_decision(
        &mut self,
        _event: &DetailReleaseMappingRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_mark_release_invalid_handle
        todo!(
            "TODO: port action `effect_mark_release_invalid_handle` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_mark_resource_exhausted(
        &mut self,
        _event: &DetailMapTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_mark_resource_exhausted
        todo!(
            "TODO: port action `effect_mark_resource_exhausted` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_mark_unmap_failed_and_release_slot(
        &mut self,
        _event: &DetailReleaseMappingRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_mark_unmap_failed_and_release_slot
        todo!(
            "TODO: port action `effect_mark_unmap_failed_and_release_slot` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_mark_unsupported_file(&mut self, _event: &DetailMapTensorRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_mark_unsupported_file
        todo!(
            "TODO: port action `effect_mark_unsupported_file` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_mark_unsupported_layout(
        &mut self,
        _event: &DetailMapTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_mark_unsupported_layout
        todo!(
            "TODO: port action `effect_mark_unsupported_layout` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_mark_unsupported_length(
        &mut self,
        _event: &DetailMapTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_mark_unsupported_length
        todo!(
            "TODO: port action `effect_mark_unsupported_length` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_mark_unsupported_offset(
        &mut self,
        _event: &DetailMapTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_mark_unsupported_offset
        todo!(
            "TODO: port action `effect_mark_unsupported_offset` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_mark_unsupported_platform(
        &mut self,
        _event: &DetailMapTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_mark_unsupported_platform
        todo!(
            "TODO: port action `effect_mark_unsupported_platform` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_measure_open_file_size(&mut self, _event: &DetailMapTensorRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_measure_open_file_size
        todo!(
            "TODO: port action `effect_measure_open_file_size` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_advise_attempt_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_on_unexpected_from_state_advise_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_on_unexpected_from_state_advise_done_callback(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_on_unexpected_from_state_advise_error_callback(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_on_unexpected_from_state_advise_failed_error_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_on_unexpected_from_state_advise_invalid_handle_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_on_unexpected_from_state_advise_invalid_kind_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_on_unexpected_from_state_advise_invalid_range_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_on_unexpected_from_state_advise_kind_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_on_unexpected_from_state_advise_owned_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_on_unexpected_from_state_advise_platform_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_on_unexpected_from_state_advise_publish_done_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_on_unexpected_from_state_advise_range_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_on_unexpected_from_state_advise_unsupported_platform_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_on_unexpected_from_state_done_callback(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_on_unexpected_from_state_error_callback(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_on_unexpected_from_state_file_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_on_unexpected_from_state_file_open_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_on_unexpected_from_state_file_open_failed_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_on_unexpected_from_state_file_path_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_on_unexpected_from_state_file_size_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_on_unexpected_from_state_invalid_request_error_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_on_unexpected_from_state_layout_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_on_unexpected_from_state_length_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_on_unexpected_from_state_mapping_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_on_unexpected_from_state_mapping_failed_error_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_on_unexpected_from_state_offset_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_on_unexpected_from_state_platform_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_on_unexpected_from_state_ready(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_on_unexpected_from_state_release_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_on_unexpected_from_state_release_done_callback(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_on_unexpected_from_state_release_error_callback(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_on_unexpected_from_state_release_in_use_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_on_unexpected_from_state_release_invalid_handle_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_on_unexpected_from_state_release_publish_done_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_on_unexpected_from_state_request_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_on_unexpected_from_state_resource_exhausted_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_on_unexpected_from_state_slot_reservation_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_on_unexpected_from_state_unmap_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_on_unexpected_from_state_unmap_failed_error_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_on_unexpected_from_state_unsupported_platform_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_on_unexpected_from_state_unsupported_resource_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_on_unexpected
        todo!("TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/mmap/actions.hpp")
    }
    fn effect_publish_advise_mapping_done(
        &mut self,
        _event: &DetailAdviseMappingRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_publish_advise_mapping_done
        todo!(
            "TODO: port action `effect_publish_advise_mapping_done` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_publish_advise_mapping_error_from_state_advise_failed_error_decision(
        &mut self,
        _event: &DetailAdviseMappingRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_publish_advise_mapping_error
        todo!(
            "TODO: port action `effect_publish_advise_mapping_error` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_publish_advise_mapping_error_from_state_advise_invalid_handle_error_decision(
        &mut self,
        _event: &DetailAdviseMappingRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_publish_advise_mapping_error
        todo!(
            "TODO: port action `effect_publish_advise_mapping_error` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_publish_advise_mapping_error_from_state_advise_invalid_kind_error_decision(
        &mut self,
        _event: &DetailAdviseMappingRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_publish_advise_mapping_error
        todo!(
            "TODO: port action `effect_publish_advise_mapping_error` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_publish_advise_mapping_error_from_state_advise_invalid_range_error_decision(
        &mut self,
        _event: &DetailAdviseMappingRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_publish_advise_mapping_error
        todo!(
            "TODO: port action `effect_publish_advise_mapping_error` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_publish_advise_mapping_error_from_state_advise_unsupported_platform_error_decision(
        &mut self,
        _event: &DetailAdviseMappingRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_publish_advise_mapping_error
        todo!(
            "TODO: port action `effect_publish_advise_mapping_error` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_publish_map_tensor_done(
        &mut self,
        _event: &DetailMapTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_publish_map_tensor_done
        todo!(
            "TODO: port action `effect_publish_map_tensor_done` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_publish_map_tensor_error_from_state_file_open_failed_error_decision(
        &mut self,
        _event: &DetailMapTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_publish_map_tensor_error
        todo!(
            "TODO: port action `effect_publish_map_tensor_error` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_publish_map_tensor_error_from_state_invalid_request_error_decision(
        &mut self,
        _event: &DetailMapTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_publish_map_tensor_error
        todo!(
            "TODO: port action `effect_publish_map_tensor_error` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_publish_map_tensor_error_from_state_mapping_failed_error_decision(
        &mut self,
        _event: &DetailMapTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_publish_map_tensor_error
        todo!(
            "TODO: port action `effect_publish_map_tensor_error` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_publish_map_tensor_error_from_state_resource_exhausted_error_decision(
        &mut self,
        _event: &DetailMapTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_publish_map_tensor_error
        todo!(
            "TODO: port action `effect_publish_map_tensor_error` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_publish_map_tensor_error_from_state_unsupported_platform_error_decision(
        &mut self,
        _event: &DetailMapTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_publish_map_tensor_error
        todo!(
            "TODO: port action `effect_publish_map_tensor_error` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_publish_map_tensor_error_from_state_unsupported_resource_error_decision(
        &mut self,
        _event: &DetailMapTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_publish_map_tensor_error
        todo!(
            "TODO: port action `effect_publish_map_tensor_error` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_publish_release_mapping_done(
        &mut self,
        _event: &DetailReleaseMappingRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_publish_release_mapping_done
        todo!(
            "TODO: port action `effect_publish_release_mapping_done` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_publish_release_mapping_error_from_state_release_invalid_handle_error_decision(
        &mut self,
        _event: &DetailReleaseMappingRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_publish_release_mapping_error
        todo!(
            "TODO: port action `effect_publish_release_mapping_error` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_publish_release_mapping_error_from_state_unmap_failed_error_decision(
        &mut self,
        _event: &DetailReleaseMappingRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_publish_release_mapping_error
        todo!(
            "TODO: port action `effect_publish_release_mapping_error` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_record_advise_mapping_done_from_state_advise_done_callback(
        &mut self,
        _event: &DetailAdviseMappingRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_record_advise_mapping_done
        todo!(
            "TODO: port action `effect_record_advise_mapping_done` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_record_advise_mapping_done_from_state_advise_publish_done_decision(
        &mut self,
        _event: &DetailAdviseMappingRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_record_advise_mapping_done
        todo!(
            "TODO: port action `effect_record_advise_mapping_done` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_record_advise_mapping_error_from_state_advise_error_callback(
        &mut self,
        _event: &DetailAdviseMappingRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_record_advise_mapping_error
        todo!(
            "TODO: port action `effect_record_advise_mapping_error` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_record_advise_mapping_error_from_state_advise_failed_error_decision(
        &mut self,
        _event: &DetailAdviseMappingRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_record_advise_mapping_error
        todo!(
            "TODO: port action `effect_record_advise_mapping_error` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_record_advise_mapping_error_from_state_advise_invalid_handle_error_decision(
        &mut self,
        _event: &DetailAdviseMappingRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_record_advise_mapping_error
        todo!(
            "TODO: port action `effect_record_advise_mapping_error` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_record_advise_mapping_error_from_state_advise_invalid_kind_error_decision(
        &mut self,
        _event: &DetailAdviseMappingRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_record_advise_mapping_error
        todo!(
            "TODO: port action `effect_record_advise_mapping_error` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_record_advise_mapping_error_from_state_advise_invalid_range_error_decision(
        &mut self,
        _event: &DetailAdviseMappingRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_record_advise_mapping_error
        todo!(
            "TODO: port action `effect_record_advise_mapping_error` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_record_advise_mapping_error_from_state_advise_unsupported_platform_error_decision(
        &mut self,
        _event: &DetailAdviseMappingRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_record_advise_mapping_error
        todo!(
            "TODO: port action `effect_record_advise_mapping_error` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_record_map_tensor_error_from_state_error_callback(
        &mut self,
        _event: &DetailMapTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_record_map_tensor_error
        todo!(
            "TODO: port action `effect_record_map_tensor_error` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_record_map_tensor_error_from_state_file_open_failed_error_decision(
        &mut self,
        _event: &DetailMapTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_record_map_tensor_error
        todo!(
            "TODO: port action `effect_record_map_tensor_error` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_record_map_tensor_error_from_state_invalid_request_error_decision(
        &mut self,
        _event: &DetailMapTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_record_map_tensor_error
        todo!(
            "TODO: port action `effect_record_map_tensor_error` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_record_map_tensor_error_from_state_mapping_failed_error_decision(
        &mut self,
        _event: &DetailMapTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_record_map_tensor_error
        todo!(
            "TODO: port action `effect_record_map_tensor_error` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_record_map_tensor_error_from_state_resource_exhausted_error_decision(
        &mut self,
        _event: &DetailMapTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_record_map_tensor_error
        todo!(
            "TODO: port action `effect_record_map_tensor_error` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_record_map_tensor_error_from_state_unsupported_platform_error_decision(
        &mut self,
        _event: &DetailMapTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_record_map_tensor_error
        todo!(
            "TODO: port action `effect_record_map_tensor_error` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_record_map_tensor_error_from_state_unsupported_resource_error_decision(
        &mut self,
        _event: &DetailMapTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_record_map_tensor_error
        todo!(
            "TODO: port action `effect_record_map_tensor_error` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_record_release_mapping_done_from_state_release_done_callback(
        &mut self,
        _event: &DetailReleaseMappingRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_record_release_mapping_done
        todo!(
            "TODO: port action `effect_record_release_mapping_done` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_record_release_mapping_done_from_state_release_publish_done_decision(
        &mut self,
        _event: &DetailReleaseMappingRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_record_release_mapping_done
        todo!(
            "TODO: port action `effect_record_release_mapping_done` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_record_release_mapping_error_from_state_release_error_callback(
        &mut self,
        _event: &DetailReleaseMappingRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_record_release_mapping_error
        todo!(
            "TODO: port action `effect_record_release_mapping_error` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_record_release_mapping_error_from_state_release_invalid_handle_error_decision(
        &mut self,
        _event: &DetailReleaseMappingRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_record_release_mapping_error
        todo!(
            "TODO: port action `effect_record_release_mapping_error` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_record_release_mapping_error_from_state_unmap_failed_error_decision(
        &mut self,
        _event: &DetailReleaseMappingRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_record_release_mapping_error
        todo!(
            "TODO: port action `effect_record_release_mapping_error` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_release_reserved_slot_on_open_failure(
        &mut self,
        _event: &DetailMapTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_release_reserved_slot_on_open_failure
        todo!(
            "TODO: port action `effect_release_reserved_slot_on_open_failure` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_release_slot_after_unmap(
        &mut self,
        _event: &DetailReleaseMappingRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_release_slot_after_unmap
        todo!(
            "TODO: port action `effect_release_slot_after_unmap` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn effect_reserve_top_free_slot_then_attempt_open(
        &mut self,
        _event: &DetailMapTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/actions.hpp::effect_reserve_top_free_slot_then_attempt_open
        todo!(
            "TODO: port action `effect_reserve_top_free_slot_then_attempt_open` from emel.cpp/src/emel/io/mmap/actions.hpp"
        )
    }
    fn error_callback_absent(&self, _event: &DetailMapTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::error_callback_absent
        todo!("TODO: port guard `error_callback_absent` from emel.cpp/src/emel/io/mmap/guards.hpp")
    }
    fn error_callback_present(&self, _event: &DetailMapTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::error_callback_present
        todo!("TODO: port guard `error_callback_present` from emel.cpp/src/emel/io/mmap/guards.hpp")
    }
    fn file_index_invalid(&self, _event: &DetailMapTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::file_index_invalid
        todo!("TODO: port guard `file_index_invalid` from emel.cpp/src/emel/io/mmap/guards.hpp")
    }
    fn file_index_valid(&self, _event: &DetailMapTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::file_index_valid
        todo!("TODO: port guard `file_index_valid` from emel.cpp/src/emel/io/mmap/guards.hpp")
    }
    fn file_open_failed(&self, _event: &DetailMapTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::file_open_failed
        todo!("TODO: port guard `file_open_failed` from emel.cpp/src/emel/io/mmap/guards.hpp")
    }
    fn file_open_succeeded(&self, _event: &DetailMapTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::file_open_succeeded
        todo!("TODO: port guard `file_open_succeeded` from emel.cpp/src/emel/io/mmap/guards.hpp")
    }
    fn file_path_invalid(&self, _event: &DetailMapTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::file_path_invalid
        todo!("TODO: port guard `file_path_invalid` from emel.cpp/src/emel/io/mmap/guards.hpp")
    }
    fn file_path_valid(&self, _event: &DetailMapTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::file_path_valid
        todo!("TODO: port guard `file_path_valid` from emel.cpp/src/emel/io/mmap/guards.hpp")
    }
    fn file_span_exceeds_file(&self, _event: &DetailMapTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::file_span_exceeds_file
        todo!("TODO: port guard `file_span_exceeds_file` from emel.cpp/src/emel/io/mmap/guards.hpp")
    }
    fn file_span_within_file(&self, _event: &DetailMapTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::file_span_within_file
        todo!("TODO: port guard `file_span_within_file` from emel.cpp/src/emel/io/mmap/guards.hpp")
    }
    fn guard_advise_done_callback_absent(
        &self,
        _event: &DetailAdviseMappingRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::guard_advise_done_callback_absent
        todo!(
            "TODO: port guard `guard_advise_done_callback_absent` from emel.cpp/src/emel/io/mmap/guards.hpp"
        )
    }
    fn guard_advise_done_callback_present(
        &self,
        _event: &DetailAdviseMappingRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::guard_advise_done_callback_present
        todo!(
            "TODO: port guard `guard_advise_done_callback_present` from emel.cpp/src/emel/io/mmap/guards.hpp"
        )
    }
    fn guard_advise_error_callback_absent(
        &self,
        _event: &DetailAdviseMappingRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::guard_advise_error_callback_absent
        todo!(
            "TODO: port guard `guard_advise_error_callback_absent` from emel.cpp/src/emel/io/mmap/guards.hpp"
        )
    }
    fn guard_advise_error_callback_present(
        &self,
        _event: &DetailAdviseMappingRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::guard_advise_error_callback_present
        todo!(
            "TODO: port guard `guard_advise_error_callback_present` from emel.cpp/src/emel/io/mmap/guards.hpp"
        )
    }
    fn guard_advise_failed(&self, _event: &DetailAdviseMappingRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::guard_advise_failed
        todo!("TODO: port guard `guard_advise_failed` from emel.cpp/src/emel/io/mmap/guards.hpp")
    }
    fn guard_advise_handle_in_range(
        &self,
        _event: &DetailAdviseMappingRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::guard_advise_handle_in_range
        todo!(
            "TODO: port guard `guard_advise_handle_in_range` from emel.cpp/src/emel/io/mmap/guards.hpp"
        )
    }
    fn guard_advise_handle_out_of_range(
        &self,
        _event: &DetailAdviseMappingRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::guard_advise_handle_out_of_range
        todo!(
            "TODO: port guard `guard_advise_handle_out_of_range` from emel.cpp/src/emel/io/mmap/guards.hpp"
        )
    }
    fn guard_advise_kind_dontneed(&self, _event: &DetailAdviseMappingRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::guard_advise_kind_dontneed
        todo!(
            "TODO: port guard `guard_advise_kind_dontneed` from emel.cpp/src/emel/io/mmap/guards.hpp"
        )
    }
    fn guard_advise_kind_invalid(&self, _event: &DetailAdviseMappingRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::guard_advise_kind_invalid
        todo!(
            "TODO: port guard `guard_advise_kind_invalid` from emel.cpp/src/emel/io/mmap/guards.hpp"
        )
    }
    fn guard_advise_kind_sequential(
        &self,
        _event: &DetailAdviseMappingRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::guard_advise_kind_sequential
        todo!(
            "TODO: port guard `guard_advise_kind_sequential` from emel.cpp/src/emel/io/mmap/guards.hpp"
        )
    }
    fn guard_advise_kind_willneed(&self, _event: &DetailAdviseMappingRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::guard_advise_kind_willneed
        todo!(
            "TODO: port guard `guard_advise_kind_willneed` from emel.cpp/src/emel/io/mmap/guards.hpp"
        )
    }
    fn guard_advise_range_outside_mapping(
        &self,
        _event: &DetailAdviseMappingRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::guard_advise_range_outside_mapping
        todo!(
            "TODO: port guard `guard_advise_range_outside_mapping` from emel.cpp/src/emel/io/mmap/guards.hpp"
        )
    }
    fn guard_advise_range_within_mapping(
        &self,
        _event: &DetailAdviseMappingRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::guard_advise_range_within_mapping
        todo!(
            "TODO: port guard `guard_advise_range_within_mapping` from emel.cpp/src/emel/io/mmap/guards.hpp"
        )
    }
    fn guard_advise_slot_in_use_owned_by_tensor(
        &self,
        _event: &DetailAdviseMappingRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::guard_advise_slot_in_use_owned_by_tensor
        todo!(
            "TODO: port guard `guard_advise_slot_in_use_owned_by_tensor` from emel.cpp/src/emel/io/mmap/guards.hpp"
        )
    }
    fn guard_advise_slot_unavailable(
        &self,
        _event: &DetailAdviseMappingRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::guard_advise_slot_unavailable
        todo!(
            "TODO: port guard `guard_advise_slot_unavailable` from emel.cpp/src/emel/io/mmap/guards.hpp"
        )
    }
    fn guard_advise_succeeded(&self, _event: &DetailAdviseMappingRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::guard_advise_succeeded
        todo!("TODO: port guard `guard_advise_succeeded` from emel.cpp/src/emel/io/mmap/guards.hpp")
    }
    fn guard_platform_advise_supported(
        &self,
        _event: &DetailAdviseMappingRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::guard_platform_advise_supported
        todo!(
            "TODO: port guard `guard_platform_advise_supported` from emel.cpp/src/emel/io/mmap/guards.hpp"
        )
    }
    fn guard_platform_advise_unsupported(
        &self,
        _event: &DetailAdviseMappingRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::guard_platform_advise_unsupported
        todo!(
            "TODO: port guard `guard_platform_advise_unsupported` from emel.cpp/src/emel/io/mmap/guards.hpp"
        )
    }
    fn layout_supported(&self, _event: &DetailMapTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::layout_supported
        todo!("TODO: port guard `layout_supported` from emel.cpp/src/emel/io/mmap/guards.hpp")
    }
    fn layout_unsupported(&self, _event: &DetailMapTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::layout_unsupported
        todo!("TODO: port guard `layout_unsupported` from emel.cpp/src/emel/io/mmap/guards.hpp")
    }
    fn length_overflow(&self, _event: &DetailMapTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::length_overflow
        todo!("TODO: port guard `length_overflow` from emel.cpp/src/emel/io/mmap/guards.hpp")
    }
    fn length_within_bounds(&self, _event: &DetailMapTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::length_within_bounds
        todo!("TODO: port guard `length_within_bounds` from emel.cpp/src/emel/io/mmap/guards.hpp")
    }
    fn mapping_failed(&self, _event: &DetailMapTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::mapping_failed
        todo!("TODO: port guard `mapping_failed` from emel.cpp/src/emel/io/mmap/guards.hpp")
    }
    fn mapping_succeeded(&self, _event: &DetailMapTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::mapping_succeeded
        todo!("TODO: port guard `mapping_succeeded` from emel.cpp/src/emel/io/mmap/guards.hpp")
    }
    fn offset_aligned(&self, _event: &DetailMapTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::offset_aligned
        todo!("TODO: port guard `offset_aligned` from emel.cpp/src/emel/io/mmap/guards.hpp")
    }
    fn offset_unaligned(&self, _event: &DetailMapTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::offset_unaligned
        todo!("TODO: port guard `offset_unaligned` from emel.cpp/src/emel/io/mmap/guards.hpp")
    }
    fn platform_mmap_supported(&self, _event: &DetailMapTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::platform_mmap_supported
        todo!(
            "TODO: port guard `platform_mmap_supported` from emel.cpp/src/emel/io/mmap/guards.hpp"
        )
    }
    fn platform_mmap_unsupported(&self, _event: &DetailMapTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::platform_mmap_unsupported
        todo!(
            "TODO: port guard `platform_mmap_unsupported` from emel.cpp/src/emel/io/mmap/guards.hpp"
        )
    }
    fn release_done_callback_absent(
        &self,
        _event: &DetailReleaseMappingRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::release_done_callback_absent
        todo!(
            "TODO: port guard `release_done_callback_absent` from emel.cpp/src/emel/io/mmap/guards.hpp"
        )
    }
    fn release_done_callback_present(
        &self,
        _event: &DetailReleaseMappingRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::release_done_callback_present
        todo!(
            "TODO: port guard `release_done_callback_present` from emel.cpp/src/emel/io/mmap/guards.hpp"
        )
    }
    fn release_error_callback_absent(
        &self,
        _event: &DetailReleaseMappingRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::release_error_callback_absent
        todo!(
            "TODO: port guard `release_error_callback_absent` from emel.cpp/src/emel/io/mmap/guards.hpp"
        )
    }
    fn release_error_callback_present(
        &self,
        _event: &DetailReleaseMappingRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::release_error_callback_present
        todo!(
            "TODO: port guard `release_error_callback_present` from emel.cpp/src/emel/io/mmap/guards.hpp"
        )
    }
    fn release_handle_in_range(&self, _event: &DetailReleaseMappingRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::release_handle_in_range
        todo!(
            "TODO: port guard `release_handle_in_range` from emel.cpp/src/emel/io/mmap/guards.hpp"
        )
    }
    fn release_handle_out_of_range(
        &self,
        _event: &DetailReleaseMappingRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::release_handle_out_of_range
        todo!(
            "TODO: port guard `release_handle_out_of_range` from emel.cpp/src/emel/io/mmap/guards.hpp"
        )
    }
    fn release_slot_in_use_not_owned_by_tensor(
        &self,
        _event: &DetailReleaseMappingRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::release_slot_in_use_not_owned_by_tensor
        todo!(
            "TODO: port guard `release_slot_in_use_not_owned_by_tensor` from emel.cpp/src/emel/io/mmap/guards.hpp"
        )
    }
    fn release_slot_in_use_owned_by_tensor(
        &self,
        _event: &DetailReleaseMappingRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::release_slot_in_use_owned_by_tensor
        todo!(
            "TODO: port guard `release_slot_in_use_owned_by_tensor` from emel.cpp/src/emel/io/mmap/guards.hpp"
        )
    }
    fn release_slot_not_in_use(&self, _event: &DetailReleaseMappingRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::release_slot_not_in_use
        todo!(
            "TODO: port guard `release_slot_not_in_use` from emel.cpp/src/emel/io/mmap/guards.hpp"
        )
    }
    fn request_span_invalid(&self, _event: &DetailMapTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::request_span_invalid
        todo!("TODO: port guard `request_span_invalid` from emel.cpp/src/emel/io/mmap/guards.hpp")
    }
    fn request_span_valid(&self, _event: &DetailMapTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::request_span_valid
        todo!("TODO: port guard `request_span_valid` from emel.cpp/src/emel/io/mmap/guards.hpp")
    }
    fn slot_capacity_available(&self, _event: &DetailMapTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::slot_capacity_available
        todo!(
            "TODO: port guard `slot_capacity_available` from emel.cpp/src/emel/io/mmap/guards.hpp"
        )
    }
    fn slot_pool_exhausted(&self, _event: &DetailMapTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::slot_pool_exhausted
        todo!("TODO: port guard `slot_pool_exhausted` from emel.cpp/src/emel/io/mmap/guards.hpp")
    }
    fn unmap_failed(&self, _event: &DetailReleaseMappingRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::unmap_failed
        todo!("TODO: port guard `unmap_failed` from emel.cpp/src/emel/io/mmap/guards.hpp")
    }
    fn unmap_succeeded(&self, _event: &DetailReleaseMappingRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/mmap/guards.hpp::unmap_succeeded
        todo!("TODO: port guard `unmap_succeeded` from emel.cpp/src/emel/io/mmap/guards.hpp")
    }
}
