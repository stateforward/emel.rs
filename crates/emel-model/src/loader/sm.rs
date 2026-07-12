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

// --- machine ModelLoader from emel.cpp/src/emel/model/loader/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventLoadRuntime;

sml! {
    ModelLoader {
        "request_decision"_s <= *"ready"_s + event<EventLoadRuntime> / begin_load,
        "parsing"_s <= "request_decision"_s + completion<EventLoadRuntime> [valid_request],
        "errored"_s <= "request_decision"_s + completion<EventLoadRuntime> [invalid_request] / mark_invalid_request_from_request_decision,
        "parse_decision"_s <= "parsing"_s + completion<EventLoadRuntime> / run_parse,
        "parse_phase_decision"_s <= "parse_decision"_s + completion<EventLoadRuntime>,
        "parse_load_tensors_policy_decision"_s <= "parse_phase_decision"_s + completion<EventLoadRuntime> [error_none],
        "errored"_s <= "parse_phase_decision"_s + completion<EventLoadRuntime> [error_invalid_request],
        "errored"_s <= "parse_phase_decision"_s + completion<EventLoadRuntime> [error_parse_failed],
        "errored"_s <= "parse_phase_decision"_s + completion<EventLoadRuntime> [error_backend_error],
        "errored"_s <= "parse_phase_decision"_s + completion<EventLoadRuntime> [error_model_invalid],
        "errored"_s <= "parse_phase_decision"_s + completion<EventLoadRuntime> [error_internal_error],
        "errored"_s <= "parse_phase_decision"_s + completion<EventLoadRuntime> [error_untracked],
        "errored"_s <= "parse_phase_decision"_s + completion<EventLoadRuntime> [error_io_strategy_unavailable],
        "errored"_s <= "parse_phase_decision"_s + completion<EventLoadRuntime> [error_unclassified_code],
        "parse_load_tensors_handler_decision"_s <= "parse_load_tensors_policy_decision"_s + completion<EventLoadRuntime> [should_load_tensors],
        "structure_decision"_s <= "parse_load_tensors_policy_decision"_s + completion<EventLoadRuntime> [skip_load_tensors],
        "errored"_s <= "parse_load_tensors_policy_decision"_s + completion<EventLoadRuntime> / mark_internal_error_from_parse_load_tensors_policy_decision,
        "loading_tensors"_s <= "parse_load_tensors_handler_decision"_s + completion<EventLoadRuntime> [can_load_tensors],
        "errored"_s <= "parse_load_tensors_handler_decision"_s + completion<EventLoadRuntime> [model_has_no_tensors] / mark_model_invalid,
        "errored"_s <= "parse_load_tensors_handler_decision"_s + completion<EventLoadRuntime> [cannot_load_tensors] / mark_invalid_request_from_parse_load_tensors_handler_decision,
        "errored"_s <= "parse_load_tensors_handler_decision"_s + completion<EventLoadRuntime> / mark_internal_error_from_parse_load_tensors_handler_decision,
        "state_tensor_bind_decision"_s <= "loading_tensors"_s + completion<EventLoadRuntime> / effect_dispatch_tensor_bind_storage,
        "state_tensor_plan_dispatch"_s <= "state_tensor_bind_decision"_s + completion<EventLoadRuntime> [tensor_bind_done_raised],
        "errored"_s <= "state_tensor_bind_decision"_s + completion<EventLoadRuntime> [tensor_bind_error_raised] / effect_mark_tensor_bind_error,
        "errored"_s <= "state_tensor_bind_decision"_s + completion<EventLoadRuntime> [tensor_bind_unhandled] / mark_internal_error_from_state_tensor_bind_decision,
        "state_tensor_plan_decision"_s <= "state_tensor_plan_dispatch"_s + completion<EventLoadRuntime> / effect_dispatch_tensor_plan_load,
        "state_tensor_apply_dispatch"_s <= "state_tensor_plan_decision"_s + completion<EventLoadRuntime> [tensor_plan_done_without_io_strategy],
        "state_tensor_effect_error_cleanup"_s <= "state_tensor_plan_decision"_s + completion<EventLoadRuntime> [tensor_plan_done_with_io_strategy_without_loader] / effect_dispatch_tensor_apply_error_results_from_state_tensor_plan_decision,
        "state_io_load_dispatch"_s <= "state_tensor_plan_decision"_s + completion<EventLoadRuntime> [tensor_plan_done_with_io_strategy_with_loader_and_batch_span_ready],
        "state_io_read_copy_load_dispatch"_s <= "state_tensor_plan_decision"_s + completion<EventLoadRuntime> [tensor_plan_done_with_storage_backed_strategy_with_loader_and_storage_ready],
        "state_tensor_effect_error_cleanup"_s <= "state_tensor_plan_decision"_s + completion<EventLoadRuntime> [tensor_plan_done_with_storage_backed_strategy_with_loader_and_storage_missing] / effect_dispatch_tensor_apply_error_results_from_state_tensor_plan_decision,
        "state_tensor_effect_error_cleanup"_s <= "state_tensor_plan_decision"_s + completion<EventLoadRuntime> [tensor_plan_done_with_io_strategy_with_loader_and_batch_span_missing] / effect_dispatch_tensor_apply_error_results_from_state_tensor_plan_decision,
        "errored"_s <= "state_tensor_plan_decision"_s + completion<EventLoadRuntime> [tensor_plan_error_raised] / effect_mark_tensor_plan_error,
        "errored"_s <= "state_tensor_plan_decision"_s + completion<EventLoadRuntime> [tensor_plan_unhandled] / mark_internal_error_from_state_tensor_plan_decision,
        "state_io_load_decision"_s <= "state_io_load_dispatch"_s + completion<EventLoadRuntime> / effect_dispatch_io_load_batch,
        "state_io_load_decision"_s <= "state_io_read_copy_load_dispatch"_s + completion<EventLoadRuntime> / effect_dispatch_io_read_copy_load_batch,
        "state_tensor_apply_dispatch"_s <= "state_io_load_decision"_s + completion<EventLoadRuntime> [io_load_done_all] / effect_mark_io_strategy_used,
        "state_tensor_effect_error_cleanup"_s <= "state_io_load_decision"_s + completion<EventLoadRuntime> [io_load_error_raised] / effect_dispatch_tensor_apply_error_results_from_state_io_load_decision,
        "state_tensor_effect_error_cleanup"_s <= "state_io_load_decision"_s + completion<EventLoadRuntime> [io_load_unhandled] / effect_dispatch_tensor_apply_error_results_from_state_io_load_decision,
        "errored"_s <= "state_tensor_effect_error_cleanup"_s + completion<EventLoadRuntime> [tensor_plan_done_with_io_strategy_without_loader] / effect_mark_io_strategy_unavailable_from_state_tensor_effect_error_cleanup,
        "errored"_s <= "state_tensor_effect_error_cleanup"_s + completion<EventLoadRuntime> [tensor_plan_done_with_io_strategy_with_loader_and_batch_span_missing] / effect_mark_io_strategy_unavailable_from_state_tensor_effect_error_cleanup,
        "errored"_s <= "state_tensor_effect_error_cleanup"_s + completion<EventLoadRuntime> [tensor_plan_done_with_storage_backed_strategy_with_loader_and_storage_missing] / mark_invalid_request_from_state_tensor_effect_error_cleanup,
        "errored"_s <= "state_tensor_effect_error_cleanup"_s + completion<EventLoadRuntime> [io_load_error_invalid_request] / mark_invalid_request_from_state_tensor_effect_error_cleanup,
        "errored"_s <= "state_tensor_effect_error_cleanup"_s + completion<EventLoadRuntime> [io_load_error_strategy_unavailable] / effect_mark_io_strategy_unavailable_from_state_tensor_effect_error_cleanup,
        "errored"_s <= "state_tensor_effect_error_cleanup"_s + completion<EventLoadRuntime> [io_load_error_internal] / mark_internal_error_from_state_tensor_effect_error_cleanup,
        "errored"_s <= "state_tensor_effect_error_cleanup"_s + completion<EventLoadRuntime> [io_load_error_untracked] / mark_untracked,
        "errored"_s <= "state_tensor_effect_error_cleanup"_s + completion<EventLoadRuntime> [io_load_error_unclassified] / mark_internal_error_from_state_tensor_effect_error_cleanup,
        "errored"_s <= "state_tensor_effect_error_cleanup"_s + completion<EventLoadRuntime> [io_load_unhandled] / mark_internal_error_from_state_tensor_effect_error_cleanup,
        "errored"_s <= "state_tensor_effect_error_cleanup"_s + completion<EventLoadRuntime> / mark_internal_error_from_state_tensor_effect_error_cleanup,
        "state_tensor_apply_decision"_s <= "state_tensor_apply_dispatch"_s + completion<EventLoadRuntime> / effect_dispatch_tensor_apply_results,
        "load_map_policy_decision"_s <= "state_tensor_apply_decision"_s + completion<EventLoadRuntime> [tensor_apply_done_with_file_image] / effect_publish_tensor_load_done_from_file_image,
        "load_map_policy_decision"_s <= "state_tensor_apply_decision"_s + completion<EventLoadRuntime> [tensor_apply_done_without_file_image] / effect_publish_tensor_load_done_from_model_data,
        "errored"_s <= "state_tensor_apply_decision"_s + completion<EventLoadRuntime> [tensor_apply_error_raised] / effect_mark_tensor_apply_error,
        "errored"_s <= "state_tensor_apply_decision"_s + completion<EventLoadRuntime> [tensor_apply_unhandled] / mark_internal_error_from_state_tensor_apply_decision,
        "mapping_layers"_s <= "load_map_policy_decision"_s + completion<EventLoadRuntime> [can_map_layers],
        "errored"_s <= "load_map_policy_decision"_s + completion<EventLoadRuntime> [cannot_map_layers] / mark_invalid_request_from_load_map_policy_decision,
        "errored"_s <= "load_map_policy_decision"_s + completion<EventLoadRuntime> / mark_internal_error_from_load_map_policy_decision,
        "map_layers_decision"_s <= "mapping_layers"_s + completion<EventLoadRuntime> / run_map_layers,
        "structure_decision"_s <= "map_layers_decision"_s + completion<EventLoadRuntime> [error_none],
        "errored"_s <= "map_layers_decision"_s + completion<EventLoadRuntime> [error_invalid_request],
        "errored"_s <= "map_layers_decision"_s + completion<EventLoadRuntime> [error_parse_failed],
        "errored"_s <= "map_layers_decision"_s + completion<EventLoadRuntime> [error_backend_error],
        "errored"_s <= "map_layers_decision"_s + completion<EventLoadRuntime> [error_model_invalid],
        "errored"_s <= "map_layers_decision"_s + completion<EventLoadRuntime> [error_internal_error],
        "errored"_s <= "map_layers_decision"_s + completion<EventLoadRuntime> [error_untracked],
        "errored"_s <= "map_layers_decision"_s + completion<EventLoadRuntime> [error_io_strategy_unavailable],
        "errored"_s <= "map_layers_decision"_s + completion<EventLoadRuntime> [error_unclassified_code],
        "structure_policy_decision"_s <= "structure_decision"_s + completion<EventLoadRuntime>,
        "architecture_decision"_s <= "structure_policy_decision"_s + completion<EventLoadRuntime> [skip_validate_structure],
        "validating_structure"_s <= "structure_policy_decision"_s + completion<EventLoadRuntime> [can_validate_structure],
        "errored"_s <= "structure_policy_decision"_s + completion<EventLoadRuntime> [cannot_validate_structure] / mark_invalid_request_from_structure_policy_decision,
        "errored"_s <= "structure_policy_decision"_s + completion<EventLoadRuntime> / mark_internal_error_from_structure_policy_decision,
        "structure_validation_decision"_s <= "validating_structure"_s + completion<EventLoadRuntime> / run_validate_structure,
        "architecture_decision"_s <= "structure_validation_decision"_s + completion<EventLoadRuntime> [error_none],
        "errored"_s <= "structure_validation_decision"_s + completion<EventLoadRuntime> [error_invalid_request],
        "errored"_s <= "structure_validation_decision"_s + completion<EventLoadRuntime> [error_parse_failed],
        "errored"_s <= "structure_validation_decision"_s + completion<EventLoadRuntime> [error_backend_error],
        "errored"_s <= "structure_validation_decision"_s + completion<EventLoadRuntime> [error_model_invalid],
        "errored"_s <= "structure_validation_decision"_s + completion<EventLoadRuntime> [error_internal_error],
        "errored"_s <= "structure_validation_decision"_s + completion<EventLoadRuntime> [error_untracked],
        "errored"_s <= "structure_validation_decision"_s + completion<EventLoadRuntime> [error_io_strategy_unavailable],
        "errored"_s <= "structure_validation_decision"_s + completion<EventLoadRuntime> [error_unclassified_code],
        "architecture_policy_decision"_s <= "architecture_decision"_s + completion<EventLoadRuntime>,
        "done"_s <= "architecture_policy_decision"_s + completion<EventLoadRuntime> [skip_validate_architecture],
        "validating_architecture"_s <= "architecture_policy_decision"_s + completion<EventLoadRuntime> [can_validate_architecture],
        "errored"_s <= "architecture_policy_decision"_s + completion<EventLoadRuntime> [cannot_validate_architecture] / mark_invalid_request_from_architecture_policy_decision,
        "errored"_s <= "architecture_policy_decision"_s + completion<EventLoadRuntime> / mark_internal_error_from_architecture_policy_decision,
        "architecture_validation_decision"_s <= "validating_architecture"_s + completion<EventLoadRuntime> / run_validate_architecture,
        "done"_s <= "architecture_validation_decision"_s + completion<EventLoadRuntime> [error_none],
        "errored"_s <= "architecture_validation_decision"_s + completion<EventLoadRuntime> [error_invalid_request],
        "errored"_s <= "architecture_validation_decision"_s + completion<EventLoadRuntime> [error_parse_failed],
        "errored"_s <= "architecture_validation_decision"_s + completion<EventLoadRuntime> [error_backend_error],
        "errored"_s <= "architecture_validation_decision"_s + completion<EventLoadRuntime> [error_model_invalid],
        "errored"_s <= "architecture_validation_decision"_s + completion<EventLoadRuntime> [error_internal_error],
        "errored"_s <= "architecture_validation_decision"_s + completion<EventLoadRuntime> [error_untracked],
        "errored"_s <= "architecture_validation_decision"_s + completion<EventLoadRuntime> [error_io_strategy_unavailable],
        "errored"_s <= "architecture_validation_decision"_s + completion<EventLoadRuntime> [error_unclassified_code],
        "ready"_s <= "done"_s + completion<EventLoadRuntime> [done_callback_present] / publish_done,
        "ready"_s <= "done"_s + completion<EventLoadRuntime> [done_callback_absent] / publish_done_noop,
        "ready"_s <= "errored"_s + completion<EventLoadRuntime> [error_callback_present] / publish_error,
        "ready"_s <= "errored"_s + completion<EventLoadRuntime> [error_callback_absent] / publish_error_noop,
        "ready"_s <= "ready"_s + unexpected_event<_> / on_unexpected_from_ready,
        "ready"_s <= "request_decision"_s + unexpected_event<_> / on_unexpected_from_request_decision,
        "ready"_s <= "parsing"_s + unexpected_event<_> / on_unexpected_from_parsing,
        "ready"_s <= "parse_decision"_s + unexpected_event<_> / on_unexpected_from_parse_decision,
        "ready"_s <= "parse_phase_decision"_s + unexpected_event<_> / on_unexpected_from_parse_phase_decision,
        "ready"_s <= "parse_load_tensors_policy_decision"_s + unexpected_event<_> / on_unexpected_from_parse_load_tensors_policy_decision,
        "ready"_s <= "parse_load_tensors_handler_decision"_s + unexpected_event<_> / on_unexpected_from_parse_load_tensors_handler_decision,
        "ready"_s <= "loading_tensors"_s + unexpected_event<_> / on_unexpected_from_loading_tensors,
        "ready"_s <= "state_tensor_bind_decision"_s + unexpected_event<_> / on_unexpected_from_state_tensor_bind_decision,
        "ready"_s <= "state_tensor_plan_dispatch"_s + unexpected_event<_> / on_unexpected_from_state_tensor_plan_dispatch,
        "ready"_s <= "state_tensor_plan_decision"_s + unexpected_event<_> / on_unexpected_from_state_tensor_plan_decision,
        "ready"_s <= "state_io_load_dispatch"_s + unexpected_event<_> / on_unexpected_from_state_io_load_dispatch,
        "ready"_s <= "state_io_read_copy_load_dispatch"_s + unexpected_event<_> / on_unexpected_from_state_io_read_copy_load_dispatch,
        "ready"_s <= "state_io_load_decision"_s + unexpected_event<_> / on_unexpected_from_state_io_load_decision,
        "ready"_s <= "state_tensor_effect_error_cleanup"_s + unexpected_event<_> / on_unexpected_from_state_tensor_effect_error_cleanup,
        "ready"_s <= "state_tensor_apply_dispatch"_s + unexpected_event<_> / on_unexpected_from_state_tensor_apply_dispatch,
        "ready"_s <= "state_tensor_apply_decision"_s + unexpected_event<_> / on_unexpected_from_state_tensor_apply_decision,
        "ready"_s <= "load_map_policy_decision"_s + unexpected_event<_> / on_unexpected_from_load_map_policy_decision,
        "ready"_s <= "mapping_layers"_s + unexpected_event<_> / on_unexpected_from_mapping_layers,
        "ready"_s <= "map_layers_decision"_s + unexpected_event<_> / on_unexpected_from_map_layers_decision,
        "ready"_s <= "structure_decision"_s + unexpected_event<_> / on_unexpected_from_structure_decision,
        "ready"_s <= "structure_policy_decision"_s + unexpected_event<_> / on_unexpected_from_structure_policy_decision,
        "ready"_s <= "validating_structure"_s + unexpected_event<_> / on_unexpected_from_validating_structure,
        "ready"_s <= "structure_validation_decision"_s + unexpected_event<_> / on_unexpected_from_structure_validation_decision,
        "ready"_s <= "architecture_decision"_s + unexpected_event<_> / on_unexpected_from_architecture_decision,
        "ready"_s <= "architecture_policy_decision"_s + unexpected_event<_> / on_unexpected_from_architecture_policy_decision,
        "ready"_s <= "validating_architecture"_s + unexpected_event<_> / on_unexpected_from_validating_architecture,
        "ready"_s <= "architecture_validation_decision"_s + unexpected_event<_> / on_unexpected_from_architecture_validation_decision,
        "ready"_s <= "done"_s + unexpected_event<_> / on_unexpected_from_done,
        "ready"_s <= "errored"_s + unexpected_event<_> / on_unexpected_from_errored,
    }
}

/// Context for `ModelLoader` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct ModelLoaderContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl ModelLoaderStateMachineContext for ModelLoaderContext {
    fn begin_load(&mut self, _event: &EventLoadRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::begin_load
        todo!("TODO: port action `begin_load` from emel.cpp/src/emel/model/loader/actions.hpp")
    }
    fn can_load_tensors(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::can_load_tensors
        todo!("TODO: port guard `can_load_tensors` from emel.cpp/src/emel/model/loader/guards.hpp")
    }
    fn can_map_layers(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::can_map_layers
        todo!("TODO: port guard `can_map_layers` from emel.cpp/src/emel/model/loader/guards.hpp")
    }
    fn can_validate_architecture(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::can_validate_architecture
        todo!(
            "TODO: port guard `can_validate_architecture` from emel.cpp/src/emel/model/loader/guards.hpp"
        )
    }
    fn can_validate_structure(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::can_validate_structure
        todo!(
            "TODO: port guard `can_validate_structure` from emel.cpp/src/emel/model/loader/guards.hpp"
        )
    }
    fn cannot_load_tensors(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::cannot_load_tensors
        todo!(
            "TODO: port guard `cannot_load_tensors` from emel.cpp/src/emel/model/loader/guards.hpp"
        )
    }
    fn cannot_map_layers(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::cannot_map_layers
        todo!("TODO: port guard `cannot_map_layers` from emel.cpp/src/emel/model/loader/guards.hpp")
    }
    fn cannot_validate_architecture(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::cannot_validate_architecture
        todo!(
            "TODO: port guard `cannot_validate_architecture` from emel.cpp/src/emel/model/loader/guards.hpp"
        )
    }
    fn cannot_validate_structure(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::cannot_validate_structure
        todo!(
            "TODO: port guard `cannot_validate_structure` from emel.cpp/src/emel/model/loader/guards.hpp"
        )
    }
    fn done_callback_absent(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::done_callback_absent
        todo!(
            "TODO: port guard `done_callback_absent` from emel.cpp/src/emel/model/loader/guards.hpp"
        )
    }
    fn done_callback_present(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::done_callback_present
        todo!(
            "TODO: port guard `done_callback_present` from emel.cpp/src/emel/model/loader/guards.hpp"
        )
    }
    fn effect_dispatch_io_load_batch(&mut self, _event: &EventLoadRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::effect_dispatch_io_load_batch
        todo!(
            "TODO: port action `effect_dispatch_io_load_batch` from emel.cpp/src/emel/model/loader/actions.hpp"
        )
    }
    fn effect_dispatch_io_read_copy_load_batch(
        &mut self,
        _event: &EventLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::effect_dispatch_io_read_copy_load_batch
        todo!(
            "TODO: port action `effect_dispatch_io_read_copy_load_batch` from emel.cpp/src/emel/model/loader/actions.hpp"
        )
    }
    fn effect_dispatch_tensor_apply_error_results_from_state_io_load_decision(
        &mut self,
        _event: &EventLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::effect_dispatch_tensor_apply_error_results
        todo!(
            "TODO: port action `effect_dispatch_tensor_apply_error_results` from emel.cpp/src/emel/model/loader/actions.hpp"
        )
    }
    fn effect_dispatch_tensor_apply_error_results_from_state_tensor_plan_decision(
        &mut self,
        _event: &EventLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::effect_dispatch_tensor_apply_error_results
        todo!(
            "TODO: port action `effect_dispatch_tensor_apply_error_results` from emel.cpp/src/emel/model/loader/actions.hpp"
        )
    }
    fn effect_dispatch_tensor_apply_results(
        &mut self,
        _event: &EventLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::effect_dispatch_tensor_apply_results
        todo!(
            "TODO: port action `effect_dispatch_tensor_apply_results` from emel.cpp/src/emel/model/loader/actions.hpp"
        )
    }
    fn effect_dispatch_tensor_bind_storage(&mut self, _event: &EventLoadRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::effect_dispatch_tensor_bind_storage
        todo!(
            "TODO: port action `effect_dispatch_tensor_bind_storage` from emel.cpp/src/emel/model/loader/actions.hpp"
        )
    }
    fn effect_dispatch_tensor_plan_load(&mut self, _event: &EventLoadRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::effect_dispatch_tensor_plan_load
        todo!(
            "TODO: port action `effect_dispatch_tensor_plan_load` from emel.cpp/src/emel/model/loader/actions.hpp"
        )
    }
    fn effect_mark_io_strategy_unavailable_from_state_tensor_effect_error_cleanup(
        &mut self,
        _event: &EventLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::effect_mark_io_strategy_unavailable
        todo!(
            "TODO: port action `effect_mark_io_strategy_unavailable` from emel.cpp/src/emel/model/loader/actions.hpp"
        )
    }
    fn effect_mark_io_strategy_used(&mut self, _event: &EventLoadRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::effect_mark_io_strategy_used
        todo!(
            "TODO: port action `effect_mark_io_strategy_used` from emel.cpp/src/emel/model/loader/actions.hpp"
        )
    }
    fn effect_mark_tensor_apply_error(&mut self, _event: &EventLoadRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::effect_mark_tensor_apply_error
        todo!(
            "TODO: port action `effect_mark_tensor_apply_error` from emel.cpp/src/emel/model/loader/actions.hpp"
        )
    }
    fn effect_mark_tensor_bind_error(&mut self, _event: &EventLoadRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::effect_mark_tensor_bind_error
        todo!(
            "TODO: port action `effect_mark_tensor_bind_error` from emel.cpp/src/emel/model/loader/actions.hpp"
        )
    }
    fn effect_mark_tensor_plan_error(&mut self, _event: &EventLoadRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::effect_mark_tensor_plan_error
        todo!(
            "TODO: port action `effect_mark_tensor_plan_error` from emel.cpp/src/emel/model/loader/actions.hpp"
        )
    }
    fn effect_publish_tensor_load_done_from_file_image(
        &mut self,
        _event: &EventLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::effect_publish_tensor_load_done_from_file_image
        todo!(
            "TODO: port action `effect_publish_tensor_load_done_from_file_image` from emel.cpp/src/emel/model/loader/actions.hpp"
        )
    }
    fn effect_publish_tensor_load_done_from_model_data(
        &mut self,
        _event: &EventLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::effect_publish_tensor_load_done_from_model_data
        todo!(
            "TODO: port action `effect_publish_tensor_load_done_from_model_data` from emel.cpp/src/emel/model/loader/actions.hpp"
        )
    }
    fn error_backend_error(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::error_backend_error
        todo!(
            "TODO: port guard `error_backend_error` from emel.cpp/src/emel/model/loader/guards.hpp"
        )
    }
    fn error_callback_absent(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::error_callback_absent
        todo!(
            "TODO: port guard `error_callback_absent` from emel.cpp/src/emel/model/loader/guards.hpp"
        )
    }
    fn error_callback_present(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::error_callback_present
        todo!(
            "TODO: port guard `error_callback_present` from emel.cpp/src/emel/model/loader/guards.hpp"
        )
    }
    fn error_internal_error(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::error_internal_error
        todo!(
            "TODO: port guard `error_internal_error` from emel.cpp/src/emel/model/loader/guards.hpp"
        )
    }
    fn error_invalid_request(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::error_invalid_request
        todo!(
            "TODO: port guard `error_invalid_request` from emel.cpp/src/emel/model/loader/guards.hpp"
        )
    }
    fn error_io_strategy_unavailable(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::error_io_strategy_unavailable
        todo!(
            "TODO: port guard `error_io_strategy_unavailable` from emel.cpp/src/emel/model/loader/guards.hpp"
        )
    }
    fn error_model_invalid(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::error_model_invalid
        todo!(
            "TODO: port guard `error_model_invalid` from emel.cpp/src/emel/model/loader/guards.hpp"
        )
    }
    fn error_none(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::error_none
        todo!("TODO: port guard `error_none` from emel.cpp/src/emel/model/loader/guards.hpp")
    }
    fn error_parse_failed(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::error_parse_failed
        todo!(
            "TODO: port guard `error_parse_failed` from emel.cpp/src/emel/model/loader/guards.hpp"
        )
    }
    fn error_unclassified_code(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::error_unclassified_code
        todo!(
            "TODO: port guard `error_unclassified_code` from emel.cpp/src/emel/model/loader/guards.hpp"
        )
    }
    fn error_untracked(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::error_untracked
        todo!("TODO: port guard `error_untracked` from emel.cpp/src/emel/model/loader/guards.hpp")
    }
    fn invalid_request(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::invalid_request
        todo!("TODO: port guard `invalid_request` from emel.cpp/src/emel/model/loader/guards.hpp")
    }
    fn io_load_done_all(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::io_load_done_all
        todo!("TODO: port guard `io_load_done_all` from emel.cpp/src/emel/model/loader/guards.hpp")
    }
    fn io_load_error_internal(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::io_load_error_internal
        todo!(
            "TODO: port guard `io_load_error_internal` from emel.cpp/src/emel/model/loader/guards.hpp"
        )
    }
    fn io_load_error_invalid_request(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::io_load_error_invalid_request
        todo!(
            "TODO: port guard `io_load_error_invalid_request` from emel.cpp/src/emel/model/loader/guards.hpp"
        )
    }
    fn io_load_error_raised(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::io_load_error_raised
        todo!(
            "TODO: port guard `io_load_error_raised` from emel.cpp/src/emel/model/loader/guards.hpp"
        )
    }
    fn io_load_error_strategy_unavailable(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::io_load_error_strategy_unavailable
        todo!(
            "TODO: port guard `io_load_error_strategy_unavailable` from emel.cpp/src/emel/model/loader/guards.hpp"
        )
    }
    fn io_load_error_unclassified(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::io_load_error_unclassified
        todo!(
            "TODO: port guard `io_load_error_unclassified` from emel.cpp/src/emel/model/loader/guards.hpp"
        )
    }
    fn io_load_error_untracked(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::io_load_error_untracked
        todo!(
            "TODO: port guard `io_load_error_untracked` from emel.cpp/src/emel/model/loader/guards.hpp"
        )
    }
    fn io_load_unhandled(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::io_load_unhandled
        todo!("TODO: port guard `io_load_unhandled` from emel.cpp/src/emel/model/loader/guards.hpp")
    }
    fn mark_internal_error_from_architecture_policy_decision(
        &mut self,
        _event: &EventLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::mark_internal_error
        todo!(
            "TODO: port action `mark_internal_error` from emel.cpp/src/emel/model/loader/actions.hpp"
        )
    }
    fn mark_internal_error_from_load_map_policy_decision(
        &mut self,
        _event: &EventLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::mark_internal_error
        todo!(
            "TODO: port action `mark_internal_error` from emel.cpp/src/emel/model/loader/actions.hpp"
        )
    }
    fn mark_internal_error_from_parse_load_tensors_handler_decision(
        &mut self,
        _event: &EventLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::mark_internal_error
        todo!(
            "TODO: port action `mark_internal_error` from emel.cpp/src/emel/model/loader/actions.hpp"
        )
    }
    fn mark_internal_error_from_parse_load_tensors_policy_decision(
        &mut self,
        _event: &EventLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::mark_internal_error
        todo!(
            "TODO: port action `mark_internal_error` from emel.cpp/src/emel/model/loader/actions.hpp"
        )
    }
    fn mark_internal_error_from_state_tensor_apply_decision(
        &mut self,
        _event: &EventLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::mark_internal_error
        todo!(
            "TODO: port action `mark_internal_error` from emel.cpp/src/emel/model/loader/actions.hpp"
        )
    }
    fn mark_internal_error_from_state_tensor_bind_decision(
        &mut self,
        _event: &EventLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::mark_internal_error
        todo!(
            "TODO: port action `mark_internal_error` from emel.cpp/src/emel/model/loader/actions.hpp"
        )
    }
    fn mark_internal_error_from_state_tensor_effect_error_cleanup(
        &mut self,
        _event: &EventLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::mark_internal_error
        todo!(
            "TODO: port action `mark_internal_error` from emel.cpp/src/emel/model/loader/actions.hpp"
        )
    }
    fn mark_internal_error_from_state_tensor_plan_decision(
        &mut self,
        _event: &EventLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::mark_internal_error
        todo!(
            "TODO: port action `mark_internal_error` from emel.cpp/src/emel/model/loader/actions.hpp"
        )
    }
    fn mark_internal_error_from_structure_policy_decision(
        &mut self,
        _event: &EventLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::mark_internal_error
        todo!(
            "TODO: port action `mark_internal_error` from emel.cpp/src/emel/model/loader/actions.hpp"
        )
    }
    fn mark_invalid_request_from_architecture_policy_decision(
        &mut self,
        _event: &EventLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/model/loader/actions.hpp"
        )
    }
    fn mark_invalid_request_from_load_map_policy_decision(
        &mut self,
        _event: &EventLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/model/loader/actions.hpp"
        )
    }
    fn mark_invalid_request_from_parse_load_tensors_handler_decision(
        &mut self,
        _event: &EventLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/model/loader/actions.hpp"
        )
    }
    fn mark_invalid_request_from_request_decision(
        &mut self,
        _event: &EventLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/model/loader/actions.hpp"
        )
    }
    fn mark_invalid_request_from_state_tensor_effect_error_cleanup(
        &mut self,
        _event: &EventLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/model/loader/actions.hpp"
        )
    }
    fn mark_invalid_request_from_structure_policy_decision(
        &mut self,
        _event: &EventLoadRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/model/loader/actions.hpp"
        )
    }
    fn mark_model_invalid(&mut self, _event: &EventLoadRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::mark_model_invalid
        todo!(
            "TODO: port action `mark_model_invalid` from emel.cpp/src/emel/model/loader/actions.hpp"
        )
    }
    fn mark_untracked(&mut self, _event: &EventLoadRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::mark_untracked
        todo!("TODO: port action `mark_untracked` from emel.cpp/src/emel/model/loader/actions.hpp")
    }
    fn model_has_no_tensors(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::model_has_no_tensors
        todo!(
            "TODO: port guard `model_has_no_tensors` from emel.cpp/src/emel/model/loader/guards.hpp"
        )
    }
    fn on_unexpected_from_architecture_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/loader/actions.hpp")
    }
    fn on_unexpected_from_architecture_policy_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/loader/actions.hpp")
    }
    fn on_unexpected_from_architecture_validation_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/loader/actions.hpp")
    }
    fn on_unexpected_from_done(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/loader/actions.hpp")
    }
    fn on_unexpected_from_errored(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/loader/actions.hpp")
    }
    fn on_unexpected_from_load_map_policy_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/loader/actions.hpp")
    }
    fn on_unexpected_from_loading_tensors(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/loader/actions.hpp")
    }
    fn on_unexpected_from_map_layers_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/loader/actions.hpp")
    }
    fn on_unexpected_from_mapping_layers(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/loader/actions.hpp")
    }
    fn on_unexpected_from_parse_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/loader/actions.hpp")
    }
    fn on_unexpected_from_parse_load_tensors_handler_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/loader/actions.hpp")
    }
    fn on_unexpected_from_parse_load_tensors_policy_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/loader/actions.hpp")
    }
    fn on_unexpected_from_parse_phase_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/loader/actions.hpp")
    }
    fn on_unexpected_from_parsing(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/loader/actions.hpp")
    }
    fn on_unexpected_from_ready(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/loader/actions.hpp")
    }
    fn on_unexpected_from_request_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/loader/actions.hpp")
    }
    fn on_unexpected_from_state_io_load_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/loader/actions.hpp")
    }
    fn on_unexpected_from_state_io_load_dispatch(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/loader/actions.hpp")
    }
    fn on_unexpected_from_state_io_read_copy_load_dispatch(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/loader/actions.hpp")
    }
    fn on_unexpected_from_state_tensor_apply_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/loader/actions.hpp")
    }
    fn on_unexpected_from_state_tensor_apply_dispatch(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/loader/actions.hpp")
    }
    fn on_unexpected_from_state_tensor_bind_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/loader/actions.hpp")
    }
    fn on_unexpected_from_state_tensor_effect_error_cleanup(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/loader/actions.hpp")
    }
    fn on_unexpected_from_state_tensor_plan_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/loader/actions.hpp")
    }
    fn on_unexpected_from_state_tensor_plan_dispatch(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/loader/actions.hpp")
    }
    fn on_unexpected_from_structure_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/loader/actions.hpp")
    }
    fn on_unexpected_from_structure_policy_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/loader/actions.hpp")
    }
    fn on_unexpected_from_structure_validation_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/loader/actions.hpp")
    }
    fn on_unexpected_from_validating_architecture(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/loader/actions.hpp")
    }
    fn on_unexpected_from_validating_structure(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/model/loader/actions.hpp")
    }
    fn publish_done(&mut self, _event: &EventLoadRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::publish_done
        todo!("TODO: port action `publish_done` from emel.cpp/src/emel/model/loader/actions.hpp")
    }
    fn publish_done_noop(&mut self, _event: &EventLoadRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::publish_done_noop
        todo!(
            "TODO: port action `publish_done_noop` from emel.cpp/src/emel/model/loader/actions.hpp"
        )
    }
    fn publish_error(&mut self, _event: &EventLoadRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::publish_error
        todo!("TODO: port action `publish_error` from emel.cpp/src/emel/model/loader/actions.hpp")
    }
    fn publish_error_noop(&mut self, _event: &EventLoadRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::publish_error_noop
        todo!(
            "TODO: port action `publish_error_noop` from emel.cpp/src/emel/model/loader/actions.hpp"
        )
    }
    fn run_map_layers(&mut self, _event: &EventLoadRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::run_map_layers
        todo!("TODO: port action `run_map_layers` from emel.cpp/src/emel/model/loader/actions.hpp")
    }
    fn run_parse(&mut self, _event: &EventLoadRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::run_parse
        todo!("TODO: port action `run_parse` from emel.cpp/src/emel/model/loader/actions.hpp")
    }
    fn run_validate_architecture(&mut self, _event: &EventLoadRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::run_validate_architecture
        todo!(
            "TODO: port action `run_validate_architecture` from emel.cpp/src/emel/model/loader/actions.hpp"
        )
    }
    fn run_validate_structure(&mut self, _event: &EventLoadRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/actions.hpp::run_validate_structure
        todo!(
            "TODO: port action `run_validate_structure` from emel.cpp/src/emel/model/loader/actions.hpp"
        )
    }
    fn should_load_tensors(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::should_load_tensors
        todo!(
            "TODO: port guard `should_load_tensors` from emel.cpp/src/emel/model/loader/guards.hpp"
        )
    }
    fn skip_load_tensors(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::skip_load_tensors
        todo!("TODO: port guard `skip_load_tensors` from emel.cpp/src/emel/model/loader/guards.hpp")
    }
    fn skip_validate_architecture(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::skip_validate_architecture
        todo!(
            "TODO: port guard `skip_validate_architecture` from emel.cpp/src/emel/model/loader/guards.hpp"
        )
    }
    fn skip_validate_structure(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::skip_validate_structure
        todo!(
            "TODO: port guard `skip_validate_structure` from emel.cpp/src/emel/model/loader/guards.hpp"
        )
    }
    fn tensor_apply_done_with_file_image(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::tensor_apply_done_with_file_image
        todo!(
            "TODO: port guard `tensor_apply_done_with_file_image` from emel.cpp/src/emel/model/loader/guards.hpp"
        )
    }
    fn tensor_apply_done_without_file_image(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::tensor_apply_done_without_file_image
        todo!(
            "TODO: port guard `tensor_apply_done_without_file_image` from emel.cpp/src/emel/model/loader/guards.hpp"
        )
    }
    fn tensor_apply_error_raised(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::tensor_apply_error_raised
        todo!(
            "TODO: port guard `tensor_apply_error_raised` from emel.cpp/src/emel/model/loader/guards.hpp"
        )
    }
    fn tensor_apply_unhandled(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::tensor_apply_unhandled
        todo!(
            "TODO: port guard `tensor_apply_unhandled` from emel.cpp/src/emel/model/loader/guards.hpp"
        )
    }
    fn tensor_bind_done_raised(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::tensor_bind_done_raised
        todo!(
            "TODO: port guard `tensor_bind_done_raised` from emel.cpp/src/emel/model/loader/guards.hpp"
        )
    }
    fn tensor_bind_error_raised(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::tensor_bind_error_raised
        todo!(
            "TODO: port guard `tensor_bind_error_raised` from emel.cpp/src/emel/model/loader/guards.hpp"
        )
    }
    fn tensor_bind_unhandled(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::tensor_bind_unhandled
        todo!(
            "TODO: port guard `tensor_bind_unhandled` from emel.cpp/src/emel/model/loader/guards.hpp"
        )
    }
    fn tensor_plan_done_with_io_strategy_with_loader_and_batch_span_missing(
        &self,
        _event: &EventLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::tensor_plan_done_with_io_strategy_with_loader_and_batch_span_missing
        todo!(
            "TODO: port guard `tensor_plan_done_with_io_strategy_with_loader_and_batch_span_missing` from emel.cpp/src/emel/model/loader/guards.hpp"
        )
    }
    fn tensor_plan_done_with_io_strategy_with_loader_and_batch_span_ready(
        &self,
        _event: &EventLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::tensor_plan_done_with_io_strategy_with_loader_and_batch_span_ready
        todo!(
            "TODO: port guard `tensor_plan_done_with_io_strategy_with_loader_and_batch_span_ready` from emel.cpp/src/emel/model/loader/guards.hpp"
        )
    }
    fn tensor_plan_done_with_io_strategy_without_loader(
        &self,
        _event: &EventLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::tensor_plan_done_with_io_strategy_without_loader
        todo!(
            "TODO: port guard `tensor_plan_done_with_io_strategy_without_loader` from emel.cpp/src/emel/model/loader/guards.hpp"
        )
    }
    fn tensor_plan_done_with_storage_backed_strategy_with_loader_and_storage_missing(
        &self,
        _event: &EventLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::tensor_plan_done_with_storage_backed_strategy_with_loader_and_storage_missing
        todo!(
            "TODO: port guard `tensor_plan_done_with_storage_backed_strategy_with_loader_and_storage_missing` from emel.cpp/src/emel/model/loader/guards.hpp"
        )
    }
    fn tensor_plan_done_with_storage_backed_strategy_with_loader_and_storage_ready(
        &self,
        _event: &EventLoadRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::tensor_plan_done_with_storage_backed_strategy_with_loader_and_storage_ready
        todo!(
            "TODO: port guard `tensor_plan_done_with_storage_backed_strategy_with_loader_and_storage_ready` from emel.cpp/src/emel/model/loader/guards.hpp"
        )
    }
    fn tensor_plan_done_without_io_strategy(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::tensor_plan_done_without_io_strategy
        todo!(
            "TODO: port guard `tensor_plan_done_without_io_strategy` from emel.cpp/src/emel/model/loader/guards.hpp"
        )
    }
    fn tensor_plan_error_raised(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::tensor_plan_error_raised
        todo!(
            "TODO: port guard `tensor_plan_error_raised` from emel.cpp/src/emel/model/loader/guards.hpp"
        )
    }
    fn tensor_plan_unhandled(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::tensor_plan_unhandled
        todo!(
            "TODO: port guard `tensor_plan_unhandled` from emel.cpp/src/emel/model/loader/guards.hpp"
        )
    }
    fn valid_request(&self, _event: &EventLoadRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/model/loader/guards.hpp::valid_request
        todo!("TODO: port guard `valid_request` from emel.cpp/src/emel/model/loader/guards.hpp")
    }
}
