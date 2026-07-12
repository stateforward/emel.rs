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

// --- machine TextGeneratorInitializer from emel.cpp/src/emel/text/generator/initializer/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventRun;

sml! {
    TextGeneratorInitializer {
        "preparing_backend"_s <= *"idle"_s + event<EventRun> / begin_initialize,
        "binding_conditioner"_s <= "preparing_backend"_s + completion<EventRun> [guard_backend_reuse_allowed],
        "idle"_s <= "preparing_backend"_s + completion<EventRun> [guard_generation_contract_invalid] / mark_invalid_request_from_preparing_backend,
        "preparing_backend_decision"_s <= "preparing_backend"_s + completion<EventRun> [guard_backend_prepare_allowed] / request_backend_prepare,
        "binding_conditioner"_s <= "preparing_backend_decision"_s + completion<EventRun> [backend_already_ready],
        "binding_conditioner"_s <= "preparing_backend_decision"_s + completion<EventRun> [backend_prepare_ok] / accept_prepared_backend,
        "idle"_s <= "preparing_backend_decision"_s + completion<EventRun> [backend_prepare_invalid_request] / mark_invalid_request_from_preparing_backend_decision,
        "idle"_s <= "preparing_backend_decision"_s + completion<EventRun> [backend_prepare_backend_error] / mark_backend_error_from_preparing_backend_decision,
        "binding_conditioner_decision"_s <= "binding_conditioner"_s + completion<EventRun> / request_conditioner_bind,
        "initializing_renderer"_s <= "binding_conditioner_decision"_s + completion<EventRun> [conditioner_bind_ok],
        "idle"_s <= "binding_conditioner_decision"_s + completion<EventRun> [conditioner_bind_invalid_request] / mark_invalid_request_from_binding_conditioner_decision,
        "idle"_s <= "binding_conditioner_decision"_s + completion<EventRun> [conditioner_bind_backend_error] / mark_backend_error_from_binding_conditioner_decision,
        "initializing_renderer_decision"_s <= "initializing_renderer"_s + completion<EventRun> / request_renderer_initialize,
        "reserving_memory"_s <= "initializing_renderer_decision"_s + completion<EventRun> [renderer_initialize_ok],
        "idle"_s <= "initializing_renderer_decision"_s + completion<EventRun> [renderer_initialize_invalid_request] / mark_invalid_request_from_initializing_renderer_decision,
        "idle"_s <= "initializing_renderer_decision"_s + completion<EventRun> [renderer_initialize_backend_error] / mark_backend_error_from_initializing_renderer_decision,
        "reserving_memory_decision"_s <= "reserving_memory"_s + completion<EventRun> [guard_memory_geometry_fits_backend] / request_memory_reserve,
        "idle"_s <= "reserving_memory"_s + completion<EventRun> [guard_memory_geometry_gap] / mark_invalid_request_from_reserving_memory,
        "configuring_sampling_mode_decision"_s <= "reserving_memory_decision"_s + completion<EventRun> [memory_reserve_with_existing_graph],
        "reserving_graph"_s <= "reserving_memory_decision"_s + completion<EventRun> [memory_reserve_with_missing_graph],
        "idle"_s <= "reserving_memory_decision"_s + completion<EventRun> [memory_reserve_invalid_request] / mark_invalid_request_from_reserving_memory_decision,
        "idle"_s <= "reserving_memory_decision"_s + completion<EventRun> [memory_reserve_backend_error] / mark_backend_error_from_reserving_memory_decision,
        "reserving_graph_decision"_s <= "reserving_graph"_s + completion<EventRun> / request_graph_reserve,
        "configuring_sampling_mode_decision"_s <= "reserving_graph_decision"_s + completion<EventRun> [graph_reserve_ok],
        "idle"_s <= "reserving_graph_decision"_s + completion<EventRun> [graph_reserve_invalid_request] / mark_invalid_request_from_reserving_graph_decision,
        "idle"_s <= "reserving_graph_decision"_s + completion<EventRun> [graph_reserve_backend_error] / mark_backend_error_from_reserving_graph_decision,
        "configuring_sampler"_s <= "configuring_sampling_mode_decision"_s + completion<EventRun> [uses_materialized_logits],
        "configure_preselected_argmax"_s <= "configuring_sampling_mode_decision"_s + completion<EventRun> [uses_preselected_argmax],
        "configuring_sampler_decision"_s <= "configuring_sampler"_s + completion<EventRun> / configure_sampler,
        "idle"_s <= "configuring_sampler_decision"_s + completion<EventRun> [sampler_configured],
        "idle"_s <= "configuring_sampler_decision"_s + completion<EventRun> [sampler_config_failed] / mark_backend_error_from_configuring_sampler_decision,
        "configure_preselected_argmax_decision"_s <= "configure_preselected_argmax"_s + completion<EventRun> / configure_preselected_argmax,
        "idle"_s <= "configure_preselected_argmax_decision"_s + completion<EventRun> [sampler_configured],
        "idle"_s <= "configure_preselected_argmax_decision"_s + completion<EventRun> [sampler_config_failed] / mark_backend_error_from_configure_preselected_argmax_decision,
        "idle"_s <= "idle"_s + unexpected_event<_> / on_unexpected_from_idle,
        "idle"_s <= "preparing_backend"_s + unexpected_event<_> / on_unexpected_from_preparing_backend,
        "idle"_s <= "preparing_backend_decision"_s + unexpected_event<_> / on_unexpected_from_preparing_backend_decision,
        "idle"_s <= "binding_conditioner"_s + unexpected_event<_> / on_unexpected_from_binding_conditioner,
        "idle"_s <= "binding_conditioner_decision"_s + unexpected_event<_> / on_unexpected_from_binding_conditioner_decision,
        "idle"_s <= "initializing_renderer"_s + unexpected_event<_> / on_unexpected_from_initializing_renderer,
        "idle"_s <= "initializing_renderer_decision"_s + unexpected_event<_> / on_unexpected_from_initializing_renderer_decision,
        "idle"_s <= "reserving_memory"_s + unexpected_event<_> / on_unexpected_from_reserving_memory,
        "idle"_s <= "reserving_memory_decision"_s + unexpected_event<_> / on_unexpected_from_reserving_memory_decision,
        "idle"_s <= "reserving_graph"_s + unexpected_event<_> / on_unexpected_from_reserving_graph,
        "idle"_s <= "reserving_graph_decision"_s + unexpected_event<_> / on_unexpected_from_reserving_graph_decision,
        "idle"_s <= "configuring_sampling_mode_decision"_s + unexpected_event<_> / on_unexpected_from_configuring_sampling_mode_decision,
        "idle"_s <= "configuring_sampler"_s + unexpected_event<_> / on_unexpected_from_configuring_sampler,
        "idle"_s <= "configuring_sampler_decision"_s + unexpected_event<_> / on_unexpected_from_configuring_sampler_decision,
        "idle"_s <= "configure_preselected_argmax"_s + unexpected_event<_> / on_unexpected_from_configure_preselected_argmax,
        "idle"_s <= "configure_preselected_argmax_decision"_s + unexpected_event<_> / on_unexpected_from_configure_preselected_argmax_decision,
    }
}

/// Context for `TextGeneratorInitializer` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct TextGeneratorInitializerContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl TextGeneratorInitializerStateMachineContext for TextGeneratorInitializerContext {
    fn accept_prepared_backend(&mut self, _event: &EventRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/actions.hpp::accept_prepared_backend
        todo!(
            "TODO: port action `accept_prepared_backend` from emel.cpp/src/emel/text/generator/initializer/actions.hpp"
        )
    }
    fn backend_already_ready(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/guards.hpp::backend_already_ready
        todo!(
            "TODO: port guard `backend_already_ready` from emel.cpp/src/emel/text/generator/initializer/guards.hpp"
        )
    }
    fn backend_prepare_backend_error(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/guards.hpp::backend_prepare_backend_error
        todo!(
            "TODO: port guard `backend_prepare_backend_error` from emel.cpp/src/emel/text/generator/initializer/guards.hpp"
        )
    }
    fn backend_prepare_invalid_request(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/guards.hpp::backend_prepare_invalid_request
        todo!(
            "TODO: port guard `backend_prepare_invalid_request` from emel.cpp/src/emel/text/generator/initializer/guards.hpp"
        )
    }
    fn backend_prepare_ok(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/guards.hpp::backend_prepare_ok
        todo!(
            "TODO: port guard `backend_prepare_ok` from emel.cpp/src/emel/text/generator/initializer/guards.hpp"
        )
    }
    fn begin_initialize(&mut self, _event: &EventRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/actions.hpp::begin_initialize
        todo!(
            "TODO: port action `begin_initialize` from emel.cpp/src/emel/text/generator/initializer/actions.hpp"
        )
    }
    fn conditioner_bind_backend_error(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/guards.hpp::conditioner_bind_backend_error
        todo!(
            "TODO: port guard `conditioner_bind_backend_error` from emel.cpp/src/emel/text/generator/initializer/guards.hpp"
        )
    }
    fn conditioner_bind_invalid_request(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/guards.hpp::conditioner_bind_invalid_request
        todo!(
            "TODO: port guard `conditioner_bind_invalid_request` from emel.cpp/src/emel/text/generator/initializer/guards.hpp"
        )
    }
    fn conditioner_bind_ok(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/guards.hpp::conditioner_bind_ok
        todo!(
            "TODO: port guard `conditioner_bind_ok` from emel.cpp/src/emel/text/generator/initializer/guards.hpp"
        )
    }
    fn configure_preselected_argmax(&mut self, _event: &EventRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/actions.hpp::configure_preselected_argmax
        todo!(
            "TODO: port action `configure_preselected_argmax` from emel.cpp/src/emel/text/generator/initializer/actions.hpp"
        )
    }
    fn configure_sampler(&mut self, _event: &EventRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/actions.hpp::configure_sampler
        todo!(
            "TODO: port action `configure_sampler` from emel.cpp/src/emel/text/generator/initializer/actions.hpp"
        )
    }
    fn graph_reserve_backend_error(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/guards.hpp::graph_reserve_backend_error
        todo!(
            "TODO: port guard `graph_reserve_backend_error` from emel.cpp/src/emel/text/generator/initializer/guards.hpp"
        )
    }
    fn graph_reserve_invalid_request(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/guards.hpp::graph_reserve_invalid_request
        todo!(
            "TODO: port guard `graph_reserve_invalid_request` from emel.cpp/src/emel/text/generator/initializer/guards.hpp"
        )
    }
    fn graph_reserve_ok(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/guards.hpp::graph_reserve_ok
        todo!(
            "TODO: port guard `graph_reserve_ok` from emel.cpp/src/emel/text/generator/initializer/guards.hpp"
        )
    }
    fn guard_backend_prepare_allowed(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/guards.hpp::guard_backend_prepare_allowed
        todo!(
            "TODO: port guard `guard_backend_prepare_allowed` from emel.cpp/src/emel/text/generator/initializer/guards.hpp"
        )
    }
    fn guard_backend_reuse_allowed(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/guards.hpp::guard_backend_reuse_allowed
        todo!(
            "TODO: port guard `guard_backend_reuse_allowed` from emel.cpp/src/emel/text/generator/initializer/guards.hpp"
        )
    }
    fn guard_generation_contract_invalid(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/guards.hpp::guard_generation_contract_invalid
        todo!(
            "TODO: port guard `guard_generation_contract_invalid` from emel.cpp/src/emel/text/generator/initializer/guards.hpp"
        )
    }
    fn guard_memory_geometry_fits_backend(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/guards.hpp::guard_memory_geometry_fits_backend
        todo!(
            "TODO: port guard `guard_memory_geometry_fits_backend` from emel.cpp/src/emel/text/generator/initializer/guards.hpp"
        )
    }
    fn guard_memory_geometry_gap(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/guards.hpp::guard_memory_geometry_gap
        todo!(
            "TODO: port guard `guard_memory_geometry_gap` from emel.cpp/src/emel/text/generator/initializer/guards.hpp"
        )
    }
    fn mark_backend_error_from_binding_conditioner_decision(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/actions.hpp::mark_backend_error
        todo!(
            "TODO: port action `mark_backend_error` from emel.cpp/src/emel/text/generator/initializer/actions.hpp"
        )
    }
    fn mark_backend_error_from_configure_preselected_argmax_decision(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/actions.hpp::mark_backend_error
        todo!(
            "TODO: port action `mark_backend_error` from emel.cpp/src/emel/text/generator/initializer/actions.hpp"
        )
    }
    fn mark_backend_error_from_configuring_sampler_decision(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/actions.hpp::mark_backend_error
        todo!(
            "TODO: port action `mark_backend_error` from emel.cpp/src/emel/text/generator/initializer/actions.hpp"
        )
    }
    fn mark_backend_error_from_initializing_renderer_decision(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/actions.hpp::mark_backend_error
        todo!(
            "TODO: port action `mark_backend_error` from emel.cpp/src/emel/text/generator/initializer/actions.hpp"
        )
    }
    fn mark_backend_error_from_preparing_backend_decision(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/actions.hpp::mark_backend_error
        todo!(
            "TODO: port action `mark_backend_error` from emel.cpp/src/emel/text/generator/initializer/actions.hpp"
        )
    }
    fn mark_backend_error_from_reserving_graph_decision(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/actions.hpp::mark_backend_error
        todo!(
            "TODO: port action `mark_backend_error` from emel.cpp/src/emel/text/generator/initializer/actions.hpp"
        )
    }
    fn mark_backend_error_from_reserving_memory_decision(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/actions.hpp::mark_backend_error
        todo!(
            "TODO: port action `mark_backend_error` from emel.cpp/src/emel/text/generator/initializer/actions.hpp"
        )
    }
    fn mark_invalid_request_from_binding_conditioner_decision(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/text/generator/initializer/actions.hpp"
        )
    }
    fn mark_invalid_request_from_initializing_renderer_decision(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/text/generator/initializer/actions.hpp"
        )
    }
    fn mark_invalid_request_from_preparing_backend(&mut self, _event: &EventRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/text/generator/initializer/actions.hpp"
        )
    }
    fn mark_invalid_request_from_preparing_backend_decision(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/text/generator/initializer/actions.hpp"
        )
    }
    fn mark_invalid_request_from_reserving_graph_decision(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/text/generator/initializer/actions.hpp"
        )
    }
    fn mark_invalid_request_from_reserving_memory(&mut self, _event: &EventRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/text/generator/initializer/actions.hpp"
        )
    }
    fn mark_invalid_request_from_reserving_memory_decision(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/text/generator/initializer/actions.hpp"
        )
    }
    fn memory_reserve_backend_error(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/guards.hpp::memory_reserve_backend_error
        todo!(
            "TODO: port guard `memory_reserve_backend_error` from emel.cpp/src/emel/text/generator/initializer/guards.hpp"
        )
    }
    fn memory_reserve_invalid_request(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/guards.hpp::memory_reserve_invalid_request
        todo!(
            "TODO: port guard `memory_reserve_invalid_request` from emel.cpp/src/emel/text/generator/initializer/guards.hpp"
        )
    }
    fn memory_reserve_with_existing_graph(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/guards.hpp::memory_reserve_with_existing_graph
        todo!(
            "TODO: port guard `memory_reserve_with_existing_graph` from emel.cpp/src/emel/text/generator/initializer/guards.hpp"
        )
    }
    fn memory_reserve_with_missing_graph(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/guards.hpp::memory_reserve_with_missing_graph
        todo!(
            "TODO: port guard `memory_reserve_with_missing_graph` from emel.cpp/src/emel/text/generator/initializer/guards.hpp"
        )
    }
    fn on_unexpected_from_binding_conditioner(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/initializer/actions.hpp"
        )
    }
    fn on_unexpected_from_binding_conditioner_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/initializer/actions.hpp"
        )
    }
    fn on_unexpected_from_configure_preselected_argmax(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/initializer/actions.hpp"
        )
    }
    fn on_unexpected_from_configure_preselected_argmax_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/initializer/actions.hpp"
        )
    }
    fn on_unexpected_from_configuring_sampler(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/initializer/actions.hpp"
        )
    }
    fn on_unexpected_from_configuring_sampler_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/initializer/actions.hpp"
        )
    }
    fn on_unexpected_from_configuring_sampling_mode_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/initializer/actions.hpp"
        )
    }
    fn on_unexpected_from_idle(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/initializer/actions.hpp"
        )
    }
    fn on_unexpected_from_initializing_renderer(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/initializer/actions.hpp"
        )
    }
    fn on_unexpected_from_initializing_renderer_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/initializer/actions.hpp"
        )
    }
    fn on_unexpected_from_preparing_backend(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/initializer/actions.hpp"
        )
    }
    fn on_unexpected_from_preparing_backend_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/initializer/actions.hpp"
        )
    }
    fn on_unexpected_from_reserving_graph(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/initializer/actions.hpp"
        )
    }
    fn on_unexpected_from_reserving_graph_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/initializer/actions.hpp"
        )
    }
    fn on_unexpected_from_reserving_memory(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/initializer/actions.hpp"
        )
    }
    fn on_unexpected_from_reserving_memory_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/initializer/actions.hpp"
        )
    }
    fn renderer_initialize_backend_error(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/guards.hpp::renderer_initialize_backend_error
        todo!(
            "TODO: port guard `renderer_initialize_backend_error` from emel.cpp/src/emel/text/generator/initializer/guards.hpp"
        )
    }
    fn renderer_initialize_invalid_request(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/guards.hpp::renderer_initialize_invalid_request
        todo!(
            "TODO: port guard `renderer_initialize_invalid_request` from emel.cpp/src/emel/text/generator/initializer/guards.hpp"
        )
    }
    fn renderer_initialize_ok(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/guards.hpp::renderer_initialize_ok
        todo!(
            "TODO: port guard `renderer_initialize_ok` from emel.cpp/src/emel/text/generator/initializer/guards.hpp"
        )
    }
    fn request_backend_prepare(&mut self, _event: &EventRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/actions.hpp::request_backend_prepare
        todo!(
            "TODO: port action `request_backend_prepare` from emel.cpp/src/emel/text/generator/initializer/actions.hpp"
        )
    }
    fn request_conditioner_bind(&mut self, _event: &EventRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/actions.hpp::request_conditioner_bind
        todo!(
            "TODO: port action `request_conditioner_bind` from emel.cpp/src/emel/text/generator/initializer/actions.hpp"
        )
    }
    fn request_graph_reserve(&mut self, _event: &EventRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/actions.hpp::request_graph_reserve
        todo!(
            "TODO: port action `request_graph_reserve` from emel.cpp/src/emel/text/generator/initializer/actions.hpp"
        )
    }
    fn request_memory_reserve(&mut self, _event: &EventRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/actions.hpp::request_memory_reserve
        todo!(
            "TODO: port action `request_memory_reserve` from emel.cpp/src/emel/text/generator/initializer/actions.hpp"
        )
    }
    fn request_renderer_initialize(&mut self, _event: &EventRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/actions.hpp::request_renderer_initialize
        todo!(
            "TODO: port action `request_renderer_initialize` from emel.cpp/src/emel/text/generator/initializer/actions.hpp"
        )
    }
    fn sampler_config_failed(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/guards.hpp::sampler_config_failed
        todo!(
            "TODO: port guard `sampler_config_failed` from emel.cpp/src/emel/text/generator/initializer/guards.hpp"
        )
    }
    fn sampler_configured(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/guards.hpp::sampler_configured
        todo!(
            "TODO: port guard `sampler_configured` from emel.cpp/src/emel/text/generator/initializer/guards.hpp"
        )
    }
    fn uses_materialized_logits(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/guards.hpp::uses_materialized_logits
        todo!(
            "TODO: port guard `uses_materialized_logits` from emel.cpp/src/emel/text/generator/initializer/guards.hpp"
        )
    }
    fn uses_preselected_argmax(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/initializer/guards.hpp::uses_preselected_argmax
        todo!(
            "TODO: port guard `uses_preselected_argmax` from emel.cpp/src/emel/text/generator/initializer/guards.hpp"
        )
    }
}
