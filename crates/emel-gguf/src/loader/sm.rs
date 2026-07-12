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

// --- machine GgufLoader from emel.cpp/src/emel/gguf/loader/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventBindRuntime;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventParseRuntime;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventProbeRuntime;

sml! {
    GgufLoader {
        "probe_request_decision"_s <= *"uninitialized"_s + event<EventProbeRuntime> / begin_probe_from_uninitialized,
        "probe_request_decision"_s <= "probed"_s + event<EventProbeRuntime> / begin_probe_from_probed,
        "probe_request_decision"_s <= "bound"_s + event<EventProbeRuntime> / begin_probe_from_bound,
        "probe_request_decision"_s <= "parsed"_s + event<EventProbeRuntime> / begin_probe_from_parsed,
        "probe_request_decision"_s <= "errored"_s + event<EventProbeRuntime> / begin_probe_from_errored,
        "probe_outcome_dispatch"_s <= "probe_request_decision"_s + completion<EventProbeRuntime> [probe_valid_request] / exec_probe,
        "probe_outcome_dispatch"_s <= "probe_request_decision"_s + completion<EventProbeRuntime> [probe_invalid_request] / mark_probe_invalid_request,
        "probe_requirements_dispatch"_s <= "probe_outcome_dispatch"_s + completion<EventProbeRuntime> [probe_error_none] / commit_probe_requirements,
        "errored"_s <= "probe_outcome_dispatch"_s + completion<EventProbeRuntime> [probe_error_invalid_request] / publish_probe_error_from_probe_outcome_dispatch,
        "errored"_s <= "probe_outcome_dispatch"_s + completion<EventProbeRuntime> [probe_error_model_invalid] / publish_probe_error_from_probe_outcome_dispatch,
        "errored"_s <= "probe_outcome_dispatch"_s + completion<EventProbeRuntime> [probe_error_capacity] / publish_probe_error_from_probe_outcome_dispatch,
        "errored"_s <= "probe_outcome_dispatch"_s + completion<EventProbeRuntime> [probe_error_parse_failed] / publish_probe_error_from_probe_outcome_dispatch,
        "errored"_s <= "probe_outcome_dispatch"_s + completion<EventProbeRuntime> [probe_error_internal_error] / publish_probe_error_from_probe_outcome_dispatch,
        "errored"_s <= "probe_outcome_dispatch"_s + completion<EventProbeRuntime> [probe_error_untracked] / publish_probe_error_from_probe_outcome_dispatch,
        "errored"_s <= "probe_outcome_dispatch"_s + completion<EventProbeRuntime> [probe_error_unknown] / publish_probe_error_from_probe_outcome_dispatch,
        "probed"_s <= "probe_requirements_dispatch"_s + completion<EventProbeRuntime> / publish_probe_done,
        "bind_request_decision"_s <= "probed"_s + event<EventBindRuntime> / begin_bind_from_probed,
        "bind_request_decision"_s <= "bound"_s + event<EventBindRuntime> / begin_bind_from_bound,
        "bind_request_decision"_s <= "parsed"_s + event<EventBindRuntime> / begin_bind_from_parsed,
        "bind_outcome_dispatch"_s <= "uninitialized"_s + event<EventBindRuntime> / mark_bind_invalid_request_from_uninitialized,
        "bind_outcome_dispatch"_s <= "errored"_s + event<EventBindRuntime> / mark_bind_invalid_request_from_errored,
        "bind_request_shape_decision"_s <= "bind_request_decision"_s + completion<EventBindRuntime>,
        "bind_capacity_decision"_s <= "bind_request_shape_decision"_s + completion<EventBindRuntime> [bind_valid_request],
        "bind_outcome_dispatch"_s <= "bind_request_shape_decision"_s + completion<EventBindRuntime> [bind_invalid_request] / mark_bind_invalid_request_from_bind_request_shape_decision,
        "bind_outcome_dispatch"_s <= "bind_request_shape_decision"_s + completion<EventBindRuntime> / mark_bind_invalid_request_from_bind_request_shape_decision,
        "bind_outcome_dispatch"_s <= "bind_capacity_decision"_s + completion<EventBindRuntime> [bind_capacity_sufficient] / exec_bind,
        "bind_outcome_dispatch"_s <= "bind_capacity_decision"_s + completion<EventBindRuntime> [bind_capacity_insufficient] / mark_bind_capacity_from_bind_capacity_decision,
        "bind_outcome_dispatch"_s <= "bind_capacity_decision"_s + completion<EventBindRuntime> / mark_bind_capacity_from_bind_capacity_decision,
        "bound"_s <= "bind_outcome_dispatch"_s + completion<EventBindRuntime> [bind_error_none] / publish_bind_done,
        "errored"_s <= "bind_outcome_dispatch"_s + completion<EventBindRuntime> [bind_error_invalid_request] / publish_bind_error_from_bind_outcome_dispatch,
        "errored"_s <= "bind_outcome_dispatch"_s + completion<EventBindRuntime> [bind_error_model_invalid] / publish_bind_error_from_bind_outcome_dispatch,
        "errored"_s <= "bind_outcome_dispatch"_s + completion<EventBindRuntime> [bind_error_capacity] / publish_bind_error_from_bind_outcome_dispatch,
        "errored"_s <= "bind_outcome_dispatch"_s + completion<EventBindRuntime> [bind_error_parse_failed] / publish_bind_error_from_bind_outcome_dispatch,
        "errored"_s <= "bind_outcome_dispatch"_s + completion<EventBindRuntime> [bind_error_internal_error] / publish_bind_error_from_bind_outcome_dispatch,
        "errored"_s <= "bind_outcome_dispatch"_s + completion<EventBindRuntime> [bind_error_untracked] / publish_bind_error_from_bind_outcome_dispatch,
        "errored"_s <= "bind_outcome_dispatch"_s + completion<EventBindRuntime> [bind_error_unknown] / publish_bind_error_from_bind_outcome_dispatch,
        "parse_request_decision"_s <= "bound"_s + event<EventParseRuntime> / begin_parse_from_bound,
        "parse_request_decision"_s <= "parsed"_s + event<EventParseRuntime> / begin_parse_from_parsed,
        "parse_outcome_dispatch"_s <= "uninitialized"_s + event<EventParseRuntime> / mark_parse_invalid_request_from_uninitialized,
        "parse_outcome_dispatch"_s <= "probed"_s + event<EventParseRuntime> / mark_parse_invalid_request_from_probed,
        "parse_outcome_dispatch"_s <= "errored"_s + event<EventParseRuntime> / mark_parse_invalid_request_from_errored,
        "parse_file_image_decision"_s <= "parse_request_decision"_s + completion<EventParseRuntime>,
        "parse_bound_storage_decision"_s <= "parse_file_image_decision"_s + completion<EventParseRuntime> [parse_has_file_image],
        "parse_outcome_dispatch"_s <= "parse_file_image_decision"_s + completion<EventParseRuntime> [parse_missing_file_image] / mark_parse_invalid_request_from_parse_file_image_decision,
        "parse_outcome_dispatch"_s <= "parse_file_image_decision"_s + completion<EventParseRuntime> / mark_parse_invalid_request_from_parse_file_image_decision,
        "parse_capacity_decision"_s <= "parse_bound_storage_decision"_s + completion<EventParseRuntime> [parse_has_bound_storage],
        "parse_outcome_dispatch"_s <= "parse_bound_storage_decision"_s + completion<EventParseRuntime> [parse_missing_bound_storage] / mark_parse_invalid_request_from_parse_bound_storage_decision,
        "parse_outcome_dispatch"_s <= "parse_bound_storage_decision"_s + completion<EventParseRuntime> / mark_parse_invalid_request_from_parse_bound_storage_decision,
        "parse_outcome_dispatch"_s <= "parse_capacity_decision"_s + completion<EventParseRuntime> [parse_bound_capacity_sufficient] / exec_parse,
        "parse_outcome_dispatch"_s <= "parse_capacity_decision"_s + completion<EventParseRuntime> [parse_bound_capacity_insufficient] / mark_parse_capacity_from_parse_capacity_decision,
        "parse_outcome_dispatch"_s <= "parse_capacity_decision"_s + completion<EventParseRuntime> / mark_parse_capacity_from_parse_capacity_decision,
        "parsed"_s <= "parse_outcome_dispatch"_s + completion<EventParseRuntime> [parse_error_none] / publish_parse_done,
        "errored"_s <= "parse_outcome_dispatch"_s + completion<EventParseRuntime> [parse_error_invalid_request] / publish_parse_error_from_parse_outcome_dispatch,
        "errored"_s <= "parse_outcome_dispatch"_s + completion<EventParseRuntime> [parse_error_model_invalid] / publish_parse_error_from_parse_outcome_dispatch,
        "errored"_s <= "parse_outcome_dispatch"_s + completion<EventParseRuntime> [parse_error_capacity] / publish_parse_error_from_parse_outcome_dispatch,
        "errored"_s <= "parse_outcome_dispatch"_s + completion<EventParseRuntime> [parse_error_parse_failed] / publish_parse_error_from_parse_outcome_dispatch,
        "errored"_s <= "parse_outcome_dispatch"_s + completion<EventParseRuntime> [parse_error_internal_error] / publish_parse_error_from_parse_outcome_dispatch,
        "errored"_s <= "parse_outcome_dispatch"_s + completion<EventParseRuntime> [parse_error_untracked] / publish_parse_error_from_parse_outcome_dispatch,
        "errored"_s <= "parse_outcome_dispatch"_s + completion<EventParseRuntime> [parse_error_unknown] / publish_parse_error_from_parse_outcome_dispatch,
        "errored"_s <= "uninitialized"_s + unexpected_event<_> / on_unexpected_from_uninitialized,
        "errored"_s <= "probed"_s + unexpected_event<_> / on_unexpected_from_probed,
        "errored"_s <= "bound"_s + unexpected_event<_> / on_unexpected_from_bound,
        "errored"_s <= "parsed"_s + unexpected_event<_> / on_unexpected_from_parsed,
        "errored"_s <= "errored"_s + unexpected_event<_> / on_unexpected_from_errored,
        "errored"_s <= "probe_request_decision"_s + unexpected_event<_> / on_unexpected_from_probe_request_decision,
        "errored"_s <= "probe_outcome_dispatch"_s + unexpected_event<_> / on_unexpected_from_probe_outcome_dispatch,
        "errored"_s <= "probe_requirements_dispatch"_s + unexpected_event<_> / on_unexpected_from_probe_requirements_dispatch,
        "errored"_s <= "bind_request_decision"_s + unexpected_event<_> / on_unexpected_from_bind_request_decision,
        "errored"_s <= "bind_request_shape_decision"_s + unexpected_event<_> / on_unexpected_from_bind_request_shape_decision,
        "errored"_s <= "bind_capacity_decision"_s + unexpected_event<_> / on_unexpected_from_bind_capacity_decision,
        "errored"_s <= "bind_outcome_dispatch"_s + unexpected_event<_> / on_unexpected_from_bind_outcome_dispatch,
        "errored"_s <= "parse_request_decision"_s + unexpected_event<_> / on_unexpected_from_parse_request_decision,
        "errored"_s <= "parse_file_image_decision"_s + unexpected_event<_> / on_unexpected_from_parse_file_image_decision,
        "errored"_s <= "parse_bound_storage_decision"_s + unexpected_event<_> / on_unexpected_from_parse_bound_storage_decision,
        "errored"_s <= "parse_capacity_decision"_s + unexpected_event<_> / on_unexpected_from_parse_capacity_decision,
        "errored"_s <= "parse_outcome_dispatch"_s + unexpected_event<_> / on_unexpected_from_parse_outcome_dispatch,
    }
}

/// Context for `GgufLoader` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct GgufLoaderContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl GgufLoaderStateMachineContext for GgufLoaderContext {
    fn begin_bind_from_bound(&mut self, _event: &EventBindRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::begin_bind
        todo!("TODO: port action `begin_bind` from emel.cpp/src/emel/gguf/loader/actions.hpp")
    }
    fn begin_bind_from_parsed(&mut self, _event: &EventBindRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::begin_bind
        todo!("TODO: port action `begin_bind` from emel.cpp/src/emel/gguf/loader/actions.hpp")
    }
    fn begin_bind_from_probed(&mut self, _event: &EventBindRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::begin_bind
        todo!("TODO: port action `begin_bind` from emel.cpp/src/emel/gguf/loader/actions.hpp")
    }
    fn begin_parse_from_bound(&mut self, _event: &EventParseRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::begin_parse
        todo!("TODO: port action `begin_parse` from emel.cpp/src/emel/gguf/loader/actions.hpp")
    }
    fn begin_parse_from_parsed(&mut self, _event: &EventParseRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::begin_parse
        todo!("TODO: port action `begin_parse` from emel.cpp/src/emel/gguf/loader/actions.hpp")
    }
    fn begin_probe_from_bound(&mut self, _event: &EventProbeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::begin_probe
        todo!("TODO: port action `begin_probe` from emel.cpp/src/emel/gguf/loader/actions.hpp")
    }
    fn begin_probe_from_errored(&mut self, _event: &EventProbeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::begin_probe
        todo!("TODO: port action `begin_probe` from emel.cpp/src/emel/gguf/loader/actions.hpp")
    }
    fn begin_probe_from_parsed(&mut self, _event: &EventProbeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::begin_probe
        todo!("TODO: port action `begin_probe` from emel.cpp/src/emel/gguf/loader/actions.hpp")
    }
    fn begin_probe_from_probed(&mut self, _event: &EventProbeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::begin_probe
        todo!("TODO: port action `begin_probe` from emel.cpp/src/emel/gguf/loader/actions.hpp")
    }
    fn begin_probe_from_uninitialized(&mut self, _event: &EventProbeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::begin_probe
        todo!("TODO: port action `begin_probe` from emel.cpp/src/emel/gguf/loader/actions.hpp")
    }
    fn bind_capacity_insufficient(&self, _event: &EventBindRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/guards.hpp::bind_capacity_insufficient
        todo!(
            "TODO: port guard `bind_capacity_insufficient` from emel.cpp/src/emel/gguf/loader/guards.hpp"
        )
    }
    fn bind_capacity_sufficient(&self, _event: &EventBindRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/guards.hpp::bind_capacity_sufficient
        todo!(
            "TODO: port guard `bind_capacity_sufficient` from emel.cpp/src/emel/gguf/loader/guards.hpp"
        )
    }
    fn bind_error_capacity(&self, _event: &EventBindRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/guards.hpp::bind_error_capacity
        todo!(
            "TODO: port guard `bind_error_capacity` from emel.cpp/src/emel/gguf/loader/guards.hpp"
        )
    }
    fn bind_error_internal_error(&self, _event: &EventBindRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/guards.hpp::bind_error_internal_error
        todo!(
            "TODO: port guard `bind_error_internal_error` from emel.cpp/src/emel/gguf/loader/guards.hpp"
        )
    }
    fn bind_error_invalid_request(&self, _event: &EventBindRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/guards.hpp::bind_error_invalid_request
        todo!(
            "TODO: port guard `bind_error_invalid_request` from emel.cpp/src/emel/gguf/loader/guards.hpp"
        )
    }
    fn bind_error_model_invalid(&self, _event: &EventBindRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/guards.hpp::bind_error_model_invalid
        todo!(
            "TODO: port guard `bind_error_model_invalid` from emel.cpp/src/emel/gguf/loader/guards.hpp"
        )
    }
    fn bind_error_none(&self, _event: &EventBindRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/guards.hpp::bind_error_none
        todo!("TODO: port guard `bind_error_none` from emel.cpp/src/emel/gguf/loader/guards.hpp")
    }
    fn bind_error_parse_failed(&self, _event: &EventBindRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/guards.hpp::bind_error_parse_failed
        todo!(
            "TODO: port guard `bind_error_parse_failed` from emel.cpp/src/emel/gguf/loader/guards.hpp"
        )
    }
    fn bind_error_unknown(&self, _event: &EventBindRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/guards.hpp::bind_error_unknown
        todo!("TODO: port guard `bind_error_unknown` from emel.cpp/src/emel/gguf/loader/guards.hpp")
    }
    fn bind_error_untracked(&self, _event: &EventBindRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/guards.hpp::bind_error_untracked
        todo!(
            "TODO: port guard `bind_error_untracked` from emel.cpp/src/emel/gguf/loader/guards.hpp"
        )
    }
    fn bind_invalid_request(&self, _event: &EventBindRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/guards.hpp::bind_invalid_request
        todo!(
            "TODO: port guard `bind_invalid_request` from emel.cpp/src/emel/gguf/loader/guards.hpp"
        )
    }
    fn bind_valid_request(&self, _event: &EventBindRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/guards.hpp::bind_valid_request
        todo!("TODO: port guard `bind_valid_request` from emel.cpp/src/emel/gguf/loader/guards.hpp")
    }
    fn commit_probe_requirements(&mut self, _event: &EventProbeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::commit_probe_requirements
        todo!(
            "TODO: port action `commit_probe_requirements` from emel.cpp/src/emel/gguf/loader/actions.hpp"
        )
    }
    fn exec_bind(&mut self, _event: &EventBindRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::exec_bind
        todo!("TODO: port action `exec_bind` from emel.cpp/src/emel/gguf/loader/actions.hpp")
    }
    fn exec_parse(&mut self, _event: &EventParseRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::exec_parse
        todo!("TODO: port action `exec_parse` from emel.cpp/src/emel/gguf/loader/actions.hpp")
    }
    fn exec_probe(&mut self, _event: &EventProbeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::exec_probe
        todo!("TODO: port action `exec_probe` from emel.cpp/src/emel/gguf/loader/actions.hpp")
    }
    fn mark_bind_capacity_from_bind_capacity_decision(
        &mut self,
        _event: &EventBindRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::mark_bind_capacity
        todo!(
            "TODO: port action `mark_bind_capacity` from emel.cpp/src/emel/gguf/loader/actions.hpp"
        )
    }
    fn mark_bind_invalid_request_from_bind_request_shape_decision(
        &mut self,
        _event: &EventBindRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::mark_bind_invalid_request
        todo!(
            "TODO: port action `mark_bind_invalid_request` from emel.cpp/src/emel/gguf/loader/actions.hpp"
        )
    }
    fn mark_bind_invalid_request_from_errored(
        &mut self,
        _event: &EventBindRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::mark_bind_invalid_request
        todo!(
            "TODO: port action `mark_bind_invalid_request` from emel.cpp/src/emel/gguf/loader/actions.hpp"
        )
    }
    fn mark_bind_invalid_request_from_uninitialized(
        &mut self,
        _event: &EventBindRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::mark_bind_invalid_request
        todo!(
            "TODO: port action `mark_bind_invalid_request` from emel.cpp/src/emel/gguf/loader/actions.hpp"
        )
    }
    fn mark_parse_capacity_from_parse_capacity_decision(
        &mut self,
        _event: &EventParseRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::mark_parse_capacity
        todo!(
            "TODO: port action `mark_parse_capacity` from emel.cpp/src/emel/gguf/loader/actions.hpp"
        )
    }
    fn mark_parse_invalid_request_from_errored(
        &mut self,
        _event: &EventParseRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::mark_parse_invalid_request
        todo!(
            "TODO: port action `mark_parse_invalid_request` from emel.cpp/src/emel/gguf/loader/actions.hpp"
        )
    }
    fn mark_parse_invalid_request_from_parse_bound_storage_decision(
        &mut self,
        _event: &EventParseRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::mark_parse_invalid_request
        todo!(
            "TODO: port action `mark_parse_invalid_request` from emel.cpp/src/emel/gguf/loader/actions.hpp"
        )
    }
    fn mark_parse_invalid_request_from_parse_file_image_decision(
        &mut self,
        _event: &EventParseRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::mark_parse_invalid_request
        todo!(
            "TODO: port action `mark_parse_invalid_request` from emel.cpp/src/emel/gguf/loader/actions.hpp"
        )
    }
    fn mark_parse_invalid_request_from_probed(
        &mut self,
        _event: &EventParseRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::mark_parse_invalid_request
        todo!(
            "TODO: port action `mark_parse_invalid_request` from emel.cpp/src/emel/gguf/loader/actions.hpp"
        )
    }
    fn mark_parse_invalid_request_from_uninitialized(
        &mut self,
        _event: &EventParseRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::mark_parse_invalid_request
        todo!(
            "TODO: port action `mark_parse_invalid_request` from emel.cpp/src/emel/gguf/loader/actions.hpp"
        )
    }
    fn mark_probe_invalid_request(&mut self, _event: &EventProbeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::mark_probe_invalid_request
        todo!(
            "TODO: port action `mark_probe_invalid_request` from emel.cpp/src/emel/gguf/loader/actions.hpp"
        )
    }
    fn on_unexpected_from_bind_capacity_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/gguf/loader/actions.hpp")
    }
    fn on_unexpected_from_bind_outcome_dispatch(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/gguf/loader/actions.hpp")
    }
    fn on_unexpected_from_bind_request_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/gguf/loader/actions.hpp")
    }
    fn on_unexpected_from_bind_request_shape_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/gguf/loader/actions.hpp")
    }
    fn on_unexpected_from_bound(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/gguf/loader/actions.hpp")
    }
    fn on_unexpected_from_errored(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/gguf/loader/actions.hpp")
    }
    fn on_unexpected_from_parse_bound_storage_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/gguf/loader/actions.hpp")
    }
    fn on_unexpected_from_parse_capacity_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/gguf/loader/actions.hpp")
    }
    fn on_unexpected_from_parse_file_image_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/gguf/loader/actions.hpp")
    }
    fn on_unexpected_from_parse_outcome_dispatch(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/gguf/loader/actions.hpp")
    }
    fn on_unexpected_from_parse_request_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/gguf/loader/actions.hpp")
    }
    fn on_unexpected_from_parsed(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/gguf/loader/actions.hpp")
    }
    fn on_unexpected_from_probe_outcome_dispatch(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/gguf/loader/actions.hpp")
    }
    fn on_unexpected_from_probe_request_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/gguf/loader/actions.hpp")
    }
    fn on_unexpected_from_probe_requirements_dispatch(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/gguf/loader/actions.hpp")
    }
    fn on_unexpected_from_probed(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/gguf/loader/actions.hpp")
    }
    fn on_unexpected_from_uninitialized(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/gguf/loader/actions.hpp")
    }
    fn parse_bound_capacity_insufficient(&self, _event: &EventParseRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/guards.hpp::parse_bound_capacity_insufficient
        todo!(
            "TODO: port guard `parse_bound_capacity_insufficient` from emel.cpp/src/emel/gguf/loader/guards.hpp"
        )
    }
    fn parse_bound_capacity_sufficient(&self, _event: &EventParseRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/guards.hpp::parse_bound_capacity_sufficient
        todo!(
            "TODO: port guard `parse_bound_capacity_sufficient` from emel.cpp/src/emel/gguf/loader/guards.hpp"
        )
    }
    fn parse_error_capacity(&self, _event: &EventParseRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/guards.hpp::parse_error_capacity
        todo!(
            "TODO: port guard `parse_error_capacity` from emel.cpp/src/emel/gguf/loader/guards.hpp"
        )
    }
    fn parse_error_internal_error(&self, _event: &EventParseRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/guards.hpp::parse_error_internal_error
        todo!(
            "TODO: port guard `parse_error_internal_error` from emel.cpp/src/emel/gguf/loader/guards.hpp"
        )
    }
    fn parse_error_invalid_request(&self, _event: &EventParseRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/guards.hpp::parse_error_invalid_request
        todo!(
            "TODO: port guard `parse_error_invalid_request` from emel.cpp/src/emel/gguf/loader/guards.hpp"
        )
    }
    fn parse_error_model_invalid(&self, _event: &EventParseRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/guards.hpp::parse_error_model_invalid
        todo!(
            "TODO: port guard `parse_error_model_invalid` from emel.cpp/src/emel/gguf/loader/guards.hpp"
        )
    }
    fn parse_error_none(&self, _event: &EventParseRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/guards.hpp::parse_error_none
        todo!("TODO: port guard `parse_error_none` from emel.cpp/src/emel/gguf/loader/guards.hpp")
    }
    fn parse_error_parse_failed(&self, _event: &EventParseRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/guards.hpp::parse_error_parse_failed
        todo!(
            "TODO: port guard `parse_error_parse_failed` from emel.cpp/src/emel/gguf/loader/guards.hpp"
        )
    }
    fn parse_error_unknown(&self, _event: &EventParseRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/guards.hpp::parse_error_unknown
        todo!(
            "TODO: port guard `parse_error_unknown` from emel.cpp/src/emel/gguf/loader/guards.hpp"
        )
    }
    fn parse_error_untracked(&self, _event: &EventParseRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/guards.hpp::parse_error_untracked
        todo!(
            "TODO: port guard `parse_error_untracked` from emel.cpp/src/emel/gguf/loader/guards.hpp"
        )
    }
    fn parse_has_bound_storage(&self, _event: &EventParseRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/guards.hpp::parse_has_bound_storage
        todo!(
            "TODO: port guard `parse_has_bound_storage` from emel.cpp/src/emel/gguf/loader/guards.hpp"
        )
    }
    fn parse_has_file_image(&self, _event: &EventParseRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/guards.hpp::parse_has_file_image
        todo!(
            "TODO: port guard `parse_has_file_image` from emel.cpp/src/emel/gguf/loader/guards.hpp"
        )
    }
    fn parse_missing_bound_storage(&self, _event: &EventParseRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/guards.hpp::parse_missing_bound_storage
        todo!(
            "TODO: port guard `parse_missing_bound_storage` from emel.cpp/src/emel/gguf/loader/guards.hpp"
        )
    }
    fn parse_missing_file_image(&self, _event: &EventParseRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/guards.hpp::parse_missing_file_image
        todo!(
            "TODO: port guard `parse_missing_file_image` from emel.cpp/src/emel/gguf/loader/guards.hpp"
        )
    }
    fn probe_error_capacity(&self, _event: &EventProbeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/guards.hpp::probe_error_capacity
        todo!(
            "TODO: port guard `probe_error_capacity` from emel.cpp/src/emel/gguf/loader/guards.hpp"
        )
    }
    fn probe_error_internal_error(&self, _event: &EventProbeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/guards.hpp::probe_error_internal_error
        todo!(
            "TODO: port guard `probe_error_internal_error` from emel.cpp/src/emel/gguf/loader/guards.hpp"
        )
    }
    fn probe_error_invalid_request(&self, _event: &EventProbeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/guards.hpp::probe_error_invalid_request
        todo!(
            "TODO: port guard `probe_error_invalid_request` from emel.cpp/src/emel/gguf/loader/guards.hpp"
        )
    }
    fn probe_error_model_invalid(&self, _event: &EventProbeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/guards.hpp::probe_error_model_invalid
        todo!(
            "TODO: port guard `probe_error_model_invalid` from emel.cpp/src/emel/gguf/loader/guards.hpp"
        )
    }
    fn probe_error_none(&self, _event: &EventProbeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/guards.hpp::probe_error_none
        todo!("TODO: port guard `probe_error_none` from emel.cpp/src/emel/gguf/loader/guards.hpp")
    }
    fn probe_error_parse_failed(&self, _event: &EventProbeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/guards.hpp::probe_error_parse_failed
        todo!(
            "TODO: port guard `probe_error_parse_failed` from emel.cpp/src/emel/gguf/loader/guards.hpp"
        )
    }
    fn probe_error_unknown(&self, _event: &EventProbeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/guards.hpp::probe_error_unknown
        todo!(
            "TODO: port guard `probe_error_unknown` from emel.cpp/src/emel/gguf/loader/guards.hpp"
        )
    }
    fn probe_error_untracked(&self, _event: &EventProbeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/guards.hpp::probe_error_untracked
        todo!(
            "TODO: port guard `probe_error_untracked` from emel.cpp/src/emel/gguf/loader/guards.hpp"
        )
    }
    fn probe_invalid_request(&self, _event: &EventProbeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/guards.hpp::probe_invalid_request
        todo!(
            "TODO: port guard `probe_invalid_request` from emel.cpp/src/emel/gguf/loader/guards.hpp"
        )
    }
    fn probe_valid_request(&self, _event: &EventProbeRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/guards.hpp::probe_valid_request
        todo!(
            "TODO: port guard `probe_valid_request` from emel.cpp/src/emel/gguf/loader/guards.hpp"
        )
    }
    fn publish_bind_done(&mut self, _event: &EventBindRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::publish_bind_done
        todo!(
            "TODO: port action `publish_bind_done` from emel.cpp/src/emel/gguf/loader/actions.hpp"
        )
    }
    fn publish_bind_error_from_bind_outcome_dispatch(
        &mut self,
        _event: &EventBindRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::publish_bind_error
        todo!(
            "TODO: port action `publish_bind_error` from emel.cpp/src/emel/gguf/loader/actions.hpp"
        )
    }
    fn publish_parse_done(&mut self, _event: &EventParseRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::publish_parse_done
        todo!(
            "TODO: port action `publish_parse_done` from emel.cpp/src/emel/gguf/loader/actions.hpp"
        )
    }
    fn publish_parse_error_from_parse_outcome_dispatch(
        &mut self,
        _event: &EventParseRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::publish_parse_error
        todo!(
            "TODO: port action `publish_parse_error` from emel.cpp/src/emel/gguf/loader/actions.hpp"
        )
    }
    fn publish_probe_done(&mut self, _event: &EventProbeRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::publish_probe_done
        todo!(
            "TODO: port action `publish_probe_done` from emel.cpp/src/emel/gguf/loader/actions.hpp"
        )
    }
    fn publish_probe_error_from_probe_outcome_dispatch(
        &mut self,
        _event: &EventProbeRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gguf/loader/actions.hpp::publish_probe_error
        todo!(
            "TODO: port action `publish_probe_error` from emel.cpp/src/emel/gguf/loader/actions.hpp"
        )
    }
}
