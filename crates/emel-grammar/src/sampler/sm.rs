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

// --- machine GbnfSampler from emel.cpp/src/emel/gbnf/sampler/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventSampleRuntime;

sml! {
    GbnfSampler {
        "request_decision"_s <= *"ready"_s + event<EventSampleRuntime> / begin_sample,
        "filter_candidates"_s <= "request_decision"_s + completion<EventSampleRuntime> [valid_sample_request],
        "errored"_s <= "request_decision"_s + completion<EventSampleRuntime> [invalid_sample_request] / mark_invalid_request,
        "finalize_decision"_s <= "filter_candidates"_s + completion<EventSampleRuntime> / filter_candidates,
        "done"_s <= "finalize_decision"_s + completion<EventSampleRuntime> [filtered_candidates_available],
        "errored"_s <= "finalize_decision"_s + completion<EventSampleRuntime> [no_filtered_candidates] / mark_parse_failed,
        "ready"_s <= "done"_s + completion<EventSampleRuntime> / publish_done,
        "ready"_s <= "errored"_s + completion<EventSampleRuntime> / publish_error,
        "ready"_s <= "ready"_s + unexpected_event<_> / on_unexpected_from_ready,
        "ready"_s <= "request_decision"_s + unexpected_event<_> / on_unexpected_from_request_decision,
        "ready"_s <= "filter_candidates"_s + unexpected_event<_> / on_unexpected_from_filter_candidates,
        "ready"_s <= "finalize_decision"_s + unexpected_event<_> / on_unexpected_from_finalize_decision,
        "ready"_s <= "done"_s + unexpected_event<_> / on_unexpected_from_done,
        "ready"_s <= "errored"_s + unexpected_event<_> / on_unexpected_from_errored,
    }
}

/// Context for `GbnfSampler` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct GbnfSamplerContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl GbnfSamplerStateMachineContext for GbnfSamplerContext {
    fn begin_sample(&mut self, _event: &EventSampleRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/actions.hpp::begin_sample
        todo!("TODO: port action `begin_sample` from emel.cpp/src/emel/gbnf/sampler/actions.hpp")
    }
    fn filter_candidates(&mut self, _event: &EventSampleRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/actions.hpp::filter_candidates
        todo!(
            "TODO: port action `filter_candidates` from emel.cpp/src/emel/gbnf/sampler/actions.hpp"
        )
    }
    fn filtered_candidates_available(&self, _event: &EventSampleRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/guards.hpp::filtered_candidates_available
        todo!(
            "TODO: port guard `filtered_candidates_available` from emel.cpp/src/emel/gbnf/sampler/guards.hpp"
        )
    }
    fn invalid_sample_request(&self, _event: &EventSampleRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/guards.hpp::invalid_sample_request
        todo!(
            "TODO: port guard `invalid_sample_request` from emel.cpp/src/emel/gbnf/sampler/guards.hpp"
        )
    }
    fn mark_invalid_request(&mut self, _event: &EventSampleRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/gbnf/sampler/actions.hpp"
        )
    }
    fn mark_parse_failed(&mut self, _event: &EventSampleRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/actions.hpp::mark_parse_failed
        todo!(
            "TODO: port action `mark_parse_failed` from emel.cpp/src/emel/gbnf/sampler/actions.hpp"
        )
    }
    fn no_filtered_candidates(&self, _event: &EventSampleRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/guards.hpp::no_filtered_candidates
        todo!(
            "TODO: port guard `no_filtered_candidates` from emel.cpp/src/emel/gbnf/sampler/guards.hpp"
        )
    }
    fn on_unexpected_from_done(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/sampler/actions.hpp")
    }
    fn on_unexpected_from_errored(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/sampler/actions.hpp")
    }
    fn on_unexpected_from_filter_candidates(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/sampler/actions.hpp")
    }
    fn on_unexpected_from_finalize_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/sampler/actions.hpp")
    }
    fn on_unexpected_from_ready(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/sampler/actions.hpp")
    }
    fn on_unexpected_from_request_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/sampler/actions.hpp")
    }
    fn publish_done(&mut self, _event: &EventSampleRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/actions.hpp::publish_done
        todo!("TODO: port action `publish_done` from emel.cpp/src/emel/gbnf/sampler/actions.hpp")
    }
    fn publish_error(&mut self, _event: &EventSampleRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/actions.hpp::publish_error
        todo!("TODO: port action `publish_error` from emel.cpp/src/emel/gbnf/sampler/actions.hpp")
    }
    fn valid_sample_request(&self, _event: &EventSampleRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/guards.hpp::valid_sample_request
        todo!(
            "TODO: port guard `valid_sample_request` from emel.cpp/src/emel/gbnf/sampler/guards.hpp"
        )
    }
}
