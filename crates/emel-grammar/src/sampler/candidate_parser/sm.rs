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

// --- machine GbnfSamplerCandidateParser from emel.cpp/src/emel/gbnf/sampler/candidate_parser/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct SamplerEventSampleRuntime;

sml! {
    GbnfSamplerCandidateParser {
        "parsed"_s <= *"deciding"_s + completion<SamplerEventSampleRuntime> [has_apply_text] / consume_text,
        "parsed"_s <= "deciding"_s + completion<SamplerEventSampleRuntime> [has_empty_apply_text] / consume_empty,
        "parse_failed"_s <= "deciding"_s + completion<SamplerEventSampleRuntime> [parse_failed] / dispatch_parse_failed,
        "unexpected_event"_s <= "deciding"_s + unexpected_event<_> / on_unexpected_from_deciding,
        "unexpected_event"_s <= "parsed"_s + unexpected_event<_> / on_unexpected_from_parsed,
        "unexpected_event"_s <= "parse_failed"_s + unexpected_event<_> / on_unexpected_from_parse_failed,
        "unexpected_event"_s <= "unexpected_event"_s + unexpected_event<_> / on_unexpected_from_unexpected_event,
        "parsed"_s = X,
        "parse_failed"_s = X,
    }
}

/// Context for `GbnfSamplerCandidateParser` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct GbnfSamplerCandidateParserContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl GbnfSamplerCandidateParserStateMachineContext for GbnfSamplerCandidateParserContext {
    fn consume_empty(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/candidate_parser/actions.hpp::consume_empty
        todo!(
            "TODO: port action `consume_empty` from emel.cpp/src/emel/gbnf/sampler/candidate_parser/actions.hpp"
        )
    }
    fn consume_text(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/candidate_parser/actions.hpp::consume_text
        todo!(
            "TODO: port action `consume_text` from emel.cpp/src/emel/gbnf/sampler/candidate_parser/actions.hpp"
        )
    }
    fn dispatch_parse_failed(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/candidate_parser/actions.hpp::dispatch_parse_failed
        todo!(
            "TODO: port action `dispatch_parse_failed` from emel.cpp/src/emel/gbnf/sampler/candidate_parser/actions.hpp"
        )
    }
    fn has_apply_text(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/candidate_parser/guards.hpp::has_apply_text
        todo!(
            "TODO: port guard `has_apply_text` from emel.cpp/src/emel/gbnf/sampler/candidate_parser/guards.hpp"
        )
    }
    fn has_empty_apply_text(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/candidate_parser/guards.hpp::has_empty_apply_text
        todo!(
            "TODO: port guard `has_empty_apply_text` from emel.cpp/src/emel/gbnf/sampler/candidate_parser/guards.hpp"
        )
    }
    fn on_unexpected_from_deciding(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/candidate_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/sampler/candidate_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_parse_failed(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/candidate_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/sampler/candidate_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_parsed(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/candidate_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/sampler/candidate_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_unexpected_event(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/candidate_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/sampler/candidate_parser/actions.hpp"
        )
    }
    fn parse_failed(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/candidate_parser/guards.hpp::parse_failed
        todo!(
            "TODO: port guard `parse_failed` from emel.cpp/src/emel/gbnf/sampler/candidate_parser/guards.hpp"
        )
    }
}
