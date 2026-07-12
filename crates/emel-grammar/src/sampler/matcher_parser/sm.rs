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

// --- machine GbnfSamplerMatcherParser from emel.cpp/src/emel/gbnf/sampler/matcher_parser/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct SamplerEventSampleRuntime;

sml! {
    GbnfSamplerMatcherParser {
        "parsed"_s <= *"deciding"_s + completion<SamplerEventSampleRuntime> [token_text] / consume_match_accepted,
        "parsed"_s <= "deciding"_s + completion<SamplerEventSampleRuntime> [token_empty] / consume_match_rejected,
        "parse_failed"_s <= "deciding"_s + completion<SamplerEventSampleRuntime> [parse_failed] / dispatch_parse_failed,
        "unexpected_event"_s <= "deciding"_s + unexpected_event<_> / on_unexpected_from_deciding,
        "unexpected_event"_s <= "parsed"_s + unexpected_event<_> / on_unexpected_from_parsed,
        "unexpected_event"_s <= "parse_failed"_s + unexpected_event<_> / on_unexpected_from_parse_failed,
        "unexpected_event"_s <= "unexpected_event"_s + unexpected_event<_> / on_unexpected_from_unexpected_event,
        "parsed"_s = X,
        "parse_failed"_s = X,
    }
}

/// Context for `GbnfSamplerMatcherParser` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct GbnfSamplerMatcherParserContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl GbnfSamplerMatcherParserStateMachineContext for GbnfSamplerMatcherParserContext {
    fn consume_match_accepted(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/matcher_parser/actions.hpp::consume_match_accepted
        todo!(
            "TODO: port action `consume_match_accepted` from emel.cpp/src/emel/gbnf/sampler/matcher_parser/actions.hpp"
        )
    }
    fn consume_match_rejected(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/matcher_parser/actions.hpp::consume_match_rejected
        todo!(
            "TODO: port action `consume_match_rejected` from emel.cpp/src/emel/gbnf/sampler/matcher_parser/actions.hpp"
        )
    }
    fn dispatch_parse_failed(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/matcher_parser/actions.hpp::dispatch_parse_failed
        todo!(
            "TODO: port action `dispatch_parse_failed` from emel.cpp/src/emel/gbnf/sampler/matcher_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_deciding(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/matcher_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/sampler/matcher_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_parse_failed(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/matcher_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/sampler/matcher_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_parsed(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/matcher_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/sampler/matcher_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_unexpected_event(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/matcher_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/sampler/matcher_parser/actions.hpp"
        )
    }
    fn parse_failed(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/matcher_parser/guards.hpp::parse_failed
        todo!(
            "TODO: port guard `parse_failed` from emel.cpp/src/emel/gbnf/sampler/matcher_parser/guards.hpp"
        )
    }
    fn token_empty(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/matcher_parser/guards.hpp::token_empty
        todo!(
            "TODO: port guard `token_empty` from emel.cpp/src/emel/gbnf/sampler/matcher_parser/guards.hpp"
        )
    }
    fn token_text(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/matcher_parser/guards.hpp::token_text
        todo!(
            "TODO: port guard `token_text` from emel.cpp/src/emel/gbnf/sampler/matcher_parser/guards.hpp"
        )
    }
}
