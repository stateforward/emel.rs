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

// --- machine GbnfSamplerTokenParser from emel.cpp/src/emel/gbnf/sampler/token_parser/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct SamplerEventSampleRuntime;

sml! {
    GbnfSamplerTokenParser {
        "parsed"_s <= *"deciding"_s + completion<SamplerEventSampleRuntime> [candidate_text] / consume_text_token,
        "parsed"_s <= "deciding"_s + completion<SamplerEventSampleRuntime> [candidate_empty] / consume_empty_token,
        "parse_failed"_s <= "deciding"_s + completion<SamplerEventSampleRuntime> [parse_failed] / dispatch_parse_failed,
        "unexpected_event"_s <= "deciding"_s + unexpected_event<_> / on_unexpected_from_deciding,
        "unexpected_event"_s <= "parsed"_s + unexpected_event<_> / on_unexpected_from_parsed,
        "unexpected_event"_s <= "parse_failed"_s + unexpected_event<_> / on_unexpected_from_parse_failed,
        "unexpected_event"_s <= "unexpected_event"_s + unexpected_event<_> / on_unexpected_from_unexpected_event,
        "parsed"_s = X,
        "parse_failed"_s = X,
    }
}

/// Context for `GbnfSamplerTokenParser` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct GbnfSamplerTokenParserContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl GbnfSamplerTokenParserStateMachineContext for GbnfSamplerTokenParserContext {
    fn candidate_empty(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/token_parser/guards.hpp::candidate_empty
        todo!(
            "TODO: port guard `candidate_empty` from emel.cpp/src/emel/gbnf/sampler/token_parser/guards.hpp"
        )
    }
    fn candidate_text(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/token_parser/guards.hpp::candidate_text
        todo!(
            "TODO: port guard `candidate_text` from emel.cpp/src/emel/gbnf/sampler/token_parser/guards.hpp"
        )
    }
    fn consume_empty_token(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/token_parser/actions.hpp::consume_empty_token
        todo!(
            "TODO: port action `consume_empty_token` from emel.cpp/src/emel/gbnf/sampler/token_parser/actions.hpp"
        )
    }
    fn consume_text_token(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/token_parser/actions.hpp::consume_text_token
        todo!(
            "TODO: port action `consume_text_token` from emel.cpp/src/emel/gbnf/sampler/token_parser/actions.hpp"
        )
    }
    fn dispatch_parse_failed(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/token_parser/actions.hpp::dispatch_parse_failed
        todo!(
            "TODO: port action `dispatch_parse_failed` from emel.cpp/src/emel/gbnf/sampler/token_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_deciding(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/token_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/sampler/token_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_parse_failed(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/token_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/sampler/token_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_parsed(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/token_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/sampler/token_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_unexpected_event(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/token_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/sampler/token_parser/actions.hpp"
        )
    }
    fn parse_failed(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/token_parser/guards.hpp::parse_failed
        todo!(
            "TODO: port guard `parse_failed` from emel.cpp/src/emel/gbnf/sampler/token_parser/guards.hpp"
        )
    }
}
