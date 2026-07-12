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

// --- machine GbnfSamplerAcceptParser from emel.cpp/src/emel/gbnf/sampler/accept_parser/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct SamplerEventSampleRuntime;

sml! {
    GbnfSamplerAcceptParser {
        "parsed"_s <= *"deciding"_s + completion<SamplerEventSampleRuntime> [token_accepted_by_grammar] / consume_accepted,
        "parsed"_s <= "deciding"_s + completion<SamplerEventSampleRuntime> [token_rejected_by_grammar] / consume_rejected,
        "parse_failed"_s <= "deciding"_s + completion<SamplerEventSampleRuntime> [parse_failed] / dispatch_parse_failed,
        "parse_failed"_s <= "deciding"_s + unexpected_event<_> / on_unexpected_from_deciding,
        "parse_failed"_s <= "parsed"_s + unexpected_event<_> / on_unexpected_from_parsed,
        "parse_failed"_s <= "parse_failed"_s + unexpected_event<_> / on_unexpected_from_parse_failed,
        "parsed"_s = X,
        "parse_failed"_s = X,
    }
}

/// Context for `GbnfSamplerAcceptParser` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct GbnfSamplerAcceptParserContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl GbnfSamplerAcceptParserStateMachineContext for GbnfSamplerAcceptParserContext {
    fn consume_accepted(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/accept_parser/actions.hpp::consume_accepted
        todo!(
            "TODO: port action `consume_accepted` from emel.cpp/src/emel/gbnf/sampler/accept_parser/actions.hpp"
        )
    }
    fn consume_rejected(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/accept_parser/actions.hpp::consume_rejected
        todo!(
            "TODO: port action `consume_rejected` from emel.cpp/src/emel/gbnf/sampler/accept_parser/actions.hpp"
        )
    }
    fn dispatch_parse_failed(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/accept_parser/actions.hpp::dispatch_parse_failed
        todo!(
            "TODO: port action `dispatch_parse_failed` from emel.cpp/src/emel/gbnf/sampler/accept_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_deciding(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/accept_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/sampler/accept_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_parse_failed(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/accept_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/sampler/accept_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_parsed(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/accept_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/sampler/accept_parser/actions.hpp"
        )
    }
    fn parse_failed(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/accept_parser/guards.hpp::parse_failed
        todo!(
            "TODO: port guard `parse_failed` from emel.cpp/src/emel/gbnf/sampler/accept_parser/guards.hpp"
        )
    }
    fn token_accepted_by_grammar(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/accept_parser/guards.hpp::token_accepted_by_grammar
        todo!(
            "TODO: port guard `token_accepted_by_grammar` from emel.cpp/src/emel/gbnf/sampler/accept_parser/guards.hpp"
        )
    }
    fn token_rejected_by_grammar(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/sampler/accept_parser/guards.hpp::token_rejected_by_grammar
        todo!(
            "TODO: port guard `token_rejected_by_grammar` from emel.cpp/src/emel/gbnf/sampler/accept_parser/guards.hpp"
        )
    }
}
