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

// --- machine GbnfRuleParserExpressionParser from emel.cpp/src/emel/gbnf/rule_parser/expression_parser/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct RuleParserEventParseRules;

sml! {
    GbnfRuleParserExpressionParser {
        "parsed_identifier"_s <= *"deciding"_s + completion<RuleParserEventParseRules> [token_identifier] / consume_identifier,
        "parsed_non_identifier"_s <= "deciding"_s + completion<RuleParserEventParseRules> [token_non_identifier] / consume_non_identifier,
        "parse_failed"_s <= "deciding"_s + completion<RuleParserEventParseRules> [parse_failed] / dispatch_parse_failed,
        "unexpected_event"_s <= "deciding"_s + unexpected_event<_> / on_unexpected_from_deciding,
        "unexpected_event"_s <= "parsed_identifier"_s + unexpected_event<_> / on_unexpected_from_parsed_identifier,
        "unexpected_event"_s <= "parsed_non_identifier"_s + unexpected_event<_> / on_unexpected_from_parsed_non_identifier,
        "unexpected_event"_s <= "parse_failed"_s + unexpected_event<_> / on_unexpected_from_parse_failed,
        "unexpected_event"_s <= "unexpected_event"_s + unexpected_event<_> / on_unexpected_from_unexpected_event,
        "parsed_identifier"_s = X,
        "parsed_non_identifier"_s = X,
        "parse_failed"_s = X,
    }
}

/// Context for `GbnfRuleParserExpressionParser` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct GbnfRuleParserExpressionParserContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl GbnfRuleParserExpressionParserStateMachineContext for GbnfRuleParserExpressionParserContext {
    fn consume_identifier(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/expression_parser/actions.hpp::consume_identifier
        todo!(
            "TODO: port action `consume_identifier` from emel.cpp/src/emel/gbnf/rule_parser/expression_parser/actions.hpp"
        )
    }
    fn consume_non_identifier(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/expression_parser/actions.hpp::consume_non_identifier
        todo!(
            "TODO: port action `consume_non_identifier` from emel.cpp/src/emel/gbnf/rule_parser/expression_parser/actions.hpp"
        )
    }
    fn dispatch_parse_failed(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/expression_parser/actions.hpp::dispatch_parse_failed
        todo!(
            "TODO: port action `dispatch_parse_failed` from emel.cpp/src/emel/gbnf/rule_parser/expression_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_deciding(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/expression_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/rule_parser/expression_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_parse_failed(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/expression_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/rule_parser/expression_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_parsed_identifier(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/expression_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/rule_parser/expression_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_parsed_non_identifier(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/expression_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/rule_parser/expression_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_unexpected_event(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/expression_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/rule_parser/expression_parser/actions.hpp"
        )
    }
    fn parse_failed(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/expression_parser/guards.hpp::parse_failed
        todo!(
            "TODO: port guard `parse_failed` from emel.cpp/src/emel/gbnf/rule_parser/expression_parser/guards.hpp"
        )
    }
    fn token_identifier(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/expression_parser/guards.hpp::token_identifier
        todo!(
            "TODO: port guard `token_identifier` from emel.cpp/src/emel/gbnf/rule_parser/expression_parser/guards.hpp"
        )
    }
    fn token_non_identifier(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/expression_parser/guards.hpp::token_non_identifier
        todo!(
            "TODO: port guard `token_non_identifier` from emel.cpp/src/emel/gbnf/rule_parser/expression_parser/guards.hpp"
        )
    }
}
