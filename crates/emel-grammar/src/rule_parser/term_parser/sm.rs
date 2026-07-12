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

// --- machine GbnfRuleParserTermParser from emel.cpp/src/emel/gbnf/rule_parser/term_parser/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct RuleParserEventParseRules;

sml! {
    GbnfRuleParserTermParser {
        "parsed"_s <= *"deciding"_s + completion<RuleParserEventParseRules> [token_string_literal] / consume_string_literal,
        "parsed"_s <= "deciding"_s + completion<RuleParserEventParseRules> [token_character_class] / consume_character_class,
        "parsed"_s <= "deciding"_s + completion<RuleParserEventParseRules> [token_rule_reference] / consume_rule_reference,
        "parsed"_s <= "deciding"_s + completion<RuleParserEventParseRules> [token_dot] / consume_dot,
        "parsed"_s <= "deciding"_s + completion<RuleParserEventParseRules> [token_open_group] / consume_open_group,
        "parsed"_s <= "deciding"_s + completion<RuleParserEventParseRules> [token_close_group] / consume_close_group,
        "parsed"_s <= "deciding"_s + completion<RuleParserEventParseRules> [token_quantifier] / consume_quantifier,
        "parsed"_s <= "deciding"_s + completion<RuleParserEventParseRules> [token_alternation] / consume_alternation,
        "parsed"_s <= "deciding"_s + completion<RuleParserEventParseRules> [token_newline] / consume_newline,
        "parse_failed"_s <= "deciding"_s + completion<RuleParserEventParseRules> [parse_failed] / dispatch_parse_failed,
        "unexpected_event"_s <= "deciding"_s + unexpected_event<_> / on_unexpected_from_deciding,
        "unexpected_event"_s <= "parsed"_s + unexpected_event<_> / on_unexpected_from_parsed,
        "unexpected_event"_s <= "parse_failed"_s + unexpected_event<_> / on_unexpected_from_parse_failed,
        "unexpected_event"_s <= "unexpected_event"_s + unexpected_event<_> / on_unexpected_from_unexpected_event,
        "parsed"_s = X,
        "parse_failed"_s = X,
    }
}

/// Context for `GbnfRuleParserTermParser` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct GbnfRuleParserTermParserContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl GbnfRuleParserTermParserStateMachineContext for GbnfRuleParserTermParserContext {
    fn consume_alternation(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/term_parser/actions.hpp::consume_alternation
        todo!(
            "TODO: port action `consume_alternation` from emel.cpp/src/emel/gbnf/rule_parser/term_parser/actions.hpp"
        )
    }
    fn consume_character_class(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/term_parser/actions.hpp::consume_character_class
        todo!(
            "TODO: port action `consume_character_class` from emel.cpp/src/emel/gbnf/rule_parser/term_parser/actions.hpp"
        )
    }
    fn consume_close_group(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/term_parser/actions.hpp::consume_close_group
        todo!(
            "TODO: port action `consume_close_group` from emel.cpp/src/emel/gbnf/rule_parser/term_parser/actions.hpp"
        )
    }
    fn consume_dot(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/term_parser/actions.hpp::consume_dot
        todo!(
            "TODO: port action `consume_dot` from emel.cpp/src/emel/gbnf/rule_parser/term_parser/actions.hpp"
        )
    }
    fn consume_newline(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/term_parser/actions.hpp::consume_newline
        todo!(
            "TODO: port action `consume_newline` from emel.cpp/src/emel/gbnf/rule_parser/term_parser/actions.hpp"
        )
    }
    fn consume_open_group(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/term_parser/actions.hpp::consume_open_group
        todo!(
            "TODO: port action `consume_open_group` from emel.cpp/src/emel/gbnf/rule_parser/term_parser/actions.hpp"
        )
    }
    fn consume_quantifier(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/term_parser/actions.hpp::consume_quantifier
        todo!(
            "TODO: port action `consume_quantifier` from emel.cpp/src/emel/gbnf/rule_parser/term_parser/actions.hpp"
        )
    }
    fn consume_rule_reference(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/term_parser/actions.hpp::consume_rule_reference
        todo!(
            "TODO: port action `consume_rule_reference` from emel.cpp/src/emel/gbnf/rule_parser/term_parser/actions.hpp"
        )
    }
    fn consume_string_literal(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/term_parser/actions.hpp::consume_string_literal
        todo!(
            "TODO: port action `consume_string_literal` from emel.cpp/src/emel/gbnf/rule_parser/term_parser/actions.hpp"
        )
    }
    fn dispatch_parse_failed(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/term_parser/actions.hpp::dispatch_parse_failed
        todo!(
            "TODO: port action `dispatch_parse_failed` from emel.cpp/src/emel/gbnf/rule_parser/term_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_deciding(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/term_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/rule_parser/term_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_parse_failed(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/term_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/rule_parser/term_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_parsed(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/term_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/rule_parser/term_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_unexpected_event(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/term_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/rule_parser/term_parser/actions.hpp"
        )
    }
    fn parse_failed(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/term_parser/guards.hpp::parse_failed
        todo!(
            "TODO: port guard `parse_failed` from emel.cpp/src/emel/gbnf/rule_parser/term_parser/guards.hpp"
        )
    }
    fn token_alternation(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/term_parser/guards.hpp::token_alternation
        todo!(
            "TODO: port guard `token_alternation` from emel.cpp/src/emel/gbnf/rule_parser/term_parser/guards.hpp"
        )
    }
    fn token_character_class(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/term_parser/guards.hpp::token_character_class
        todo!(
            "TODO: port guard `token_character_class` from emel.cpp/src/emel/gbnf/rule_parser/term_parser/guards.hpp"
        )
    }
    fn token_close_group(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/term_parser/guards.hpp::token_close_group
        todo!(
            "TODO: port guard `token_close_group` from emel.cpp/src/emel/gbnf/rule_parser/term_parser/guards.hpp"
        )
    }
    fn token_dot(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/term_parser/guards.hpp::token_dot
        todo!(
            "TODO: port guard `token_dot` from emel.cpp/src/emel/gbnf/rule_parser/term_parser/guards.hpp"
        )
    }
    fn token_newline(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/term_parser/guards.hpp::token_newline
        todo!(
            "TODO: port guard `token_newline` from emel.cpp/src/emel/gbnf/rule_parser/term_parser/guards.hpp"
        )
    }
    fn token_open_group(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/term_parser/guards.hpp::token_open_group
        todo!(
            "TODO: port guard `token_open_group` from emel.cpp/src/emel/gbnf/rule_parser/term_parser/guards.hpp"
        )
    }
    fn token_quantifier(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/term_parser/guards.hpp::token_quantifier
        todo!(
            "TODO: port guard `token_quantifier` from emel.cpp/src/emel/gbnf/rule_parser/term_parser/guards.hpp"
        )
    }
    fn token_rule_reference(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/term_parser/guards.hpp::token_rule_reference
        todo!(
            "TODO: port guard `token_rule_reference` from emel.cpp/src/emel/gbnf/rule_parser/term_parser/guards.hpp"
        )
    }
    fn token_string_literal(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/term_parser/guards.hpp::token_string_literal
        todo!(
            "TODO: port guard `token_string_literal` from emel.cpp/src/emel/gbnf/rule_parser/term_parser/guards.hpp"
        )
    }
}
