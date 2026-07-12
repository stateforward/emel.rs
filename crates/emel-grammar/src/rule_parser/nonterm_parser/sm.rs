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

// --- machine GbnfRuleParserNontermParser from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct RuleParserEventParseRules;

sml! {
    GbnfRuleParserNontermParser {
        "definition_lookup_exec"_s <= *"deciding"_s + completion<RuleParserEventParseRules> [token_identifier_definition],
        "reference_lookup_exec"_s <= "deciding"_s + completion<RuleParserEventParseRules> [token_identifier_reference],
        "parse_failed"_s <= "deciding"_s + completion<RuleParserEventParseRules> / dispatch_parse_failed_from_deciding,
        "definition_lookup_decision"_s <= "definition_lookup_exec"_s + completion<RuleParserEventParseRules> / lookup_definition_candidate,
        "reference_lookup_decision"_s <= "reference_lookup_exec"_s + completion<RuleParserEventParseRules> / lookup_reference_candidate,
        "parsed"_s <= "definition_lookup_decision"_s + completion<RuleParserEventParseRules> [definition_existing_valid] / consume_definition_existing,
        "parsed"_s <= "definition_lookup_decision"_s + completion<RuleParserEventParseRules> [definition_new_valid] / consume_definition_new,
        "parse_failed"_s <= "definition_lookup_decision"_s + completion<RuleParserEventParseRules> [definition_failed] / dispatch_parse_failed_from_definition_lookup_decision,
        "parsed"_s <= "reference_lookup_decision"_s + completion<RuleParserEventParseRules> [reference_existing_valid] / consume_reference_existing,
        "parsed"_s <= "reference_lookup_decision"_s + completion<RuleParserEventParseRules> [reference_new_valid] / consume_reference_new,
        "parse_failed"_s <= "reference_lookup_decision"_s + completion<RuleParserEventParseRules> [reference_failed] / dispatch_parse_failed_from_reference_lookup_decision,
        "unexpected_event"_s <= "deciding"_s + unexpected_event<_> / on_unexpected_from_deciding,
        "unexpected_event"_s <= "definition_lookup_exec"_s + unexpected_event<_> / on_unexpected_from_definition_lookup_exec,
        "unexpected_event"_s <= "definition_lookup_decision"_s + unexpected_event<_> / on_unexpected_from_definition_lookup_decision,
        "unexpected_event"_s <= "reference_lookup_exec"_s + unexpected_event<_> / on_unexpected_from_reference_lookup_exec,
        "unexpected_event"_s <= "reference_lookup_decision"_s + unexpected_event<_> / on_unexpected_from_reference_lookup_decision,
        "unexpected_event"_s <= "parsed"_s + unexpected_event<_> / on_unexpected_from_parsed,
        "unexpected_event"_s <= "parse_failed"_s + unexpected_event<_> / on_unexpected_from_parse_failed,
        "unexpected_event"_s <= "unexpected_event"_s + unexpected_event<_> / on_unexpected_from_unexpected_event,
        "parsed"_s = X,
        "parse_failed"_s = X,
    }
}

/// Context for `GbnfRuleParserNontermParser` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct GbnfRuleParserNontermParserContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl GbnfRuleParserNontermParserStateMachineContext for GbnfRuleParserNontermParserContext {
    fn consume_definition_existing(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/actions.hpp::consume_definition_existing
        todo!(
            "TODO: port action `consume_definition_existing` from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/actions.hpp"
        )
    }
    fn consume_definition_new(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/actions.hpp::consume_definition_new
        todo!(
            "TODO: port action `consume_definition_new` from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/actions.hpp"
        )
    }
    fn consume_reference_existing(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/actions.hpp::consume_reference_existing
        todo!(
            "TODO: port action `consume_reference_existing` from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/actions.hpp"
        )
    }
    fn consume_reference_new(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/actions.hpp::consume_reference_new
        todo!(
            "TODO: port action `consume_reference_new` from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/actions.hpp"
        )
    }
    fn definition_existing_valid(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/guards.hpp::definition_existing_valid
        todo!(
            "TODO: port guard `definition_existing_valid` from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/guards.hpp"
        )
    }
    fn definition_failed(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/guards.hpp::definition_failed
        todo!(
            "TODO: port guard `definition_failed` from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/guards.hpp"
        )
    }
    fn definition_new_valid(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/guards.hpp::definition_new_valid
        todo!(
            "TODO: port guard `definition_new_valid` from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/guards.hpp"
        )
    }
    fn dispatch_parse_failed_from_deciding(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/actions.hpp::dispatch_parse_failed
        todo!(
            "TODO: port action `dispatch_parse_failed` from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/actions.hpp"
        )
    }
    fn dispatch_parse_failed_from_definition_lookup_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/actions.hpp::dispatch_parse_failed
        todo!(
            "TODO: port action `dispatch_parse_failed` from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/actions.hpp"
        )
    }
    fn dispatch_parse_failed_from_reference_lookup_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/actions.hpp::dispatch_parse_failed
        todo!(
            "TODO: port action `dispatch_parse_failed` from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/actions.hpp"
        )
    }
    fn lookup_definition_candidate(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/actions.hpp::lookup_definition_candidate
        todo!(
            "TODO: port action `lookup_definition_candidate` from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/actions.hpp"
        )
    }
    fn lookup_reference_candidate(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/actions.hpp::lookup_reference_candidate
        todo!(
            "TODO: port action `lookup_reference_candidate` from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_deciding(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_definition_lookup_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_definition_lookup_exec(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_parse_failed(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_parsed(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_reference_lookup_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_reference_lookup_exec(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_unexpected_event(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/actions.hpp"
        )
    }
    fn reference_existing_valid(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/guards.hpp::reference_existing_valid
        todo!(
            "TODO: port guard `reference_existing_valid` from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/guards.hpp"
        )
    }
    fn reference_failed(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/guards.hpp::reference_failed
        todo!(
            "TODO: port guard `reference_failed` from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/guards.hpp"
        )
    }
    fn reference_new_valid(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/guards.hpp::reference_new_valid
        todo!(
            "TODO: port guard `reference_new_valid` from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/guards.hpp"
        )
    }
    fn token_identifier_definition(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/guards.hpp::token_identifier_definition
        todo!(
            "TODO: port guard `token_identifier_definition` from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/guards.hpp"
        )
    }
    fn token_identifier_reference(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/guards.hpp::token_identifier_reference
        todo!(
            "TODO: port guard `token_identifier_reference` from emel.cpp/src/emel/gbnf/rule_parser/nonterm_parser/guards.hpp"
        )
    }
}
