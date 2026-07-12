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

// --- machine GbnfRuleParser from emel.cpp/src/emel/gbnf/rule_parser/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventParseRules;

sml! {
    GbnfRuleParser {
        "expect_rule_name"_s <= *"ready"_s + event<EventParseRules> [valid_parse] / begin_parse,
        "ready"_s <= "ready"_s + event<EventParseRules> [invalid_parse_with_dispatchable_grammar] / reject_invalid_parse_with_dispatch,
        "ready"_s <= "ready"_s + event<EventParseRules> [invalid_parse_with_grammar_only] / reject_invalid_parse_with_grammar_only,
        "ready"_s <= "ready"_s + event<EventParseRules> [invalid_parse_without_grammar] / reject_invalid_parse_without_grammar,
        "expect_rule_name_decision"_s <= "expect_rule_name"_s + completion<EventParseRules> / request_next_token_from_expect_rule_name,
        "parse_decision"_s <= "expect_rule_name_decision"_s + completion<EventParseRules> [lexer_failed],
        "eof_symbols_decision"_s <= "expect_rule_name_decision"_s + completion<EventParseRules> [lexer_at_eof],
        "nonterm_parser_model"_s <= "expect_rule_name_decision"_s + completion<EventParseRules> [token_identifier] / set_nonterm_mode_definition,
        "expect_rule_name"_s <= "expect_rule_name_decision"_s + completion<EventParseRules> [token_newline],
        "parse_decision"_s <= "expect_rule_name_decision"_s + completion<EventParseRules> / consume_token_invalid_from_expect_rule_name_decision,
        "parse_decision"_s <= "nonterm_parser_model"_s + completion<EventParseRules> [nonterm_failed],
        "expect_definition"_s <= "nonterm_parser_model"_s + completion<EventParseRules> [nonterm_definition_done] / apply_nonterm_definition,
        "in_rule_expression_after_term"_s <= "nonterm_parser_model"_s + completion<EventParseRules> [nonterm_reference_done] / apply_nonterm_reference,
        "parse_decision"_s <= "nonterm_parser_model"_s + completion<EventParseRules> / consume_token_invalid_from_nonterm_parser_model,
        "expect_definition_decision"_s <= "expect_definition"_s + completion<EventParseRules> / request_next_token_from_expect_definition,
        "parse_decision"_s <= "expect_definition_decision"_s + completion<EventParseRules> [lexer_failed],
        "parse_decision"_s <= "expect_definition_decision"_s + completion<EventParseRules> [lexer_at_eof] / fail_eof_in_expect_definition,
        "definition_parser_model"_s <= "expect_definition_decision"_s + completion<EventParseRules> [lexer_has_token],
        "parse_decision"_s <= "expect_definition_decision"_s + completion<EventParseRules> / consume_token_invalid_from_expect_definition_decision,
        "parse_decision"_s <= "definition_parser_model"_s + completion<EventParseRules> [definition_failed],
        "in_rule_expression_need_term"_s <= "definition_parser_model"_s + completion<EventParseRules> [definition_done] / consume_token_definition_operator,
        "parse_decision"_s <= "definition_parser_model"_s + completion<EventParseRules> / consume_token_invalid_from_definition_parser_model,
        "in_rule_expression_need_term_decision"_s <= "in_rule_expression_need_term"_s + completion<EventParseRules> / request_next_token_from_in_rule_expression_need_term,
        "parse_decision"_s <= "in_rule_expression_need_term_decision"_s + completion<EventParseRules> [lexer_failed],
        "expression_parser_model"_s <= "in_rule_expression_need_term_decision"_s + completion<EventParseRules> [lexer_has_token] / set_term_origin_need_term,
        "parse_decision"_s <= "in_rule_expression_need_term_decision"_s + completion<EventParseRules> / consume_token_invalid_from_in_rule_expression_need_term_decision,
        "in_rule_expression_after_term_decision"_s <= "in_rule_expression_after_term"_s + completion<EventParseRules> / request_next_token_from_in_rule_expression_after_term,
        "parse_decision"_s <= "in_rule_expression_after_term_decision"_s + completion<EventParseRules> [lexer_failed],
        "eof_symbols_decision"_s <= "in_rule_expression_after_term_decision"_s + completion<EventParseRules> [eof_can_finalize_active_rule] / finalize_active_rule_on_eof_from_in_rule_expression_after_term_decision,
        "parse_decision"_s <= "in_rule_expression_after_term_decision"_s + completion<EventParseRules> [eof_cannot_finalize_active_rule] / consume_token_invalid_from_in_rule_expression_after_term_decision,
        "expression_parser_model"_s <= "in_rule_expression_after_term_decision"_s + completion<EventParseRules> [lexer_has_token] / set_term_origin_after_term,
        "parse_decision"_s <= "in_rule_expression_after_term_decision"_s + completion<EventParseRules> / consume_token_invalid_from_in_rule_expression_after_term_decision,
        "parse_decision"_s <= "expression_parser_model"_s + completion<EventParseRules> [expression_failed],
        "nonterm_parser_model"_s <= "expression_parser_model"_s + completion<EventParseRules> [expression_done_identifier] / set_nonterm_mode_reference,
        "term_parser_model"_s <= "expression_parser_model"_s + completion<EventParseRules> [expression_done_non_identifier],
        "parse_decision"_s <= "expression_parser_model"_s + completion<EventParseRules> / consume_token_invalid_from_expression_parser_model,
        "parse_decision"_s <= "term_parser_model"_s + completion<EventParseRules> [term_failed],
        "in_rule_expression_after_term"_s <= "term_parser_model"_s + completion<EventParseRules> [term_need_literal_valid] / consume_token_literal_from_term_parser_model,
        "in_rule_expression_after_term"_s <= "term_parser_model"_s + completion<EventParseRules> [term_need_character_class_valid] / consume_token_character_class_from_term_parser_model,
        "in_rule_expression_after_term"_s <= "term_parser_model"_s + completion<EventParseRules> [term_after_literal_valid] / consume_token_literal_from_term_parser_model,
        "in_rule_expression_after_term"_s <= "term_parser_model"_s + completion<EventParseRules> [term_after_character_class_valid] / consume_token_character_class_from_term_parser_model,
        "rule_reference_decision"_s <= "term_parser_model"_s + completion<EventParseRules> [term_need_rule_reference_candidate],
        "in_rule_expression_after_term"_s <= "term_parser_model"_s + completion<EventParseRules> [term_need_dot_valid] / consume_token_dot_from_term_parser_model,
        "in_rule_expression_need_term"_s <= "term_parser_model"_s + completion<EventParseRules> [term_need_open_group_valid] / consume_token_open_group_from_term_parser_model,
        "in_rule_expression_need_term"_s <= "term_parser_model"_s + completion<EventParseRules> [term_need_newline_with_group_depth_nonzero],
        "parse_decision"_s <= "term_parser_model"_s + completion<EventParseRules> [term_from_need_term] / consume_token_invalid_from_term_parser_model,
        "rule_reference_decision"_s <= "term_parser_model"_s + completion<EventParseRules> [term_after_rule_reference_candidate],
        "in_rule_expression_after_term"_s <= "term_parser_model"_s + completion<EventParseRules> [term_after_dot_valid] / consume_token_dot_from_term_parser_model,
        "in_rule_expression_need_term"_s <= "term_parser_model"_s + completion<EventParseRules> [term_after_open_group_valid] / consume_token_open_group_from_term_parser_model,
        "in_rule_expression_need_term"_s <= "term_parser_model"_s + completion<EventParseRules> [term_after_alternation_valid] / consume_token_alternation,
        "in_rule_expression_after_term"_s <= "term_parser_model"_s + completion<EventParseRules> [term_after_newline_with_group_depth_nonzero],
        "expect_rule_name"_s <= "term_parser_model"_s + completion<EventParseRules> [term_after_newline_with_group_depth_zero_valid] / finalize_active_rule_on_eof_from_term_parser_model,
        "in_rule_expression_after_term"_s <= "term_parser_model"_s + completion<EventParseRules> [term_after_close_group_valid] / consume_token_close_group,
        "quantifier_decision"_s <= "term_parser_model"_s + completion<EventParseRules> [term_after_quantifier_candidate],
        "parse_decision"_s <= "term_parser_model"_s + completion<EventParseRules> [term_from_after_term] / consume_token_invalid_from_term_parser_model,
        "rule_reference_plain_exec"_s <= "rule_reference_decision"_s + completion<EventParseRules> [rule_reference_plain_envelope_valid] / consume_token_rule_reference_plain,
        "rule_reference_negated_exec"_s <= "rule_reference_decision"_s + completion<EventParseRules> [rule_reference_negated_envelope_valid] / consume_token_rule_reference_negated,
        "parse_decision"_s <= "rule_reference_decision"_s + completion<EventParseRules> / consume_token_invalid_from_rule_reference_decision,
        "in_rule_expression_after_term"_s <= "rule_reference_plain_exec"_s + completion<EventParseRules> [parse_error_none],
        "parse_decision"_s <= "rule_reference_plain_exec"_s + completion<EventParseRules> / consume_token_invalid_from_rule_reference_plain_exec,
        "in_rule_expression_after_term"_s <= "rule_reference_negated_exec"_s + completion<EventParseRules> [parse_error_none],
        "parse_decision"_s <= "rule_reference_negated_exec"_s + completion<EventParseRules> / consume_token_invalid_from_rule_reference_negated_exec,
        "quantifier_star_exec"_s <= "quantifier_decision"_s + completion<EventParseRules> [quantifier_token_star] / consume_token_quantifier_star,
        "quantifier_plus_exec"_s <= "quantifier_decision"_s + completion<EventParseRules> [quantifier_token_plus] / consume_token_quantifier_plus,
        "quantifier_question_exec"_s <= "quantifier_decision"_s + completion<EventParseRules> [quantifier_token_question] / consume_token_quantifier_question,
        "quantifier_braced_exact_exec"_s <= "quantifier_decision"_s + completion<EventParseRules> [quantifier_braced_exact_shape] / consume_token_quantifier_braced_exact,
        "quantifier_braced_open_exec"_s <= "quantifier_decision"_s + completion<EventParseRules> [quantifier_braced_open_shape] / consume_token_quantifier_braced_open,
        "quantifier_braced_range_exec"_s <= "quantifier_decision"_s + completion<EventParseRules> [quantifier_braced_range_shape] / consume_token_quantifier_braced_range,
        "parse_decision"_s <= "quantifier_decision"_s + completion<EventParseRules> / consume_token_invalid_from_quantifier_decision,
        "in_rule_expression_after_term"_s <= "quantifier_star_exec"_s + completion<EventParseRules> [parse_error_none],
        "parse_decision"_s <= "quantifier_star_exec"_s + completion<EventParseRules> / consume_token_invalid_from_quantifier_star_exec,
        "in_rule_expression_after_term"_s <= "quantifier_plus_exec"_s + completion<EventParseRules> [parse_error_none],
        "parse_decision"_s <= "quantifier_plus_exec"_s + completion<EventParseRules> / consume_token_invalid_from_quantifier_plus_exec,
        "in_rule_expression_after_term"_s <= "quantifier_question_exec"_s + completion<EventParseRules> [parse_error_none],
        "parse_decision"_s <= "quantifier_question_exec"_s + completion<EventParseRules> / consume_token_invalid_from_quantifier_question_exec,
        "in_rule_expression_after_term"_s <= "quantifier_braced_exact_exec"_s + completion<EventParseRules> [parse_error_none],
        "parse_decision"_s <= "quantifier_braced_exact_exec"_s + completion<EventParseRules> / consume_token_invalid_from_quantifier_braced_exact_exec,
        "in_rule_expression_after_term"_s <= "quantifier_braced_open_exec"_s + completion<EventParseRules> [parse_error_none],
        "parse_decision"_s <= "quantifier_braced_open_exec"_s + completion<EventParseRules> / consume_token_invalid_from_quantifier_braced_open_exec,
        "in_rule_expression_after_term"_s <= "quantifier_braced_range_exec"_s + completion<EventParseRules> [parse_error_none],
        "parse_decision"_s <= "quantifier_braced_range_exec"_s + completion<EventParseRules> / consume_token_invalid_from_quantifier_braced_range_exec,
        "parse_decision"_s <= "eof_symbols_decision"_s + completion<EventParseRules> [eof_can_finalize_symbols],
        "parse_decision"_s <= "eof_symbols_decision"_s + completion<EventParseRules> [eof_cannot_finalize_symbols] / consume_token_invalid_from_eof_symbols_decision,
        "ready"_s <= "parse_decision"_s + completion<EventParseRules> [parse_error_none] / dispatch_done,
        "ready"_s <= "parse_decision"_s + completion<EventParseRules> [parse_error_invalid_request] / dispatch_error_from_parse_decision,
        "ready"_s <= "parse_decision"_s + completion<EventParseRules> [parse_error_parse_failed] / dispatch_error_from_parse_decision,
        "ready"_s <= "parse_decision"_s + completion<EventParseRules> [parse_error_internal_error] / dispatch_error_from_parse_decision,
        "ready"_s <= "parse_decision"_s + completion<EventParseRules> [parse_error_untracked] / dispatch_error_from_parse_decision,
        "ready"_s <= "parse_decision"_s + completion<EventParseRules> [parse_error_unknown] / dispatch_error_from_parse_decision,
        "ready"_s <= "ready"_s + unexpected_event<_> / on_unexpected_from_ready,
        "parse_decision"_s <= "expect_rule_name"_s + unexpected_event<_> / on_unexpected_from_expect_rule_name,
        "parse_decision"_s <= "expect_rule_name_decision"_s + unexpected_event<_> / on_unexpected_from_expect_rule_name_decision,
        "parse_decision"_s <= "expect_definition"_s + unexpected_event<_> / on_unexpected_from_expect_definition,
        "parse_decision"_s <= "expect_definition_decision"_s + unexpected_event<_> / on_unexpected_from_expect_definition_decision,
        "parse_decision"_s <= "in_rule_expression_need_term"_s + unexpected_event<_> / on_unexpected_from_in_rule_expression_need_term,
        "parse_decision"_s <= "in_rule_expression_need_term_decision"_s + unexpected_event<_> / on_unexpected_from_in_rule_expression_need_term_decision,
        "parse_decision"_s <= "in_rule_expression_after_term"_s + unexpected_event<_> / on_unexpected_from_in_rule_expression_after_term,
        "parse_decision"_s <= "in_rule_expression_after_term_decision"_s + unexpected_event<_> / on_unexpected_from_in_rule_expression_after_term_decision,
        "parse_decision"_s <= "rule_reference_decision"_s + unexpected_event<_> / on_unexpected_from_rule_reference_decision,
        "parse_decision"_s <= "rule_reference_plain_exec"_s + unexpected_event<_> / on_unexpected_from_rule_reference_plain_exec,
        "parse_decision"_s <= "rule_reference_negated_exec"_s + unexpected_event<_> / on_unexpected_from_rule_reference_negated_exec,
        "parse_decision"_s <= "quantifier_decision"_s + unexpected_event<_> / on_unexpected_from_quantifier_decision,
        "parse_decision"_s <= "quantifier_star_exec"_s + unexpected_event<_> / on_unexpected_from_quantifier_star_exec,
        "parse_decision"_s <= "quantifier_plus_exec"_s + unexpected_event<_> / on_unexpected_from_quantifier_plus_exec,
        "parse_decision"_s <= "quantifier_question_exec"_s + unexpected_event<_> / on_unexpected_from_quantifier_question_exec,
        "parse_decision"_s <= "quantifier_braced_exact_exec"_s + unexpected_event<_> / on_unexpected_from_quantifier_braced_exact_exec,
        "parse_decision"_s <= "quantifier_braced_open_exec"_s + unexpected_event<_> / on_unexpected_from_quantifier_braced_open_exec,
        "parse_decision"_s <= "quantifier_braced_range_exec"_s + unexpected_event<_> / on_unexpected_from_quantifier_braced_range_exec,
        "parse_decision"_s <= "eof_symbols_decision"_s + unexpected_event<_> / on_unexpected_from_eof_symbols_decision,
        "ready"_s <= "parse_decision"_s + unexpected_event<_> / on_unexpected_from_parse_decision,
    }
}

/// Context for `GbnfRuleParser` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct GbnfRuleParserContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl GbnfRuleParserStateMachineContext for GbnfRuleParserContext {
    fn apply_nonterm_definition(&mut self, _event: &EventParseRules) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::apply_nonterm_definition
        todo!(
            "TODO: port action `apply_nonterm_definition` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn apply_nonterm_reference(&mut self, _event: &EventParseRules) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::apply_nonterm_reference
        todo!(
            "TODO: port action `apply_nonterm_reference` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn begin_parse(&mut self, _event: &EventParseRules) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::begin_parse
        todo!("TODO: port action `begin_parse` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp")
    }
    fn consume_token_alternation(&mut self, _event: &EventParseRules) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::consume_token_alternation
        todo!(
            "TODO: port action `consume_token_alternation` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn consume_token_character_class_from_term_parser_model(
        &mut self,
        _event: &EventParseRules,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::consume_token_character_class
        todo!(
            "TODO: port action `consume_token_character_class` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn consume_token_close_group(&mut self, _event: &EventParseRules) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::consume_token_close_group
        todo!(
            "TODO: port action `consume_token_close_group` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn consume_token_definition_operator(&mut self, _event: &EventParseRules) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::consume_token_definition_operator
        todo!(
            "TODO: port action `consume_token_definition_operator` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn consume_token_dot_from_term_parser_model(
        &mut self,
        _event: &EventParseRules,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::consume_token_dot
        todo!(
            "TODO: port action `consume_token_dot` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn consume_token_invalid_from_definition_parser_model(
        &mut self,
        _event: &EventParseRules,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::consume_token_invalid
        todo!(
            "TODO: port action `consume_token_invalid` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn consume_token_invalid_from_eof_symbols_decision(
        &mut self,
        _event: &EventParseRules,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::consume_token_invalid
        todo!(
            "TODO: port action `consume_token_invalid` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn consume_token_invalid_from_expect_definition_decision(
        &mut self,
        _event: &EventParseRules,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::consume_token_invalid
        todo!(
            "TODO: port action `consume_token_invalid` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn consume_token_invalid_from_expect_rule_name_decision(
        &mut self,
        _event: &EventParseRules,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::consume_token_invalid
        todo!(
            "TODO: port action `consume_token_invalid` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn consume_token_invalid_from_expression_parser_model(
        &mut self,
        _event: &EventParseRules,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::consume_token_invalid
        todo!(
            "TODO: port action `consume_token_invalid` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn consume_token_invalid_from_in_rule_expression_after_term_decision(
        &mut self,
        _event: &EventParseRules,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::consume_token_invalid
        todo!(
            "TODO: port action `consume_token_invalid` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn consume_token_invalid_from_in_rule_expression_need_term_decision(
        &mut self,
        _event: &EventParseRules,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::consume_token_invalid
        todo!(
            "TODO: port action `consume_token_invalid` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn consume_token_invalid_from_nonterm_parser_model(
        &mut self,
        _event: &EventParseRules,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::consume_token_invalid
        todo!(
            "TODO: port action `consume_token_invalid` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn consume_token_invalid_from_quantifier_braced_exact_exec(
        &mut self,
        _event: &EventParseRules,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::consume_token_invalid
        todo!(
            "TODO: port action `consume_token_invalid` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn consume_token_invalid_from_quantifier_braced_open_exec(
        &mut self,
        _event: &EventParseRules,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::consume_token_invalid
        todo!(
            "TODO: port action `consume_token_invalid` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn consume_token_invalid_from_quantifier_braced_range_exec(
        &mut self,
        _event: &EventParseRules,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::consume_token_invalid
        todo!(
            "TODO: port action `consume_token_invalid` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn consume_token_invalid_from_quantifier_decision(
        &mut self,
        _event: &EventParseRules,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::consume_token_invalid
        todo!(
            "TODO: port action `consume_token_invalid` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn consume_token_invalid_from_quantifier_plus_exec(
        &mut self,
        _event: &EventParseRules,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::consume_token_invalid
        todo!(
            "TODO: port action `consume_token_invalid` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn consume_token_invalid_from_quantifier_question_exec(
        &mut self,
        _event: &EventParseRules,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::consume_token_invalid
        todo!(
            "TODO: port action `consume_token_invalid` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn consume_token_invalid_from_quantifier_star_exec(
        &mut self,
        _event: &EventParseRules,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::consume_token_invalid
        todo!(
            "TODO: port action `consume_token_invalid` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn consume_token_invalid_from_rule_reference_decision(
        &mut self,
        _event: &EventParseRules,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::consume_token_invalid
        todo!(
            "TODO: port action `consume_token_invalid` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn consume_token_invalid_from_rule_reference_negated_exec(
        &mut self,
        _event: &EventParseRules,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::consume_token_invalid
        todo!(
            "TODO: port action `consume_token_invalid` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn consume_token_invalid_from_rule_reference_plain_exec(
        &mut self,
        _event: &EventParseRules,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::consume_token_invalid
        todo!(
            "TODO: port action `consume_token_invalid` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn consume_token_invalid_from_term_parser_model(
        &mut self,
        _event: &EventParseRules,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::consume_token_invalid
        todo!(
            "TODO: port action `consume_token_invalid` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn consume_token_literal_from_term_parser_model(
        &mut self,
        _event: &EventParseRules,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::consume_token_literal
        todo!(
            "TODO: port action `consume_token_literal` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn consume_token_open_group_from_term_parser_model(
        &mut self,
        _event: &EventParseRules,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::consume_token_open_group
        todo!(
            "TODO: port action `consume_token_open_group` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn consume_token_quantifier_braced_exact(
        &mut self,
        _event: &EventParseRules,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::consume_token_quantifier_braced_exact
        todo!(
            "TODO: port action `consume_token_quantifier_braced_exact` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn consume_token_quantifier_braced_open(&mut self, _event: &EventParseRules) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::consume_token_quantifier_braced_open
        todo!(
            "TODO: port action `consume_token_quantifier_braced_open` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn consume_token_quantifier_braced_range(
        &mut self,
        _event: &EventParseRules,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::consume_token_quantifier_braced_range
        todo!(
            "TODO: port action `consume_token_quantifier_braced_range` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn consume_token_quantifier_plus(&mut self, _event: &EventParseRules) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::consume_token_quantifier_plus
        todo!(
            "TODO: port action `consume_token_quantifier_plus` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn consume_token_quantifier_question(&mut self, _event: &EventParseRules) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::consume_token_quantifier_question
        todo!(
            "TODO: port action `consume_token_quantifier_question` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn consume_token_quantifier_star(&mut self, _event: &EventParseRules) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::consume_token_quantifier_star
        todo!(
            "TODO: port action `consume_token_quantifier_star` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn consume_token_rule_reference_negated(&mut self, _event: &EventParseRules) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::consume_token_rule_reference_negated
        todo!(
            "TODO: port action `consume_token_rule_reference_negated` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn consume_token_rule_reference_plain(&mut self, _event: &EventParseRules) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::consume_token_rule_reference_plain
        todo!(
            "TODO: port action `consume_token_rule_reference_plain` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn definition_done(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::definition_done
        todo!(
            "TODO: port guard `definition_done` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn definition_failed(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::definition_failed
        todo!(
            "TODO: port guard `definition_failed` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn dispatch_done(&mut self, _event: &EventParseRules) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::dispatch_done
        todo!(
            "TODO: port action `dispatch_done` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn dispatch_error_from_parse_decision(&mut self, _event: &EventParseRules) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::dispatch_error
        todo!(
            "TODO: port action `dispatch_error` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn eof_can_finalize_active_rule(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::eof_can_finalize_active_rule
        todo!(
            "TODO: port guard `eof_can_finalize_active_rule` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn eof_can_finalize_symbols(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::eof_can_finalize_symbols
        todo!(
            "TODO: port guard `eof_can_finalize_symbols` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn eof_cannot_finalize_active_rule(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::eof_cannot_finalize_active_rule
        todo!(
            "TODO: port guard `eof_cannot_finalize_active_rule` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn eof_cannot_finalize_symbols(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::eof_cannot_finalize_symbols
        todo!(
            "TODO: port guard `eof_cannot_finalize_symbols` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn expression_done_identifier(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::expression_done_identifier
        todo!(
            "TODO: port guard `expression_done_identifier` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn expression_done_non_identifier(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::expression_done_non_identifier
        todo!(
            "TODO: port guard `expression_done_non_identifier` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn expression_failed(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::expression_failed
        todo!(
            "TODO: port guard `expression_failed` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn fail_eof_in_expect_definition(&mut self, _event: &EventParseRules) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::fail_eof_in_expect_definition
        todo!(
            "TODO: port action `fail_eof_in_expect_definition` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn finalize_active_rule_on_eof_from_in_rule_expression_after_term_decision(
        &mut self,
        _event: &EventParseRules,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::finalize_active_rule_on_eof
        todo!(
            "TODO: port action `finalize_active_rule_on_eof` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn finalize_active_rule_on_eof_from_term_parser_model(
        &mut self,
        _event: &EventParseRules,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::finalize_active_rule_on_eof
        todo!(
            "TODO: port action `finalize_active_rule_on_eof` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn invalid_parse_with_dispatchable_grammar(
        &self,
        _event: &EventParseRules,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::invalid_parse_with_dispatchable_grammar
        todo!(
            "TODO: port guard `invalid_parse_with_dispatchable_grammar` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn invalid_parse_with_grammar_only(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::invalid_parse_with_grammar_only
        todo!(
            "TODO: port guard `invalid_parse_with_grammar_only` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn invalid_parse_without_grammar(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::invalid_parse_without_grammar
        todo!(
            "TODO: port guard `invalid_parse_without_grammar` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn lexer_at_eof(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::lexer_at_eof
        todo!("TODO: port guard `lexer_at_eof` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp")
    }
    fn lexer_failed(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::lexer_failed
        todo!("TODO: port guard `lexer_failed` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp")
    }
    fn lexer_has_token(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::lexer_has_token
        todo!(
            "TODO: port guard `lexer_has_token` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn nonterm_definition_done(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::nonterm_definition_done
        todo!(
            "TODO: port guard `nonterm_definition_done` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn nonterm_failed(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::nonterm_failed
        todo!(
            "TODO: port guard `nonterm_failed` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn nonterm_reference_done(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::nonterm_reference_done
        todo!(
            "TODO: port guard `nonterm_reference_done` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn on_unexpected_from_eof_symbols_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_expect_definition(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_expect_definition_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_expect_rule_name(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_expect_rule_name_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_in_rule_expression_after_term(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_in_rule_expression_after_term_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_in_rule_expression_need_term(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_in_rule_expression_need_term_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_parse_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_quantifier_braced_exact_exec(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_quantifier_braced_open_exec(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_quantifier_braced_range_exec(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_quantifier_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_quantifier_plus_exec(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_quantifier_question_exec(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_quantifier_star_exec(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_ready(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_rule_reference_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_rule_reference_negated_exec(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_rule_reference_plain_exec(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn parse_error_internal_error(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::parse_error_internal_error
        todo!(
            "TODO: port guard `parse_error_internal_error` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn parse_error_invalid_request(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::parse_error_invalid_request
        todo!(
            "TODO: port guard `parse_error_invalid_request` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn parse_error_none(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::parse_error_none
        todo!(
            "TODO: port guard `parse_error_none` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn parse_error_parse_failed(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::parse_error_parse_failed
        todo!(
            "TODO: port guard `parse_error_parse_failed` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn parse_error_unknown(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::parse_error_unknown
        todo!(
            "TODO: port guard `parse_error_unknown` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn parse_error_untracked(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::parse_error_untracked
        todo!(
            "TODO: port guard `parse_error_untracked` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn quantifier_braced_exact_shape(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::quantifier_braced_exact_shape
        todo!(
            "TODO: port guard `quantifier_braced_exact_shape` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn quantifier_braced_open_shape(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::quantifier_braced_open_shape
        todo!(
            "TODO: port guard `quantifier_braced_open_shape` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn quantifier_braced_range_shape(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::quantifier_braced_range_shape
        todo!(
            "TODO: port guard `quantifier_braced_range_shape` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn quantifier_token_plus(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::quantifier_token_plus
        todo!(
            "TODO: port guard `quantifier_token_plus` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn quantifier_token_question(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::quantifier_token_question
        todo!(
            "TODO: port guard `quantifier_token_question` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn quantifier_token_star(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::quantifier_token_star
        todo!(
            "TODO: port guard `quantifier_token_star` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn reject_invalid_parse_with_dispatch(&mut self, _event: &EventParseRules) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::reject_invalid_parse_with_dispatch
        todo!(
            "TODO: port action `reject_invalid_parse_with_dispatch` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn reject_invalid_parse_with_grammar_only(
        &mut self,
        _event: &EventParseRules,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::reject_invalid_parse_with_grammar_only
        todo!(
            "TODO: port action `reject_invalid_parse_with_grammar_only` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn reject_invalid_parse_without_grammar(&mut self, _event: &EventParseRules) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::reject_invalid_parse_without_grammar
        todo!(
            "TODO: port action `reject_invalid_parse_without_grammar` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn request_next_token_from_expect_definition(
        &mut self,
        _event: &EventParseRules,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::request_next_token
        todo!(
            "TODO: port action `request_next_token` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn request_next_token_from_expect_rule_name(
        &mut self,
        _event: &EventParseRules,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::request_next_token
        todo!(
            "TODO: port action `request_next_token` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn request_next_token_from_in_rule_expression_after_term(
        &mut self,
        _event: &EventParseRules,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::request_next_token
        todo!(
            "TODO: port action `request_next_token` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn request_next_token_from_in_rule_expression_need_term(
        &mut self,
        _event: &EventParseRules,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::request_next_token
        todo!(
            "TODO: port action `request_next_token` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn rule_reference_negated_envelope_valid(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::rule_reference_negated_envelope_valid
        todo!(
            "TODO: port guard `rule_reference_negated_envelope_valid` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn rule_reference_plain_envelope_valid(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::rule_reference_plain_envelope_valid
        todo!(
            "TODO: port guard `rule_reference_plain_envelope_valid` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn set_nonterm_mode_definition(&mut self, _event: &EventParseRules) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::set_nonterm_mode_definition
        todo!(
            "TODO: port action `set_nonterm_mode_definition` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn set_nonterm_mode_reference(&mut self, _event: &EventParseRules) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::set_nonterm_mode_reference
        todo!(
            "TODO: port action `set_nonterm_mode_reference` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn set_term_origin_after_term(&mut self, _event: &EventParseRules) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::set_term_origin_after_term
        todo!(
            "TODO: port action `set_term_origin_after_term` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn set_term_origin_need_term(&mut self, _event: &EventParseRules) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp::set_term_origin_need_term
        todo!(
            "TODO: port action `set_term_origin_need_term` from emel.cpp/src/emel/gbnf/rule_parser/actions.hpp"
        )
    }
    fn term_after_alternation_valid(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::term_after_alternation_valid
        todo!(
            "TODO: port guard `term_after_alternation_valid` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn term_after_character_class_valid(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::term_after_character_class_valid
        todo!(
            "TODO: port guard `term_after_character_class_valid` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn term_after_close_group_valid(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::term_after_close_group_valid
        todo!(
            "TODO: port guard `term_after_close_group_valid` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn term_after_dot_valid(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::term_after_dot_valid
        todo!(
            "TODO: port guard `term_after_dot_valid` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn term_after_literal_valid(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::term_after_literal_valid
        todo!(
            "TODO: port guard `term_after_literal_valid` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn term_after_newline_with_group_depth_nonzero(
        &self,
        _event: &EventParseRules,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::term_after_newline_with_group_depth_nonzero
        todo!(
            "TODO: port guard `term_after_newline_with_group_depth_nonzero` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn term_after_newline_with_group_depth_zero_valid(
        &self,
        _event: &EventParseRules,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::term_after_newline_with_group_depth_zero_valid
        todo!(
            "TODO: port guard `term_after_newline_with_group_depth_zero_valid` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn term_after_open_group_valid(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::term_after_open_group_valid
        todo!(
            "TODO: port guard `term_after_open_group_valid` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn term_after_quantifier_candidate(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::term_after_quantifier_candidate
        todo!(
            "TODO: port guard `term_after_quantifier_candidate` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn term_after_rule_reference_candidate(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::term_after_rule_reference_candidate
        todo!(
            "TODO: port guard `term_after_rule_reference_candidate` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn term_failed(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::term_failed
        todo!("TODO: port guard `term_failed` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp")
    }
    fn term_from_after_term(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::term_from_after_term
        todo!(
            "TODO: port guard `term_from_after_term` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn term_from_need_term(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::term_from_need_term
        todo!(
            "TODO: port guard `term_from_need_term` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn term_need_character_class_valid(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::term_need_character_class_valid
        todo!(
            "TODO: port guard `term_need_character_class_valid` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn term_need_dot_valid(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::term_need_dot_valid
        todo!(
            "TODO: port guard `term_need_dot_valid` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn term_need_literal_valid(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::term_need_literal_valid
        todo!(
            "TODO: port guard `term_need_literal_valid` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn term_need_newline_with_group_depth_nonzero(
        &self,
        _event: &EventParseRules,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::term_need_newline_with_group_depth_nonzero
        todo!(
            "TODO: port guard `term_need_newline_with_group_depth_nonzero` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn term_need_open_group_valid(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::term_need_open_group_valid
        todo!(
            "TODO: port guard `term_need_open_group_valid` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn term_need_rule_reference_candidate(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::term_need_rule_reference_candidate
        todo!(
            "TODO: port guard `term_need_rule_reference_candidate` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn token_identifier(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::token_identifier
        todo!(
            "TODO: port guard `token_identifier` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp"
        )
    }
    fn token_newline(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::token_newline
        todo!("TODO: port guard `token_newline` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp")
    }
    fn valid_parse(&self, _event: &EventParseRules) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp::valid_parse
        todo!("TODO: port guard `valid_parse` from emel.cpp/src/emel/gbnf/rule_parser/guards.hpp")
    }
}
