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

// --- machine TextJinjaParserClassifierParser from emel.cpp/src/emel/text/jinja/parser/classifier_parser/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventParseRuntime;

sml! {
    TextJinjaParserClassifierParser {
        "statement_decision"_s <= *"deciding"_s + completion<EventParseRuntime> / begin_classification,
        "classification_result_decision"_s <= "statement_decision"_s + completion<EventParseRuntime> [no_tokens] / set_statement_unknown_from_statement_decision,
        "classification_result_decision"_s <= "statement_decision"_s + completion<EventParseRuntime> [token_text] / set_statement_text,
        "classification_result_decision"_s <= "statement_decision"_s + completion<EventParseRuntime> [token_comment] / set_statement_comment,
        "expression_decision"_s <= "statement_decision"_s + completion<EventParseRuntime> [token_open_expression] / set_statement_expression,
        "classification_result_decision"_s <= "statement_decision"_s + completion<EventParseRuntime> [token_open_statement] / set_statement_statement,
        "classification_result_decision"_s <= "statement_decision"_s + completion<EventParseRuntime> / set_statement_unknown_from_statement_decision,
        "classification_result_decision"_s <= "expression_decision"_s + completion<EventParseRuntime> [expr_no_token] / set_expression_unknown_from_expression_decision,
        "classification_result_decision"_s <= "expression_decision"_s + completion<EventParseRuntime> [expr_token_literal] / set_expression_literal,
        "classification_result_decision"_s <= "expression_decision"_s + completion<EventParseRuntime> [expr_token_identifier] / set_expression_identifier,
        "classification_result_decision"_s <= "expression_decision"_s + completion<EventParseRuntime> [expr_token_unary] / set_expression_unary,
        "classification_result_decision"_s <= "expression_decision"_s + completion<EventParseRuntime> [expr_token_compound] / set_expression_compound,
        "classification_result_decision"_s <= "expression_decision"_s + completion<EventParseRuntime> / set_expression_unknown_from_expression_decision,
        "done"_s <= "classification_result_decision"_s + completion<EventParseRuntime> [parse_error_none],
        "errored"_s <= "classification_result_decision"_s + completion<EventParseRuntime> [parse_error_invalid_request],
        "errored"_s <= "classification_result_decision"_s + completion<EventParseRuntime> [parse_error_parse_failed],
        "errored"_s <= "classification_result_decision"_s + completion<EventParseRuntime> [parse_error_internal_error],
        "errored"_s <= "classification_result_decision"_s + completion<EventParseRuntime> [parse_error_untracked],
        "errored"_s <= "classification_result_decision"_s + completion<EventParseRuntime>,
        "unexpected_event"_s <= "deciding"_s + unexpected_event<_> / on_unexpected_from_deciding,
        "unexpected_event"_s <= "statement_decision"_s + unexpected_event<_> / on_unexpected_from_statement_decision,
        "unexpected_event"_s <= "expression_decision"_s + unexpected_event<_> / on_unexpected_from_expression_decision,
        "unexpected_event"_s <= "classification_result_decision"_s + unexpected_event<_> / on_unexpected_from_classification_result_decision,
        "unexpected_event"_s <= "done"_s + unexpected_event<_> / on_unexpected_from_done,
        "unexpected_event"_s <= "errored"_s + unexpected_event<_> / on_unexpected_from_errored,
        "unexpected_event"_s <= "unexpected_event"_s + unexpected_event<_> / on_unexpected_from_unexpected_event,
        "done"_s = X,
        "errored"_s = X,
    }
}

/// Context for `TextJinjaParserClassifierParser` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct TextJinjaParserClassifierParserContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl TextJinjaParserClassifierParserStateMachineContext for TextJinjaParserClassifierParserContext {
    fn begin_classification(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/classifier_parser/actions.hpp::begin_classification
        todo!(
            "TODO: port action `begin_classification` from emel.cpp/src/emel/text/jinja/parser/classifier_parser/actions.hpp"
        )
    }
    fn expr_no_token(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/classifier_parser/guards.hpp::expr_no_token
        todo!(
            "TODO: port guard `expr_no_token` from emel.cpp/src/emel/text/jinja/parser/classifier_parser/guards.hpp"
        )
    }
    fn expr_token_compound(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/classifier_parser/guards.hpp::expr_token_compound
        todo!(
            "TODO: port guard `expr_token_compound` from emel.cpp/src/emel/text/jinja/parser/classifier_parser/guards.hpp"
        )
    }
    fn expr_token_identifier(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/classifier_parser/guards.hpp::expr_token_identifier
        todo!(
            "TODO: port guard `expr_token_identifier` from emel.cpp/src/emel/text/jinja/parser/classifier_parser/guards.hpp"
        )
    }
    fn expr_token_literal(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/classifier_parser/guards.hpp::expr_token_literal
        todo!(
            "TODO: port guard `expr_token_literal` from emel.cpp/src/emel/text/jinja/parser/classifier_parser/guards.hpp"
        )
    }
    fn expr_token_unary(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/classifier_parser/guards.hpp::expr_token_unary
        todo!(
            "TODO: port guard `expr_token_unary` from emel.cpp/src/emel/text/jinja/parser/classifier_parser/guards.hpp"
        )
    }
    fn no_tokens(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/classifier_parser/guards.hpp::no_tokens
        todo!(
            "TODO: port guard `no_tokens` from emel.cpp/src/emel/text/jinja/parser/classifier_parser/guards.hpp"
        )
    }
    fn on_unexpected_from_classification_result_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/classifier_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/parser/classifier_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_deciding(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/classifier_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/parser/classifier_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_done(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/classifier_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/parser/classifier_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_errored(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/classifier_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/parser/classifier_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_expression_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/classifier_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/parser/classifier_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_statement_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/classifier_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/parser/classifier_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_unexpected_event(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/classifier_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/parser/classifier_parser/actions.hpp"
        )
    }
    fn parse_error_internal_error(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/classifier_parser/guards.hpp::parse_error_internal_error
        todo!(
            "TODO: port guard `parse_error_internal_error` from emel.cpp/src/emel/text/jinja/parser/classifier_parser/guards.hpp"
        )
    }
    fn parse_error_invalid_request(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/classifier_parser/guards.hpp::parse_error_invalid_request
        todo!(
            "TODO: port guard `parse_error_invalid_request` from emel.cpp/src/emel/text/jinja/parser/classifier_parser/guards.hpp"
        )
    }
    fn parse_error_none(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/classifier_parser/guards.hpp::parse_error_none
        todo!(
            "TODO: port guard `parse_error_none` from emel.cpp/src/emel/text/jinja/parser/classifier_parser/guards.hpp"
        )
    }
    fn parse_error_parse_failed(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/classifier_parser/guards.hpp::parse_error_parse_failed
        todo!(
            "TODO: port guard `parse_error_parse_failed` from emel.cpp/src/emel/text/jinja/parser/classifier_parser/guards.hpp"
        )
    }
    fn parse_error_untracked(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/classifier_parser/guards.hpp::parse_error_untracked
        todo!(
            "TODO: port guard `parse_error_untracked` from emel.cpp/src/emel/text/jinja/parser/classifier_parser/guards.hpp"
        )
    }
    fn set_expression_compound(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/classifier_parser/actions.hpp::set_expression_compound
        todo!(
            "TODO: port action `set_expression_compound` from emel.cpp/src/emel/text/jinja/parser/classifier_parser/actions.hpp"
        )
    }
    fn set_expression_identifier(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/classifier_parser/actions.hpp::set_expression_identifier
        todo!(
            "TODO: port action `set_expression_identifier` from emel.cpp/src/emel/text/jinja/parser/classifier_parser/actions.hpp"
        )
    }
    fn set_expression_literal(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/classifier_parser/actions.hpp::set_expression_literal
        todo!(
            "TODO: port action `set_expression_literal` from emel.cpp/src/emel/text/jinja/parser/classifier_parser/actions.hpp"
        )
    }
    fn set_expression_unary(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/classifier_parser/actions.hpp::set_expression_unary
        todo!(
            "TODO: port action `set_expression_unary` from emel.cpp/src/emel/text/jinja/parser/classifier_parser/actions.hpp"
        )
    }
    fn set_expression_unknown_from_expression_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/classifier_parser/actions.hpp::set_expression_unknown
        todo!(
            "TODO: port action `set_expression_unknown` from emel.cpp/src/emel/text/jinja/parser/classifier_parser/actions.hpp"
        )
    }
    fn set_statement_comment(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/classifier_parser/actions.hpp::set_statement_comment
        todo!(
            "TODO: port action `set_statement_comment` from emel.cpp/src/emel/text/jinja/parser/classifier_parser/actions.hpp"
        )
    }
    fn set_statement_expression(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/classifier_parser/actions.hpp::set_statement_expression
        todo!(
            "TODO: port action `set_statement_expression` from emel.cpp/src/emel/text/jinja/parser/classifier_parser/actions.hpp"
        )
    }
    fn set_statement_statement(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/classifier_parser/actions.hpp::set_statement_statement
        todo!(
            "TODO: port action `set_statement_statement` from emel.cpp/src/emel/text/jinja/parser/classifier_parser/actions.hpp"
        )
    }
    fn set_statement_text(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/classifier_parser/actions.hpp::set_statement_text
        todo!(
            "TODO: port action `set_statement_text` from emel.cpp/src/emel/text/jinja/parser/classifier_parser/actions.hpp"
        )
    }
    fn set_statement_unknown_from_statement_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/classifier_parser/actions.hpp::set_statement_unknown
        todo!(
            "TODO: port action `set_statement_unknown` from emel.cpp/src/emel/text/jinja/parser/classifier_parser/actions.hpp"
        )
    }
    fn token_comment(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/classifier_parser/guards.hpp::token_comment
        todo!(
            "TODO: port guard `token_comment` from emel.cpp/src/emel/text/jinja/parser/classifier_parser/guards.hpp"
        )
    }
    fn token_open_expression(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/classifier_parser/guards.hpp::token_open_expression
        todo!(
            "TODO: port guard `token_open_expression` from emel.cpp/src/emel/text/jinja/parser/classifier_parser/guards.hpp"
        )
    }
    fn token_open_statement(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/classifier_parser/guards.hpp::token_open_statement
        todo!(
            "TODO: port guard `token_open_statement` from emel.cpp/src/emel/text/jinja/parser/classifier_parser/guards.hpp"
        )
    }
    fn token_text(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/classifier_parser/guards.hpp::token_text
        todo!(
            "TODO: port guard `token_text` from emel.cpp/src/emel/text/jinja/parser/classifier_parser/guards.hpp"
        )
    }
}
