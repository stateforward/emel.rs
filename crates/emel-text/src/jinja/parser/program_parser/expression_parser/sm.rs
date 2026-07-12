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

// --- machine TextJinjaParserProgramParserExpressionParser from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventParseRuntime;

sml! {
    TextJinjaParserProgramParserExpressionParser {
        "expression_first_decision"_s <= *"deciding"_s + completion<EventParseRuntime> / begin_expression_parse,
        "parse_failed"_s <= "expression_first_decision"_s + completion<EventParseRuntime> [expr_scan_eof] / fail_expression_start_token_from_expression_first_decision,
        "parse_failed"_s <= "expression_first_decision"_s + completion<EventParseRuntime> [expr_first_is_close] / fail_expression_close_token,
        "parsed"_s <= "expression_first_decision"_s + completion<EventParseRuntime> [expr_first_identifier_followed_by_close] / consume_expression_identifier_and_close,
        "expression_scan"_s <= "expression_first_decision"_s + completion<EventParseRuntime> [expr_first_is_identifier] / consume_expression_identifier,
        "expression_scan"_s <= "expression_first_decision"_s + completion<EventParseRuntime> [expr_first_is_literal] / consume_expression_literal,
        "expression_scan"_s <= "expression_first_decision"_s + completion<EventParseRuntime> [expr_first_is_unary] / consume_expression_unary,
        "expression_scan"_s <= "expression_first_decision"_s + completion<EventParseRuntime> [expr_first_is_other_content] / consume_expression_compound,
        "expression_emit_decision"_s <= "expression_scan"_s + completion<EventParseRuntime> [expr_scan_at_close],
        "expression_scan"_s <= "expression_scan"_s + completion<EventParseRuntime> [expr_scan_continue] / consume_expression_token,
        "parse_failed"_s <= "expression_scan"_s + completion<EventParseRuntime> [expr_scan_eof] / fail_expression_start_token_from_expression_scan,
        "expression_close"_s <= "expression_emit_decision"_s + completion<EventParseRuntime> [expression_identifier] / emit_expression_identifier,
        "expression_close"_s <= "expression_emit_decision"_s + completion<EventParseRuntime> [expression_non_identifier] / emit_expression_generic,
        "parsed"_s <= "expression_close"_s + completion<EventParseRuntime> [expr_scan_at_close] / consume_expression_close,
        "parse_failed"_s <= "expression_close"_s + completion<EventParseRuntime> [expr_scan_eof] / fail_expression_start_token_from_expression_close,
        "unexpected_event"_s <= "deciding"_s + unexpected_event<_> / on_unexpected_from_deciding,
        "unexpected_event"_s <= "expression_first_decision"_s + unexpected_event<_> / on_unexpected_from_expression_first_decision,
        "unexpected_event"_s <= "expression_scan"_s + unexpected_event<_> / on_unexpected_from_expression_scan,
        "unexpected_event"_s <= "expression_emit_decision"_s + unexpected_event<_> / on_unexpected_from_expression_emit_decision,
        "unexpected_event"_s <= "expression_close"_s + unexpected_event<_> / on_unexpected_from_expression_close,
        "unexpected_event"_s <= "parsed"_s + unexpected_event<_> / on_unexpected_from_parsed,
        "unexpected_event"_s <= "parse_failed"_s + unexpected_event<_> / on_unexpected_from_parse_failed,
        "unexpected_event"_s <= "unexpected_event"_s + unexpected_event<_> / on_unexpected_from_unexpected_event,
        "parsed"_s = X,
        "parse_failed"_s = X,
    }
}

/// Context for `TextJinjaParserProgramParserExpressionParser` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct TextJinjaParserProgramParserExpressionParserContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl TextJinjaParserProgramParserExpressionParserStateMachineContext
    for TextJinjaParserProgramParserExpressionParserContext
{
    fn begin_expression_parse(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/actions.hpp::begin_expression_parse
        todo!(
            "TODO: port action `begin_expression_parse` from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/actions.hpp"
        )
    }
    fn consume_expression_close(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/actions.hpp::consume_expression_close
        todo!(
            "TODO: port action `consume_expression_close` from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/actions.hpp"
        )
    }
    fn consume_expression_compound(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/actions.hpp::consume_expression_compound
        todo!(
            "TODO: port action `consume_expression_compound` from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/actions.hpp"
        )
    }
    fn consume_expression_identifier(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/actions.hpp::consume_expression_identifier
        todo!(
            "TODO: port action `consume_expression_identifier` from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/actions.hpp"
        )
    }
    fn consume_expression_identifier_and_close(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/actions.hpp::consume_expression_identifier_and_close
        todo!(
            "TODO: port action `consume_expression_identifier_and_close` from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/actions.hpp"
        )
    }
    fn consume_expression_literal(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/actions.hpp::consume_expression_literal
        todo!(
            "TODO: port action `consume_expression_literal` from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/actions.hpp"
        )
    }
    fn consume_expression_token(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/actions.hpp::consume_expression_token
        todo!(
            "TODO: port action `consume_expression_token` from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/actions.hpp"
        )
    }
    fn consume_expression_unary(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/actions.hpp::consume_expression_unary
        todo!(
            "TODO: port action `consume_expression_unary` from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/actions.hpp"
        )
    }
    fn emit_expression_generic(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/actions.hpp::emit_expression_generic
        todo!(
            "TODO: port action `emit_expression_generic` from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/actions.hpp"
        )
    }
    fn emit_expression_identifier(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/actions.hpp::emit_expression_identifier
        todo!(
            "TODO: port action `emit_expression_identifier` from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/actions.hpp"
        )
    }
    fn expr_first_identifier_followed_by_close(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/guards.hpp::expr_first_identifier_followed_by_close
        todo!(
            "TODO: port guard `expr_first_identifier_followed_by_close` from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/guards.hpp"
        )
    }
    fn expr_first_is_close(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/guards.hpp::expr_first_is_close
        todo!(
            "TODO: port guard `expr_first_is_close` from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/guards.hpp"
        )
    }
    fn expr_first_is_identifier(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/guards.hpp::expr_first_is_identifier
        todo!(
            "TODO: port guard `expr_first_is_identifier` from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/guards.hpp"
        )
    }
    fn expr_first_is_literal(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/guards.hpp::expr_first_is_literal
        todo!(
            "TODO: port guard `expr_first_is_literal` from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/guards.hpp"
        )
    }
    fn expr_first_is_other_content(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/guards.hpp::expr_first_is_other_content
        todo!(
            "TODO: port guard `expr_first_is_other_content` from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/guards.hpp"
        )
    }
    fn expr_first_is_unary(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/guards.hpp::expr_first_is_unary
        todo!(
            "TODO: port guard `expr_first_is_unary` from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/guards.hpp"
        )
    }
    fn expr_scan_at_close(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/guards.hpp::expr_scan_at_close
        todo!(
            "TODO: port guard `expr_scan_at_close` from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/guards.hpp"
        )
    }
    fn expr_scan_continue(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/guards.hpp::expr_scan_continue
        todo!(
            "TODO: port guard `expr_scan_continue` from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/guards.hpp"
        )
    }
    fn expr_scan_eof(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/guards.hpp::expr_scan_eof
        todo!(
            "TODO: port guard `expr_scan_eof` from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/guards.hpp"
        )
    }
    fn expression_identifier(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/guards.hpp::expression_identifier
        todo!(
            "TODO: port guard `expression_identifier` from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/guards.hpp"
        )
    }
    fn expression_non_identifier(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/guards.hpp::expression_non_identifier
        todo!(
            "TODO: port guard `expression_non_identifier` from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/guards.hpp"
        )
    }
    fn fail_expression_close_token(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/actions.hpp::fail_expression_close_token
        todo!(
            "TODO: port action `fail_expression_close_token` from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/actions.hpp"
        )
    }
    fn fail_expression_start_token_from_expression_close(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/actions.hpp::fail_expression_start_token
        todo!(
            "TODO: port action `fail_expression_start_token` from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/actions.hpp"
        )
    }
    fn fail_expression_start_token_from_expression_first_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/actions.hpp::fail_expression_start_token
        todo!(
            "TODO: port action `fail_expression_start_token` from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/actions.hpp"
        )
    }
    fn fail_expression_start_token_from_expression_scan(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/actions.hpp::fail_expression_start_token
        todo!(
            "TODO: port action `fail_expression_start_token` from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_deciding(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_expression_close(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_expression_emit_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_expression_first_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_expression_scan(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_parse_failed(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_parsed(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_unexpected_event(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/parser/program_parser/expression_parser/actions.hpp"
        )
    }
}
