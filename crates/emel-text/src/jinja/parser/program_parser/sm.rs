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

// --- machine TextJinjaParserProgramParser from emel.cpp/src/emel/text/jinja/parser/program_parser/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventParseRuntime;

sml! {
    TextJinjaParserProgramParser {
        "parse_begin"_s <= *"deciding"_s + completion<EventParseRuntime> / start_program_parse,
        "dispatch_decision"_s <= "parse_begin"_s + completion<EventParseRuntime>,
        "parsed"_s <= "dispatch_decision"_s + completion<EventParseRuntime> [at_eof] / finish_parsed,
        "text_emit"_s <= "dispatch_decision"_s + completion<EventParseRuntime> [token_text],
        "comment_emit"_s <= "dispatch_decision"_s + completion<EventParseRuntime> [token_comment],
        "statement_parser_model"_s <= "dispatch_decision"_s + completion<EventParseRuntime> [token_open_statement],
        "expression_parser_model"_s <= "dispatch_decision"_s + completion<EventParseRuntime> [token_open_expression],
        "parse_failed"_s <= "dispatch_decision"_s + completion<EventParseRuntime> [token_unexpected] / fail_current_token,
        "dispatch_decision"_s <= "text_emit"_s + completion<EventParseRuntime> / consume_text,
        "dispatch_decision"_s <= "comment_emit"_s + completion<EventParseRuntime> / consume_comment,
        "statement_parse_result_decision"_s <= "statement_parser_model"_s + completion<EventParseRuntime>,
        "dispatch_decision"_s <= "statement_parse_result_decision"_s + completion<EventParseRuntime> [parse_error_none],
        "parse_failed"_s <= "statement_parse_result_decision"_s + completion<EventParseRuntime> [parse_error_invalid_request],
        "parse_failed"_s <= "statement_parse_result_decision"_s + completion<EventParseRuntime> [parse_error_parse_failed],
        "parse_failed"_s <= "statement_parse_result_decision"_s + completion<EventParseRuntime> [parse_error_internal_error],
        "parse_failed"_s <= "statement_parse_result_decision"_s + completion<EventParseRuntime> [parse_error_untracked],
        "parse_failed"_s <= "statement_parse_result_decision"_s + completion<EventParseRuntime> [parse_error_unknown],
        "expression_parse_result_decision"_s <= "expression_parser_model"_s + completion<EventParseRuntime>,
        "dispatch_decision"_s <= "expression_parse_result_decision"_s + completion<EventParseRuntime> [parse_error_none],
        "parse_failed"_s <= "expression_parse_result_decision"_s + completion<EventParseRuntime> [parse_error_invalid_request],
        "parse_failed"_s <= "expression_parse_result_decision"_s + completion<EventParseRuntime> [parse_error_parse_failed],
        "parse_failed"_s <= "expression_parse_result_decision"_s + completion<EventParseRuntime> [parse_error_internal_error],
        "parse_failed"_s <= "expression_parse_result_decision"_s + completion<EventParseRuntime> [parse_error_untracked],
        "parse_failed"_s <= "expression_parse_result_decision"_s + completion<EventParseRuntime> [parse_error_unknown],
        "unexpected_event"_s <= "deciding"_s + unexpected_event<_> / on_unexpected_from_deciding,
        "unexpected_event"_s <= "parse_begin"_s + unexpected_event<_> / on_unexpected_from_parse_begin,
        "unexpected_event"_s <= "dispatch_decision"_s + unexpected_event<_> / on_unexpected_from_dispatch_decision,
        "unexpected_event"_s <= "text_emit"_s + unexpected_event<_> / on_unexpected_from_text_emit,
        "unexpected_event"_s <= "comment_emit"_s + unexpected_event<_> / on_unexpected_from_comment_emit,
        "unexpected_event"_s <= "statement_parse_result_decision"_s + unexpected_event<_> / on_unexpected_from_statement_parse_result_decision,
        "unexpected_event"_s <= "expression_parse_result_decision"_s + unexpected_event<_> / on_unexpected_from_expression_parse_result_decision,
        "unexpected_event"_s <= "parsed"_s + unexpected_event<_> / on_unexpected_from_parsed,
        "unexpected_event"_s <= "parse_failed"_s + unexpected_event<_> / on_unexpected_from_parse_failed,
        "unexpected_event"_s <= "unexpected_event"_s + unexpected_event<_> / on_unexpected_from_unexpected_event,
        "parsed"_s = X,
        "parse_failed"_s = X,
    }
}

/// Context for `TextJinjaParserProgramParser` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct TextJinjaParserProgramParserContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl TextJinjaParserProgramParserStateMachineContext for TextJinjaParserProgramParserContext {
    fn at_eof(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/guards.hpp::at_eof
        todo!(
            "TODO: port guard `at_eof` from emel.cpp/src/emel/text/jinja/parser/program_parser/guards.hpp"
        )
    }
    fn consume_comment(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/actions.hpp::consume_comment
        todo!(
            "TODO: port action `consume_comment` from emel.cpp/src/emel/text/jinja/parser/program_parser/actions.hpp"
        )
    }
    fn consume_text(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/actions.hpp::consume_text
        todo!(
            "TODO: port action `consume_text` from emel.cpp/src/emel/text/jinja/parser/program_parser/actions.hpp"
        )
    }
    fn fail_current_token(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/actions.hpp::fail_current_token
        todo!(
            "TODO: port action `fail_current_token` from emel.cpp/src/emel/text/jinja/parser/program_parser/actions.hpp"
        )
    }
    fn finish_parsed(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/actions.hpp::finish_parsed
        todo!(
            "TODO: port action `finish_parsed` from emel.cpp/src/emel/text/jinja/parser/program_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_comment_emit(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/parser/program_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_deciding(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/parser/program_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_dispatch_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/parser/program_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_expression_parse_result_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/parser/program_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_parse_begin(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/parser/program_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_parse_failed(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/parser/program_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_parsed(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/parser/program_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_statement_parse_result_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/parser/program_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_text_emit(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/parser/program_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_unexpected_event(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/parser/program_parser/actions.hpp"
        )
    }
    fn parse_error_internal_error(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/guards.hpp::parse_error_internal_error
        todo!(
            "TODO: port guard `parse_error_internal_error` from emel.cpp/src/emel/text/jinja/parser/program_parser/guards.hpp"
        )
    }
    fn parse_error_invalid_request(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/guards.hpp::parse_error_invalid_request
        todo!(
            "TODO: port guard `parse_error_invalid_request` from emel.cpp/src/emel/text/jinja/parser/program_parser/guards.hpp"
        )
    }
    fn parse_error_none(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/guards.hpp::parse_error_none
        todo!(
            "TODO: port guard `parse_error_none` from emel.cpp/src/emel/text/jinja/parser/program_parser/guards.hpp"
        )
    }
    fn parse_error_parse_failed(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/guards.hpp::parse_error_parse_failed
        todo!(
            "TODO: port guard `parse_error_parse_failed` from emel.cpp/src/emel/text/jinja/parser/program_parser/guards.hpp"
        )
    }
    fn parse_error_unknown(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/guards.hpp::parse_error_unknown
        todo!(
            "TODO: port guard `parse_error_unknown` from emel.cpp/src/emel/text/jinja/parser/program_parser/guards.hpp"
        )
    }
    fn parse_error_untracked(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/guards.hpp::parse_error_untracked
        todo!(
            "TODO: port guard `parse_error_untracked` from emel.cpp/src/emel/text/jinja/parser/program_parser/guards.hpp"
        )
    }
    fn start_program_parse(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/actions.hpp::start_program_parse
        todo!(
            "TODO: port action `start_program_parse` from emel.cpp/src/emel/text/jinja/parser/program_parser/actions.hpp"
        )
    }
    fn token_comment(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/guards.hpp::token_comment
        todo!(
            "TODO: port guard `token_comment` from emel.cpp/src/emel/text/jinja/parser/program_parser/guards.hpp"
        )
    }
    fn token_open_expression(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/guards.hpp::token_open_expression
        todo!(
            "TODO: port guard `token_open_expression` from emel.cpp/src/emel/text/jinja/parser/program_parser/guards.hpp"
        )
    }
    fn token_open_statement(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/guards.hpp::token_open_statement
        todo!(
            "TODO: port guard `token_open_statement` from emel.cpp/src/emel/text/jinja/parser/program_parser/guards.hpp"
        )
    }
    fn token_text(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/guards.hpp::token_text
        todo!(
            "TODO: port guard `token_text` from emel.cpp/src/emel/text/jinja/parser/program_parser/guards.hpp"
        )
    }
    fn token_unexpected(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/guards.hpp::token_unexpected
        todo!(
            "TODO: port guard `token_unexpected` from emel.cpp/src/emel/text/jinja/parser/program_parser/guards.hpp"
        )
    }
}
