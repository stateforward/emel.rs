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

// --- machine TextJinjaParserProgramParserStatementParser from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventParseRuntime;

sml! {
    TextJinjaParserProgramParserStatementParser {
        "statement_kind_decision"_s <= *"deciding"_s + completion<EventParseRuntime>,
        "statement_scan"_s <= "statement_kind_decision"_s + completion<EventParseRuntime> [statement_name_set] / begin_statement_scan_from_statement_kind_decision,
        "statement_scan"_s <= "statement_kind_decision"_s + completion<EventParseRuntime> [statement_name_if] / begin_statement_scan_from_statement_kind_decision,
        "statement_scan"_s <= "statement_kind_decision"_s + completion<EventParseRuntime> [statement_name_elif] / begin_statement_scan_from_statement_kind_decision,
        "statement_scan"_s <= "statement_kind_decision"_s + completion<EventParseRuntime> [statement_name_else] / begin_statement_scan_from_statement_kind_decision,
        "statement_scan"_s <= "statement_kind_decision"_s + completion<EventParseRuntime> [statement_name_endif] / begin_statement_scan_from_statement_kind_decision,
        "statement_scan"_s <= "statement_kind_decision"_s + completion<EventParseRuntime> [statement_name_for] / begin_statement_scan_from_statement_kind_decision,
        "statement_scan"_s <= "statement_kind_decision"_s + completion<EventParseRuntime> [statement_name_endfor] / begin_statement_scan_from_statement_kind_decision,
        "statement_scan"_s <= "statement_kind_decision"_s + completion<EventParseRuntime> [statement_name_macro] / begin_statement_scan_from_statement_kind_decision,
        "statement_scan"_s <= "statement_kind_decision"_s + completion<EventParseRuntime> [statement_name_endmacro] / begin_statement_scan_from_statement_kind_decision,
        "statement_scan"_s <= "statement_kind_decision"_s + completion<EventParseRuntime> [statement_name_call] / begin_statement_scan_from_statement_kind_decision,
        "statement_scan"_s <= "statement_kind_decision"_s + completion<EventParseRuntime> [statement_name_endcall] / begin_statement_scan_from_statement_kind_decision,
        "statement_scan"_s <= "statement_kind_decision"_s + completion<EventParseRuntime> [statement_name_filter] / begin_statement_scan_from_statement_kind_decision,
        "statement_scan"_s <= "statement_kind_decision"_s + completion<EventParseRuntime> [statement_name_endfilter] / begin_statement_scan_from_statement_kind_decision,
        "statement_scan"_s <= "statement_kind_decision"_s + completion<EventParseRuntime> [statement_name_break] / begin_statement_scan_from_statement_kind_decision,
        "statement_scan"_s <= "statement_kind_decision"_s + completion<EventParseRuntime> [statement_name_continue] / begin_statement_scan_from_statement_kind_decision,
        "statement_scan"_s <= "statement_kind_decision"_s + completion<EventParseRuntime> [statement_name_generation] / begin_statement_scan_from_statement_kind_decision,
        "statement_scan"_s <= "statement_kind_decision"_s + completion<EventParseRuntime> [statement_name_endgeneration] / begin_statement_scan_from_statement_kind_decision,
        "statement_scan"_s <= "statement_kind_decision"_s + completion<EventParseRuntime> [statement_name_endset] / begin_statement_scan_from_statement_kind_decision,
        "parse_failed"_s <= "statement_kind_decision"_s + completion<EventParseRuntime> [statement_identifier_missing] / fail_statement_open_token,
        "parse_failed"_s <= "statement_kind_decision"_s + completion<EventParseRuntime> [statement_name_unknown] / fail_statement_name_token,
        "parsed"_s <= "statement_scan"_s + completion<EventParseRuntime> [statement_scan_at_close] / consume_statement_close_and_emit,
        "statement_scan"_s <= "statement_scan"_s + completion<EventParseRuntime> [statement_scan_continue] / consume_statement_token,
        "parse_failed"_s <= "statement_scan"_s + completion<EventParseRuntime> [statement_scan_eof] / fail_statement_start_token,
        "unexpected_event"_s <= "deciding"_s + unexpected_event<_> / on_unexpected_from_deciding,
        "unexpected_event"_s <= "statement_kind_decision"_s + unexpected_event<_> / on_unexpected_from_statement_kind_decision,
        "unexpected_event"_s <= "statement_scan"_s + unexpected_event<_> / on_unexpected_from_statement_scan,
        "unexpected_event"_s <= "parsed"_s + unexpected_event<_> / on_unexpected_from_parsed,
        "unexpected_event"_s <= "parse_failed"_s + unexpected_event<_> / on_unexpected_from_parse_failed,
        "unexpected_event"_s <= "unexpected_event"_s + unexpected_event<_> / on_unexpected_from_unexpected_event,
        "parsed"_s = X,
        "parse_failed"_s = X,
    }
}

/// Context for `TextJinjaParserProgramParserStatementParser` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct TextJinjaParserProgramParserStatementParserContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl TextJinjaParserProgramParserStatementParserStateMachineContext
    for TextJinjaParserProgramParserStatementParserContext
{
    fn begin_statement_scan_from_statement_kind_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/actions.hpp::begin_statement_scan
        todo!(
            "TODO: port action `begin_statement_scan` from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/actions.hpp"
        )
    }
    fn consume_statement_close_and_emit(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/actions.hpp::consume_statement_close_and_emit
        todo!(
            "TODO: port action `consume_statement_close_and_emit` from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/actions.hpp"
        )
    }
    fn consume_statement_token(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/actions.hpp::consume_statement_token
        todo!(
            "TODO: port action `consume_statement_token` from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/actions.hpp"
        )
    }
    fn fail_statement_name_token(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/actions.hpp::fail_statement_name_token
        todo!(
            "TODO: port action `fail_statement_name_token` from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/actions.hpp"
        )
    }
    fn fail_statement_open_token(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/actions.hpp::fail_statement_open_token
        todo!(
            "TODO: port action `fail_statement_open_token` from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/actions.hpp"
        )
    }
    fn fail_statement_start_token(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/actions.hpp::fail_statement_start_token
        todo!(
            "TODO: port action `fail_statement_start_token` from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_deciding(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_parse_failed(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_parsed(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_statement_kind_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_statement_scan(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/actions.hpp"
        )
    }
    fn on_unexpected_from_unexpected_event(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/actions.hpp"
        )
    }
    fn statement_identifier_missing(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp::statement_identifier_missing
        todo!(
            "TODO: port guard `statement_identifier_missing` from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp"
        )
    }
    fn statement_name_break(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp::statement_name_break
        todo!(
            "TODO: port guard `statement_name_break` from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp"
        )
    }
    fn statement_name_call(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp::statement_name_call
        todo!(
            "TODO: port guard `statement_name_call` from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp"
        )
    }
    fn statement_name_continue(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp::statement_name_continue
        todo!(
            "TODO: port guard `statement_name_continue` from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp"
        )
    }
    fn statement_name_elif(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp::statement_name_elif
        todo!(
            "TODO: port guard `statement_name_elif` from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp"
        )
    }
    fn statement_name_else(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp::statement_name_else
        todo!(
            "TODO: port guard `statement_name_else` from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp"
        )
    }
    fn statement_name_endcall(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp::statement_name_endcall
        todo!(
            "TODO: port guard `statement_name_endcall` from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp"
        )
    }
    fn statement_name_endfilter(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp::statement_name_endfilter
        todo!(
            "TODO: port guard `statement_name_endfilter` from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp"
        )
    }
    fn statement_name_endfor(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp::statement_name_endfor
        todo!(
            "TODO: port guard `statement_name_endfor` from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp"
        )
    }
    fn statement_name_endgeneration(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp::statement_name_endgeneration
        todo!(
            "TODO: port guard `statement_name_endgeneration` from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp"
        )
    }
    fn statement_name_endif(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp::statement_name_endif
        todo!(
            "TODO: port guard `statement_name_endif` from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp"
        )
    }
    fn statement_name_endmacro(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp::statement_name_endmacro
        todo!(
            "TODO: port guard `statement_name_endmacro` from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp"
        )
    }
    fn statement_name_endset(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp::statement_name_endset
        todo!(
            "TODO: port guard `statement_name_endset` from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp"
        )
    }
    fn statement_name_filter(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp::statement_name_filter
        todo!(
            "TODO: port guard `statement_name_filter` from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp"
        )
    }
    fn statement_name_for(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp::statement_name_for
        todo!(
            "TODO: port guard `statement_name_for` from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp"
        )
    }
    fn statement_name_generation(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp::statement_name_generation
        todo!(
            "TODO: port guard `statement_name_generation` from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp"
        )
    }
    fn statement_name_if(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp::statement_name_if
        todo!(
            "TODO: port guard `statement_name_if` from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp"
        )
    }
    fn statement_name_macro(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp::statement_name_macro
        todo!(
            "TODO: port guard `statement_name_macro` from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp"
        )
    }
    fn statement_name_set(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp::statement_name_set
        todo!(
            "TODO: port guard `statement_name_set` from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp"
        )
    }
    fn statement_name_unknown(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp::statement_name_unknown
        todo!(
            "TODO: port guard `statement_name_unknown` from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp"
        )
    }
    fn statement_scan_at_close(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp::statement_scan_at_close
        todo!(
            "TODO: port guard `statement_scan_at_close` from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp"
        )
    }
    fn statement_scan_continue(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp::statement_scan_continue
        todo!(
            "TODO: port guard `statement_scan_continue` from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp"
        )
    }
    fn statement_scan_eof(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp::statement_scan_eof
        todo!(
            "TODO: port guard `statement_scan_eof` from emel.cpp/src/emel/text/jinja/parser/program_parser/statement_parser/guards.hpp"
        )
    }
}
