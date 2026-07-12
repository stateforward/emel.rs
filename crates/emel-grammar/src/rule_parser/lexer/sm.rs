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

// --- machine GbnfRuleParserLexer from emel.cpp/src/emel/gbnf/rule_parser/lexer/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventScanNext;

sml! {
    GbnfRuleParserLexer {
        "initialized"_s <= *"initialized"_s + event<EventScanNext> [invalid_next] / reject_invalid_next_from_initialized,
        "initialized"_s <= "initialized"_s + event<EventScanNext> [invalid_cursor_position] / reject_invalid_cursor_from_initialized,
        "scan_ready"_s <= "initialized"_s + event<EventScanNext> [valid_cursor_position] / prepare_scan_from_initialized,
        "scanning"_s <= "scanning"_s + event<EventScanNext> [invalid_next] / reject_invalid_next_from_scanning,
        "scanning"_s <= "scanning"_s + event<EventScanNext> [invalid_cursor_position] / reject_invalid_cursor_from_scanning,
        "scan_ready"_s <= "scanning"_s + event<EventScanNext> [valid_cursor_position] / prepare_scan_from_scanning,
        "scanning"_s <= "scan_ready"_s + completion<EventScanNext> [at_eof] / emit_eof,
        "scanning"_s <= "scan_ready"_s + completion<EventScanNext> [layout_exhausted] / emit_layout_exhausted_unknown,
        "scanning"_s <= "scan_ready"_s + completion<EventScanNext> [starts_newline_crlf] / emit_newline_crlf_token,
        "scanning"_s <= "scan_ready"_s + completion<EventScanNext> [starts_newline_single] / emit_newline_single_token,
        "scanning"_s <= "scan_ready"_s + completion<EventScanNext> [starts_definition_operator] / emit_definition_operator,
        "scanning"_s <= "scan_ready"_s + completion<EventScanNext> [starts_alternation] / emit_alternation,
        "scanning"_s <= "scan_ready"_s + completion<EventScanNext> [starts_dot] / emit_dot,
        "scanning"_s <= "scan_ready"_s + completion<EventScanNext> [starts_open_group] / emit_open_group,
        "scanning"_s <= "scan_ready"_s + completion<EventScanNext> [starts_close_group] / emit_close_group,
        "scanning"_s <= "scan_ready"_s + completion<EventScanNext> [starts_quantifier] / emit_quantifier,
        "scanning"_s <= "scan_ready"_s + completion<EventScanNext> [starts_string_literal] / emit_string_literal,
        "scanning"_s <= "scan_ready"_s + completion<EventScanNext> [starts_character_class] / emit_character_class,
        "scanning"_s <= "scan_ready"_s + completion<EventScanNext> [starts_braced_quantifier] / emit_braced_quantifier,
        "scanning"_s <= "scan_ready"_s + completion<EventScanNext> [parsed_rule_reference_negated_valid] / emit_rule_reference_negated,
        "scanning"_s <= "scan_ready"_s + completion<EventScanNext> [parsed_rule_reference_negated_invalid] / emit_unknown_from_scan_ready,
        "scanning"_s <= "scan_ready"_s + completion<EventScanNext> [parsed_rule_reference_plain_valid] / emit_rule_reference_plain,
        "scanning"_s <= "scan_ready"_s + completion<EventScanNext> [parsed_rule_reference_plain_invalid] / emit_unknown_from_scan_ready,
        "scanning"_s <= "scan_ready"_s + completion<EventScanNext> [starts_identifier] / emit_identifier,
        "scanning"_s <= "scan_ready"_s + completion<EventScanNext> / emit_unknown_from_scan_ready,
        "initialized"_s <= "initialized"_s + unexpected_event<_> [unexpected_has_error_callback] / dispatch_unexpected_error_from_initialized,
        "initialized"_s <= "initialized"_s + unexpected_event<_> / ignore_unexpected_from_initialized,
        "scanning"_s <= "scanning"_s + unexpected_event<_> [unexpected_has_error_callback] / dispatch_unexpected_error_from_scanning,
        "scanning"_s <= "scanning"_s + unexpected_event<_> / ignore_unexpected_from_scanning,
        "scan_ready"_s <= "scan_ready"_s + unexpected_event<_> [unexpected_has_error_callback] / dispatch_unexpected_error_from_scan_ready,
        "scan_ready"_s <= "scan_ready"_s + unexpected_event<_> / ignore_unexpected_from_scan_ready,
    }
}

/// Context for `GbnfRuleParserLexer` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct GbnfRuleParserLexerContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl GbnfRuleParserLexerStateMachineContext for GbnfRuleParserLexerContext {
    fn at_eof(&self, _event: &EventScanNext) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/guards.hpp::at_eof
        todo!("TODO: port guard `at_eof` from emel.cpp/src/emel/gbnf/rule_parser/lexer/guards.hpp")
    }
    fn dispatch_unexpected_error_from_initialized(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp::dispatch_unexpected_error
        todo!(
            "TODO: port action `dispatch_unexpected_error` from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp"
        )
    }
    fn dispatch_unexpected_error_from_scan_ready(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp::dispatch_unexpected_error
        todo!(
            "TODO: port action `dispatch_unexpected_error` from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp"
        )
    }
    fn dispatch_unexpected_error_from_scanning(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp::dispatch_unexpected_error
        todo!(
            "TODO: port action `dispatch_unexpected_error` from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp"
        )
    }
    fn emit_alternation(&mut self, _event: &EventScanNext) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp::emit_alternation
        todo!(
            "TODO: port action `emit_alternation` from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp"
        )
    }
    fn emit_braced_quantifier(&mut self, _event: &EventScanNext) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp::emit_braced_quantifier
        todo!(
            "TODO: port action `emit_braced_quantifier` from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp"
        )
    }
    fn emit_character_class(&mut self, _event: &EventScanNext) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp::emit_character_class
        todo!(
            "TODO: port action `emit_character_class` from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp"
        )
    }
    fn emit_close_group(&mut self, _event: &EventScanNext) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp::emit_close_group
        todo!(
            "TODO: port action `emit_close_group` from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp"
        )
    }
    fn emit_definition_operator(&mut self, _event: &EventScanNext) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp::emit_definition_operator
        todo!(
            "TODO: port action `emit_definition_operator` from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp"
        )
    }
    fn emit_dot(&mut self, _event: &EventScanNext) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp::emit_dot
        todo!(
            "TODO: port action `emit_dot` from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp"
        )
    }
    fn emit_eof(&mut self, _event: &EventScanNext) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp::emit_eof
        todo!(
            "TODO: port action `emit_eof` from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp"
        )
    }
    fn emit_identifier(&mut self, _event: &EventScanNext) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp::emit_identifier
        todo!(
            "TODO: port action `emit_identifier` from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp"
        )
    }
    fn emit_layout_exhausted_unknown(&mut self, _event: &EventScanNext) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp::emit_layout_exhausted_unknown
        todo!(
            "TODO: port action `emit_layout_exhausted_unknown` from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp"
        )
    }
    fn emit_newline_crlf_token(&mut self, _event: &EventScanNext) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp::emit_newline_crlf_token
        todo!(
            "TODO: port action `emit_newline_crlf_token` from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp"
        )
    }
    fn emit_newline_single_token(&mut self, _event: &EventScanNext) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp::emit_newline_single_token
        todo!(
            "TODO: port action `emit_newline_single_token` from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp"
        )
    }
    fn emit_open_group(&mut self, _event: &EventScanNext) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp::emit_open_group
        todo!(
            "TODO: port action `emit_open_group` from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp"
        )
    }
    fn emit_quantifier(&mut self, _event: &EventScanNext) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp::emit_quantifier
        todo!(
            "TODO: port action `emit_quantifier` from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp"
        )
    }
    fn emit_rule_reference_negated(&mut self, _event: &EventScanNext) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp::emit_rule_reference_negated
        todo!(
            "TODO: port action `emit_rule_reference_negated` from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp"
        )
    }
    fn emit_rule_reference_plain(&mut self, _event: &EventScanNext) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp::emit_rule_reference_plain
        todo!(
            "TODO: port action `emit_rule_reference_plain` from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp"
        )
    }
    fn emit_string_literal(&mut self, _event: &EventScanNext) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp::emit_string_literal
        todo!(
            "TODO: port action `emit_string_literal` from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp"
        )
    }
    fn emit_unknown_from_scan_ready(&mut self, _event: &EventScanNext) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp::emit_unknown
        todo!(
            "TODO: port action `emit_unknown` from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp"
        )
    }
    fn ignore_unexpected_from_initialized(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp::ignore_unexpected
        todo!(
            "TODO: port action `ignore_unexpected` from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp"
        )
    }
    fn ignore_unexpected_from_scan_ready(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp::ignore_unexpected
        todo!(
            "TODO: port action `ignore_unexpected` from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp"
        )
    }
    fn ignore_unexpected_from_scanning(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp::ignore_unexpected
        todo!(
            "TODO: port action `ignore_unexpected` from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp"
        )
    }
    fn invalid_cursor_position(&self, _event: &EventScanNext) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/guards.hpp::invalid_cursor_position
        todo!(
            "TODO: port guard `invalid_cursor_position` from emel.cpp/src/emel/gbnf/rule_parser/lexer/guards.hpp"
        )
    }
    fn invalid_next(&self, _event: &EventScanNext) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/guards.hpp::invalid_next
        todo!(
            "TODO: port guard `invalid_next` from emel.cpp/src/emel/gbnf/rule_parser/lexer/guards.hpp"
        )
    }
    fn layout_exhausted(&self, _event: &EventScanNext) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/guards.hpp::layout_exhausted
        todo!(
            "TODO: port guard `layout_exhausted` from emel.cpp/src/emel/gbnf/rule_parser/lexer/guards.hpp"
        )
    }
    fn parsed_rule_reference_negated_invalid(&self, _event: &EventScanNext) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/guards.hpp::parsed_rule_reference_negated_invalid
        todo!(
            "TODO: port guard `parsed_rule_reference_negated_invalid` from emel.cpp/src/emel/gbnf/rule_parser/lexer/guards.hpp"
        )
    }
    fn parsed_rule_reference_negated_valid(&self, _event: &EventScanNext) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/guards.hpp::parsed_rule_reference_negated_valid
        todo!(
            "TODO: port guard `parsed_rule_reference_negated_valid` from emel.cpp/src/emel/gbnf/rule_parser/lexer/guards.hpp"
        )
    }
    fn parsed_rule_reference_plain_invalid(&self, _event: &EventScanNext) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/guards.hpp::parsed_rule_reference_plain_invalid
        todo!(
            "TODO: port guard `parsed_rule_reference_plain_invalid` from emel.cpp/src/emel/gbnf/rule_parser/lexer/guards.hpp"
        )
    }
    fn parsed_rule_reference_plain_valid(&self, _event: &EventScanNext) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/guards.hpp::parsed_rule_reference_plain_valid
        todo!(
            "TODO: port guard `parsed_rule_reference_plain_valid` from emel.cpp/src/emel/gbnf/rule_parser/lexer/guards.hpp"
        )
    }
    fn prepare_scan_from_initialized(&mut self, _event: &EventScanNext) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp::prepare_scan
        todo!(
            "TODO: port action `prepare_scan` from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp"
        )
    }
    fn prepare_scan_from_scanning(&mut self, _event: &EventScanNext) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp::prepare_scan
        todo!(
            "TODO: port action `prepare_scan` from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp"
        )
    }
    fn reject_invalid_cursor_from_initialized(&mut self, _event: &EventScanNext) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp::reject_invalid_cursor
        todo!(
            "TODO: port action `reject_invalid_cursor` from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp"
        )
    }
    fn reject_invalid_cursor_from_scanning(&mut self, _event: &EventScanNext) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp::reject_invalid_cursor
        todo!(
            "TODO: port action `reject_invalid_cursor` from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp"
        )
    }
    fn reject_invalid_next_from_initialized(&mut self, _event: &EventScanNext) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp::reject_invalid_next
        todo!(
            "TODO: port action `reject_invalid_next` from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp"
        )
    }
    fn reject_invalid_next_from_scanning(&mut self, _event: &EventScanNext) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp::reject_invalid_next
        todo!(
            "TODO: port action `reject_invalid_next` from emel.cpp/src/emel/gbnf/rule_parser/lexer/actions.hpp"
        )
    }
    fn starts_alternation(&self, _event: &EventScanNext) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/guards.hpp::starts_alternation
        todo!(
            "TODO: port guard `starts_alternation` from emel.cpp/src/emel/gbnf/rule_parser/lexer/guards.hpp"
        )
    }
    fn starts_braced_quantifier(&self, _event: &EventScanNext) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/guards.hpp::starts_braced_quantifier
        todo!(
            "TODO: port guard `starts_braced_quantifier` from emel.cpp/src/emel/gbnf/rule_parser/lexer/guards.hpp"
        )
    }
    fn starts_character_class(&self, _event: &EventScanNext) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/guards.hpp::starts_character_class
        todo!(
            "TODO: port guard `starts_character_class` from emel.cpp/src/emel/gbnf/rule_parser/lexer/guards.hpp"
        )
    }
    fn starts_close_group(&self, _event: &EventScanNext) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/guards.hpp::starts_close_group
        todo!(
            "TODO: port guard `starts_close_group` from emel.cpp/src/emel/gbnf/rule_parser/lexer/guards.hpp"
        )
    }
    fn starts_definition_operator(&self, _event: &EventScanNext) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/guards.hpp::starts_definition_operator
        todo!(
            "TODO: port guard `starts_definition_operator` from emel.cpp/src/emel/gbnf/rule_parser/lexer/guards.hpp"
        )
    }
    fn starts_dot(&self, _event: &EventScanNext) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/guards.hpp::starts_dot
        todo!(
            "TODO: port guard `starts_dot` from emel.cpp/src/emel/gbnf/rule_parser/lexer/guards.hpp"
        )
    }
    fn starts_identifier(&self, _event: &EventScanNext) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/guards.hpp::starts_identifier
        todo!(
            "TODO: port guard `starts_identifier` from emel.cpp/src/emel/gbnf/rule_parser/lexer/guards.hpp"
        )
    }
    fn starts_newline_crlf(&self, _event: &EventScanNext) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/guards.hpp::starts_newline_crlf
        todo!(
            "TODO: port guard `starts_newline_crlf` from emel.cpp/src/emel/gbnf/rule_parser/lexer/guards.hpp"
        )
    }
    fn starts_newline_single(&self, _event: &EventScanNext) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/guards.hpp::starts_newline_single
        todo!(
            "TODO: port guard `starts_newline_single` from emel.cpp/src/emel/gbnf/rule_parser/lexer/guards.hpp"
        )
    }
    fn starts_open_group(&self, _event: &EventScanNext) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/guards.hpp::starts_open_group
        todo!(
            "TODO: port guard `starts_open_group` from emel.cpp/src/emel/gbnf/rule_parser/lexer/guards.hpp"
        )
    }
    fn starts_quantifier(&self, _event: &EventScanNext) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/guards.hpp::starts_quantifier
        todo!(
            "TODO: port guard `starts_quantifier` from emel.cpp/src/emel/gbnf/rule_parser/lexer/guards.hpp"
        )
    }
    fn starts_string_literal(&self, _event: &EventScanNext) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/guards.hpp::starts_string_literal
        todo!(
            "TODO: port guard `starts_string_literal` from emel.cpp/src/emel/gbnf/rule_parser/lexer/guards.hpp"
        )
    }
    fn unexpected_has_error_callback(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/guards.hpp::unexpected_has_error_callback
        todo!(
            "TODO: port guard `unexpected_has_error_callback` from emel.cpp/src/emel/gbnf/rule_parser/lexer/guards.hpp"
        )
    }
    fn valid_cursor_position(&self, _event: &EventScanNext) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/gbnf/rule_parser/lexer/guards.hpp::valid_cursor_position
        todo!(
            "TODO: port guard `valid_cursor_position` from emel.cpp/src/emel/gbnf/rule_parser/lexer/guards.hpp"
        )
    }
}
