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

// --- machine TextJinjaParser from emel.cpp/src/emel/text/jinja/parser/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventParseRuntime;

sml! {
    TextJinjaParser {
        "request_decision"_s <= *"initialized"_s + event<EventParseRuntime> [valid_parse] / begin_parse_from_initialized,
        "parse_result_decision"_s <= "initialized"_s + event<EventParseRuntime> [invalid_parse_with_callbacks] / reject_invalid_parse_from_initialized,
        "errored"_s <= "initialized"_s + event<EventParseRuntime> [invalid_parse_without_callbacks] / reject_invalid_parse_from_initialized,
        "request_decision"_s <= "done"_s + event<EventParseRuntime> [valid_parse] / begin_parse_from_done,
        "parse_result_decision"_s <= "done"_s + event<EventParseRuntime> [invalid_parse_with_callbacks] / reject_invalid_parse_from_done,
        "errored"_s <= "done"_s + event<EventParseRuntime> [invalid_parse_without_callbacks] / reject_invalid_parse_from_done,
        "request_decision"_s <= "errored"_s + event<EventParseRuntime> [valid_parse] / begin_parse_from_errored,
        "parse_result_decision"_s <= "errored"_s + event<EventParseRuntime> [invalid_parse_with_callbacks] / reject_invalid_parse_from_errored,
        "errored"_s <= "errored"_s + event<EventParseRuntime> [invalid_parse_without_callbacks] / reject_invalid_parse_from_errored,
        "request_decision"_s <= "unexpected"_s + event<EventParseRuntime> [valid_parse] / begin_parse_from_unexpected,
        "parse_result_decision"_s <= "unexpected"_s + event<EventParseRuntime> [invalid_parse_with_callbacks] / reject_invalid_parse_from_unexpected,
        "errored"_s <= "unexpected"_s + event<EventParseRuntime> [invalid_parse_without_callbacks] / reject_invalid_parse_from_unexpected,
        "tokenize_begin"_s <= "request_decision"_s + completion<EventParseRuntime> / begin_tokenization,
        "tokenize_next"_s <= "tokenize_begin"_s + completion<EventParseRuntime> / request_next_lex_token_from_tokenize_begin,
        "tokenize_result_decision"_s <= "tokenize_next"_s + completion<EventParseRuntime>,
        "program_parser_model"_s <= "tokenize_result_decision"_s + completion<EventParseRuntime> [lexer_at_eof],
        "tokenize_append"_s <= "tokenize_result_decision"_s + completion<EventParseRuntime> [lexer_has_token] / append_lex_token,
        "parse_result_decision"_s <= "tokenize_result_decision"_s + completion<EventParseRuntime> [parse_error_invalid_request] / commit_lex_error_from_tokenize_result_decision,
        "parse_result_decision"_s <= "tokenize_result_decision"_s + completion<EventParseRuntime> [parse_error_parse_failed] / commit_lex_error_from_tokenize_result_decision,
        "parse_result_decision"_s <= "tokenize_result_decision"_s + completion<EventParseRuntime> [parse_error_internal_error] / commit_lex_error_from_tokenize_result_decision,
        "parse_result_decision"_s <= "tokenize_result_decision"_s + completion<EventParseRuntime> [parse_error_untracked] / commit_lex_error_from_tokenize_result_decision,
        "parse_result_decision"_s <= "tokenize_result_decision"_s + completion<EventParseRuntime> [parse_error_unknown] / commit_lex_error_from_tokenize_result_decision,
        "tokenize_next"_s <= "tokenize_append"_s + completion<EventParseRuntime> / request_next_lex_token_from_tokenize_append,
        "parse_result_decision"_s <= "program_parser_model"_s + completion<EventParseRuntime>,
        "done"_s <= "parse_result_decision"_s + completion<EventParseRuntime> [parse_error_none] / dispatch_done,
        "errored"_s <= "parse_result_decision"_s + completion<EventParseRuntime> [parse_error_invalid_request] / dispatch_error_from_parse_result_decision,
        "errored"_s <= "parse_result_decision"_s + completion<EventParseRuntime> [parse_error_parse_failed] / dispatch_error_from_parse_result_decision,
        "errored"_s <= "parse_result_decision"_s + completion<EventParseRuntime> [parse_error_internal_error] / dispatch_error_from_parse_result_decision,
        "errored"_s <= "parse_result_decision"_s + completion<EventParseRuntime> [parse_error_untracked] / dispatch_error_from_parse_result_decision,
        "errored"_s <= "parse_result_decision"_s + completion<EventParseRuntime> [parse_error_unknown] / dispatch_error_from_parse_result_decision,
        "unexpected"_s <= "initialized"_s + unexpected_event<_> / on_unexpected_from_initialized,
        "unexpected"_s <= "request_decision"_s + unexpected_event<_> / on_unexpected_from_request_decision,
        "unexpected"_s <= "tokenize_begin"_s + unexpected_event<_> / on_unexpected_from_tokenize_begin,
        "unexpected"_s <= "tokenize_next"_s + unexpected_event<_> / on_unexpected_from_tokenize_next,
        "unexpected"_s <= "tokenize_result_decision"_s + unexpected_event<_> / on_unexpected_from_tokenize_result_decision,
        "unexpected"_s <= "tokenize_append"_s + unexpected_event<_> / on_unexpected_from_tokenize_append,
        "unexpected"_s <= "parse_result_decision"_s + unexpected_event<_> / on_unexpected_from_parse_result_decision,
        "unexpected"_s <= "done"_s + unexpected_event<_> / on_unexpected_from_done,
        "unexpected"_s <= "errored"_s + unexpected_event<_> / on_unexpected_from_errored,
        "unexpected"_s <= "unexpected"_s + unexpected_event<_> / on_unexpected_from_unexpected,
    }
}

/// Context for `TextJinjaParser` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct TextJinjaParserContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl TextJinjaParserStateMachineContext for TextJinjaParserContext {
    fn append_lex_token(&mut self, _event: &EventParseRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/actions.hpp::append_lex_token
        todo!(
            "TODO: port action `append_lex_token` from emel.cpp/src/emel/text/jinja/parser/actions.hpp"
        )
    }
    fn begin_parse_from_done(&mut self, _event: &EventParseRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/actions.hpp::begin_parse
        todo!(
            "TODO: port action `begin_parse` from emel.cpp/src/emel/text/jinja/parser/actions.hpp"
        )
    }
    fn begin_parse_from_errored(&mut self, _event: &EventParseRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/actions.hpp::begin_parse
        todo!(
            "TODO: port action `begin_parse` from emel.cpp/src/emel/text/jinja/parser/actions.hpp"
        )
    }
    fn begin_parse_from_initialized(&mut self, _event: &EventParseRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/actions.hpp::begin_parse
        todo!(
            "TODO: port action `begin_parse` from emel.cpp/src/emel/text/jinja/parser/actions.hpp"
        )
    }
    fn begin_parse_from_unexpected(&mut self, _event: &EventParseRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/actions.hpp::begin_parse
        todo!(
            "TODO: port action `begin_parse` from emel.cpp/src/emel/text/jinja/parser/actions.hpp"
        )
    }
    fn begin_tokenization(&mut self, _event: &EventParseRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/actions.hpp::begin_tokenization
        todo!(
            "TODO: port action `begin_tokenization` from emel.cpp/src/emel/text/jinja/parser/actions.hpp"
        )
    }
    fn commit_lex_error_from_tokenize_result_decision(
        &mut self,
        _event: &EventParseRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/actions.hpp::commit_lex_error
        todo!(
            "TODO: port action `commit_lex_error` from emel.cpp/src/emel/text/jinja/parser/actions.hpp"
        )
    }
    fn dispatch_done(&mut self, _event: &EventParseRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/actions.hpp::dispatch_done
        todo!(
            "TODO: port action `dispatch_done` from emel.cpp/src/emel/text/jinja/parser/actions.hpp"
        )
    }
    fn dispatch_error_from_parse_result_decision(
        &mut self,
        _event: &EventParseRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/actions.hpp::dispatch_error
        todo!(
            "TODO: port action `dispatch_error` from emel.cpp/src/emel/text/jinja/parser/actions.hpp"
        )
    }
    fn invalid_parse_with_callbacks(&self, _event: &EventParseRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/guards.hpp::invalid_parse_with_callbacks
        todo!(
            "TODO: port guard `invalid_parse_with_callbacks` from emel.cpp/src/emel/text/jinja/parser/guards.hpp"
        )
    }
    fn invalid_parse_without_callbacks(&self, _event: &EventParseRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/guards.hpp::invalid_parse_without_callbacks
        todo!(
            "TODO: port guard `invalid_parse_without_callbacks` from emel.cpp/src/emel/text/jinja/parser/guards.hpp"
        )
    }
    fn lexer_at_eof(&self, _event: &EventParseRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/guards.hpp::lexer_at_eof
        todo!("TODO: port guard `lexer_at_eof` from emel.cpp/src/emel/text/jinja/parser/guards.hpp")
    }
    fn lexer_has_token(&self, _event: &EventParseRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/guards.hpp::lexer_has_token
        todo!(
            "TODO: port guard `lexer_has_token` from emel.cpp/src/emel/text/jinja/parser/guards.hpp"
        )
    }
    fn on_unexpected_from_done(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/parser/actions.hpp"
        )
    }
    fn on_unexpected_from_errored(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/parser/actions.hpp"
        )
    }
    fn on_unexpected_from_initialized(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/parser/actions.hpp"
        )
    }
    fn on_unexpected_from_parse_result_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/parser/actions.hpp"
        )
    }
    fn on_unexpected_from_request_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/parser/actions.hpp"
        )
    }
    fn on_unexpected_from_tokenize_append(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/parser/actions.hpp"
        )
    }
    fn on_unexpected_from_tokenize_begin(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/parser/actions.hpp"
        )
    }
    fn on_unexpected_from_tokenize_next(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/parser/actions.hpp"
        )
    }
    fn on_unexpected_from_tokenize_result_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/parser/actions.hpp"
        )
    }
    fn on_unexpected_from_unexpected(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/parser/actions.hpp"
        )
    }
    fn parse_error_internal_error(&self, _event: &EventParseRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/guards.hpp::parse_error_internal_error
        todo!(
            "TODO: port guard `parse_error_internal_error` from emel.cpp/src/emel/text/jinja/parser/guards.hpp"
        )
    }
    fn parse_error_invalid_request(&self, _event: &EventParseRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/guards.hpp::parse_error_invalid_request
        todo!(
            "TODO: port guard `parse_error_invalid_request` from emel.cpp/src/emel/text/jinja/parser/guards.hpp"
        )
    }
    fn parse_error_none(&self, _event: &EventParseRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/guards.hpp::parse_error_none
        todo!(
            "TODO: port guard `parse_error_none` from emel.cpp/src/emel/text/jinja/parser/guards.hpp"
        )
    }
    fn parse_error_parse_failed(&self, _event: &EventParseRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/guards.hpp::parse_error_parse_failed
        todo!(
            "TODO: port guard `parse_error_parse_failed` from emel.cpp/src/emel/text/jinja/parser/guards.hpp"
        )
    }
    fn parse_error_unknown(&self, _event: &EventParseRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/guards.hpp::parse_error_unknown
        todo!(
            "TODO: port guard `parse_error_unknown` from emel.cpp/src/emel/text/jinja/parser/guards.hpp"
        )
    }
    fn parse_error_untracked(&self, _event: &EventParseRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/guards.hpp::parse_error_untracked
        todo!(
            "TODO: port guard `parse_error_untracked` from emel.cpp/src/emel/text/jinja/parser/guards.hpp"
        )
    }
    fn reject_invalid_parse_from_done(&mut self, _event: &EventParseRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/actions.hpp::reject_invalid_parse
        todo!(
            "TODO: port action `reject_invalid_parse` from emel.cpp/src/emel/text/jinja/parser/actions.hpp"
        )
    }
    fn reject_invalid_parse_from_errored(&mut self, _event: &EventParseRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/actions.hpp::reject_invalid_parse
        todo!(
            "TODO: port action `reject_invalid_parse` from emel.cpp/src/emel/text/jinja/parser/actions.hpp"
        )
    }
    fn reject_invalid_parse_from_initialized(
        &mut self,
        _event: &EventParseRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/actions.hpp::reject_invalid_parse
        todo!(
            "TODO: port action `reject_invalid_parse` from emel.cpp/src/emel/text/jinja/parser/actions.hpp"
        )
    }
    fn reject_invalid_parse_from_unexpected(
        &mut self,
        _event: &EventParseRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/actions.hpp::reject_invalid_parse
        todo!(
            "TODO: port action `reject_invalid_parse` from emel.cpp/src/emel/text/jinja/parser/actions.hpp"
        )
    }
    fn request_next_lex_token_from_tokenize_append(
        &mut self,
        _event: &EventParseRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/actions.hpp::request_next_lex_token
        todo!(
            "TODO: port action `request_next_lex_token` from emel.cpp/src/emel/text/jinja/parser/actions.hpp"
        )
    }
    fn request_next_lex_token_from_tokenize_begin(
        &mut self,
        _event: &EventParseRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/actions.hpp::request_next_lex_token
        todo!(
            "TODO: port action `request_next_lex_token` from emel.cpp/src/emel/text/jinja/parser/actions.hpp"
        )
    }
    fn valid_parse(&self, _event: &EventParseRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/guards.hpp::valid_parse
        todo!("TODO: port guard `valid_parse` from emel.cpp/src/emel/text/jinja/parser/guards.hpp")
    }
}
