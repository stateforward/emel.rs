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

// --- machine TextJinjaParserLexer from emel.cpp/src/emel/text/jinja/parser/lexer/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventNextRuntime;

sml! {
    TextJinjaParserLexer {
        "initialized"_s <= *"initialized"_s + event<EventNextRuntime> [invalid_next] / reject_invalid_next_from_initialized,
        "initialized"_s <= "initialized"_s + event<EventNextRuntime> [invalid_cursor_position] / reject_invalid_cursor_from_initialized,
        "text_boundary_candidate_decision"_s <= "initialized"_s + event<EventNextRuntime> / begin_scan_from_initialized,
        "scanning"_s <= "scanning"_s + event<EventNextRuntime> [invalid_next] / reject_invalid_next_from_scanning,
        "scanning"_s <= "scanning"_s + event<EventNextRuntime> [invalid_cursor_position] / reject_invalid_cursor_from_scanning,
        "text_boundary_candidate_decision"_s <= "scanning"_s + event<EventNextRuntime> / begin_scan_from_scanning,
        "text_scan_exec"_s <= "text_boundary_candidate_decision"_s + completion<EventNextRuntime> [at_text_boundary] / scan_text_boundary,
        "comment_candidate_decision"_s <= "text_boundary_candidate_decision"_s + completion<EventNextRuntime>,
        "text_opening_block_decision"_s <= "text_scan_exec"_s + completion<EventNextRuntime>,
        "text_trim_opening_block_exec"_s <= "text_opening_block_decision"_s + completion<EventNextRuntime> [text_opening_block_ahead],
        "text_materialize_exec"_s <= "text_opening_block_decision"_s + completion<EventNextRuntime>,
        "text_trim_opening_block_result_decision"_s <= "text_trim_opening_block_exec"_s + completion<EventNextRuntime> / probe_text_opening_trim,
        "text_materialize_exec"_s <= "text_trim_opening_block_result_decision"_s + completion<EventNextRuntime> [text_opening_trim_stopped_on_newline] / apply_text_opening_trim_to_newline,
        "text_materialize_exec"_s <= "text_trim_opening_block_result_decision"_s + completion<EventNextRuntime> [text_opening_trim_to_zero] / apply_text_opening_trim_to_zero,
        "text_materialize_exec"_s <= "text_trim_opening_block_result_decision"_s + completion<EventNextRuntime>,
        "scanning"_s <= "text_materialize_exec"_s + completion<EventNextRuntime> [text_boundary_empty_at_end] / emit_text_boundary_eof,
        "scanning"_s <= "text_materialize_exec"_s + completion<EventNextRuntime> [text_plain_boundary_ready] / emit_plain_text_boundary_token,
        "text_finalize_exec"_s <= "text_materialize_exec"_s + completion<EventNextRuntime> / materialize_text_token,
        "text_finalize_result_decision"_s <= "text_finalize_exec"_s + completion<EventNextRuntime> [text_can_trim_leading_newline] / trim_text_leading_newline,
        "text_finalize_result_decision"_s <= "text_finalize_exec"_s + completion<EventNextRuntime>,
        "text_finalize_token_exec"_s <= "text_finalize_result_decision"_s + completion<EventNextRuntime> [text_apply_lstrip_and_rstrip] / lstrip_and_rstrip_text_token,
        "text_finalize_token_exec"_s <= "text_finalize_result_decision"_s + completion<EventNextRuntime> [text_apply_lstrip_only] / lstrip_text_token,
        "text_finalize_token_exec"_s <= "text_finalize_result_decision"_s + completion<EventNextRuntime> [text_apply_rstrip_only] / rstrip_text_token,
        "text_finalize_token_exec"_s <= "text_finalize_result_decision"_s + completion<EventNextRuntime> [text_apply_no_strip],
        "invalid_char_exec"_s <= "text_finalize_result_decision"_s + completion<EventNextRuntime> [scan_unhandled],
        "text_emit_result_decision"_s <= "text_finalize_token_exec"_s + completion<EventNextRuntime> / finalize_text_boundary_token,
        "scanning"_s <= "text_emit_result_decision"_s + completion<EventNextRuntime> [text_token_non_empty] / emit_scanned_token_from_text_emit_result_decision,
        "space_eof_exec"_s <= "text_emit_result_decision"_s + completion<EventNextRuntime> [text_token_empty_at_end] / mark_no_token_eof_from_text_emit_result_decision,
        "comment_candidate_decision"_s <= "text_emit_result_decision"_s + completion<EventNextRuntime>,
        "comment_scan_exec"_s <= "comment_candidate_decision"_s + completion<EventNextRuntime> [starts_comment] / scan_comment,
        "trim_prefix_scan_exec"_s <= "comment_candidate_decision"_s + completion<EventNextRuntime> [starts_trim_prefix] / scan_trim_prefix,
        "space_scan_exec"_s <= "comment_candidate_decision"_s + completion<EventNextRuntime> / scan_spaces_from_comment_candidate_decision,
        "comment_scan_result_decision"_s <= "comment_scan_exec"_s + completion<EventNextRuntime>,
        "scanning"_s <= "comment_scan_result_decision"_s + completion<EventNextRuntime> [parse_error_invalid_request] / emit_scan_error_from_comment_scan_result_decision,
        "scanning"_s <= "comment_scan_result_decision"_s + completion<EventNextRuntime> [parse_error_parse_failed] / emit_scan_error_from_comment_scan_result_decision,
        "scanning"_s <= "comment_scan_result_decision"_s + completion<EventNextRuntime> [parse_error_internal_error] / emit_scan_error_from_comment_scan_result_decision,
        "scanning"_s <= "comment_scan_result_decision"_s + completion<EventNextRuntime> [parse_error_untracked] / emit_scan_error_from_comment_scan_result_decision,
        "scanning"_s <= "comment_scan_result_decision"_s + completion<EventNextRuntime> [parse_error_unknown] / emit_scan_error_from_comment_scan_result_decision,
        "comment_finalize_exec"_s <= "comment_scan_result_decision"_s + completion<EventNextRuntime> [comment_terminated],
        "comment_unterminated_exec"_s <= "comment_scan_result_decision"_s + completion<EventNextRuntime>,
        "comment_finalize_result_decision"_s <= "comment_finalize_exec"_s + completion<EventNextRuntime> / finalize_comment_token,
        "scanning"_s <= "comment_finalize_result_decision"_s + completion<EventNextRuntime> [parse_error_invalid_request] / emit_scan_error_from_comment_finalize_result_decision,
        "scanning"_s <= "comment_finalize_result_decision"_s + completion<EventNextRuntime> [parse_error_parse_failed] / emit_scan_error_from_comment_finalize_result_decision,
        "scanning"_s <= "comment_finalize_result_decision"_s + completion<EventNextRuntime> [parse_error_internal_error] / emit_scan_error_from_comment_finalize_result_decision,
        "scanning"_s <= "comment_finalize_result_decision"_s + completion<EventNextRuntime> [parse_error_untracked] / emit_scan_error_from_comment_finalize_result_decision,
        "scanning"_s <= "comment_finalize_result_decision"_s + completion<EventNextRuntime> [parse_error_unknown] / emit_scan_error_from_comment_finalize_result_decision,
        "scanning"_s <= "comment_finalize_result_decision"_s + completion<EventNextRuntime> [scan_token_available] / emit_scanned_token_from_comment_finalize_result_decision,
        "scanning"_s <= "comment_finalize_result_decision"_s + completion<EventNextRuntime> [scan_no_token_eof] / emit_eof_from_comment_finalize_result_decision,
        "invalid_char_exec"_s <= "comment_finalize_result_decision"_s + completion<EventNextRuntime> [scan_unhandled],
        "comment_unterminated_result_decision"_s <= "comment_unterminated_exec"_s + completion<EventNextRuntime> / mark_comment_unterminated,
        "scanning"_s <= "comment_unterminated_result_decision"_s + completion<EventNextRuntime> [parse_error_invalid_request] / emit_scan_error_from_comment_unterminated_result_decision,
        "scanning"_s <= "comment_unterminated_result_decision"_s + completion<EventNextRuntime> [parse_error_parse_failed] / emit_scan_error_from_comment_unterminated_result_decision,
        "scanning"_s <= "comment_unterminated_result_decision"_s + completion<EventNextRuntime> [parse_error_internal_error] / emit_scan_error_from_comment_unterminated_result_decision,
        "scanning"_s <= "comment_unterminated_result_decision"_s + completion<EventNextRuntime> [parse_error_untracked] / emit_scan_error_from_comment_unterminated_result_decision,
        "scanning"_s <= "comment_unterminated_result_decision"_s + completion<EventNextRuntime> [parse_error_unknown] / emit_scan_error_from_comment_unterminated_result_decision,
        "scanning"_s <= "comment_unterminated_result_decision"_s + completion<EventNextRuntime> [scan_token_available] / emit_scanned_token_from_comment_unterminated_result_decision,
        "scanning"_s <= "comment_unterminated_result_decision"_s + completion<EventNextRuntime> [scan_no_token_eof] / emit_eof_from_comment_unterminated_result_decision,
        "invalid_char_exec"_s <= "comment_unterminated_result_decision"_s + completion<EventNextRuntime> [scan_unhandled],
        "trim_prefix_eof_exec"_s <= "trim_prefix_scan_exec"_s + completion<EventNextRuntime> [cursor_at_end] / mark_no_token_eof_from_trim_prefix_scan_exec,
        "space_scan_exec"_s <= "trim_prefix_scan_exec"_s + completion<EventNextRuntime> / scan_spaces_from_trim_prefix_scan_exec,
        "scanning"_s <= "trim_prefix_eof_exec"_s + completion<EventNextRuntime> / emit_eof_from_trim_prefix_eof_exec,
        "space_eof_exec"_s <= "space_scan_exec"_s + completion<EventNextRuntime> [cursor_at_end] / mark_no_token_eof_from_space_scan_exec,
        "unary_candidate_decision"_s <= "space_scan_exec"_s + completion<EventNextRuntime>,
        "scanning"_s <= "space_eof_exec"_s + completion<EventNextRuntime> / emit_eof_from_space_eof_exec,
        "unary_prefix_context_decision"_s <= "unary_candidate_decision"_s + completion<EventNextRuntime> [unary_candidate],
        "string_scan_exec"_s <= "unary_candidate_decision"_s + completion<EventNextRuntime> [starts_string],
        "numeric_scan_exec"_s <= "unary_candidate_decision"_s + completion<EventNextRuntime> [starts_numeric] / scan_numeric,
        "word_scan_exec"_s <= "unary_candidate_decision"_s + completion<EventNextRuntime> [starts_word] / scan_word,
        "mapping_candidate_decision"_s <= "unary_candidate_decision"_s + completion<EventNextRuntime>,
        "invalid_char_exec"_s <= "unary_prefix_context_decision"_s + completion<EventNextRuntime> [unary_prefix_context_invalid],
        "unary_prefix_allowed_decision"_s <= "unary_prefix_context_decision"_s + completion<EventNextRuntime>,
        "mapping_candidate_decision"_s <= "unary_prefix_allowed_decision"_s + completion<EventNextRuntime> [unary_prefix_disallowed],
        "unary_scan_exec"_s <= "unary_prefix_allowed_decision"_s + completion<EventNextRuntime> / scan_unary,
        "scanning"_s <= "unary_scan_exec"_s + completion<EventNextRuntime> [unary_numeric_suffix_present] / emit_unary_numeric_token,
        "scanning"_s <= "unary_scan_exec"_s + completion<EventNextRuntime> / emit_unary_operator_token,
        "mapping_close_curly_exec"_s <= "mapping_candidate_decision"_s + completion<EventNextRuntime> [mapping_close_expression_blocked_by_curly_depth] / scan_mapping_close_curly,
        "mapping_scan_exec"_s <= "mapping_candidate_decision"_s + completion<EventNextRuntime> [mapping_open_statement_trim] / scan_mapping_open_statement_trim,
        "mapping_scan_exec"_s <= "mapping_candidate_decision"_s + completion<EventNextRuntime> [mapping_open_statement] / scan_mapping_open_statement,
        "mapping_scan_exec"_s <= "mapping_candidate_decision"_s + completion<EventNextRuntime> [mapping_close_statement] / scan_mapping_close_statement,
        "mapping_scan_exec"_s <= "mapping_candidate_decision"_s + completion<EventNextRuntime> [mapping_open_expression_trim] / scan_mapping_open_expression_trim,
        "mapping_scan_exec"_s <= "mapping_candidate_decision"_s + completion<EventNextRuntime> [mapping_open_expression] / scan_mapping_open_expression,
        "mapping_scan_exec"_s <= "mapping_candidate_decision"_s + completion<EventNextRuntime> [mapping_close_expression_not_blocked] / scan_mapping_close_expression,
        "mapping_scan_exec"_s <= "mapping_candidate_decision"_s + completion<EventNextRuntime> [mapping_close_statement_trim] / scan_mapping_close_statement_trim,
        "mapping_scan_exec"_s <= "mapping_candidate_decision"_s + completion<EventNextRuntime> [mapping_close_expression_trim] / scan_mapping_close_expression_trim,
        "mapping_scan_exec"_s <= "mapping_candidate_decision"_s + completion<EventNextRuntime> [mapping_open_paren] / scan_mapping_open_paren,
        "mapping_scan_exec"_s <= "mapping_candidate_decision"_s + completion<EventNextRuntime> [mapping_close_paren] / scan_mapping_close_paren,
        "mapping_scan_exec"_s <= "mapping_candidate_decision"_s + completion<EventNextRuntime> [mapping_open_curly_bracket] / scan_mapping_open_curly_bracket,
        "mapping_scan_exec"_s <= "mapping_candidate_decision"_s + completion<EventNextRuntime> [mapping_close_curly_bracket] / scan_mapping_close_curly_bracket,
        "mapping_scan_exec"_s <= "mapping_candidate_decision"_s + completion<EventNextRuntime> [mapping_open_square_bracket] / scan_mapping_open_square_bracket,
        "mapping_scan_exec"_s <= "mapping_candidate_decision"_s + completion<EventNextRuntime> [mapping_close_square_bracket] / scan_mapping_close_square_bracket,
        "mapping_scan_exec"_s <= "mapping_candidate_decision"_s + completion<EventNextRuntime> [mapping_comma] / scan_mapping_comma,
        "mapping_scan_exec"_s <= "mapping_candidate_decision"_s + completion<EventNextRuntime> [mapping_dot] / scan_mapping_dot,
        "mapping_scan_exec"_s <= "mapping_candidate_decision"_s + completion<EventNextRuntime> [mapping_colon] / scan_mapping_colon,
        "mapping_scan_exec"_s <= "mapping_candidate_decision"_s + completion<EventNextRuntime> [mapping_pipe] / scan_mapping_pipe,
        "mapping_scan_exec"_s <= "mapping_candidate_decision"_s + completion<EventNextRuntime> [mapping_less_equal] / scan_mapping_less_equal,
        "mapping_scan_exec"_s <= "mapping_candidate_decision"_s + completion<EventNextRuntime> [mapping_greater_equal] / scan_mapping_greater_equal,
        "mapping_scan_exec"_s <= "mapping_candidate_decision"_s + completion<EventNextRuntime> [mapping_equal_equal] / scan_mapping_equal_equal,
        "mapping_scan_exec"_s <= "mapping_candidate_decision"_s + completion<EventNextRuntime> [mapping_bang_equal] / scan_mapping_bang_equal,
        "mapping_scan_exec"_s <= "mapping_candidate_decision"_s + completion<EventNextRuntime> [mapping_less] / scan_mapping_less,
        "mapping_scan_exec"_s <= "mapping_candidate_decision"_s + completion<EventNextRuntime> [mapping_greater] / scan_mapping_greater,
        "mapping_scan_exec"_s <= "mapping_candidate_decision"_s + completion<EventNextRuntime> [mapping_plus] / scan_mapping_plus,
        "mapping_scan_exec"_s <= "mapping_candidate_decision"_s + completion<EventNextRuntime> [mapping_minus] / scan_mapping_minus,
        "mapping_scan_exec"_s <= "mapping_candidate_decision"_s + completion<EventNextRuntime> [mapping_tilde] / scan_mapping_tilde,
        "mapping_scan_exec"_s <= "mapping_candidate_decision"_s + completion<EventNextRuntime> [mapping_star] / scan_mapping_star,
        "mapping_scan_exec"_s <= "mapping_candidate_decision"_s + completion<EventNextRuntime> [mapping_slash] / scan_mapping_slash,
        "mapping_scan_exec"_s <= "mapping_candidate_decision"_s + completion<EventNextRuntime> [mapping_percent] / scan_mapping_percent,
        "mapping_scan_exec"_s <= "mapping_candidate_decision"_s + completion<EventNextRuntime> [mapping_equals] / scan_mapping_equals,
        "invalid_char_exec"_s <= "mapping_candidate_decision"_s + completion<EventNextRuntime>,
        "scanning"_s <= "mapping_close_curly_exec"_s + completion<EventNextRuntime> / emit_scanned_token_from_mapping_close_curly_exec,
        "scanning"_s <= "mapping_scan_exec"_s + completion<EventNextRuntime> / emit_scanned_token_from_mapping_scan_exec,
        "string_content_scan_exec"_s <= "string_scan_exec"_s + completion<EventNextRuntime> / begin_string_scan,
        "string_content_policy_decision"_s <= "string_content_scan_exec"_s + completion<EventNextRuntime>,
        "string_scan_result_decision"_s <= "string_content_policy_decision"_s + completion<EventNextRuntime> [string_scan_immediate_termination_or_eof],
        "string_scan_result_decision"_s <= "string_content_policy_decision"_s + completion<EventNextRuntime> [string_scan_requires_content] / scan_string_content,
        "invalid_char_exec"_s <= "string_content_policy_decision"_s + completion<EventNextRuntime> [scan_unhandled],
        "string_materialize_exec"_s <= "string_scan_result_decision"_s + completion<EventNextRuntime>,
        "string_status_decision"_s <= "string_materialize_exec"_s + completion<EventNextRuntime> / materialize_string_token,
        "scanning"_s <= "string_status_decision"_s + completion<EventNextRuntime> [parse_error_invalid_request] / emit_scan_error_from_string_status_decision,
        "scanning"_s <= "string_status_decision"_s + completion<EventNextRuntime> [parse_error_parse_failed] / emit_scan_error_from_string_status_decision,
        "scanning"_s <= "string_status_decision"_s + completion<EventNextRuntime> [parse_error_internal_error] / emit_scan_error_from_string_status_decision,
        "scanning"_s <= "string_status_decision"_s + completion<EventNextRuntime> [parse_error_untracked] / emit_scan_error_from_string_status_decision,
        "scanning"_s <= "string_status_decision"_s + completion<EventNextRuntime> [parse_error_unknown] / emit_scan_error_from_string_status_decision,
        "string_unterminated_exec"_s <= "string_status_decision"_s + completion<EventNextRuntime> [string_not_terminated],
        "string_finalize_exec"_s <= "string_status_decision"_s + completion<EventNextRuntime> [string_terminated],
        "invalid_char_exec"_s <= "string_status_decision"_s + completion<EventNextRuntime> [scan_unhandled],
        "string_unterminated_result_decision"_s <= "string_unterminated_exec"_s + completion<EventNextRuntime> / mark_string_unterminated,
        "scanning"_s <= "string_unterminated_result_decision"_s + completion<EventNextRuntime> [parse_error_invalid_request] / emit_scan_error_from_string_unterminated_result_decision,
        "scanning"_s <= "string_unterminated_result_decision"_s + completion<EventNextRuntime> [parse_error_parse_failed] / emit_scan_error_from_string_unterminated_result_decision,
        "scanning"_s <= "string_unterminated_result_decision"_s + completion<EventNextRuntime> [parse_error_internal_error] / emit_scan_error_from_string_unterminated_result_decision,
        "scanning"_s <= "string_unterminated_result_decision"_s + completion<EventNextRuntime> [parse_error_untracked] / emit_scan_error_from_string_unterminated_result_decision,
        "scanning"_s <= "string_unterminated_result_decision"_s + completion<EventNextRuntime> [parse_error_unknown] / emit_scan_error_from_string_unterminated_result_decision,
        "scanning"_s <= "string_unterminated_result_decision"_s + completion<EventNextRuntime> [scan_token_available] / emit_scanned_token_from_string_unterminated_result_decision,
        "scanning"_s <= "string_unterminated_result_decision"_s + completion<EventNextRuntime> [scan_no_token_eof] / emit_eof_from_string_unterminated_result_decision,
        "invalid_char_exec"_s <= "string_unterminated_result_decision"_s + completion<EventNextRuntime> [scan_unhandled],
        "string_finalize_result_decision"_s <= "string_finalize_exec"_s + completion<EventNextRuntime> / finalize_string_token,
        "scanning"_s <= "string_finalize_result_decision"_s + completion<EventNextRuntime> [scan_token_available] / emit_scanned_token_from_string_finalize_result_decision,
        "scanning"_s <= "string_finalize_result_decision"_s + completion<EventNextRuntime> [scan_no_token_eof] / emit_eof_from_string_finalize_result_decision,
        "scanning"_s <= "string_finalize_result_decision"_s + completion<EventNextRuntime> [parse_error_invalid_request] / emit_scan_error_from_string_finalize_result_decision,
        "scanning"_s <= "string_finalize_result_decision"_s + completion<EventNextRuntime> [parse_error_parse_failed] / emit_scan_error_from_string_finalize_result_decision,
        "scanning"_s <= "string_finalize_result_decision"_s + completion<EventNextRuntime> [parse_error_internal_error] / emit_scan_error_from_string_finalize_result_decision,
        "scanning"_s <= "string_finalize_result_decision"_s + completion<EventNextRuntime> [parse_error_untracked] / emit_scan_error_from_string_finalize_result_decision,
        "scanning"_s <= "string_finalize_result_decision"_s + completion<EventNextRuntime> [parse_error_unknown] / emit_scan_error_from_string_finalize_result_decision,
        "invalid_char_exec"_s <= "string_finalize_result_decision"_s + completion<EventNextRuntime> [scan_unhandled],
        "scanning"_s <= "numeric_scan_exec"_s + completion<EventNextRuntime> / emit_scanned_token_from_numeric_scan_exec,
        "scanning"_s <= "word_scan_exec"_s + completion<EventNextRuntime> / emit_scanned_token_from_word_scan_exec,
        "invalid_char_result_decision"_s <= "invalid_char_exec"_s + completion<EventNextRuntime> / mark_invalid_character,
        "scanning"_s <= "invalid_char_result_decision"_s + completion<EventNextRuntime> [parse_error_invalid_request] / emit_scan_error_from_invalid_char_result_decision,
        "scanning"_s <= "invalid_char_result_decision"_s + completion<EventNextRuntime> [parse_error_parse_failed] / emit_scan_error_from_invalid_char_result_decision,
        "scanning"_s <= "invalid_char_result_decision"_s + completion<EventNextRuntime> [parse_error_internal_error] / emit_scan_error_from_invalid_char_result_decision,
        "scanning"_s <= "invalid_char_result_decision"_s + completion<EventNextRuntime> [parse_error_untracked] / emit_scan_error_from_invalid_char_result_decision,
        "scanning"_s <= "invalid_char_result_decision"_s + completion<EventNextRuntime> [parse_error_unknown] / emit_scan_error_from_invalid_char_result_decision,
        "scanning"_s <= "initialized"_s + unexpected_event<_> / on_unexpected_from_initialized,
        "scanning"_s <= "scanning"_s + unexpected_event<_> / on_unexpected_from_scanning,
    }
}

/// Context for `TextJinjaParserLexer` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct TextJinjaParserLexerContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl TextJinjaParserLexerStateMachineContext for TextJinjaParserLexerContext {
    fn apply_text_opening_trim_to_newline(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::apply_text_opening_trim_to_newline
        todo!(
            "TODO: port action `apply_text_opening_trim_to_newline` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn apply_text_opening_trim_to_zero(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::apply_text_opening_trim_to_zero
        todo!(
            "TODO: port action `apply_text_opening_trim_to_zero` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn at_text_boundary(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::at_text_boundary
        todo!(
            "TODO: port guard `at_text_boundary` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn begin_scan_from_initialized(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::begin_scan
        todo!(
            "TODO: port action `begin_scan` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn begin_scan_from_scanning(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::begin_scan
        todo!(
            "TODO: port action `begin_scan` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn begin_string_scan(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::begin_string_scan
        todo!(
            "TODO: port action `begin_string_scan` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn comment_terminated(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::comment_terminated
        todo!(
            "TODO: port guard `comment_terminated` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn cursor_at_end(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::cursor_at_end
        todo!(
            "TODO: port guard `cursor_at_end` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn emit_eof_from_comment_finalize_result_decision(
        &mut self,
        _event: &EventNextRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::emit_eof
        todo!(
            "TODO: port action `emit_eof` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn emit_eof_from_comment_unterminated_result_decision(
        &mut self,
        _event: &EventNextRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::emit_eof
        todo!(
            "TODO: port action `emit_eof` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn emit_eof_from_space_eof_exec(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::emit_eof
        todo!(
            "TODO: port action `emit_eof` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn emit_eof_from_string_finalize_result_decision(
        &mut self,
        _event: &EventNextRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::emit_eof
        todo!(
            "TODO: port action `emit_eof` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn emit_eof_from_string_unterminated_result_decision(
        &mut self,
        _event: &EventNextRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::emit_eof
        todo!(
            "TODO: port action `emit_eof` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn emit_eof_from_trim_prefix_eof_exec(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::emit_eof
        todo!(
            "TODO: port action `emit_eof` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn emit_plain_text_boundary_token(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::emit_plain_text_boundary_token
        todo!(
            "TODO: port action `emit_plain_text_boundary_token` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn emit_scan_error_from_comment_finalize_result_decision(
        &mut self,
        _event: &EventNextRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::emit_scan_error
        todo!(
            "TODO: port action `emit_scan_error` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn emit_scan_error_from_comment_scan_result_decision(
        &mut self,
        _event: &EventNextRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::emit_scan_error
        todo!(
            "TODO: port action `emit_scan_error` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn emit_scan_error_from_comment_unterminated_result_decision(
        &mut self,
        _event: &EventNextRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::emit_scan_error
        todo!(
            "TODO: port action `emit_scan_error` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn emit_scan_error_from_invalid_char_result_decision(
        &mut self,
        _event: &EventNextRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::emit_scan_error
        todo!(
            "TODO: port action `emit_scan_error` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn emit_scan_error_from_string_finalize_result_decision(
        &mut self,
        _event: &EventNextRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::emit_scan_error
        todo!(
            "TODO: port action `emit_scan_error` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn emit_scan_error_from_string_status_decision(
        &mut self,
        _event: &EventNextRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::emit_scan_error
        todo!(
            "TODO: port action `emit_scan_error` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn emit_scan_error_from_string_unterminated_result_decision(
        &mut self,
        _event: &EventNextRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::emit_scan_error
        todo!(
            "TODO: port action `emit_scan_error` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn emit_scanned_token_from_comment_finalize_result_decision(
        &mut self,
        _event: &EventNextRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::emit_scanned_token
        todo!(
            "TODO: port action `emit_scanned_token` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn emit_scanned_token_from_comment_unterminated_result_decision(
        &mut self,
        _event: &EventNextRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::emit_scanned_token
        todo!(
            "TODO: port action `emit_scanned_token` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn emit_scanned_token_from_mapping_close_curly_exec(
        &mut self,
        _event: &EventNextRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::emit_scanned_token
        todo!(
            "TODO: port action `emit_scanned_token` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn emit_scanned_token_from_mapping_scan_exec(
        &mut self,
        _event: &EventNextRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::emit_scanned_token
        todo!(
            "TODO: port action `emit_scanned_token` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn emit_scanned_token_from_numeric_scan_exec(
        &mut self,
        _event: &EventNextRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::emit_scanned_token
        todo!(
            "TODO: port action `emit_scanned_token` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn emit_scanned_token_from_string_finalize_result_decision(
        &mut self,
        _event: &EventNextRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::emit_scanned_token
        todo!(
            "TODO: port action `emit_scanned_token` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn emit_scanned_token_from_string_unterminated_result_decision(
        &mut self,
        _event: &EventNextRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::emit_scanned_token
        todo!(
            "TODO: port action `emit_scanned_token` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn emit_scanned_token_from_text_emit_result_decision(
        &mut self,
        _event: &EventNextRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::emit_scanned_token
        todo!(
            "TODO: port action `emit_scanned_token` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn emit_scanned_token_from_word_scan_exec(
        &mut self,
        _event: &EventNextRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::emit_scanned_token
        todo!(
            "TODO: port action `emit_scanned_token` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn emit_text_boundary_eof(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::emit_text_boundary_eof
        todo!(
            "TODO: port action `emit_text_boundary_eof` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn emit_unary_numeric_token(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::emit_unary_numeric_token
        todo!(
            "TODO: port action `emit_unary_numeric_token` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn emit_unary_operator_token(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::emit_unary_operator_token
        todo!(
            "TODO: port action `emit_unary_operator_token` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn finalize_comment_token(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::finalize_comment_token
        todo!(
            "TODO: port action `finalize_comment_token` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn finalize_string_token(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::finalize_string_token
        todo!(
            "TODO: port action `finalize_string_token` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn finalize_text_boundary_token(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::finalize_text_boundary_token
        todo!(
            "TODO: port action `finalize_text_boundary_token` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn invalid_cursor_position(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::invalid_cursor_position
        todo!(
            "TODO: port guard `invalid_cursor_position` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn invalid_next(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::invalid_next
        todo!(
            "TODO: port guard `invalid_next` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn lstrip_and_rstrip_text_token(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::lstrip_and_rstrip_text_token
        todo!(
            "TODO: port action `lstrip_and_rstrip_text_token` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn lstrip_text_token(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::lstrip_text_token
        todo!(
            "TODO: port action `lstrip_text_token` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn mapping_bang_equal(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::mapping_bang_equal
        todo!(
            "TODO: port guard `mapping_bang_equal` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn mapping_close_curly_bracket(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::mapping_close_curly_bracket
        todo!(
            "TODO: port guard `mapping_close_curly_bracket` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn mapping_close_expression_blocked_by_curly_depth(
        &self,
        _event: &EventNextRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::mapping_close_expression_blocked_by_curly_depth
        todo!(
            "TODO: port guard `mapping_close_expression_blocked_by_curly_depth` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn mapping_close_expression_not_blocked(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::mapping_close_expression_not_blocked
        todo!(
            "TODO: port guard `mapping_close_expression_not_blocked` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn mapping_close_expression_trim(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::mapping_close_expression_trim
        todo!(
            "TODO: port guard `mapping_close_expression_trim` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn mapping_close_paren(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::mapping_close_paren
        todo!(
            "TODO: port guard `mapping_close_paren` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn mapping_close_square_bracket(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::mapping_close_square_bracket
        todo!(
            "TODO: port guard `mapping_close_square_bracket` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn mapping_close_statement(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::mapping_close_statement
        todo!(
            "TODO: port guard `mapping_close_statement` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn mapping_close_statement_trim(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::mapping_close_statement_trim
        todo!(
            "TODO: port guard `mapping_close_statement_trim` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn mapping_colon(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::mapping_colon
        todo!(
            "TODO: port guard `mapping_colon` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn mapping_comma(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::mapping_comma
        todo!(
            "TODO: port guard `mapping_comma` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn mapping_dot(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::mapping_dot
        todo!(
            "TODO: port guard `mapping_dot` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn mapping_equal_equal(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::mapping_equal_equal
        todo!(
            "TODO: port guard `mapping_equal_equal` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn mapping_equals(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::mapping_equals
        todo!(
            "TODO: port guard `mapping_equals` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn mapping_greater(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::mapping_greater
        todo!(
            "TODO: port guard `mapping_greater` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn mapping_greater_equal(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::mapping_greater_equal
        todo!(
            "TODO: port guard `mapping_greater_equal` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn mapping_less(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::mapping_less
        todo!(
            "TODO: port guard `mapping_less` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn mapping_less_equal(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::mapping_less_equal
        todo!(
            "TODO: port guard `mapping_less_equal` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn mapping_minus(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::mapping_minus
        todo!(
            "TODO: port guard `mapping_minus` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn mapping_open_curly_bracket(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::mapping_open_curly_bracket
        todo!(
            "TODO: port guard `mapping_open_curly_bracket` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn mapping_open_expression(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::mapping_open_expression
        todo!(
            "TODO: port guard `mapping_open_expression` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn mapping_open_expression_trim(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::mapping_open_expression_trim
        todo!(
            "TODO: port guard `mapping_open_expression_trim` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn mapping_open_paren(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::mapping_open_paren
        todo!(
            "TODO: port guard `mapping_open_paren` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn mapping_open_square_bracket(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::mapping_open_square_bracket
        todo!(
            "TODO: port guard `mapping_open_square_bracket` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn mapping_open_statement(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::mapping_open_statement
        todo!(
            "TODO: port guard `mapping_open_statement` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn mapping_open_statement_trim(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::mapping_open_statement_trim
        todo!(
            "TODO: port guard `mapping_open_statement_trim` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn mapping_percent(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::mapping_percent
        todo!(
            "TODO: port guard `mapping_percent` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn mapping_pipe(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::mapping_pipe
        todo!(
            "TODO: port guard `mapping_pipe` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn mapping_plus(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::mapping_plus
        todo!(
            "TODO: port guard `mapping_plus` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn mapping_slash(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::mapping_slash
        todo!(
            "TODO: port guard `mapping_slash` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn mapping_star(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::mapping_star
        todo!(
            "TODO: port guard `mapping_star` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn mapping_tilde(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::mapping_tilde
        todo!(
            "TODO: port guard `mapping_tilde` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn mark_comment_unterminated(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::mark_comment_unterminated
        todo!(
            "TODO: port action `mark_comment_unterminated` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn mark_invalid_character(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::mark_invalid_character
        todo!(
            "TODO: port action `mark_invalid_character` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn mark_no_token_eof_from_space_scan_exec(
        &mut self,
        _event: &EventNextRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::mark_no_token_eof
        todo!(
            "TODO: port action `mark_no_token_eof` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn mark_no_token_eof_from_text_emit_result_decision(
        &mut self,
        _event: &EventNextRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::mark_no_token_eof
        todo!(
            "TODO: port action `mark_no_token_eof` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn mark_no_token_eof_from_trim_prefix_scan_exec(
        &mut self,
        _event: &EventNextRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::mark_no_token_eof
        todo!(
            "TODO: port action `mark_no_token_eof` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn mark_string_unterminated(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::mark_string_unterminated
        todo!(
            "TODO: port action `mark_string_unterminated` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn materialize_string_token(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::materialize_string_token
        todo!(
            "TODO: port action `materialize_string_token` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn materialize_text_token(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::materialize_text_token
        todo!(
            "TODO: port action `materialize_text_token` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn on_unexpected_from_initialized(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn on_unexpected_from_scanning(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn parse_error_internal_error(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::parse_error_internal_error
        todo!(
            "TODO: port guard `parse_error_internal_error` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn parse_error_invalid_request(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::parse_error_invalid_request
        todo!(
            "TODO: port guard `parse_error_invalid_request` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn parse_error_parse_failed(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::parse_error_parse_failed
        todo!(
            "TODO: port guard `parse_error_parse_failed` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn parse_error_unknown(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::parse_error_unknown
        todo!(
            "TODO: port guard `parse_error_unknown` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn parse_error_untracked(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::parse_error_untracked
        todo!(
            "TODO: port guard `parse_error_untracked` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn probe_text_opening_trim(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::probe_text_opening_trim
        todo!(
            "TODO: port action `probe_text_opening_trim` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn reject_invalid_cursor_from_initialized(
        &mut self,
        _event: &EventNextRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::reject_invalid_cursor
        todo!(
            "TODO: port action `reject_invalid_cursor` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn reject_invalid_cursor_from_scanning(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::reject_invalid_cursor
        todo!(
            "TODO: port action `reject_invalid_cursor` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn reject_invalid_next_from_initialized(
        &mut self,
        _event: &EventNextRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::reject_invalid_next
        todo!(
            "TODO: port action `reject_invalid_next` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn reject_invalid_next_from_scanning(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::reject_invalid_next
        todo!(
            "TODO: port action `reject_invalid_next` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn rstrip_text_token(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::rstrip_text_token
        todo!(
            "TODO: port action `rstrip_text_token` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn scan_comment(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::scan_comment
        todo!(
            "TODO: port action `scan_comment` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn scan_mapping_bang_equal(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::scan_mapping_bang_equal
        todo!(
            "TODO: port action `scan_mapping_bang_equal` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn scan_mapping_close_curly(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::scan_mapping_close_curly
        todo!(
            "TODO: port action `scan_mapping_close_curly` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn scan_mapping_close_curly_bracket(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::scan_mapping_close_curly_bracket
        todo!(
            "TODO: port action `scan_mapping_close_curly_bracket` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn scan_mapping_close_expression(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::scan_mapping_close_expression
        todo!(
            "TODO: port action `scan_mapping_close_expression` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn scan_mapping_close_expression_trim(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::scan_mapping_close_expression_trim
        todo!(
            "TODO: port action `scan_mapping_close_expression_trim` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn scan_mapping_close_paren(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::scan_mapping_close_paren
        todo!(
            "TODO: port action `scan_mapping_close_paren` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn scan_mapping_close_square_bracket(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::scan_mapping_close_square_bracket
        todo!(
            "TODO: port action `scan_mapping_close_square_bracket` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn scan_mapping_close_statement(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::scan_mapping_close_statement
        todo!(
            "TODO: port action `scan_mapping_close_statement` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn scan_mapping_close_statement_trim(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::scan_mapping_close_statement_trim
        todo!(
            "TODO: port action `scan_mapping_close_statement_trim` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn scan_mapping_colon(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::scan_mapping_colon
        todo!(
            "TODO: port action `scan_mapping_colon` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn scan_mapping_comma(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::scan_mapping_comma
        todo!(
            "TODO: port action `scan_mapping_comma` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn scan_mapping_dot(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::scan_mapping_dot
        todo!(
            "TODO: port action `scan_mapping_dot` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn scan_mapping_equal_equal(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::scan_mapping_equal_equal
        todo!(
            "TODO: port action `scan_mapping_equal_equal` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn scan_mapping_equals(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::scan_mapping_equals
        todo!(
            "TODO: port action `scan_mapping_equals` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn scan_mapping_greater(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::scan_mapping_greater
        todo!(
            "TODO: port action `scan_mapping_greater` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn scan_mapping_greater_equal(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::scan_mapping_greater_equal
        todo!(
            "TODO: port action `scan_mapping_greater_equal` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn scan_mapping_less(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::scan_mapping_less
        todo!(
            "TODO: port action `scan_mapping_less` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn scan_mapping_less_equal(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::scan_mapping_less_equal
        todo!(
            "TODO: port action `scan_mapping_less_equal` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn scan_mapping_minus(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::scan_mapping_minus
        todo!(
            "TODO: port action `scan_mapping_minus` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn scan_mapping_open_curly_bracket(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::scan_mapping_open_curly_bracket
        todo!(
            "TODO: port action `scan_mapping_open_curly_bracket` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn scan_mapping_open_expression(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::scan_mapping_open_expression
        todo!(
            "TODO: port action `scan_mapping_open_expression` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn scan_mapping_open_expression_trim(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::scan_mapping_open_expression_trim
        todo!(
            "TODO: port action `scan_mapping_open_expression_trim` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn scan_mapping_open_paren(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::scan_mapping_open_paren
        todo!(
            "TODO: port action `scan_mapping_open_paren` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn scan_mapping_open_square_bracket(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::scan_mapping_open_square_bracket
        todo!(
            "TODO: port action `scan_mapping_open_square_bracket` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn scan_mapping_open_statement(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::scan_mapping_open_statement
        todo!(
            "TODO: port action `scan_mapping_open_statement` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn scan_mapping_open_statement_trim(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::scan_mapping_open_statement_trim
        todo!(
            "TODO: port action `scan_mapping_open_statement_trim` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn scan_mapping_percent(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::scan_mapping_percent
        todo!(
            "TODO: port action `scan_mapping_percent` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn scan_mapping_pipe(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::scan_mapping_pipe
        todo!(
            "TODO: port action `scan_mapping_pipe` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn scan_mapping_plus(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::scan_mapping_plus
        todo!(
            "TODO: port action `scan_mapping_plus` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn scan_mapping_slash(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::scan_mapping_slash
        todo!(
            "TODO: port action `scan_mapping_slash` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn scan_mapping_star(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::scan_mapping_star
        todo!(
            "TODO: port action `scan_mapping_star` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn scan_mapping_tilde(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::scan_mapping_tilde
        todo!(
            "TODO: port action `scan_mapping_tilde` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn scan_no_token_eof(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::scan_no_token_eof
        todo!(
            "TODO: port guard `scan_no_token_eof` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn scan_numeric(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::scan_numeric
        todo!(
            "TODO: port action `scan_numeric` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn scan_spaces_from_comment_candidate_decision(
        &mut self,
        _event: &EventNextRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::scan_spaces
        todo!(
            "TODO: port action `scan_spaces` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn scan_spaces_from_trim_prefix_scan_exec(
        &mut self,
        _event: &EventNextRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::scan_spaces
        todo!(
            "TODO: port action `scan_spaces` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn scan_string_content(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::scan_string_content
        todo!(
            "TODO: port action `scan_string_content` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn scan_text_boundary(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::scan_text_boundary
        todo!(
            "TODO: port action `scan_text_boundary` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn scan_token_available(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::scan_token_available
        todo!(
            "TODO: port guard `scan_token_available` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn scan_trim_prefix(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::scan_trim_prefix
        todo!(
            "TODO: port action `scan_trim_prefix` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn scan_unary(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::scan_unary
        todo!(
            "TODO: port action `scan_unary` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn scan_unhandled(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::scan_unhandled
        todo!(
            "TODO: port guard `scan_unhandled` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn scan_word(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::scan_word
        todo!(
            "TODO: port action `scan_word` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn starts_comment(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::starts_comment
        todo!(
            "TODO: port guard `starts_comment` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn starts_numeric(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::starts_numeric
        todo!(
            "TODO: port guard `starts_numeric` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn starts_string(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::starts_string
        todo!(
            "TODO: port guard `starts_string` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn starts_trim_prefix(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::starts_trim_prefix
        todo!(
            "TODO: port guard `starts_trim_prefix` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn starts_word(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::starts_word
        todo!(
            "TODO: port guard `starts_word` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn string_not_terminated(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::string_not_terminated
        todo!(
            "TODO: port guard `string_not_terminated` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn string_scan_immediate_termination_or_eof(
        &self,
        _event: &EventNextRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::string_scan_immediate_termination_or_eof
        todo!(
            "TODO: port guard `string_scan_immediate_termination_or_eof` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn string_scan_requires_content(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::string_scan_requires_content
        todo!(
            "TODO: port guard `string_scan_requires_content` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn string_terminated(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::string_terminated
        todo!(
            "TODO: port guard `string_terminated` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn text_apply_lstrip_and_rstrip(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::text_apply_lstrip_and_rstrip
        todo!(
            "TODO: port guard `text_apply_lstrip_and_rstrip` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn text_apply_lstrip_only(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::text_apply_lstrip_only
        todo!(
            "TODO: port guard `text_apply_lstrip_only` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn text_apply_no_strip(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::text_apply_no_strip
        todo!(
            "TODO: port guard `text_apply_no_strip` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn text_apply_rstrip_only(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::text_apply_rstrip_only
        todo!(
            "TODO: port guard `text_apply_rstrip_only` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn text_boundary_empty_at_end(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::text_boundary_empty_at_end
        todo!(
            "TODO: port guard `text_boundary_empty_at_end` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn text_can_trim_leading_newline(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::text_can_trim_leading_newline
        todo!(
            "TODO: port guard `text_can_trim_leading_newline` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn text_opening_block_ahead(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::text_opening_block_ahead
        todo!(
            "TODO: port guard `text_opening_block_ahead` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn text_opening_trim_stopped_on_newline(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::text_opening_trim_stopped_on_newline
        todo!(
            "TODO: port guard `text_opening_trim_stopped_on_newline` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn text_opening_trim_to_zero(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::text_opening_trim_to_zero
        todo!(
            "TODO: port guard `text_opening_trim_to_zero` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn text_plain_boundary_ready(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::text_plain_boundary_ready
        todo!(
            "TODO: port guard `text_plain_boundary_ready` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn text_token_empty_at_end(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::text_token_empty_at_end
        todo!(
            "TODO: port guard `text_token_empty_at_end` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn text_token_non_empty(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::text_token_non_empty
        todo!(
            "TODO: port guard `text_token_non_empty` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn trim_text_leading_newline(&mut self, _event: &EventNextRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp::trim_text_leading_newline
        todo!(
            "TODO: port action `trim_text_leading_newline` from emel.cpp/src/emel/text/jinja/parser/lexer/actions.hpp"
        )
    }
    fn unary_candidate(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::unary_candidate
        todo!(
            "TODO: port guard `unary_candidate` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn unary_numeric_suffix_present(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::unary_numeric_suffix_present
        todo!(
            "TODO: port guard `unary_numeric_suffix_present` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn unary_prefix_context_invalid(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::unary_prefix_context_invalid
        todo!(
            "TODO: port guard `unary_prefix_context_invalid` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
    fn unary_prefix_disallowed(&self, _event: &EventNextRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp::unary_prefix_disallowed
        todo!(
            "TODO: port guard `unary_prefix_disallowed` from emel.cpp/src/emel/text/jinja/parser/lexer/guards.hpp"
        )
    }
}
