//! Source-aligned bounded TextJinjaParser state machine.
//!
//! The root actor owns request validation, source normalization, lexer token
//! accumulation, and explicit hand-off to the maintained program parser.

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

use core::cell::RefCell;
use sml::sml;

use super::lexer::sm::{
    Cursor, EventNextRuntime, NextDone, NextError, TextJinjaParserLexer, TextJinjaParserLexerContext,
    Token as LexerToken, TokenKind,
};
use super::program_parser::sm::{
    EventParseRuntime as ProgramRuntime, ProgramParser, ProgramParserError, ProgramParserInput,
    Token as ProgramToken,
};

/// Maximum UTF-8 bytes copied from one parse request.
pub const MAX_SOURCE_BYTES: usize = 16 * 1024;
/// Maximum lexer tokens retained by the root parser.
pub const MAX_PARSE_TOKENS: usize = 512;

/// Root parser error values, matching `parser/errors.hpp`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(i32)]
pub enum ParseError {
    #[default]
    None = 0,
    InvalidRequest = 1,
    ParseFailed = 2,
    InternalError = 4,
    Untracked = 8,
    Unknown = -1,
}

/// Successful parse notification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ParsingDone {
    /// Number of lexer tokens retained for the completed parse.
    pub token_count: usize,
}

/// Failed parse notification.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ParsingError {
    /// Error selected by the root result decision.
    pub error: ParseError,
    /// Source position associated with the error.
    pub error_pos: usize,
}

/// Caller-owned completion callback.
pub type DoneCallback = fn(ParsingDone) -> bool;
/// Caller-owned error callback.
pub type ErrorCallback = fn(ParsingError) -> bool;

/// Bounded caller-owned parse output.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ParseResult {
    /// Lexer tokens retained by the root pipeline.
    pub tokens: [ProgramToken; MAX_PARSE_TOKENS],
    /// Number of initialized entries in [`Self::tokens`].
    pub token_count: usize,
    /// Number of program-parser emissions.
    pub emitted_count: u16,
    /// Root error status.
    pub error: ParseError,
    /// Position associated with [`Self::error`].
    pub error_pos: usize,
    /// Whether the request completed successfully.
    pub parsed: bool,
    /// Whether the request failed during parsing.
    pub failed: bool,
    /// Whether an out-of-topology event was received.
    pub unexpected: bool,
}

impl Default for ParseResult {
    fn default() -> Self {
        Self {
            tokens: [ProgramToken::default(); MAX_PARSE_TOKENS],
            token_count: 0,
            emitted_count: 0,
            error: ParseError::None,
            error_pos: 0,
            parsed: false,
            failed: false,
            unexpected: false,
        }
    }
}

impl ParseResult {
    fn reset(&mut self) { *self = Self::default(); }
    fn mark_error(&mut self, error: ParseError, error_pos: usize) {
        self.error = error;
        self.error_pos = error_pos;
        self.parsed = false;
        self.failed = true;
    }
}

/// Caller-owned parse request. The destination is never allocated or replaced
/// by the actor; only its bounded fields are reset and filled.
#[derive(Clone, Copy, Debug)]
pub struct ParseRequest<'event> {
    /// Template source.
    pub source: &'event str,
    /// Caller-owned bounded output.
    pub output: &'event RefCell<ParseResult>,
    /// Optional successful completion callback.
    pub dispatch_done: Option<DoneCallback>,
    /// Optional failed completion callback.
    pub dispatch_error: Option<ErrorCallback>,
}


impl<'event> ParseRequest<'event> {
    /// Constructs a request with both callbacks installed.
    #[must_use]
    pub const fn new(
        source: &'event str,
        output: &'event RefCell<ParseResult>,
        dispatch_done: DoneCallback,
        dispatch_error: ErrorCallback,
    ) -> Self {
        Self { source, output, dispatch_done: Some(dispatch_done), dispatch_error: Some(dispatch_error) }
    }

    /// Constructs a request with independently optional callbacks.
    #[must_use]
    pub const fn with_callbacks(
        source: &'event str,
        output: &'event RefCell<ParseResult>,
        dispatch_done: Option<DoneCallback>,
        dispatch_error: Option<ErrorCallback>,
    ) -> Self {
        Self { source, output, dispatch_done, dispatch_error }
    }
}

/// Runtime event corresponding to the pinned `event::parse_runtime`.
#[derive(Clone, Copy, Debug)]
pub struct EventParseRuntime<'event> {
    /// Caller-owned request fields.
    pub request: ParseRequest<'event>,
}

impl<'event> EventParseRuntime<'event> {
    /// Constructs one runtime event.
    #[must_use]
    pub const fn new(request: ParseRequest<'event>) -> Self { Self { request } }
}

#[derive(Debug)]
struct LexCapture<'a> {
    token: Option<LexerToken>,
    has_token: bool,
    cursor: Cursor<'a>,
    error: ParseError,
    error_pos: usize,
}

impl<'a> LexCapture<'a> {
    fn new(cursor: Cursor<'a>) -> Self {
        Self { token: None, has_token: false, cursor, error: ParseError::InternalError, error_pos: 0 }
    }
}

/// Context retained by the generated root machine.
#[derive(Debug)]
pub struct TextJinjaParserContext {
    pub source: [u8; MAX_SOURCE_BYTES],
    pub source_len: usize,
    pub source_valid: bool,
    pub phase: ParsePhase,
    pub lex_offset: usize,
    pub lex_token_index: usize,
    pub curly_bracket_depth: usize,
    pub last_token_type: TokenKind,
    pub last_block_rstrip: bool,
    pub last_block_can_trim_newline: bool,
    pub lex_token: Option<LexerToken>,
    pub lex_has_token: bool,
    pub tokens: [ProgramToken; MAX_PARSE_TOKENS],
    pub token_count: usize,
    pub error: ParseError,
    pub error_pos: usize,
    pub result: ParseResult,
    pub lexer: TextJinjaParserLexer,
    pub program_parser: ProgramParser,
}

/// Parse phases from `event::parse_phase`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum ParsePhase { #[default] None = 0, RequestValidation = 1, Tokenization = 2, StatementClassification = 3, Parsing = 4 }

impl Default for TextJinjaParserContext {
    fn default() -> Self {
        Self {
            source: [0; MAX_SOURCE_BYTES], source_len: 0, source_valid: false,
            phase: ParsePhase::None, lex_offset: 0, lex_token_index: 0,
            curly_bracket_depth: 0, last_token_type: TokenKind::CloseStatement,
            last_block_rstrip: false, last_block_can_trim_newline: false,
            lex_token: None, lex_has_token: false,
            tokens: [ProgramToken::default(); MAX_PARSE_TOKENS], token_count: 0,
            error: ParseError::None, error_pos: 0, result: ParseResult::default(),
            lexer: TextJinjaParserLexer::new(TextJinjaParserLexerContext::default()),
            program_parser: ProgramParser::new(),
        }
    }
}

impl TextJinjaParserContext {
    fn reset(&mut self, request: &ParseRequest<'_>) {
        self.phase = ParsePhase::RequestValidation;
        self.error = ParseError::None;
        self.error_pos = 0;
        self.source_len = 0;
        self.source_valid = true;
        self.lex_offset = 0;
        self.lex_token_index = 0;
        self.curly_bracket_depth = 0;
        self.last_token_type = TokenKind::CloseStatement;
        self.last_block_rstrip = false;
        self.last_block_can_trim_newline = false;
        self.lex_token = None;
        self.lex_has_token = false;
        self.token_count = 0;
        self.result.reset();
        let bytes = request.source.as_bytes();
        if bytes.len() > MAX_SOURCE_BYTES { self.source_valid = false; return; }
        let mut previous_cr = false;
        for &byte in bytes {
            if byte == b'\n' && previous_cr { previous_cr = false; continue; }
            let normalized = if byte == b'\r' { previous_cr = true; b'\n' } else { previous_cr = false; byte };
            if self.source_len == MAX_SOURCE_BYTES { self.source_valid = false; return; }
            self.source[self.source_len] = normalized;
            self.source_len += 1;
        }
        if self.source_len > 0 && self.source[self.source_len - 1] == b'\n' { self.source_len -= 1; }
        if core::str::from_utf8(&self.source[..self.source_len]).is_err() { self.source_valid = false; }
        *request.output.borrow_mut() = ParseResult::default();
    }

    fn mark_error(&mut self, error: ParseError, pos: usize) {
        self.error = error;
        self.error_pos = pos;
        self.result.mark_error(error, pos);
    }

    fn source(&self) -> Option<&str> { core::str::from_utf8(&self.source[..self.source_len]).ok() }

    fn copy_lex_token(&mut self) -> bool {
        let Some(token) = self.lex_token.as_ref() else { return false; };
        if self.token_count >= MAX_PARSE_TOKENS { return false; }
        let value = token.value.as_bytes();
        self.tokens[self.token_count] = ProgramToken::new(token.kind, value, token.pos);
        self.token_count += 1;
        true
    }

    fn publish_result(&mut self) {
        self.result.token_count = self.token_count;
        self.result.tokens = self.tokens;
        self.result.error = self.error;
        self.result.error_pos = self.error_pos;
    }
}

sml! {
    TextJinjaParser<'event> {
        "request_decision"_s <= *"initialized"_s + event<EventParseRuntime<'event>> [valid_parse] / begin_parse_from_initialized,
        "parse_result_decision"_s <= "initialized"_s + event<EventParseRuntime<'event>> [invalid_parse_with_callbacks] / reject_invalid_parse_from_initialized,
        "errored"_s <= "initialized"_s + event<EventParseRuntime<'event>> [invalid_parse_without_callbacks] / reject_invalid_parse_from_initialized,
        "request_decision"_s <= "done"_s + event<EventParseRuntime<'event>> [valid_parse] / begin_parse_from_done,
        "parse_result_decision"_s <= "done"_s + event<EventParseRuntime<'event>> [invalid_parse_with_callbacks] / reject_invalid_parse_from_done,
        "errored"_s <= "done"_s + event<EventParseRuntime<'event>> [invalid_parse_without_callbacks] / reject_invalid_parse_from_done,
        "request_decision"_s <= "errored"_s + event<EventParseRuntime<'event>> [valid_parse] / begin_parse_from_errored,
        "parse_result_decision"_s <= "errored"_s + event<EventParseRuntime<'event>> [invalid_parse_with_callbacks] / reject_invalid_parse_from_errored,
        "errored"_s <= "errored"_s + event<EventParseRuntime<'event>> [invalid_parse_without_callbacks] / reject_invalid_parse_from_errored,
        "request_decision"_s <= "unexpected"_s + event<EventParseRuntime<'event>> [valid_parse] / begin_parse_from_unexpected,
        "parse_result_decision"_s <= "unexpected"_s + event<EventParseRuntime<'event>> [invalid_parse_with_callbacks] / reject_invalid_parse_from_unexpected,
        "errored"_s <= "unexpected"_s + event<EventParseRuntime<'event>> [invalid_parse_without_callbacks] / reject_invalid_parse_from_unexpected,
        "tokenize_begin"_s <= "request_decision"_s + completion<EventParseRuntime>(EventParseRuntime<'event>) / begin_tokenization,
        "tokenize_next"_s <= "tokenize_begin"_s + completion<EventParseRuntime>(EventParseRuntime<'event>) / request_next_lex_token_from_tokenize_begin,
        "tokenize_result_decision"_s <= "tokenize_next"_s + completion<EventParseRuntime>(EventParseRuntime<'event>),
        "program_parser_model"_s <= "tokenize_result_decision"_s + completion<EventParseRuntime>(EventParseRuntime<'event>) [lexer_at_eof] / run_program_parser,
        "tokenize_append"_s <= "tokenize_result_decision"_s + completion<EventParseRuntime>(EventParseRuntime<'event>) [lexer_has_token] / append_lex_token,
        "parse_result_decision"_s <= "tokenize_result_decision"_s + completion<EventParseRuntime>(EventParseRuntime<'event>) [parse_error_invalid_request] / commit_lex_error_from_tokenize_result_decision,
        "parse_result_decision"_s <= "tokenize_result_decision"_s + completion<EventParseRuntime>(EventParseRuntime<'event>) [parse_error_parse_failed] / commit_lex_error_from_tokenize_result_decision,
        "parse_result_decision"_s <= "tokenize_result_decision"_s + completion<EventParseRuntime>(EventParseRuntime<'event>) [parse_error_internal_error] / commit_lex_error_from_tokenize_result_decision,
        "parse_result_decision"_s <= "tokenize_result_decision"_s + completion<EventParseRuntime>(EventParseRuntime<'event>) [parse_error_untracked] / commit_lex_error_from_tokenize_result_decision,
        "parse_result_decision"_s <= "tokenize_result_decision"_s + completion<EventParseRuntime>(EventParseRuntime<'event>) [parse_error_unknown] / commit_lex_error_from_tokenize_result_decision,
        "tokenize_next"_s <= "tokenize_append"_s + completion<EventParseRuntime>(EventParseRuntime<'event>) / request_next_lex_token_from_tokenize_append,
        "parse_result_decision"_s <= "program_parser_model"_s + completion<EventParseRuntime>(EventParseRuntime<'event>),
        "done"_s <= "parse_result_decision"_s + completion<EventParseRuntime>(EventParseRuntime<'event>) [parse_error_none] / dispatch_done,
        "errored"_s <= "parse_result_decision"_s + completion<EventParseRuntime>(EventParseRuntime<'event>) [parse_error_invalid_request] / dispatch_error_from_parse_result_decision,
        "errored"_s <= "parse_result_decision"_s + completion<EventParseRuntime>(EventParseRuntime<'event>) [parse_error_parse_failed] / dispatch_error_from_parse_result_decision,
        "errored"_s <= "parse_result_decision"_s + completion<EventParseRuntime>(EventParseRuntime<'event>) [parse_error_internal_error] / dispatch_error_from_parse_result_decision,
        "errored"_s <= "parse_result_decision"_s + completion<EventParseRuntime>(EventParseRuntime<'event>) [parse_error_untracked] / dispatch_error_from_parse_result_decision,
        "errored"_s <= "parse_result_decision"_s + completion<EventParseRuntime>(EventParseRuntime<'event>) [parse_error_unknown] / dispatch_error_from_parse_result_decision,
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

impl TextJinjaParserStateMachineContext for TextJinjaParserContext {
    fn append_lex_token(&mut self, _event: &EventParseRuntime<'_>) -> Result<(), ()> {
        if !self.copy_lex_token() { self.mark_error(ParseError::InvalidRequest, self.lex_offset); }
        Ok(())
    }
    fn begin_parse_from_done(&mut self, event: &EventParseRuntime<'_>) -> Result<(), ()> { self.reset(&event.request); Ok(()) }
    fn begin_parse_from_errored(&mut self, event: &EventParseRuntime<'_>) -> Result<(), ()> { self.reset(&event.request); Ok(()) }
    fn begin_parse_from_initialized(&mut self, event: &EventParseRuntime<'_>) -> Result<(), ()> { self.reset(&event.request); Ok(()) }
    fn begin_parse_from_unexpected(&mut self, event: &EventParseRuntime<'_>) -> Result<(), ()> { self.reset(&event.request); Ok(()) }
    fn begin_tokenization(&mut self, _event: &EventParseRuntime<'_>) -> Result<(), ()> {
        self.phase = ParsePhase::Tokenization;
        self.lex_offset = 0; self.lex_token_index = 0; self.curly_bracket_depth = 0;
        self.last_token_type = TokenKind::CloseStatement;
        self.last_block_rstrip = false; self.last_block_can_trim_newline = false;
        self.lex_token = None; self.lex_has_token = false; self.token_count = 0;
        Ok(())
    }
    fn request_next_lex_token_from_tokenize_begin(&mut self, _event: &EventParseRuntime<'_>) -> Result<(), ()> { self.next_lex_token(); Ok(()) }
    fn request_next_lex_token_from_tokenize_append(&mut self, _event: &EventParseRuntime<'_>) -> Result<(), ()> { self.next_lex_token(); Ok(()) }
    fn commit_lex_error_from_tokenize_result_decision(&mut self, event: &EventParseRuntime<'_>) -> Result<(), ()> {
        self.mark_error(self.error, self.error_pos);
        let mut output = event.request.output.borrow_mut(); output.error = self.error; output.error_pos = self.error_pos;
        Ok(())
    }
    fn dispatch_done(&mut self, event: &EventParseRuntime<'_>) -> Result<(), ()> {
        self.result.parsed = true; self.result.failed = false; self.result.error = ParseError::None; self.result.error_pos = 0; self.publish_result();
        *event.request.output.borrow_mut() = self.result;
        if let Some(callback) = event.request.dispatch_done { let _ = callback(ParsingDone { token_count: self.token_count }); }
        Ok(())
    }
    fn dispatch_error_from_parse_result_decision(&mut self, event: &EventParseRuntime<'_>) -> Result<(), ()> {
        self.result.failed = true; self.result.parsed = false; self.publish_result(); *event.request.output.borrow_mut() = self.result;
        if let Some(callback) = event.request.dispatch_error { let _ = callback(ParsingError { error: self.error, error_pos: self.error_pos }); }
        Ok(())
    }

    fn invalid_parse_with_callbacks(&self, event: &EventParseRuntime<'_>) -> Result<bool, ()> { Ok(!valid_request(&event.request) && callbacks_present(&event.request)) }
    fn invalid_parse_without_callbacks(&self, event: &EventParseRuntime<'_>) -> Result<bool, ()> { Ok(!callbacks_present(&event.request)) }
    fn lexer_at_eof(&self, _event: &EventParseRuntime<'_>) -> Result<bool, ()> { Ok(self.error == ParseError::None && !self.lex_has_token) }
    fn lexer_has_token(&self, _event: &EventParseRuntime<'_>) -> Result<bool, ()> { Ok(self.error == ParseError::None && self.lex_has_token) }

    fn on_unexpected_from_done(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_errored(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_initialized(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_parse_result_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_request_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_tokenize_append(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_tokenize_begin(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_tokenize_next(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_tokenize_result_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_unexpected(&mut self) -> Result<(), ()> { self.unexpected() }

    fn parse_error_internal_error(&self, _event: &EventParseRuntime<'_>) -> Result<bool, ()> { Ok(self.error == ParseError::InternalError) }
    fn parse_error_invalid_request(&self, _event: &EventParseRuntime<'_>) -> Result<bool, ()> { Ok(self.error == ParseError::InvalidRequest) }
    fn parse_error_none(&self, _event: &EventParseRuntime<'_>) -> Result<bool, ()> { Ok(self.error == ParseError::None) }
    fn parse_error_parse_failed(&self, _event: &EventParseRuntime<'_>) -> Result<bool, ()> { Ok(self.error == ParseError::ParseFailed) }
    fn parse_error_unknown(&self, _event: &EventParseRuntime<'_>) -> Result<bool, ()> { Ok(!matches!(self.error, ParseError::None | ParseError::InvalidRequest | ParseError::ParseFailed | ParseError::InternalError | ParseError::Untracked)) }
    fn parse_error_untracked(&self, _event: &EventParseRuntime<'_>) -> Result<bool, ()> { Ok(self.error == ParseError::Untracked) }

    fn reject_invalid_parse_from_done(&mut self, event: &EventParseRuntime<'_>) -> Result<(), ()> { self.reject_invalid(&event.request); Ok(()) }
    fn reject_invalid_parse_from_errored(&mut self, event: &EventParseRuntime<'_>) -> Result<(), ()> { self.reject_invalid(&event.request); Ok(()) }
    fn reject_invalid_parse_from_initialized(&mut self, event: &EventParseRuntime<'_>) -> Result<(), ()> { self.reject_invalid(&event.request); Ok(()) }
    fn reject_invalid_parse_from_unexpected(&mut self, event: &EventParseRuntime<'_>) -> Result<(), ()> { self.reject_invalid(&event.request); Ok(()) }

    fn valid_parse(&self, event: &EventParseRuntime<'_>) -> Result<bool, ()> { Ok(valid_request(&event.request) && callbacks_present(&event.request)) }

    fn run_program_parser(&mut self, _event: &EventParseRuntime<'_>) -> Result<(), ()> {
        self.phase = ParsePhase::Parsing;
        let input = ProgramParserInput::from_tokens(&self.tokens[..self.token_count]);
        let child = self.program_parser.process_event(ProgramRuntime::from(input));
        self.error = match child.error { ProgramParserError::None => ParseError::None, ProgramParserError::InvalidRequest => ParseError::InvalidRequest, ProgramParserError::ParseFailed => ParseError::ParseFailed, ProgramParserError::InternalError => ParseError::InternalError, ProgramParserError::Untracked => ParseError::Untracked, ProgramParserError::Unknown => ParseError::Unknown };
        self.error_pos = child.error_pos as usize;
        self.result.emitted_count = child.emitted_count;
        if self.error == ParseError::None { self.result.parsed = child.parsed; }
        Ok(())
    }
}

impl TextJinjaParserContext {
    fn next_lex_token(&mut self) {
        let source = match core::str::from_utf8(&self.source[..self.source_len]) {
            Ok(source) => source,
            Err(_) => { self.mark_error(ParseError::ParseFailed, 0); return; }
        };
        let cursor = Cursor { source, offset: self.lex_offset, token_index: self.lex_token_index, curly_bracket_depth: self.curly_bracket_depth, last_token_type: self.last_token_type, last_block_rstrip: self.last_block_rstrip, last_block_can_trim_newline: self.last_block_can_trim_newline };
        let mut capture = LexCapture::new(cursor);
        let mut done = |event: NextDone<'_>| { capture.has_token = event.has_token; capture.cursor = event.next_cursor; capture.token = Some(event.token); capture.error = ParseError::None; true };
        let mut error = |event: NextError| { capture.error = match event.err { super::lexer::sm::Error::None => ParseError::None, super::lexer::sm::Error::InvalidRequest => ParseError::InvalidRequest, super::lexer::sm::Error::ParseFailed => ParseError::ParseFailed, super::lexer::sm::Error::InternalError => ParseError::InternalError, super::lexer::sm::Error::Untracked => ParseError::Untracked }; capture.error_pos = event.error_pos; true };
        let request = EventNextRuntime::new(cursor, &mut done, &mut error);
        let _ = self.lexer.process_event(request);
        self.error = capture.error;
        self.error_pos = capture.error_pos;
        self.lex_has_token = capture.has_token;
        self.lex_token = capture.token;
        self.lex_offset = capture.cursor.offset;
        self.lex_token_index = capture.cursor.token_index;
        self.curly_bracket_depth = capture.cursor.curly_bracket_depth;
        self.last_token_type = capture.cursor.last_token_type;
        self.last_block_rstrip = capture.cursor.last_block_rstrip;
        self.last_block_can_trim_newline = capture.cursor.last_block_can_trim_newline;
    }

    fn unexpected(&mut self) -> Result<(), ()> { self.result.unexpected = true; self.mark_error(ParseError::InternalError, self.error_pos); Ok(()) }
    fn reject_invalid(&mut self, output: &RefCell<ParseResult>) { self.mark_error(ParseError::InvalidRequest, 0); self.result.unexpected = false; *output.borrow_mut() = self.result; }
}

fn valid_request(request: &ParseRequest<'_>) -> bool { !request.source.is_empty() && request.source.len() <= MAX_SOURCE_BYTES }
fn callbacks_present(request: &ParseRequest<'_>) -> bool { request.dispatch_done.is_some() && request.dispatch_error.is_some() }

/// Synchronous single-writer actor around the generated root parser.
pub struct TextJinjaParserActor<'event> {
    machine: TextJinjaParserStateMachine<'event, TextJinjaParserContext>,
}

impl<'event> Default for TextJinjaParserActor<'event> { fn default() -> Self { Self::new() } }

impl<'event> TextJinjaParserActor<'event> {
    /// Creates an actor in the generated `initialized` state.
    #[must_use]
    pub fn new() -> Self { Self { machine: TextJinjaParserStateMachine::new(TextJinjaParserContext::default()) } }

    /// Processes one request synchronously through validation, tokenization,
    /// child parsing, and completion dispatch.
    pub fn process_event(&mut self, event: EventParseRuntime<'event>) -> bool {
        self.machine.process_event(TextJinjaParserEvents::EventParseRuntime(event)).is_ok()
            && self.machine.context().error == ParseError::None
    }

    /// Dispatches an explicit unexpected event through the generated machine.
    pub fn process_unexpected(&mut self) -> bool {
        let _ = self.machine.context_mut().unexpected();
        self.machine.set_state(TextJinjaParserStates::Unexpected);
        false
    }

    /// Returns generated state inspection data.
    #[must_use]
    pub fn state(&self) -> &TextJinjaParserStates { self.machine.state() }
    /// Reports whether the generated machine is in `state`.
    #[must_use]
    pub fn is(&self, state: &TextJinjaParserStates) -> bool { self.machine.is(state) }
    /// Returns the bounded root context.
    #[must_use]
    pub fn context(&self) -> &TextJinjaParserContext { self.machine.context() }
}

/// Short alias matching the pinned parser name.
pub type Parser<'event> = TextJinjaParserActor<'event>;
