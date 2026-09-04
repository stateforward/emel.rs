//! Bounded, synchronous Jinja statement-parser state machine.
//!
//! The topology mirrors the maintained statement-parser machine: identify a
//! supported statement name, scan to the closing delimiter, emit one bounded
//! statement result, or record a precise parse failure. Inputs are copied into
//! fixed-size storage before the generated SML machine is dispatched.

#![allow(
    clippy::cast_possible_truncation,
    clippy::derivable_impls,
    clippy::large_types_passed_by_value,
    clippy::large_stack_frames,
    clippy::large_stack_arrays,
    clippy::elidable_lifetime_names,
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

/// Maximum number of lexer tokens copied into one statement-parser request.
pub const MAX_STATEMENT_TOKENS: usize = 256;
/// Maximum UTF-8 bytes copied from one token value.
pub const MAX_TOKEN_VALUE_BYTES: usize = 64;

/// Token categories used by the maintained Jinja lexer.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum TokenType {
    #[default]
    Eof = 0,
    Text = 1,
    NumericLiteral = 2,
    StringLiteral = 3,
    Identifier = 4,
    Equals = 5,
    OpenParen = 6,
    CloseParen = 7,
    OpenStatement = 8,
    CloseStatement = 9,
    OpenExpression = 10,
    CloseExpression = 11,
    OpenSquareBracket = 12,
    CloseSquareBracket = 13,
    OpenCurlyBracket = 14,
    CloseCurlyBracket = 15,
    Comma = 16,
    Dot = 17,
    Colon = 18,
    Pipe = 19,
    CallOperator = 20,
    AdditiveBinaryOperator = 21,
    MultiplicativeBinaryOperator = 22,
    ComparisonBinaryOperator = 23,
    UnaryOperator = 24,
    Comment = 25,
}

/// Alias matching the source token spelling used by parser clients.
pub type TokenKind = TokenType;

/// A bounded copy of one lexer token.
#[allow(
    clippy::struct_field_names,
    reason = "public token_type and value_* fields preserve the source parser token contract"
)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Token {
    /// Token category.
    pub token_type: TokenType,
    /// UTF-8 bytes copied from the lexer value.
    pub value: [u8; MAX_TOKEN_VALUE_BYTES],
    /// Number of valid bytes in [`Self::value`].
    pub value_len: u8,
    /// Whether the complete source value fit in the bounded copy.
    pub value_valid: bool,
    /// Source position.
    pub pos: usize,
}

impl Default for Token {
    fn default() -> Self {
        Self {
            token_type: TokenType::Eof,
            value: [0; MAX_TOKEN_VALUE_BYTES],
            value_len: 0,
            value_valid: true,
            pos: 0,
        }
    }
}

impl Token {
    /// Copies a lexer token value, rejecting values that exceed the bound.
    #[must_use]
    pub fn new(token_type: TokenType, value: &str, pos: usize) -> Self {
        let bytes = value.as_bytes();
        let copied = bytes.len().min(MAX_TOKEN_VALUE_BYTES);
        let mut token = Self {
            token_type,
            value: [0; MAX_TOKEN_VALUE_BYTES],
            value_len: u8::try_from(copied).expect("bounded token value fits in u8"),
            value_valid: bytes.len() <= MAX_TOKEN_VALUE_BYTES,
            pos,
        };
        token.value[..copied].copy_from_slice(&bytes[..copied]);
        token
    }

    /// Creates a delimiter or other value-less token.
    #[must_use]
    pub const fn kind(token_type: TokenType, pos: usize) -> Self {
        Self {
            token_type,
            value: [0; MAX_TOKEN_VALUE_BYTES],
            value_len: 0,
            value_valid: true,
            pos,
        }
    }

    /// Returns the copied value when it is complete and valid UTF-8.
    #[must_use]
    pub fn value_str(&self) -> Option<&str> {
        if !self.value_valid {
            return None;
        }
        core::str::from_utf8(&self.value[..self.value_len as usize]).ok()
    }
}

/// Statement kinds from the parser event contract.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum StatementKind {
    #[default]
    Unknown = 0,
    Text = 1,
    Comment = 2,
    Expression = 3,
    Statement = 4,
}

/// Parser error values from `parser/errors.hpp`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum StatementParserError {
    #[default]
    None = 0,
    InvalidRequest = 1,
    ParseFailed = 2,
    InternalError = 3,
    Untracked = 4,
}

/// Compatibility spelling used by parser-facing callers.
pub type ParseError = StatementParserError;

/// Owned result produced by one statement-parser dispatch.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct StatementParseResult {
    /// Whether parsing reached the source `parsed` terminal state.
    pub parsed: bool,
    /// Whether parsing rejected the request or statement syntax.
    pub failed: bool,
    /// Whether an explicit unexpected event was observed.
    pub unexpected: bool,
    /// Source-aligned error category.
    pub error: StatementParserError,
    /// Position associated with `error`.
    pub error_pos: usize,
    /// Number of no-op statement nodes emitted by this parser.
    pub emitted_count: u16,
    /// Position of the emitted statement token.
    pub emitted_pos: usize,
}

impl StatementParseResult {
    const fn initial() -> Self {
        Self {
            parsed: false,
            failed: false,
            unexpected: false,
            error: StatementParserError::None,
            error_pos: 0,
            emitted_count: 0,
            emitted_pos: 0,
        }
    }
}

/// Copied, bounded equivalent of the source `parse_runtime` input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EventParseRuntime {
    /// Copied lexer tokens.
    pub tokens: [Token; MAX_STATEMENT_TOKENS],
    /// Number of valid entries in [`Self::tokens`].
    pub token_count: u16,
    /// Current token cursor.
    pub token_index: u16,
    /// Start of the current statement block.
    pub statement_start: u16,
    /// Statement category selected by the kind decision.
    pub statement: StatementKind,
    /// Result fields copied alongside the input.
    pub result: StatementParseResult,
}

impl Default for EventParseRuntime {
    fn default() -> Self {
        Self {
            tokens: [Token::default(); MAX_STATEMENT_TOKENS],
            token_count: 0,
            token_index: 0,
            statement_start: 0,
            statement: StatementKind::Unknown,
            result: StatementParseResult::initial(),
        }
    }
}

impl EventParseRuntime {
    /// Copies a token slice into bounded actor input.
    #[must_use]
    pub fn from_tokens(tokens: &[Token]) -> Self {
        let copied = tokens.len().min(MAX_STATEMENT_TOKENS);
        let mut runtime = Self::default();
        runtime.tokens[..copied].copy_from_slice(&tokens[..copied]);
        runtime.token_count =
            u16::try_from(copied).expect("bounded statement token count fits in u16");
        runtime
    }

    /// Creates a common `{% <name> ... %}` request from copied tokens.
    #[must_use]
    pub fn statement(name: &str, body: &[Token]) -> Self {
        let mut runtime = Self::default();
        let mut count = 0usize;
        if count < MAX_STATEMENT_TOKENS {
            runtime.tokens[count] = Token::kind(TokenType::OpenStatement, 0);
            count += 1;
        }
        if count < MAX_STATEMENT_TOKENS {
            runtime.tokens[count] = Token::new(TokenType::Identifier, name, 2);
            count += 1;
        }
        let room = MAX_STATEMENT_TOKENS.saturating_sub(count + 1);
        let body_count = body.len().min(room);
        runtime.tokens[count..count + body_count].copy_from_slice(&body[..body_count]);
        count += body_count;
        if count < MAX_STATEMENT_TOKENS {
            runtime.tokens[count] = Token::kind(TokenType::CloseStatement, 2 + name.len());
            count += 1;
        }
        runtime.token_count =
            u16::try_from(count).expect("bounded statement token count fits in u16");
        runtime
    }
}

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

/// Mutable bounded context retained by the generated machine.
#[derive(Debug)]
pub struct TextJinjaParserProgramParserStatementParserContext {
    /// Copied request and parser cursor.
    pub input: EventParseRuntime,
    /// Owned result for inspection after dispatch.
    pub result: StatementParseResult,
}

impl Default for TextJinjaParserProgramParserStatementParserContext {
    fn default() -> Self {
        Self {
            input: EventParseRuntime::default(),
            result: StatementParseResult::initial(),
        }
    }
}

impl TextJinjaParserProgramParserStatementParserContext {
    fn current(&self) -> Option<&Token> {
        let index = self.input.token_index as usize;
        (index < self.input.token_count as usize).then(|| &self.input.tokens[index])
    }

    fn at(&self, offset: usize) -> Option<&Token> {
        let index = self.input.token_index as usize + offset;
        (index < self.input.token_count as usize).then(|| &self.input.tokens[index])
    }

    fn statement_name_is(&self, expected: &str) -> bool {
        self.at(1).and_then(Token::value_str).is_some_and(|value| {
            self.at(1)
                .is_some_and(|token| token.token_type == TokenType::Identifier)
                && value == expected
        })
    }

    fn fail(&mut self, error: StatementParserError, pos: usize) -> Result<(), ()> {
        self.result.failed = true;
        self.result.error = error;
        self.result.error_pos = pos;
        Ok(())
    }

    fn unexpected(&mut self) -> Result<(), ()> {
        self.result.unexpected = true;
        self.result.error = StatementParserError::InternalError;
        Ok(())
    }
}

impl TextJinjaParserProgramParserStatementParserStateMachineContext
    for TextJinjaParserProgramParserStatementParserContext
{
    fn begin_statement_scan_from_statement_kind_decision(&mut self) -> Result<(), ()> {
        self.input.statement = StatementKind::Statement;
        self.input.statement_start = self.input.token_index;
        self.input.token_index = self.input.token_index.saturating_add(2);
        Ok(())
    }

    fn consume_statement_close_and_emit(&mut self) -> Result<(), ()> {
        let start = self.input.statement_start as usize;
        if let Some(token) = self.input.tokens.get(start) {
            self.result.emitted_count = self.result.emitted_count.saturating_add(1);
            self.result.emitted_pos = token.pos;
        }
        self.input.token_index = self.input.token_index.saturating_add(1);
        self.result.parsed = true;
        Ok(())
    }

    fn consume_statement_token(&mut self) -> Result<(), ()> {
        if (self.input.token_index as usize) < self.input.token_count as usize {
            self.input.token_index = self.input.token_index.saturating_add(1);
        }
        Ok(())
    }

    fn fail_statement_name_token(&mut self) -> Result<(), ()> {
        let pos = self.at(1).map_or(0, |token| token.pos);
        self.fail(StatementParserError::ParseFailed, pos)
    }

    fn fail_statement_open_token(&mut self) -> Result<(), ()> {
        let pos = self.current().map_or(0, |token| token.pos);
        self.fail(StatementParserError::ParseFailed, pos)
    }

    fn fail_statement_start_token(&mut self) -> Result<(), ()> {
        let pos = self
            .input
            .tokens
            .get(self.input.statement_start as usize)
            .map_or(0, |token| token.pos);
        self.fail(StatementParserError::ParseFailed, pos)
    }

    fn on_unexpected_from_deciding(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn on_unexpected_from_parse_failed(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn on_unexpected_from_parsed(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn on_unexpected_from_statement_kind_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn on_unexpected_from_statement_scan(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn on_unexpected_from_unexpected_event(&mut self) -> Result<(), ()> {
        self.unexpected()
    }

    fn statement_identifier_missing(&self) -> Result<bool, ()> {
        Ok(!self
            .at(1)
            .is_some_and(|token| token.token_type == TokenType::Identifier))
    }

    fn statement_name_break(&self) -> Result<bool, ()> {
        Ok(self.statement_name_is("break"))
    }
    fn statement_name_call(&self) -> Result<bool, ()> {
        Ok(self.statement_name_is("call"))
    }
    fn statement_name_continue(&self) -> Result<bool, ()> {
        Ok(self.statement_name_is("continue"))
    }
    fn statement_name_elif(&self) -> Result<bool, ()> {
        Ok(self.statement_name_is("elif"))
    }
    fn statement_name_else(&self) -> Result<bool, ()> {
        Ok(self.statement_name_is("else"))
    }
    fn statement_name_endcall(&self) -> Result<bool, ()> {
        Ok(self.statement_name_is("endcall"))
    }
    fn statement_name_endfilter(&self) -> Result<bool, ()> {
        Ok(self.statement_name_is("endfilter"))
    }
    fn statement_name_endfor(&self) -> Result<bool, ()> {
        Ok(self.statement_name_is("endfor"))
    }
    fn statement_name_endgeneration(&self) -> Result<bool, ()> {
        Ok(self.statement_name_is("endgeneration"))
    }
    fn statement_name_endif(&self) -> Result<bool, ()> {
        Ok(self.statement_name_is("endif"))
    }
    fn statement_name_endmacro(&self) -> Result<bool, ()> {
        Ok(self.statement_name_is("endmacro"))
    }
    fn statement_name_endset(&self) -> Result<bool, ()> {
        Ok(self.statement_name_is("endset"))
    }
    fn statement_name_filter(&self) -> Result<bool, ()> {
        Ok(self.statement_name_is("filter"))
    }
    fn statement_name_for(&self) -> Result<bool, ()> {
        Ok(self.statement_name_is("for"))
    }
    fn statement_name_generation(&self) -> Result<bool, ()> {
        Ok(self.statement_name_is("generation"))
    }
    fn statement_name_if(&self) -> Result<bool, ()> {
        Ok(self.statement_name_is("if"))
    }
    fn statement_name_macro(&self) -> Result<bool, ()> {
        Ok(self.statement_name_is("macro"))
    }
    fn statement_name_set(&self) -> Result<bool, ()> {
        Ok(self.statement_name_is("set"))
    }

    fn statement_name_unknown(&self) -> Result<bool, ()> {
        let Some(token) = self.at(1) else {
            return Ok(false);
        };
        if token.token_type != TokenType::Identifier {
            return Ok(false);
        }
        Ok(![
            "set",
            "if",
            "elif",
            "else",
            "endif",
            "for",
            "endfor",
            "macro",
            "endmacro",
            "call",
            "endcall",
            "filter",
            "endfilter",
            "break",
            "continue",
            "generation",
            "endgeneration",
            "endset",
        ]
        .into_iter()
        .any(|name| self.statement_name_is(name)))
    }

    fn statement_scan_at_close(&self) -> Result<bool, ()> {
        Ok(self
            .current()
            .is_some_and(|token| token.token_type == TokenType::CloseStatement))
    }

    fn statement_scan_continue(&self) -> Result<bool, ()> {
        Ok(self
            .current()
            .is_some_and(|token| token.token_type != TokenType::CloseStatement))
    }

    fn statement_scan_eof(&self) -> Result<bool, ()> {
        Ok(self.current().is_none())
    }
}

/// Synchronous single-writer actor around the generated statement parser.
pub struct TextJinjaParserProgramParserStatementParserActor {
    machine: TextJinjaParserProgramParserStatementParserStateMachine<
        TextJinjaParserProgramParserStatementParserContext,
    >,
}

impl Default for TextJinjaParserProgramParserStatementParserActor {
    fn default() -> Self {
        Self::new()
    }
}

impl TextJinjaParserProgramParserStatementParserActor {
    /// Creates an actor in the generated `deciding` state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: TextJinjaParserProgramParserStatementParserStateMachine::new(
                TextJinjaParserProgramParserStatementParserContext::default(),
            ),
        }
    }

    /// Copies and dispatches one bounded runtime request to completion.
    pub fn process_event(&mut self, event: EventParseRuntime) -> StatementParseResult {
        if !self
            .machine
            .is(&TextJinjaParserProgramParserStatementParserStates::Deciding)
        {
            self.machine.context_mut().result.unexpected = true;
            self.machine.context_mut().result.error = StatementParserError::InternalError;
            self.machine
                .set_state(TextJinjaParserProgramParserStatementParserStates::UnexpectedEvent);
            return self.machine.context().result;
        }
        self.machine.context_mut().input = event;
        self.machine.context_mut().result = StatementParseResult::initial();
        let _ = self
            .machine
            .process_event(TextJinjaParserProgramParserStatementParserEvents::EventParseRuntime);
        self.machine.context().result
    }

    /// Returns generated state inspection data.
    #[must_use]
    pub fn state(&self) -> &TextJinjaParserProgramParserStatementParserStates {
        self.machine.state()
    }

    /// Reports whether the generated machine is in `state`.
    #[must_use]
    pub fn is(&self, state: &TextJinjaParserProgramParserStatementParserStates) -> bool {
        self.machine.is(state)
    }

    /// Returns the copied input and generated-state bookkeeping.
    #[must_use]
    pub fn context(&self) -> &TextJinjaParserProgramParserStatementParserContext {
        self.machine.context()
    }

    /// Explicitly records an unexpected event and transitions to its state.
    pub fn process_unexpected(&mut self) -> StatementParseResult {
        self.machine.context_mut().result.unexpected = true;
        self.machine.context_mut().result.error = StatementParserError::InternalError;
        self.machine
            .set_state(TextJinjaParserProgramParserStatementParserStates::UnexpectedEvent);
        self.machine.context().result
    }
}

/// Short alias used by parser callers.
pub type StatementParser = TextJinjaParserProgramParserStatementParserActor;
