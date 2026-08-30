//! Source-aligned synchronous Jinja expression parser.
//!
//! The actor mirrors the pinned expression-parser transition table while using
//! bounded, copied input and output data.  The parser deliberately preserves
//! the source machine's coarse expression classification: a leading identifier
//! emits an identifier node, while every other supported leading token emits a
//! generic string-literal node.

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
    missing_docs,
    private_interfaces
)]

use sml::sml;

/// Maximum number of copied lexer tokens accepted by one expression parse.
pub const MAX_EXPRESSION_TOKENS: usize = 128;
/// Maximum number of bytes copied from one lexer token value.
pub const MAX_TOKEN_VALUE: usize = 128;

/// Token categories used by the pinned Jinja lexer.
///
/// The expression parser consumes the lexer worker's exact discriminants so
/// callers can pass categories without a translation table.
pub type TokenType = super::super::lexer::sm::TokenKind;

/// A bounded copied lexer token.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Token {
    /// Lexer category.
    pub kind: TokenType,
    /// Copied token value bytes.
    pub value: [u8; MAX_TOKEN_VALUE],
    /// Number of initialized bytes in [`Token::value`].
    pub value_len: u16,
    /// Source position.
    pub pos: u32,
}

impl Default for Token {
    fn default() -> Self {
        Self { kind: TokenType::Eof, value: [0; MAX_TOKEN_VALUE], value_len: 0, pos: 0 }
    }
}

impl Token {
    /// Constructs a token, truncating a value that exceeds the bounded copy size.
    #[must_use]
    pub fn new(kind: TokenType, value: &[u8], pos: usize) -> Self {
        let mut token = Self { kind, pos: pos.min(u32::MAX as usize) as u32, ..Self::default() };
        let count = value.len().min(MAX_TOKEN_VALUE);
        token.value[..count].copy_from_slice(&value[..count]);
        token.value_len = count as u16;
        token
    }

    /// Returns the initialized copied value.
    #[must_use]
    pub fn value(&self) -> &[u8] { &self.value[..self.value_len as usize] }
}

/// Expression kind tracked by the source parser context.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum ExpressionKind {
    #[default]
    Unknown = 0,
    Literal = 1,
    Identifier = 2,
    Unary = 3,
    Compound = 4,
}

/// Errors corresponding to the pinned parser error enum.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum ExpressionParserError {
    #[default]
    None = 0,
    InvalidRequest = 1,
    ParseFailed = 2,
    InternalError = 4,
    Untracked = 8,
}

/// Owned bounded result from one expression-parser dispatch.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExpressionParseResult {
    /// Classification of the leading expression token.
    pub expression: ExpressionKind,
    /// Token emitted by the source parser's emit action.
    pub emitted: Token,
    /// Whether an expression node was emitted.
    pub emitted_present: bool,
    /// Number of input tokens consumed, including opening and closing tokens.
    pub consumed: u16,
    /// Error status, if parsing did not complete successfully.
    pub error: ExpressionParserError,
    /// Error source position.
    pub error_pos: u32,
}

/// Bounded copied input corresponding to the source parser runtime context.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpressionParserInput {
    /// Copied lexer token sequence, beginning with `OpenExpression`.
    pub tokens: [Token; MAX_EXPRESSION_TOKENS],
    /// Number of initialized entries in [`Self::tokens`].
    pub token_count: u16,
}

impl Default for ExpressionParserInput {
    fn default() -> Self { Self { tokens: [Token::default(); MAX_EXPRESSION_TOKENS], token_count: 0 } }
}

impl ExpressionParserInput {
    /// Creates input from a bounded token slice.
    #[must_use]
    pub fn new(tokens: &[Token]) -> Self {
        let mut input = Self::default();
        let count = tokens.len().min(MAX_EXPRESSION_TOKENS);
        input.tokens[..count].copy_from_slice(&tokens[..count]);
        input.token_count = count as u16;
        input
    }

    /// Creates an empty input, which follows the source EOF failure path.
    #[must_use]
    pub const fn empty() -> Self { Self { tokens: [Token { kind: TokenType::Eof, value: [0; MAX_TOKEN_VALUE], value_len: 0, pos: 0 }; MAX_EXPRESSION_TOKENS], token_count: 0 } }
}

/// Runtime event carrying bounded copied input and result fields.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EventParseRuntime {
    /// Copied input visible to parser guards.
    pub input: ExpressionParserInput,
    /// Copied result snapshot carried by the event boundary.
    pub result: ExpressionParseResult,
}

impl From<ExpressionParserInput> for EventParseRuntime {
    fn from(input: ExpressionParserInput) -> Self {
        Self { input, result: ExpressionParseResult::default() }
    }
}

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

/// Context retained by the generated expression parser state machine.
#[derive(Clone, Copy, Debug)]
pub struct TextJinjaParserProgramParserExpressionParserContext {
    /// Copied input token sequence.
    pub input: ExpressionParserInput,
    /// Source parser statement classification.
    pub statement_is_expression: bool,
    /// Current expression classification.
    pub expression: ExpressionKind,
    /// Expression opening-token index.
    pub expression_start: u16,
    /// First expression-value token index.
    pub expression_value_index: u16,
    /// Current token index.
    pub token_index: u16,
    /// Bounded parse result.
    pub result: ExpressionParseResult,
}

impl Default for TextJinjaParserProgramParserExpressionParserContext {
    fn default() -> Self {
        Self {
            input: ExpressionParserInput::default(),
            statement_is_expression: false,
            expression: ExpressionKind::Unknown,
            expression_start: 0,
            expression_value_index: 0,
            token_index: 0,
            result: ExpressionParseResult::default(),
        }
    }
}

impl TextJinjaParserProgramParserExpressionParserContext {
    fn has_token(&self, offset: u16) -> bool {
        self.token_index.saturating_add(offset) < self.input.token_count
    }

    fn token(&self, offset: u16) -> TokenType {
        if self.has_token(offset) { self.input.tokens[(self.token_index + offset) as usize].kind } else { TokenType::Eof }
    }

    fn current(&self) -> Token { self.input.tokens[self.token_index as usize] }

    fn fail(&mut self, pos: u32) {
        self.result.error = ExpressionParserError::ParseFailed;
        self.result.error_pos = pos;
        self.result.expression = self.expression;
        self.result.consumed = self.token_index;
    }

    fn unexpected(&mut self) {
        self.result.error = ExpressionParserError::InternalError;
        self.result.error_pos = self.current().pos;
    }
}

impl TextJinjaParserProgramParserExpressionParserStateMachineContext
    for TextJinjaParserProgramParserExpressionParserContext
{
    fn begin_expression_parse(&mut self) -> Result<(), ()> {
        self.statement_is_expression = true;
        self.expression = ExpressionKind::Unknown;
        self.expression_start = self.token_index;
        self.expression_value_index = self.token_index;
        self.result = ExpressionParseResult::default();
        self.token_index = self.token_index.saturating_add(1);
        Ok(())
    }

    fn consume_expression_close(&mut self) -> Result<(), ()> {
        self.token_index = self.token_index.saturating_add(1);
        self.result.expression = self.expression;
        self.result.consumed = self.token_index;
        Ok(())
    }

    fn consume_expression_compound(&mut self) -> Result<(), ()> {
        self.expression = ExpressionKind::Compound;
        self.expression_value_index = self.token_index;
        self.token_index = self.token_index.saturating_add(1);
        Ok(())
    }

    fn consume_expression_identifier(&mut self) -> Result<(), ()> {
        self.expression = ExpressionKind::Identifier;
        self.expression_value_index = self.token_index;
        self.token_index = self.token_index.saturating_add(1);
        Ok(())
    }

    fn consume_expression_identifier_and_close(&mut self) -> Result<(), ()> {
        self.expression = ExpressionKind::Identifier;
        self.expression_value_index = self.token_index;
        self.result.emitted = self.current();
        self.result.emitted_present = true;
        self.token_index = self.token_index.saturating_add(2);
        self.result.expression = self.expression;
        self.result.consumed = self.token_index;
        Ok(())
    }

    fn consume_expression_literal(&mut self) -> Result<(), ()> {
        self.expression = ExpressionKind::Literal;
        self.expression_value_index = self.token_index;
        self.token_index = self.token_index.saturating_add(1);
        Ok(())
    }

    fn consume_expression_token(&mut self) -> Result<(), ()> {
        self.token_index = self.token_index.saturating_add(1);
        Ok(())
    }

    fn consume_expression_unary(&mut self) -> Result<(), ()> {
        self.expression = ExpressionKind::Unary;
        self.expression_value_index = self.token_index;
        self.token_index = self.token_index.saturating_add(1);
        Ok(())
    }

    fn emit_expression_generic(&mut self) -> Result<(), ()> {
        self.result.emitted = self.input.tokens[self.expression_value_index as usize];
        self.result.emitted_present = true;
        Ok(())
    }

    fn emit_expression_identifier(&mut self) -> Result<(), ()> {
        self.result.emitted = self.input.tokens[self.expression_value_index as usize];
        self.result.emitted_present = true;
        Ok(())
    }

    fn expr_first_identifier_followed_by_close(&self) -> Result<bool, ()> {
        Ok(self.token(0) == TokenType::Identifier && self.token(1) == TokenType::CloseExpression)
    }

    fn expr_first_is_close(&self) -> Result<bool, ()> { Ok(self.token(0) == TokenType::CloseExpression) }
    fn expr_first_is_identifier(&self) -> Result<bool, ()> { Ok(self.token(0) == TokenType::Identifier) }
    fn expr_first_is_literal(&self) -> Result<bool, ()> {
        Ok(matches!(self.token(0), TokenType::NumericLiteral | TokenType::StringLiteral | TokenType::OpenSquareBracket | TokenType::OpenCurlyBracket))
    }
    fn expr_first_is_other_content(&self) -> Result<bool, ()> {
        Ok(self.has_token(0) && !self.expr_first_is_close()? && !self.expr_first_is_identifier()? && !self.expr_first_is_literal()? && !self.expr_first_is_unary()?)
    }
    fn expr_first_is_unary(&self) -> Result<bool, ()> {
        Ok(matches!(self.token(0), TokenType::AdditiveBinaryOperator | TokenType::UnaryOperator))
    }
    fn expr_scan_at_close(&self) -> Result<bool, ()> { Ok(self.token(0) == TokenType::CloseExpression) }
    fn expr_scan_continue(&self) -> Result<bool, ()> { Ok(self.has_token(0) && !self.expr_scan_at_close()?) }
    fn expr_scan_eof(&self) -> Result<bool, ()> { Ok(!self.has_token(0)) }
    fn expression_identifier(&self) -> Result<bool, ()> { Ok(self.expression == ExpressionKind::Identifier) }
    fn expression_non_identifier(&self) -> Result<bool, ()> { Ok(self.expression != ExpressionKind::Identifier) }

    fn fail_expression_close_token(&mut self) -> Result<(), ()> { self.fail(self.current().pos); Ok(()) }
    fn fail_expression_start_token_from_expression_close(&mut self) -> Result<(), ()> { self.fail(self.input.tokens[self.expression_start as usize].pos); Ok(()) }
    fn fail_expression_start_token_from_expression_first_decision(&mut self) -> Result<(), ()> { self.fail(self.input.tokens[self.expression_start as usize].pos); Ok(()) }
    fn fail_expression_start_token_from_expression_scan(&mut self) -> Result<(), ()> { self.fail(self.input.tokens[self.expression_start as usize].pos); Ok(()) }

    fn on_unexpected_from_deciding(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_expression_close(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_expression_emit_decision(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_expression_first_decision(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_expression_scan(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_parse_failed(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_parsed(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_unexpected_event(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
}

/// Synchronous bounded actor around the generated expression parser machine.
pub struct TextJinjaParserProgramParserExpressionParserActor {
    machine: TextJinjaParserProgramParserExpressionParserStateMachine<TextJinjaParserProgramParserExpressionParserContext>,
}

impl Default for TextJinjaParserProgramParserExpressionParserActor {
    fn default() -> Self { Self::new() }
}

impl TextJinjaParserProgramParserExpressionParserActor {
    /// Creates an actor in the generated initial state.
    #[must_use]
    pub fn new() -> Self { Self { machine: TextJinjaParserProgramParserExpressionParserStateMachine::new(Default::default()) } }

    /// Processes one copied input to run-to-completion.
    pub fn process_event(&mut self, input: ExpressionParserInput) -> ExpressionParseResult {
        if !self.machine.is(&TextJinjaParserProgramParserExpressionParserStates::Deciding) {
            let _ = self.process_unexpected();
            return self.machine.context().result;
        }
        self.machine.context_mut().input = input;
        self.machine.context_mut().token_index = 0;
        let event = EventParseRuntime::from(input);
        let _ = self.machine.process_event(
            TextJinjaParserProgramParserExpressionParserEvents::EventParseRuntime(event),
        );
        self.machine.context().result
    }

    /// Records an explicit unexpected event.
    pub fn process_unexpected(&mut self) -> ExpressionParseResult {
        self.machine.context_mut().unexpected();
        self.machine.set_state(TextJinjaParserProgramParserExpressionParserStates::UnexpectedEvent);
        self.machine.context().result
    }

    /// Returns the generated state inspection value.
    #[must_use]
    pub fn state(&self) -> &TextJinjaParserProgramParserExpressionParserStates { self.machine.state() }

    /// Tests generated state identity.
    #[must_use]
    pub fn is(&self, state: &TextJinjaParserProgramParserExpressionParserStates) -> bool { self.machine.is(state) }

    /// Returns the current bounded context snapshot.
    #[must_use]
    pub fn context(&self) -> &TextJinjaParserProgramParserExpressionParserContext { self.machine.context() }
}

/// Short expression-parser actor alias.
pub type ExpressionParser = TextJinjaParserProgramParserExpressionParserActor;
/// Short input alias.
pub type ExpressionInput = ExpressionParserInput;
/// Short result alias.
pub type ExpressionResult = ExpressionParseResult;
