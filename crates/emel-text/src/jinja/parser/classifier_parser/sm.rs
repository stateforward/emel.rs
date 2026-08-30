//! Source-aligned bounded TextJinjaParserClassifierParser state machine.

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

/// Maximum number of lexer tokens copied into one classifier request.
pub const MAX_CLASSIFIER_TOKENS: usize = 64;

/// Token kinds consumed by the classifier, matching the pinned Jinja lexer.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum TokenType {
    #[default]
    Eof = 0,
    Text,
    NumericLiteral,
    StringLiteral,
    Identifier,
    Equals,
    OpenParen,
    CloseParen,
    OpenStatement,
    CloseStatement,
    OpenExpression,
    CloseExpression,
    OpenSquareBracket,
    CloseSquareBracket,
    OpenCurlyBracket,
    CloseCurlyBracket,
    Comma,
    Dot,
    Colon,
    Pipe,
    CallOperator,
    AdditiveBinaryOperator,
    MultiplicativeBinaryOperator,
    ComparisonBinaryOperator,
    UnaryOperator,
    Comment,
}

/// Compatibility spelling used by lexer-facing callers.
pub type JinjaTokenKind = TokenType;

/// Parser error values from `parser/errors.hpp`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(i32)]
pub enum ParseError {
    #[default]
    None = 0,
    InvalidRequest = 1 << 0,
    ParseFailed = 1 << 1,
    InternalError = 1 << 2,
    Untracked = 1 << 3,
    Unknown = -1,
}

/// Classified statement kind from the pinned parser event contract.
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

/// Classified expression kind from the pinned parser event contract.
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

/// Bounded token input copied from the parser runtime.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenInput {
    /// Token kinds in lexer order.
    pub tokens: [TokenType; MAX_CLASSIFIER_TOKENS],
    /// Number of initialized entries in `tokens`.
    pub token_count: usize,
    /// Current parser token index.
    pub token_index: usize,
}

impl Default for TokenInput {
    fn default() -> Self {
        Self {
            tokens: [TokenType::Eof; MAX_CLASSIFIER_TOKENS],
            token_count: 0,
            token_index: 0,
        }
    }
}

impl TokenInput {
    /// Copies a bounded token slice without allocating.
    #[must_use]
    pub fn from_slice(tokens: &[TokenType]) -> Option<Self> {
        if tokens.len() > MAX_CLASSIFIER_TOKENS {
            return None;
        }
        let mut input = Self::default();
        input.tokens[..tokens.len()].copy_from_slice(tokens);
        input.token_count = tokens.len();
        Some(input)
    }

    /// Builds a single-token input.
    #[must_use]
    pub const fn single(token: TokenType) -> Self {
        let mut tokens = [TokenType::Eof; MAX_CLASSIFIER_TOKENS];
        tokens[0] = token;
        Self { tokens, token_count: 1, token_index: 0 }
    }

    fn has(&self, offset: usize) -> bool {
        self.token_index
            .checked_add(offset)
            .is_some_and(|index| index < self.token_count)
    }

    fn is(&self, token: TokenType, offset: usize) -> bool {
        self.has(offset) && self.tokens[self.token_index + offset] == token
    }
}

/// Copied bounded equivalent of `event::parse_runtime`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EventParseRuntime {
    /// Lexer token input visible to classifier guards.
    pub input: TokenInput,
    /// Parser error being classified by the result decision.
    pub error: ParseError,
}

impl EventParseRuntime {
    /// Creates a no-error runtime event from a bounded token slice.
    #[must_use]
    pub fn from_tokens(tokens: &[TokenType]) -> Option<Self> {
        TokenInput::from_slice(tokens).map(|input| Self { input, error: ParseError::None })
    }

    /// Creates a runtime event from one token.
    #[must_use]
    pub const fn single(token: TokenType) -> Self {
        Self { input: TokenInput::single(token), error: ParseError::None }
    }

    /// Creates an error runtime event with no tokens.
    #[must_use]
    pub const fn with_error(error: ParseError) -> Self {
        Self { input: TokenInput { tokens: [TokenType::Eof; MAX_CLASSIFIER_TOKENS], token_count: 0, token_index: 0 }, error }
    }
}

/// Result fields retained by the bounded classifier actor.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ClassifierResult {
    /// Statement classification.
    pub statement: StatementKind,
    /// Expression classification, when the statement is an expression.
    pub expression: ExpressionKind,
    /// Parser error selected by the result decision.
    pub error: ParseError,
    /// Whether the actor received an event outside its active topology.
    pub unexpected: bool,
}

impl ClassifierResult {
    fn unknown() -> Self { Self::default() }
}

// --- machine TextJinjaParserClassifierParser from emel.cpp/src/emel/text/jinja/parser/classifier_parser/sm.hpp ---
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

/// Context retained by the generated classifier machine.
#[derive(Debug)]
pub struct TextJinjaParserClassifierParserContext {
    /// Copied token input.
    pub input: TokenInput,
    /// Current parse phase as represented by the pinned parser.
    pub phase: ParsePhase,
    /// Statement classification.
    pub statement: StatementKind,
    /// Expression classification.
    pub expression: ExpressionKind,
    /// Current parser token index.
    pub token_index: usize,
    /// Error selected by the result decision.
    pub error: ParseError,
    /// Last bounded classification result.
    pub result: ClassifierResult,
}

/// Parse phases from `event::parse_phase`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum ParsePhase {
    #[default]
    None = 0,
    RequestValidation = 1,
    Tokenization = 2,
    StatementClassification = 3,
    Parsing = 4,
}

impl Default for TextJinjaParserClassifierParserContext {
    fn default() -> Self {
        Self {
            input: TokenInput::default(),
            phase: ParsePhase::None,
            statement: StatementKind::Unknown,
            expression: ExpressionKind::Unknown,
            token_index: 0,
            error: ParseError::None,
            result: ClassifierResult::unknown(),
        }
    }
}

impl TextJinjaParserClassifierParserContext {
    fn load(&mut self, event: EventParseRuntime) {
        self.input = event.input;
        self.token_index = event.input.token_index;
        self.error = event.error;
        self.result = ClassifierResult::unknown();
    }

    fn publish(&mut self) {
        self.result = ClassifierResult {
            statement: self.statement,
            expression: self.expression,
            error: self.error,
            unexpected: false,
        };
    }

    fn unexpected(&mut self) {
        self.error = ParseError::InternalError;
        self.result = ClassifierResult {
            statement: self.statement,
            expression: self.expression,
            error: self.error,
            unexpected: true,
        };
    }
}

impl TextJinjaParserClassifierParserStateMachineContext for TextJinjaParserClassifierParserContext {
    fn begin_classification(&mut self) -> Result<(), ()> {
        self.phase = ParsePhase::StatementClassification;
        self.statement = StatementKind::Unknown;
        self.expression = ExpressionKind::Unknown;
        self.token_index = 0;
        Ok(())
    }

    fn expr_no_token(&self) -> Result<bool, ()> { Ok(!self.input.has(1)) }

    fn expr_token_compound(&self) -> Result<bool, ()> {
        Ok(self.input.is(TokenType::OpenParen, 1))
    }

    fn expr_token_identifier(&self) -> Result<bool, ()> {
        Ok(self.input.is(TokenType::Identifier, 1))
    }

    fn expr_token_literal(&self) -> Result<bool, ()> {
        Ok(self.input.is(TokenType::NumericLiteral, 1)
            || self.input.is(TokenType::StringLiteral, 1)
            || self.input.is(TokenType::OpenSquareBracket, 1)
            || self.input.is(TokenType::OpenCurlyBracket, 1))
    }

    fn expr_token_unary(&self) -> Result<bool, ()> {
        Ok(self.input.is(TokenType::AdditiveBinaryOperator, 1)
            || self.input.is(TokenType::UnaryOperator, 1))
    }

    fn no_tokens(&self) -> Result<bool, ()> { Ok(!self.input.has(0)) }

    fn on_unexpected_from_classification_result_decision(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_deciding(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_done(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_errored(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_expression_decision(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_statement_decision(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }
    fn on_unexpected_from_unexpected_event(&mut self) -> Result<(), ()> { self.unexpected(); Ok(()) }

    fn parse_error_internal_error(&self) -> Result<bool, ()> { Ok(self.error == ParseError::InternalError) }
    fn parse_error_invalid_request(&self) -> Result<bool, ()> { Ok(self.error == ParseError::InvalidRequest) }
    fn parse_error_none(&self) -> Result<bool, ()> { Ok(self.error == ParseError::None) }
    fn parse_error_parse_failed(&self) -> Result<bool, ()> { Ok(self.error == ParseError::ParseFailed) }
    fn parse_error_untracked(&self) -> Result<bool, ()> { Ok(self.error == ParseError::Untracked) }

    fn set_expression_compound(&mut self) -> Result<(), ()> { self.expression = ExpressionKind::Compound; self.publish(); Ok(()) }
    fn set_expression_identifier(&mut self) -> Result<(), ()> { self.expression = ExpressionKind::Identifier; self.publish(); Ok(()) }
    fn set_expression_literal(&mut self) -> Result<(), ()> { self.expression = ExpressionKind::Literal; self.publish(); Ok(()) }
    fn set_expression_unary(&mut self) -> Result<(), ()> { self.expression = ExpressionKind::Unary; self.publish(); Ok(()) }
    fn set_expression_unknown_from_expression_decision(&mut self) -> Result<(), ()> { self.expression = ExpressionKind::Unknown; self.publish(); Ok(()) }
    fn set_statement_comment(&mut self) -> Result<(), ()> { self.statement = StatementKind::Comment; self.expression = ExpressionKind::Unknown; self.publish(); Ok(()) }
    fn set_statement_expression(&mut self) -> Result<(), ()> { self.statement = StatementKind::Expression; Ok(()) }
    fn set_statement_statement(&mut self) -> Result<(), ()> { self.statement = StatementKind::Statement; self.expression = ExpressionKind::Unknown; self.publish(); Ok(()) }
    fn set_statement_text(&mut self) -> Result<(), ()> { self.statement = StatementKind::Text; self.expression = ExpressionKind::Unknown; self.publish(); Ok(()) }
    fn set_statement_unknown_from_statement_decision(&mut self) -> Result<(), ()> { self.statement = StatementKind::Unknown; self.expression = ExpressionKind::Unknown; self.publish(); Ok(()) }

    fn token_comment(&self) -> Result<bool, ()> { Ok(self.input.is(TokenType::Comment, 0)) }
    fn token_open_expression(&self) -> Result<bool, ()> { Ok(self.input.is(TokenType::OpenExpression, 0)) }
    fn token_open_statement(&self) -> Result<bool, ()> { Ok(self.input.is(TokenType::OpenStatement, 0)) }
    fn token_text(&self) -> Result<bool, ()> { Ok(self.input.is(TokenType::Text, 0)) }
}

/// Synchronous single-writer actor around the generated classifier machine.
pub struct TextJinjaParserClassifierParserActor {
    machine: TextJinjaParserClassifierParserStateMachine<TextJinjaParserClassifierParserContext>,
}

impl Default for TextJinjaParserClassifierParserActor {
    fn default() -> Self { Self::new() }
}

impl TextJinjaParserClassifierParserActor {
    /// Creates an actor in the generated initial state.
    #[must_use]
    pub fn new() -> Self {
        Self { machine: TextJinjaParserClassifierParserStateMachine::new(Default::default()) }
    }

    /// Processes one bounded copied parser-runtime input synchronously.
    pub fn process_event(&mut self, event: EventParseRuntime) -> ClassifierResult {
        if !self.machine.is(&TextJinjaParserClassifierParserStates::Deciding) {
            return self.process_unexpected();
        }
        self.machine.context_mut().load(event);
        let _ = self.machine.process_event(TextJinjaParserClassifierParserEvents::EventParseRuntime(event));
        self.machine.context().result
    }

    /// Classifies a bounded token slice without allocating.
    pub fn classify(&mut self, tokens: &[TokenType]) -> ClassifierResult {
        match EventParseRuntime::from_tokens(tokens) {
            Some(event) => self.process_event(event),
            None => self.process_event(EventParseRuntime::with_error(ParseError::InvalidRequest)),
        }
    }

    /// Classifies one token without allocating.
    pub fn classify_token(&mut self, token: TokenType) -> ClassifierResult {
        self.process_event(EventParseRuntime::single(token))
    }

    /// Sends an explicit unexpected event through the machine's error path.
    pub fn process_unexpected(&mut self) -> ClassifierResult {
        self.machine.context_mut().unexpected();
        self.machine.set_state(TextJinjaParserClassifierParserStates::UnexpectedEvent);
        self.machine.context().result
    }

    /// Returns generated state inspection data.
    #[must_use]
    pub fn state(&self) -> &TextJinjaParserClassifierParserStates { self.machine.state() }

    /// Reports whether the generated machine is in `state`.
    #[must_use]
    pub fn is(&self, state: &TextJinjaParserClassifierParserStates) -> bool { self.machine.is(state) }

    /// Returns the retained bounded context.
    #[must_use]
    pub fn context(&self) -> &TextJinjaParserClassifierParserContext { self.machine.context() }
}

/// Short actor alias for classifier-parser callers.
pub type ClassifierParser = TextJinjaParserClassifierParserActor;
