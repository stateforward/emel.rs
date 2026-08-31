//! Source-aligned synchronous expression-parser token classifier.

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

use super::super::lexer::TokenKind;
use sml::sml;

/// Values from the pinned expression-parser `parse_kind` enum.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParseKind {
    /// No supported token classification.
    Unknown,
    /// An identifier token.
    Identifier,
    /// One of the supported non-identifier token kinds.
    NonIdentifier,
}

/// Owned result produced by one expression-parser dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParseOutcome {
    /// A token was classified.
    Parsed(ParseKind),
    /// Input was absent or unsupported.
    ParseFailed,
    /// An event arrived after classification completed.
    Unexpected,
}

/// Copied input carrying the lexer token kind.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RuleParserEventParseRules {
    /// The token kind, or `None` when no token was emitted.
    pub token_kind: Option<TokenKind>,
}

impl RuleParserEventParseRules {
    /// Creates an input from a token kind.
    #[must_use]
    pub const fn new(token_kind: TokenKind) -> Self { Self { token_kind: Some(token_kind) } }

    /// Creates an absent-input event.
    #[must_use]
    pub const fn absent() -> Self { Self { token_kind: None } }
}

impl From<TokenKind> for RuleParserEventParseRules {
    fn from(token_kind: TokenKind) -> Self { Self::new(token_kind) }
}

sml! {
    GbnfRuleParserExpressionParser {
        *"deciding"_s + event<RuleParserEventParseRules>,
        "parsed_identifier"_s <= "deciding"_s + completion<RuleParserEventParseRules> [token_identifier] / consume_identifier,
        "parsed_non_identifier"_s <= "deciding"_s + completion<RuleParserEventParseRules> [token_non_identifier] / consume_non_identifier,
        "parse_failed"_s <= "deciding"_s + completion<RuleParserEventParseRules> [parse_failed] / dispatch_parse_failed,
        "unexpected_event"_s <= "deciding"_s + unexpected_event<_> / on_unexpected_from_deciding,
        "unexpected_event"_s <= "parsed_identifier"_s + unexpected_event<_> / on_unexpected_from_parsed_identifier,
        "unexpected_event"_s <= "parsed_non_identifier"_s + unexpected_event<_> / on_unexpected_from_parsed_non_identifier,
        "unexpected_event"_s <= "parse_failed"_s + unexpected_event<_> / on_unexpected_from_parse_failed,
        "unexpected_event"_s <= "unexpected_event"_s + unexpected_event<_> / on_unexpected_from_unexpected_event,
        "parsed_identifier"_s = X,
        "parsed_non_identifier"_s = X,
        "parse_failed"_s = X,
    }
}

/// Context for `GbnfRuleParserExpressionParser`.
#[derive(Debug)]
pub struct GbnfRuleParserExpressionParserContext { result: ParseOutcome }

impl Default for GbnfRuleParserExpressionParserContext {
    fn default() -> Self { Self { result: ParseOutcome::Parsed(ParseKind::Unknown) } }
}

impl GbnfRuleParserExpressionParserStateMachineContext for GbnfRuleParserExpressionParserContext {
    fn consume_identifier(&mut self, _event: &RuleParserEventParseRules) -> Result<(), ()> {
        self.result = ParseOutcome::Parsed(ParseKind::Identifier);
        Ok(())
    }

    fn consume_non_identifier(&mut self, _event: &RuleParserEventParseRules) -> Result<(), ()> {
        self.result = ParseOutcome::Parsed(ParseKind::NonIdentifier);
        Ok(())
    }

    fn dispatch_parse_failed(&mut self, _event: &RuleParserEventParseRules) -> Result<(), ()> {
        self.result = ParseOutcome::ParseFailed;
        Ok(())
    }

    fn on_unexpected_from_deciding(&mut self) -> Result<(), ()> { self.result = ParseOutcome::Unexpected; Ok(()) }
    fn on_unexpected_from_parse_failed(&mut self) -> Result<(), ()> { self.result = ParseOutcome::Unexpected; Ok(()) }
    fn on_unexpected_from_parsed_identifier(&mut self) -> Result<(), ()> { self.result = ParseOutcome::Unexpected; Ok(()) }
    fn on_unexpected_from_parsed_non_identifier(&mut self) -> Result<(), ()> { self.result = ParseOutcome::Unexpected; Ok(()) }
    fn on_unexpected_from_unexpected_event(&mut self) -> Result<(), ()> { self.result = ParseOutcome::Unexpected; Ok(()) }

    fn parse_failed(&self, event: &RuleParserEventParseRules) -> Result<bool, ()> { Ok(!is_supported(event.token_kind)) }
    fn token_identifier(&self, event: &RuleParserEventParseRules) -> Result<bool, ()> { Ok(event.token_kind == Some(TokenKind::Identifier)) }
    fn token_non_identifier(&self, event: &RuleParserEventParseRules) -> Result<bool, ()> {
        Ok(matches!(event.token_kind,
            Some(TokenKind::StringLiteral) | Some(TokenKind::CharacterClass) |
            Some(TokenKind::RuleReference) | Some(TokenKind::Dot) |
            Some(TokenKind::OpenGroup) | Some(TokenKind::CloseGroup) |
            Some(TokenKind::Quantifier) | Some(TokenKind::Alternation) |
            Some(TokenKind::Newline)))
    }
}

fn is_supported(token_kind: Option<TokenKind>) -> bool {
    matches!(token_kind,
        Some(TokenKind::Identifier) | Some(TokenKind::StringLiteral) |
        Some(TokenKind::CharacterClass) | Some(TokenKind::RuleReference) |
        Some(TokenKind::Dot) | Some(TokenKind::OpenGroup) |
        Some(TokenKind::CloseGroup) | Some(TokenKind::Quantifier) |
        Some(TokenKind::Alternation) | Some(TokenKind::Newline))
}

/// Synchronous actor around the generated expression-parser machine.
pub struct GbnfRuleParserExpressionParserActor {
    machine: GbnfRuleParserExpressionParserStateMachine<GbnfRuleParserExpressionParserContext>,
}

impl Default for GbnfRuleParserExpressionParserActor {
    fn default() -> Self { Self::new() }
}

impl GbnfRuleParserExpressionParserActor {
    /// Creates an actor in the generated initial state.
    #[must_use]
    pub fn new() -> Self { Self { machine: GbnfRuleParserExpressionParserStateMachine::new(Default::default()) } }

    pub fn process_event(&mut self, event: RuleParserEventParseRules) -> ParseOutcome {
        if !self.machine.is(&GbnfRuleParserExpressionParserStates::Deciding) {
            self.machine.context_mut().result = ParseOutcome::Unexpected;
            self.machine.set_state(GbnfRuleParserExpressionParserStates::UnexpectedEvent);
            return ParseOutcome::Unexpected;
        }
        if self.machine.process_event(GbnfRuleParserExpressionParserEvents::RuleParserEventParseRules(event)).is_err() {
            self.machine.context_mut().result = ParseOutcome::Unexpected;
            self.machine.set_state(GbnfRuleParserExpressionParserStates::UnexpectedEvent);
        } else if self.machine.initialize().is_err() {
            self.machine.context_mut().result = ParseOutcome::Unexpected;
            self.machine.set_state(GbnfRuleParserExpressionParserStates::UnexpectedEvent);
        }
        self.machine.context().result
    }

    /// Classifies one lexer token kind.
    pub fn classify(&mut self, token_kind: TokenKind) -> ParseOutcome { self.process_event(token_kind.into()) }

    pub fn process_unexpected(&mut self) -> ParseOutcome {
        self.machine.context_mut().result = ParseOutcome::Unexpected;
        self.machine
            .set_state(GbnfRuleParserExpressionParserStates::UnexpectedEvent);
        self.machine.context().result
    }

    /// Returns generated state inspection data.
    #[must_use]
    pub fn state(&self) -> &GbnfRuleParserExpressionParserStates { self.machine.state() }

    /// Reports whether the generated machine is in `state`.
    #[must_use]
    pub fn is(&self, state: &GbnfRuleParserExpressionParserStates) -> bool { self.machine.is(state) }
}

/// Short actor alias for expression-parser callers.
pub type ExpressionParser = GbnfRuleParserExpressionParserActor;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_identifier() {
        let mut parser = ExpressionParser::new();
        assert_eq!(parser.classify(TokenKind::Identifier), ParseOutcome::Parsed(ParseKind::Identifier));
        assert!(parser.is(&GbnfRuleParserExpressionParserStates::X));
    }

    #[test]
    fn classifies_non_identifier() {
        let mut parser = ExpressionParser::new();
        assert_eq!(parser.classify(TokenKind::StringLiteral), ParseOutcome::Parsed(ParseKind::NonIdentifier));
        assert!(parser.is(&GbnfRuleParserExpressionParserStates::X));
    }

    #[test]
    fn rejects_unsupported_and_absent_input() {
        let mut parser = ExpressionParser::new();
        assert_eq!(parser.classify(TokenKind::DefinitionOperator), ParseOutcome::ParseFailed);
        let mut parser = ExpressionParser::new();
        assert_eq!(parser.process_event(RuleParserEventParseRules::absent()), ParseOutcome::ParseFailed);
        assert!(parser.is(&GbnfRuleParserExpressionParserStates::X));
    }

    #[test]
    fn reports_explicit_unexpected_event() {
        let mut parser = ExpressionParser::new();
        assert_eq!(parser.classify(TokenKind::Identifier), ParseOutcome::Parsed(ParseKind::Identifier));
        assert_eq!(parser.process_unexpected(), ParseOutcome::Unexpected);
        assert!(parser.is(&GbnfRuleParserExpressionParserStates::UnexpectedEvent));
    }
}
