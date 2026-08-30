//! Source-aligned, allocation-free GBNF term-token classifier.

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
    private_interfaces,
)]

use super::super::lexer::TokenKind;
use sml::sml;

/// Internal term classification, matching pinned `term_kind` discriminants.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum TermKind {
    Unknown = 0,
    StringLiteral = 1,
    CharacterClass = 2,
    RuleReference = 3,
    Dot = 4,
    OpenGroup = 5,
    CloseGroup = 6,
    Quantifier = 7,
    Alternation = 8,
    Newline = 9,
}

/// Errors reported by the bounded term classifier.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TermParserError {
    /// Unsupported or absent lexer token.
    ParseFailed,
    /// Event was not valid in the current state.
    InternalError,
}

/// Copied input accepted by [`TermParser`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TermInput {
    /// Lexer category to classify.
    pub token: TokenKind,
    /// Whether the lexer supplied a token (`parse_rules::has_token`).
    pub has_token: bool,
}

impl TermInput {
    /// Constructs an input containing one lexer token.
    #[must_use]
    pub const fn new(token: TokenKind) -> Self { Self { token, has_token: true } }

    /// Constructs the no-token input used by an exhausted lexer.
    #[must_use]
    pub const fn empty() -> Self { Self { token: TokenKind::Unknown, has_token: false } }
}

/// Private completion event corresponding to pinned `parse_rules`.
/// The token is copied into the context before generated dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RuleParserEventParseRules;

// Source mapping: term_parser/sm.hpp's deciding -> parsed, deciding ->
// parse_failed, and all four unexpected-event rows.
sml! {
    GbnfRuleParserTermParser {
        *"deciding"_s + event<RuleParserEventParseRules>,
        "parsed"_s <= "deciding"_s + completion<RuleParserEventParseRules> [token_string_literal] / consume_string_literal,
        "parsed"_s <= "deciding"_s + completion<RuleParserEventParseRules> [token_character_class] / consume_character_class,
        "parsed"_s <= "deciding"_s + completion<RuleParserEventParseRules> [token_rule_reference] / consume_rule_reference,
        "parsed"_s <= "deciding"_s + completion<RuleParserEventParseRules> [token_dot] / consume_dot,
        "parsed"_s <= "deciding"_s + completion<RuleParserEventParseRules> [token_open_group] / consume_open_group,
        "parsed"_s <= "deciding"_s + completion<RuleParserEventParseRules> [token_close_group] / consume_close_group,
        "parsed"_s <= "deciding"_s + completion<RuleParserEventParseRules> [token_quantifier] / consume_quantifier,
        "parsed"_s <= "deciding"_s + completion<RuleParserEventParseRules> [token_alternation] / consume_alternation,
        "parsed"_s <= "deciding"_s + completion<RuleParserEventParseRules> [token_newline] / consume_newline,
        "parse_failed"_s <= "deciding"_s + completion<RuleParserEventParseRules> [parse_failed] / dispatch_parse_failed,
        "unexpected_event"_s <= "deciding"_s + unexpected_event<_> / on_unexpected_from_deciding,
        "unexpected_event"_s <= "parsed"_s + unexpected_event<_> / on_unexpected_from_parsed,
        "unexpected_event"_s <= "parse_failed"_s + unexpected_event<_> / on_unexpected_from_parse_failed,
        "unexpected_event"_s <= "unexpected_event"_s + unexpected_event<_> / on_unexpected_from_unexpected_event,
        "parsed"_s = X,
        "parse_failed"_s = X,
    }
}

/// Context retained by the generated state machine.
#[derive(Clone, Copy, Debug)]
pub struct GbnfRuleParserTermParserContext {
    token: TokenKind,
    has_token: bool,
    result: TermKind,
    error: Option<TermParserError>,
}

impl Default for GbnfRuleParserTermParserContext {
    fn default() -> Self {
        Self { token: TokenKind::Unknown, has_token: false, result: TermKind::Unknown, error: None }
    }
}

impl GbnfRuleParserTermParserContext {
    fn set_input(&mut self, input: TermInput) {
        self.token = input.token;
        self.has_token = input.has_token;
        self.result = TermKind::Unknown;
        self.error = None;
    }

    fn token_is(&self, kind: TokenKind) -> bool {
        self.error.is_none() && self.has_token && self.token == kind
    }

    fn consume(&mut self, kind: TermKind) -> Result<(), ()> {
        self.result = kind;
        self.error = None;
        Ok(())
    }

    fn unexpected(&mut self) -> Result<(), ()> {
        self.result = TermKind::Unknown;
        self.error = Some(TermParserError::InternalError);
        Ok(())
    }
}

impl GbnfRuleParserTermParserStateMachineContext for GbnfRuleParserTermParserContext {
    // Source mapping: actions.hpp::consume_kind<term_kind::...>; actions only
    // write the selected result.
    fn consume_alternation(&mut self, _event_data: &RuleParserEventParseRules) -> Result<(), ()> { self.consume(TermKind::Alternation) }
    fn consume_character_class(&mut self, _event_data: &RuleParserEventParseRules) -> Result<(), ()> { self.consume(TermKind::CharacterClass) }
    fn consume_close_group(&mut self, _event_data: &RuleParserEventParseRules) -> Result<(), ()> { self.consume(TermKind::CloseGroup) }
    fn consume_dot(&mut self, _event_data: &RuleParserEventParseRules) -> Result<(), ()> { self.consume(TermKind::Dot) }
    fn consume_newline(&mut self, _event_data: &RuleParserEventParseRules) -> Result<(), ()> { self.consume(TermKind::Newline) }
    fn consume_open_group(&mut self, _event_data: &RuleParserEventParseRules) -> Result<(), ()> { self.consume(TermKind::OpenGroup) }
    fn consume_quantifier(&mut self, _event_data: &RuleParserEventParseRules) -> Result<(), ()> { self.consume(TermKind::Quantifier) }
    fn consume_rule_reference(&mut self, _event_data: &RuleParserEventParseRules) -> Result<(), ()> { self.consume(TermKind::RuleReference) }
    fn consume_string_literal(&mut self, _event_data: &RuleParserEventParseRules) -> Result<(), ()> { self.consume(TermKind::StringLiteral) }

    // Source mapping: actions.hpp::dispatch_parse_failed.
    fn dispatch_parse_failed(&mut self, _event_data: &RuleParserEventParseRules) -> Result<(), ()> {
        self.result = TermKind::Unknown;
        self.error = Some(TermParserError::ParseFailed);
        Ok(())
    }

    // Source mapping: actions.hpp::on_unexpected; wildcard unexpected events
    // carry no payload in the generated callback API.
    fn on_unexpected_from_deciding(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_parse_failed(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_parsed(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_unexpected_event(&mut self) -> Result<(), ()> { self.unexpected() }

    // Source mapping: guards.hpp::token_is and guards.hpp::parse_failed.
    fn parse_failed(&self, event_data: &RuleParserEventParseRules) -> Result<bool, ()> {
        Ok(!self.token_string_literal(event_data)?
            && !self.token_character_class(event_data)?
            && !self.token_rule_reference(event_data)?
            && !self.token_dot(event_data)?
            && !self.token_open_group(event_data)?
            && !self.token_close_group(event_data)?
            && !self.token_quantifier(event_data)?
            && !self.token_alternation(event_data)?
            && !self.token_newline(event_data)?)
    }
    fn token_alternation(&self, _event_data: &RuleParserEventParseRules) -> Result<bool, ()> { Ok(self.token_is(TokenKind::Alternation)) }
    fn token_character_class(&self, _event_data: &RuleParserEventParseRules) -> Result<bool, ()> { Ok(self.token_is(TokenKind::CharacterClass)) }
    fn token_close_group(&self, _event_data: &RuleParserEventParseRules) -> Result<bool, ()> { Ok(self.token_is(TokenKind::CloseGroup)) }
    fn token_dot(&self, _event_data: &RuleParserEventParseRules) -> Result<bool, ()> { Ok(self.token_is(TokenKind::Dot)) }
    fn token_newline(&self, _event_data: &RuleParserEventParseRules) -> Result<bool, ()> { Ok(self.token_is(TokenKind::Newline)) }
    fn token_open_group(&self, _event_data: &RuleParserEventParseRules) -> Result<bool, ()> { Ok(self.token_is(TokenKind::OpenGroup)) }
    fn token_quantifier(&self, _event_data: &RuleParserEventParseRules) -> Result<bool, ()> { Ok(self.token_is(TokenKind::Quantifier)) }
    fn token_rule_reference(&self, _event_data: &RuleParserEventParseRules) -> Result<bool, ()> { Ok(self.token_is(TokenKind::RuleReference)) }
    fn token_string_literal(&self, _event_data: &RuleParserEventParseRules) -> Result<bool, ()> { Ok(self.token_is(TokenKind::StringLiteral)) }
}


/// Synchronous bounded term-token classifier actor.
pub struct TermParser {
    machine: GbnfRuleParserTermParserStateMachine<GbnfRuleParserTermParserContext>,
}

impl Default for TermParser {
    fn default() -> Self { Self::new() }
}

impl TermParser {
    /// Creates a classifier in generated `deciding` state.
    #[must_use]
    pub fn new() -> Self { Self { machine: GbnfRuleParserTermParserStateMachine::new(Default::default()) } }

    /// Dispatches one copied lexer token to completion.
    pub fn process_event(&mut self, input: TermInput) -> Result<TermKind, TermParserError> {
        if !self.machine.is(&GbnfRuleParserTermParserStates::Deciding) {
            self.machine.context_mut().error = Some(TermParserError::InternalError);
            return Err(TermParserError::InternalError);
        }
        self.machine.context_mut().set_input(input);
        if self.machine.process_event(RuleParserEventParseRules).is_err() {
            self.machine.context_mut().error = Some(TermParserError::InternalError);
            return Err(TermParserError::InternalError);
        }
        let context = self.machine.context();
        match context.error { Some(error) => Err(error), None => Ok(context.result) }
    }

    /// Records an unsupported event as an explicit internal error.
    pub fn process_unexpected_event(&mut self) -> Result<TermKind, TermParserError> {
        self.machine.context_mut().unexpected();
        self.machine.set_state(GbnfRuleParserTermParserStates::UnexpectedEvent);
        Err(TermParserError::InternalError)
    }

    /// Returns generated state inspection.
    #[must_use]
    pub fn state(&self) -> &GbnfRuleParserTermParserStates { self.machine.state() }

    /// Tests generated state identity.
    #[must_use]
    pub fn is(&self, state: GbnfRuleParserTermParserStates) -> bool { self.machine.is(&state) }

    /// Returns the result context for caller inspection.
    #[must_use]
    pub fn context(&self) -> &GbnfRuleParserTermParserContext { self.machine.context() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classify(token: TokenKind) -> TermKind { TermParser::new().process_event(TermInput::new(token)).expect("supported token") }

    #[test]
    fn classifies_each_supported_term_kind() {
        assert_eq!(classify(TokenKind::StringLiteral), TermKind::StringLiteral);
        assert_eq!(classify(TokenKind::CharacterClass), TermKind::CharacterClass);
        assert_eq!(classify(TokenKind::RuleReference), TermKind::RuleReference);
        assert_eq!(classify(TokenKind::Dot), TermKind::Dot);
        assert_eq!(classify(TokenKind::OpenGroup), TermKind::OpenGroup);
        assert_eq!(classify(TokenKind::CloseGroup), TermKind::CloseGroup);
        assert_eq!(classify(TokenKind::Quantifier), TermKind::Quantifier);
        assert_eq!(classify(TokenKind::Alternation), TermKind::Alternation);
        assert_eq!(classify(TokenKind::Newline), TermKind::Newline);
    }

    #[test]
    fn unsupported_token_is_parse_failed() {
        let mut parser = TermParser::new();
        assert_eq!(parser.process_event(TermInput::new(TokenKind::Unknown)), Err(TermParserError::ParseFailed));
        assert!(parser.is(GbnfRuleParserTermParserStates::ParseFailed));
        assert_eq!(parser.context().result, TermKind::Unknown);
    }

    #[test]
    fn no_token_is_parse_failed() {
        assert_eq!(TermParser::new().process_event(TermInput::empty()), Err(TermParserError::ParseFailed));
    }

    #[test]
    fn unexpected_event_is_explicit_internal_error() {
        let mut parser = TermParser::new();
        assert_eq!(parser.process_unexpected_event(), Err(TermParserError::InternalError));
        assert_eq!(parser.context().error, Some(TermParserError::InternalError));
    }
}
