//! Source-aligned synchronous definition-parser token classifier.

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

/// Values from the pinned definition-parser `parse_result` enum.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum ParseResult {
    /// No supported token classification.
    #[default]
    Unknown = 0,
    /// A definition operator token (`::=`).
    DefinitionOperator = 1,
}

/// Owned result produced by one definition-parser dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParseOutcome {
    /// The token was classified as a definition operator.
    Parsed(ParseResult),
    /// Input was absent or was not a definition operator.
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
    pub const fn new(token_kind: TokenKind) -> Self {
        Self {
            token_kind: Some(token_kind),
        }
    }

    /// Creates an absent-input event.
    #[must_use]
    pub const fn absent() -> Self {
        Self { token_kind: None }
    }
}

impl Default for RuleParserEventParseRules {
    fn default() -> Self {
        Self::absent()
    }
}

impl From<TokenKind> for RuleParserEventParseRules {
    fn from(token_kind: TokenKind) -> Self {
        Self::new(token_kind)
    }
}

sml! {
    GbnfRuleParserDefinitionParser {
        *"deciding"_s + event<RuleParserEventParseRules>,
        "parsed"_s <= "deciding"_s + completion<RuleParserEventParseRules> [token_definition_operator] / consume_definition_operator,
        "parse_failed"_s <= "deciding"_s + completion<RuleParserEventParseRules> [parse_failed] / dispatch_parse_failed,
        "unexpected_event"_s <= "deciding"_s + unexpected_event<_> / on_unexpected_from_deciding,
        "unexpected_event"_s <= "parsed"_s + unexpected_event<_> / on_unexpected_from_parsed,
        "unexpected_event"_s <= "parse_failed"_s + unexpected_event<_> / on_unexpected_from_parse_failed,
        "unexpected_event"_s <= "unexpected_event"_s + unexpected_event<_> / on_unexpected_from_unexpected_event,
        "parsed"_s = X,
        "parse_failed"_s = X,
    }
}

/// Context for `GbnfRuleParserDefinitionParser`.
#[derive(Debug)]
pub struct GbnfRuleParserDefinitionParserContext {
    result: ParseOutcome,
}

impl Default for GbnfRuleParserDefinitionParserContext {
    fn default() -> Self {
        Self {
            result: ParseOutcome::Parsed(ParseResult::Unknown),
        }
    }
}

impl GbnfRuleParserDefinitionParserStateMachineContext for GbnfRuleParserDefinitionParserContext {
    fn consume_definition_operator(
        &mut self,
        _event: &RuleParserEventParseRules,
    ) -> Result<(), ()> {
        self.result = ParseOutcome::Parsed(ParseResult::DefinitionOperator);
        Ok(())
    }

    fn dispatch_parse_failed(&mut self, _event: &RuleParserEventParseRules) -> Result<(), ()> {
        self.result = ParseOutcome::ParseFailed;
        Ok(())
    }

    fn on_unexpected_from_deciding(&mut self) -> Result<(), ()> {
        self.result = ParseOutcome::Unexpected;
        Ok(())
    }

    fn on_unexpected_from_parse_failed(&mut self) -> Result<(), ()> {
        self.result = ParseOutcome::Unexpected;
        Ok(())
    }

    fn on_unexpected_from_parsed(&mut self) -> Result<(), ()> {
        self.result = ParseOutcome::Unexpected;
        Ok(())
    }

    fn on_unexpected_from_unexpected_event(&mut self) -> Result<(), ()> {
        self.result = ParseOutcome::Unexpected;
        Ok(())
    }

    fn parse_failed(&self, event: &RuleParserEventParseRules) -> Result<bool, ()> {
        Ok(!self.token_definition_operator(event)?)
    }

    fn token_definition_operator(&self, event: &RuleParserEventParseRules) -> Result<bool, ()> {
        Ok(event.token_kind == Some(TokenKind::DefinitionOperator))
    }
}

/// Synchronous actor around the generated definition-parser machine.
#[allow(
    missing_debug_implementations,
    reason = "generated state-machine wrapper has no stable Debug contract"
)]
pub struct GbnfRuleParserDefinitionParserActor {
    machine: GbnfRuleParserDefinitionParserStateMachine<GbnfRuleParserDefinitionParserContext>,
}

impl Default for GbnfRuleParserDefinitionParserActor {
    fn default() -> Self {
        Self::new()
    }
}

impl GbnfRuleParserDefinitionParserActor {
    /// Creates an actor in the generated initial state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: GbnfRuleParserDefinitionParserStateMachine::new(
                GbnfRuleParserDefinitionParserContext::default(),
            ),
        }
    }

    pub fn process_event(&mut self, event: RuleParserEventParseRules) -> ParseOutcome {
        if !self
            .machine
            .is(&GbnfRuleParserDefinitionParserStates::Deciding)
        {
            self.machine.context_mut().result = ParseOutcome::Unexpected;
            self.machine
                .set_state(GbnfRuleParserDefinitionParserStates::UnexpectedEvent);
            return ParseOutcome::Unexpected;
        }
        let failed = self
            .machine
            .process_event(GbnfRuleParserDefinitionParserEvents::RuleParserEventParseRules(event))
            .is_err()
            || self.machine.initialize().is_err();
        if failed {
            self.machine.context_mut().result = ParseOutcome::Unexpected;
            self.machine
                .set_state(GbnfRuleParserDefinitionParserStates::UnexpectedEvent);
        }
        self.machine.context().result
    }
    /// Classifies one lexer token kind.
    pub fn classify(&mut self, token_kind: TokenKind) -> ParseOutcome {
        self.process_event(token_kind.into())
    }

    /// Processes an absent-input event.
    pub fn process_absent(&mut self) -> ParseOutcome {
        self.process_event(RuleParserEventParseRules::absent())
    }

    /// Processes an explicit unexpected event.
    pub fn process_unexpected(&mut self) -> ParseOutcome {
        self.machine.context_mut().result = ParseOutcome::Unexpected;
        self.machine
            .set_state(GbnfRuleParserDefinitionParserStates::UnexpectedEvent);
        self.machine.context().result
    }

    /// Returns generated state inspection data.
    #[must_use]
    pub fn state(&self) -> &GbnfRuleParserDefinitionParserStates {
        self.machine.state()
    }

    /// Reports whether the generated machine is in `state`.
    #[must_use]
    pub fn is(&self, state: &GbnfRuleParserDefinitionParserStates) -> bool {
        self.machine.is(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_definition_operator() {
        let mut parser = GbnfRuleParserDefinitionParserActor::new();
        assert_eq!(
            parser.classify(TokenKind::DefinitionOperator),
            ParseOutcome::Parsed(ParseResult::DefinitionOperator)
        );
        assert!(parser.is(&GbnfRuleParserDefinitionParserStates::X));
    }

    #[test]
    fn rejects_unsupported_and_absent_input() {
        let mut parser = GbnfRuleParserDefinitionParserActor::new();
        assert_eq!(
            parser.classify(TokenKind::Identifier),
            ParseOutcome::ParseFailed
        );
        assert!(parser.is(&GbnfRuleParserDefinitionParserStates::X));

        let mut parser = GbnfRuleParserDefinitionParserActor::new();
        assert_eq!(parser.process_absent(), ParseOutcome::ParseFailed);
        assert!(parser.is(&GbnfRuleParserDefinitionParserStates::X));
    }

    #[test]
    fn reports_explicit_unexpected_event() {
        let mut parser = GbnfRuleParserDefinitionParserActor::new();
        assert_eq!(
            parser.classify(TokenKind::DefinitionOperator),
            ParseOutcome::Parsed(ParseResult::DefinitionOperator)
        );
        assert_eq!(parser.process_unexpected(), ParseOutcome::Unexpected);
        assert!(parser.is(&GbnfRuleParserDefinitionParserStates::UnexpectedEvent));
    }
}
