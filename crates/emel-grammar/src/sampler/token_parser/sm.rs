//! Source-aligned, allocation-free GBNF sampler token parser.
//!
//! This child machine mirrors the pinned
//! `emel.cpp/src/emel/gbnf/sampler/token_parser/{sm,events,guards,actions}.hpp`
//! contract while keeping its event and result data bounded and owned.

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

/// Sampler errors relevant to the token-parser boundary.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum TokenParserError {
    /// No error was reported by the preceding sampler stage.
    #[default]
    None = 0,
    /// The candidate could not be classified as text or empty.
    ParseFailed = 1,
    /// An event was delivered outside the supported machine contract.
    InternalError = 2,
    /// The caller supplied an invalid request.
    InvalidRequest = 4,
    /// The source pipeline reported an untracked error.
    Untracked = 8,
}

/// Candidate classification consumed by the pinned token-parser guards.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum CandidateKind {
    /// No candidate classification is available.
    #[default]
    Unknown = 0,
    /// Candidate contains text to be sampled.
    Text = 1,
    /// Candidate is empty.
    Empty = 2,
}

/// Token classification written by the pinned token-parser actions.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum TokenKind {
    /// No token classification is available.
    #[default]
    Unknown = 0,
    /// Candidate was classified as a text token.
    TextToken = 1,
    /// Candidate was classified as an empty token.
    EmptyToken = 2,
}

/// Copied, bounded input for one token-parser dispatch.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TokenParserInput {
    /// Candidate classification supplied by the candidate parser.
    pub candidate_kind: CandidateKind,
    /// Error carried into this parser from the preceding sampler stage.
    pub error: TokenParserError,
}

impl TokenParserInput {
    /// Creates an error-free input from a candidate classification.
    #[must_use]
    pub const fn new(candidate_kind: CandidateKind) -> Self {
        Self { candidate_kind, error: TokenParserError::None }
    }

    /// Creates an input carrying a preceding sampler error.
    #[must_use]
    pub const fn failed(error: TokenParserError) -> Self {
        Self { candidate_kind: CandidateKind::Unknown, error }
    }

    /// Creates an explicitly invalid request input.
    #[must_use]
    pub const fn invalid() -> Self { Self::failed(TokenParserError::InvalidRequest) }
}

/// Copied runtime event corresponding to `sampler::event::sample_runtime`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SamplerEventSampleRuntime {
    /// Bounded input copied into the actor before dispatch.
    pub input: TokenParserInput,
}

impl From<TokenParserInput> for SamplerEventSampleRuntime {
    fn from(input: TokenParserInput) -> Self { Self { input } }
}

// Source mapping: token_parser/sm.hpp's deciding -> parsed, deciding ->
// parse_failed, terminal states, and all four explicit unexpected-event rows.
sml! {
    GbnfSamplerTokenParser {
        *"deciding"_s + event<SamplerEventSampleRuntime>,
        "parsed"_s <= "deciding"_s + completion<SamplerEventSampleRuntime> [candidate_text] / consume_text_token,
        "parsed"_s <= "deciding"_s + completion<SamplerEventSampleRuntime> [candidate_empty] / consume_empty_token,
        "parse_failed"_s <= "deciding"_s + completion<SamplerEventSampleRuntime> [parse_failed] / dispatch_parse_failed,
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
pub struct GbnfSamplerTokenParserContext {
    /// Last copied candidate classification.
    pub candidate_kind: CandidateKind,
    /// Error output corresponding to `sample_ctx::err`.
    pub error: TokenParserError,
    /// Token output corresponding to `sample_ctx::token_kind`.
    pub token_kind: TokenKind,
}

impl Default for GbnfSamplerTokenParserContext {
    fn default() -> Self {
        Self {
            candidate_kind: CandidateKind::Unknown,
            error: TokenParserError::None,
            token_kind: TokenKind::Unknown,
        }
    }
}

impl GbnfSamplerTokenParserContext {
    fn set_input(&mut self, input: TokenParserInput) {
        self.candidate_kind = input.candidate_kind;
        self.error = input.error;
        self.token_kind = TokenKind::Unknown;
    }

    fn unexpected(&mut self) -> Result<(), ()> {
        self.error = TokenParserError::InternalError;
        self.token_kind = TokenKind::Unknown;
        Ok(())
    }
}

impl GbnfSamplerTokenParserStateMachineContext for GbnfSamplerTokenParserContext {
    // Source mapping: actions.hpp::consume_text_token.
    fn consume_text_token(&mut self, _event_data: SamplerEventSampleRuntime) -> Result<(), ()> {
        self.error = TokenParserError::None;
        self.token_kind = TokenKind::TextToken;
        Ok(())
    }

    // Source mapping: actions.hpp::consume_empty_token.
    fn consume_empty_token(&mut self, _event_data: SamplerEventSampleRuntime) -> Result<(), ()> {
        self.error = TokenParserError::None;
        self.token_kind = TokenKind::EmptyToken;
        Ok(())
    }

    // Source mapping: actions.hpp::dispatch_parse_failed.
    fn dispatch_parse_failed(&mut self, _event_data: SamplerEventSampleRuntime) -> Result<(), ()> {
        self.error = TokenParserError::ParseFailed;
        self.token_kind = TokenKind::Unknown;
        Ok(())
    }

    // Source mapping: actions.hpp::on_unexpected, origin-explicit per row.
    fn on_unexpected_from_deciding(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_parsed(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_parse_failed(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_unexpected_event(&mut self) -> Result<(), ()> { self.unexpected() }

    // Source mapping: guards.hpp::candidate_text.
    fn candidate_text(&self, event_data: &SamplerEventSampleRuntime) -> Result<bool, ()> {
        Ok(event_data.input.error == TokenParserError::None
            && event_data.input.candidate_kind == CandidateKind::Text)
    }

    // Source mapping: guards.hpp::candidate_empty.
    fn candidate_empty(&self, event_data: &SamplerEventSampleRuntime) -> Result<bool, ()> {
        Ok(event_data.input.error == TokenParserError::None
            && event_data.input.candidate_kind == CandidateKind::Empty)
    }

    // Source mapping: guards.hpp::parse_failed.
    fn parse_failed(&self, event_data: &SamplerEventSampleRuntime) -> Result<bool, ()> {
        Ok(event_data.input.error == TokenParserError::None
            && !self.candidate_text(event_data)?
            && !self.candidate_empty(event_data)?)
    }
}

/// Synchronous bounded actor around the generated token-parser machine.
pub struct GbnfSamplerTokenParserActor {
    machine: GbnfSamplerTokenParserStateMachine<GbnfSamplerTokenParserContext>,
}

impl Default for GbnfSamplerTokenParserActor {
    fn default() -> Self { Self::new() }
}

impl GbnfSamplerTokenParserActor {
    /// Creates an actor in generated `deciding` state.
    #[must_use]
    pub fn new() -> Self {
        Self { machine: GbnfSamplerTokenParserStateMachine::new(Default::default()) }
    }

    /// Dispatches one copied runtime event to completion.
    pub fn process_event(&mut self, input: TokenParserInput) -> Result<TokenKind, TokenParserError> {
        self.machine.context_mut().set_input(input);
        if self
            .machine
            .process_event(GbnfSamplerTokenParserEvents::SamplerEventSampleRuntime(input.into()))
            .is_err()
        {
            self.machine.context_mut().error = TokenParserError::InternalError;
            self.machine.context_mut().token_kind = TokenKind::Unknown;
        }
        self.outcome()
    }

    /// Dispatches an explicit unexpected event.
    pub fn process_unexpected(&mut self) -> Result<TokenKind, TokenParserError> {
        let _ = self.machine.context_mut().unexpected();
        self.machine.set_state(GbnfSamplerTokenParserStates::UnexpectedEvent);
        self.outcome()
    }

    fn outcome(&self) -> Result<TokenKind, TokenParserError> {
        let context = self.machine.context();
        match context.error {
            TokenParserError::None => Ok(context.token_kind),
            error => Err(error),
        }
    }

    /// Returns generated state inspection data.
    #[must_use]
    pub fn state(&self) -> &GbnfSamplerTokenParserStates { self.machine.state() }

    /// Reports whether the generated machine is in `state`.
    #[must_use]
    pub fn is(&self, state: &GbnfSamplerTokenParserStates) -> bool { self.machine.is(state) }

    /// Returns the actor context for result inspection.
    #[must_use]
    pub fn context(&self) -> &GbnfSamplerTokenParserContext { self.machine.context() }
}

/// Short actor alias for token-parser callers.
pub type TokenParser = GbnfSamplerTokenParserActor;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_text_and_empty_candidates() {
        let mut parser = TokenParser::new();
        assert_eq!(parser.process_event(TokenParserInput::new(CandidateKind::Text)), Ok(TokenKind::TextToken));
        assert!(parser.is(&GbnfSamplerTokenParserStates::Parsed));

        let mut parser = TokenParser::new();
        assert_eq!(parser.process_event(TokenParserInput::new(CandidateKind::Empty)), Ok(TokenKind::EmptyToken));
        assert!(parser.is(&GbnfSamplerTokenParserStates::Parsed));
    }
    #[test]
    fn unknown_candidate_is_parse_failed() {
        let mut parser = TokenParser::new();
        assert_eq!(
            parser.process_event(TokenParserInput::new(CandidateKind::Unknown)),
            Err(TokenParserError::ParseFailed)
        );
        assert!(parser.is(&GbnfSamplerTokenParserStates::ParseFailed));
        assert_eq!(parser.context().token_kind, TokenKind::Unknown);
    }

    #[test]
    fn prior_errors_are_rejected_without_parse_failed_transition() {
        for input in [
            TokenParserInput::failed(TokenParserError::ParseFailed),
            TokenParserInput::invalid(),
        ] {
            let mut parser = TokenParser::new();
            assert_eq!(parser.process_event(input), Err(TokenParserError::InternalError));
            assert!(parser.is(&GbnfSamplerTokenParserStates::Deciding));
            assert_eq!(parser.context().token_kind, TokenKind::Unknown);
        }
    }

    #[test]
    fn explicit_unexpected_event_is_internal_error() {
        let mut parser = TokenParser::new();
        assert_eq!(parser.process_unexpected(), Err(TokenParserError::InternalError));
        assert!(parser.is(&GbnfSamplerTokenParserStates::UnexpectedEvent));
    }
}
