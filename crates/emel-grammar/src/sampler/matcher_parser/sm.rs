//! Source-aligned, allocation-free GBNF sampler matcher parser.
//!
//! Mirrors the pinned matcher-parser state machine while using copied bounded
//! input and owned result values at the Rust actor boundary.

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

/// Token categories consumed by the pinned matcher parser.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum TokenKind {
    /// No token was classified.
    #[default]
    Unknown = 0,
    /// A text token matched by the grammar.
    Text = 1,
    /// An empty token rejected by the matcher.
    Empty = 2,
}

/// Values corresponding to the pinned `match_result` enum.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum MatchResult {
    /// No match decision has been produced.
    #[default]
    Unknown = 0,
    /// The text token was accepted.
    Accepted = 1,
    /// The empty token was rejected.
    Rejected = 2,
}

/// Values corresponding to the sampler accept-parser result.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum AcceptResult {
    /// No acceptance decision has been produced.
    #[default]
    Unknown = 0,
    /// The candidate is accepted.
    Accepted = 1,
    /// The candidate is rejected.
    Rejected = 2,
}

/// Errors represented by the pinned sampler matcher-parser boundary.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum MatcherParserError {
    /// No error was reported by the preceding sampler stage.
    #[default]
    None = 0,
    /// The caller supplied an invalid request.
    InvalidRequest = 1,
    /// The token kind could not be matched.
    ParseFailed = 2,
    /// An event arrived outside the supported machine contract.
    InternalError = 4,
    /// The preceding sampler stage reported an untracked error.
    Untracked = 8,
}

/// Copied, bounded input for one matcher-parser dispatch.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MatcherInput {
    /// Token category produced by the token parser.
    pub token_kind: TokenKind,
    /// Error carried into this parser from the preceding sampler stage.
    pub error: MatcherParserError,
}

impl MatcherInput {
    /// Creates an error-free matcher input.
    #[must_use]
    pub const fn new(token_kind: TokenKind) -> Self {
        Self { token_kind, error: MatcherParserError::None }
    }

    /// Creates an input carrying a preceding sampler error.
    #[must_use]
    pub const fn failed(error: MatcherParserError) -> Self {
        Self { token_kind: TokenKind::Unknown, error }
    }

    /// Creates an explicitly invalid request input.
    #[must_use]
    pub const fn invalid() -> Self { Self::failed(MatcherParserError::InvalidRequest) }
}

/// Copied runtime event corresponding to `sampler::event::sample_runtime`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SamplerEventSampleRuntime {
    /// Bounded input copied into the actor before dispatch.
    pub input: MatcherInput,
}

impl From<MatcherInput> for SamplerEventSampleRuntime {
    fn from(input: MatcherInput) -> Self { Self { input } }
}

// Source mapping: matcher_parser/sm.hpp's deciding -> parsed, deciding ->
// parse_failed, terminal states, and all four explicit unexpected-event rows.
sml! {
    GbnfSamplerMatcherParser {
        *"deciding"_s + event<SamplerEventSampleRuntime>,
        "parsed"_s <= "deciding"_s + completion<SamplerEventSampleRuntime> [token_text] / consume_match_accepted,
        "parsed"_s <= "deciding"_s + completion<SamplerEventSampleRuntime> [token_empty] / consume_match_rejected,
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
pub struct GbnfSamplerMatcherParserContext {
    /// Last copied token category.
    pub token_kind: TokenKind,
    /// Error output corresponding to `sample_ctx::err`.
    pub error: MatcherParserError,
    /// Match output corresponding to `sample_ctx::match_result`.
    pub match_result: MatchResult,
    /// Candidate allowance corresponding to `sample_ctx::candidate_allowed`.
    pub candidate_allowed: bool,
    /// Acceptance output corresponding to `sample_ctx::accept_result`.
    pub accept_result: AcceptResult,
}

impl Default for GbnfSamplerMatcherParserContext {
    fn default() -> Self {
        Self {
            token_kind: TokenKind::Unknown,
            error: MatcherParserError::None,
            match_result: MatchResult::Unknown,
            candidate_allowed: false,
            accept_result: AcceptResult::Unknown,
        }
    }
}

impl GbnfSamplerMatcherParserContext {
    fn set_input(&mut self, input: MatcherInput) {
        self.token_kind = input.token_kind;
        self.error = input.error;
        self.match_result = MatchResult::Unknown;
        self.candidate_allowed = false;
        self.accept_result = AcceptResult::Unknown;
    }

    fn unexpected(&mut self) -> Result<(), ()> {
        // Source mapping: actions.hpp::on_unexpected only writes `err`.
        self.error = MatcherParserError::InternalError;
        Ok(())
    }
}

impl GbnfSamplerMatcherParserStateMachineContext for GbnfSamplerMatcherParserContext {
    // Source mapping: actions.hpp::consume_match_accepted.
    fn consume_match_accepted(&mut self, _event_data: &SamplerEventSampleRuntime) -> Result<(), ()> {
        self.error = MatcherParserError::None;
        self.match_result = MatchResult::Accepted;
        self.candidate_allowed = true;
        self.accept_result = AcceptResult::Accepted;
        Ok(())
    }

    // Source mapping: actions.hpp::consume_match_rejected.
    fn consume_match_rejected(&mut self, _event_data: &SamplerEventSampleRuntime) -> Result<(), ()> {
        self.error = MatcherParserError::None;
        self.match_result = MatchResult::Rejected;
        self.candidate_allowed = false;
        self.accept_result = AcceptResult::Rejected;
        Ok(())
    }

    // Source mapping: actions.hpp::dispatch_parse_failed.
    fn dispatch_parse_failed(&mut self, _event_data: &SamplerEventSampleRuntime) -> Result<(), ()> {
        self.error = MatcherParserError::ParseFailed;
        self.match_result = MatchResult::Unknown;
        self.candidate_allowed = false;
        self.accept_result = AcceptResult::Unknown;
        Ok(())
    }

    // Source mapping: actions.hpp::on_unexpected, origin-explicit per row.
    fn on_unexpected_from_deciding(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_parsed(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_parse_failed(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_unexpected_event(&mut self) -> Result<(), ()> { self.unexpected() }

    // Source mapping: guards.hpp::parse_failed.
    fn parse_failed(&self, event_data: &SamplerEventSampleRuntime) -> Result<bool, ()> {
        Ok(event_data.input.error == MatcherParserError::None
            && !self.token_text(event_data)?
            && !self.token_empty(event_data)?)
    }

    // Source mapping: guards.hpp::token_empty.
    fn token_empty(&self, event_data: &SamplerEventSampleRuntime) -> Result<bool, ()> {
        Ok(event_data.input.error == MatcherParserError::None
            && event_data.input.token_kind == TokenKind::Empty)
    }

    // Source mapping: guards.hpp::token_text.
    fn token_text(&self, event_data: &SamplerEventSampleRuntime) -> Result<bool, ()> {
        Ok(event_data.input.error == MatcherParserError::None
            && event_data.input.token_kind == TokenKind::Text)
    }
}

/// Synchronous bounded actor around the generated matcher-parser machine.
pub struct GbnfSamplerMatcherParserActor {
    machine: GbnfSamplerMatcherParserStateMachine<GbnfSamplerMatcherParserContext>,
}

impl Default for GbnfSamplerMatcherParserActor {
    fn default() -> Self { Self::new() }
}

impl GbnfSamplerMatcherParserActor {
    /// Creates an actor in generated `deciding` state.
    #[must_use]
    pub fn new() -> Self {
        Self { machine: GbnfSamplerMatcherParserStateMachine::new(Default::default()) }
    }

    /// Dispatches one copied runtime event to completion.
    pub fn process_event(&mut self, input: MatcherInput) -> Result<MatchResult, MatcherParserError> {
        if !self.machine.is(&GbnfSamplerMatcherParserStates::Deciding) {
            self.machine.context_mut().error = MatcherParserError::InternalError;
            return self.outcome();
        }
        self.machine.context_mut().set_input(input);
        if self
            .machine
            .process_event(GbnfSamplerMatcherParserEvents::SamplerEventSampleRuntime(input.into()))
            .is_err()
        {
            self.machine.context_mut().error = MatcherParserError::InternalError;
        }
        self.outcome()
    }

    /// Dispatches an explicit unexpected event.
    pub fn process_unexpected(&mut self) -> Result<MatchResult, MatcherParserError> {
        let _ = self.machine.context_mut().unexpected();
        self.machine.set_state(GbnfSamplerMatcherParserStates::UnexpectedEvent);
        self.outcome()
    }

    fn outcome(&self) -> Result<MatchResult, MatcherParserError> {
        let context = self.machine.context();
        match context.error {
            MatcherParserError::None => Ok(context.match_result),
            error => Err(error),
        }
    }

    /// Returns generated state inspection data.
    #[must_use]
    pub fn state(&self) -> &GbnfSamplerMatcherParserStates { self.machine.state() }

    /// Reports whether the generated machine is in `state`.
    #[must_use]
    pub fn is(&self, state: &GbnfSamplerMatcherParserStates) -> bool { self.machine.is(state) }

    /// Returns the actor context for result inspection.
    #[must_use]
    pub fn context(&self) -> &GbnfSamplerMatcherParserContext { self.machine.context() }
}

/// Short actor alias for matcher-parser callers.
pub type MatcherParser = GbnfSamplerMatcherParserActor;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_text_and_allows_candidate() {
        let mut parser = MatcherParser::new();
        assert_eq!(parser.process_event(MatcherInput::new(TokenKind::Text)), Ok(MatchResult::Accepted));
        assert!(parser.is(&GbnfSamplerMatcherParserStates::Parsed));
        assert_eq!(parser.context().match_result, MatchResult::Accepted);
        assert!(parser.context().candidate_allowed);
        assert_eq!(parser.context().accept_result, AcceptResult::Accepted);
    }

    #[test]
    fn rejects_empty_and_disallows_candidate() {
        let mut parser = MatcherParser::new();
        assert_eq!(parser.process_event(MatcherInput::new(TokenKind::Empty)), Ok(MatchResult::Rejected));
        assert!(parser.is(&GbnfSamplerMatcherParserStates::Parsed));
        assert!(!parser.context().candidate_allowed);
        assert_eq!(parser.context().accept_result, AcceptResult::Rejected);
    }

    #[test]
    fn unknown_token_is_parse_failed() {
        let mut parser = MatcherParser::new();
        assert_eq!(parser.process_event(MatcherInput::new(TokenKind::Unknown)), Err(MatcherParserError::ParseFailed));
        assert!(parser.is(&GbnfSamplerMatcherParserStates::ParseFailed));
        assert_eq!(parser.context().match_result, MatchResult::Unknown);
        assert!(!parser.context().candidate_allowed);
        assert_eq!(parser.context().accept_result, AcceptResult::Unknown);
    }

    #[test]
    fn explicit_unexpected_event_is_internal_error() {
        let mut parser = MatcherParser::new();
        assert_eq!(parser.process_unexpected(), Err(MatcherParserError::InternalError));
        assert!(parser.is(&GbnfSamplerMatcherParserStates::UnexpectedEvent));
    }
}
