//! Source-aligned, allocation-free GBNF sampler accept parser.
//!
//! This child machine mirrors the pinned
//! `emel.cpp/src/emel/gbnf/sampler/accept_parser/{sm,events,guards,actions}.hpp`
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

/// Sampler errors relevant to the accept-parser boundary.
///
/// The discriminants mirror `sampler::error` in the pinned source.  The
/// parser itself turns every non-`None` event error into `ParseFailed`, as
/// `actions.hpp::dispatch_parse_failed` does.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum AcceptParserError {
    /// No error was reported by the preceding sampler stage.
    #[default]
    None = 0,
    /// The caller supplied an invalid request.
    InvalidRequest = 1,
    /// The preceding sampler stage failed to parse its input.
    ParseFailed = 2,
    /// An event was delivered outside the supported machine contract.
    InternalError = 4,
    /// The source pipeline reported an untracked error.
    Untracked = 8,
}

/// Result values from the pinned accept-parser `accept_result` enum.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum AcceptResult {
    /// No acceptance decision has been produced.
    #[default]
    Unknown = 0,
    /// The token ID is within the grammar's bounded rule/token range.
    Accepted = 1,
    /// The token ID is outside that range.
    Rejected = 2,
}

/// Copied, bounded input for one accept-parser dispatch.
///
/// This is the relevant projection of the pinned `sample_runtime` event:
/// `grammar.rule_count`, `ctx.current_token_id`, and `ctx.err`.  It contains
/// no references or heap-backed payloads, so dispatch remains synchronous and
/// allocation-free.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AcceptInput {
    /// Number of grammar rules available for token acceptance.
    pub grammar_rule_count: u32,
    /// Candidate token ID, or a negative value when no token was selected.
    pub current_token_id: i32,
    /// Error carried into this parser from the preceding sampler stage.
    pub error: AcceptParserError,
}

impl AcceptInput {
    /// Creates an error-free accept/reject input.
    #[must_use]
    pub const fn new(grammar_rule_count: u32, current_token_id: i32) -> Self {
        Self { grammar_rule_count, current_token_id, error: AcceptParserError::None }
    }

    /// Creates an input carrying a preceding sampler error.
    #[must_use]
    pub const fn failed(error: AcceptParserError) -> Self {
        Self { grammar_rule_count: 0, current_token_id: -1, error }
    }

    /// Creates an explicitly invalid request input.
    #[must_use]
    pub const fn invalid() -> Self { Self::failed(AcceptParserError::InvalidRequest) }
}

/// Copied runtime event corresponding to `sampler::event::sample_runtime`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SamplerEventSampleRuntime {
    /// Bounded input copied into the actor before dispatch.
    pub input: AcceptInput,
}

impl From<AcceptInput> for SamplerEventSampleRuntime {
    fn from(input: AcceptInput) -> Self { Self { input } }
}

// Source mapping: accept_parser/sm.hpp's deciding -> parsed, deciding ->
// parse_failed, terminal states, and all three explicit unexpected-event rows.
sml! {
    GbnfSamplerAcceptParser {
        *"deciding"_s + event<SamplerEventSampleRuntime>,
        "parsed"_s <= "deciding"_s + completion<SamplerEventSampleRuntime> [token_accepted_by_grammar] / consume_accepted,
        "parsed"_s <= "deciding"_s + completion<SamplerEventSampleRuntime> [token_rejected_by_grammar] / consume_rejected,
        "parse_failed"_s <= "deciding"_s + completion<SamplerEventSampleRuntime> [parse_failed] / dispatch_parse_failed,
        "parse_failed"_s <= "deciding"_s + unexpected_event<_> / on_unexpected_from_deciding,
        "parse_failed"_s <= "parsed"_s + unexpected_event<_> / on_unexpected_from_parsed,
        "parse_failed"_s <= "parse_failed"_s + unexpected_event<_> / on_unexpected_from_parse_failed,
        "parsed"_s = X,
        "parse_failed"_s = X,
    }
}

/// Context retained by the generated state machine.
#[derive(Clone, Copy, Debug)]
pub struct GbnfSamplerAcceptParserContext {
    /// Last copied grammar rule count.
    pub grammar_rule_count: u32,
    /// Last copied candidate token ID.
    pub current_token_id: i32,
    /// Error output corresponding to `sample_ctx::err`.
    pub error: AcceptParserError,
    /// Acceptance output corresponding to `sample_ctx::accept_result`.
    pub accept_result: AcceptResult,
}

impl Default for GbnfSamplerAcceptParserContext {
    fn default() -> Self {
        Self {
            grammar_rule_count: 0,
            current_token_id: -1,
            error: AcceptParserError::None,
            accept_result: AcceptResult::Unknown,
        }
    }
}

impl GbnfSamplerAcceptParserContext {
    fn set_input(&mut self, input: AcceptInput) {
        self.grammar_rule_count = input.grammar_rule_count;
        self.current_token_id = input.current_token_id;
        self.error = input.error;
        self.accept_result = AcceptResult::Unknown;
    }

    fn token_accepted(&self, input: &SamplerEventSampleRuntime) -> bool {
        input.input.error == AcceptParserError::None
            && input.input.current_token_id >= 0
            && (input.input.current_token_id as u32) < input.input.grammar_rule_count
    }

    fn unexpected(&mut self) -> Result<(), ()> {
        self.error = AcceptParserError::InternalError;
        self.accept_result = AcceptResult::Unknown;
        Ok(())
    }
}

impl GbnfSamplerAcceptParserStateMachineContext for GbnfSamplerAcceptParserContext {
    // Source mapping: actions.hpp::consume_accepted.
    fn consume_accepted(&mut self, _event_data: &SamplerEventSampleRuntime) -> Result<(), ()> {
        self.error = AcceptParserError::None;
        self.accept_result = AcceptResult::Accepted;
        Ok(())
    }

    // Source mapping: actions.hpp::consume_rejected.
    fn consume_rejected(&mut self, _event_data: &SamplerEventSampleRuntime) -> Result<(), ()> {
        self.error = AcceptParserError::None;
        self.accept_result = AcceptResult::Rejected;
        Ok(())
    }

    // Source mapping: actions.hpp::dispatch_parse_failed.
    fn dispatch_parse_failed(&mut self, _event_data: &SamplerEventSampleRuntime) -> Result<(), ()> {
        self.error = AcceptParserError::ParseFailed;
        self.accept_result = AcceptResult::Unknown;
        Ok(())
    }

    // Source mapping: actions.hpp::on_unexpected, origin-explicit per row.
    fn on_unexpected_from_deciding(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_parsed(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_parse_failed(&mut self) -> Result<(), ()> { self.unexpected() }

    // Source mapping: guards.hpp::parse_failed.
    fn parse_failed(&self, event_data: &SamplerEventSampleRuntime) -> Result<bool, ()> {
        Ok(event_data.input.error != AcceptParserError::None)
    }

    // Source mapping: guards.hpp::token_accepted_by_grammar.
    fn token_accepted_by_grammar(
        &self,
        event_data: &SamplerEventSampleRuntime,
    ) -> Result<bool, ()> {
        Ok(self.token_accepted(event_data))
    }

    // Source mapping: guards.hpp::token_rejected_by_grammar.
    fn token_rejected_by_grammar(
        &self,
        event_data: &SamplerEventSampleRuntime,
    ) -> Result<bool, ()> {
        Ok(event_data.input.error == AcceptParserError::None && !self.token_accepted(event_data))
    }
}

/// Synchronous bounded actor around the generated accept-parser machine.
pub struct GbnfSamplerAcceptParserActor {
    machine: GbnfSamplerAcceptParserStateMachine<GbnfSamplerAcceptParserContext>,
}

impl Default for GbnfSamplerAcceptParserActor {
    fn default() -> Self { Self::new() }
}

impl GbnfSamplerAcceptParserActor {
    /// Creates an actor in generated `deciding` state.
    #[must_use]
    pub fn new() -> Self {
        Self { machine: GbnfSamplerAcceptParserStateMachine::new(Default::default()) }
    }

    /// Dispatches one copied runtime event to completion.
    pub fn process_event(&mut self, input: AcceptInput) -> Result<AcceptResult, AcceptParserError> {
        if !self.machine.is(&GbnfSamplerAcceptParserStates::Deciding) {
            self.machine.context_mut().error = AcceptParserError::InternalError;
            self.machine.context_mut().accept_result = AcceptResult::Unknown;
            return self.outcome();
        }
        self.machine.context_mut().set_input(input);
        if self.machine.process_event(GbnfSamplerAcceptParserEvents::SamplerEventSampleRuntime(input.into())).is_err() {
            self.machine.context_mut().error = AcceptParserError::InternalError;
            self.machine.context_mut().accept_result = AcceptResult::Unknown;
        } else if self.machine.initialize().is_err() {
            self.machine.context_mut().error = AcceptParserError::InternalError;
            self.machine.context_mut().accept_result = AcceptResult::Unknown;
        }
        self.outcome()
    }

    /// Dispatches an explicit unexpected event.
    pub fn process_unexpected(&mut self) -> Result<AcceptResult, AcceptParserError> {
        let _ = self.machine.context_mut().unexpected();
        self.machine.set_state(GbnfSamplerAcceptParserStates::ParseFailed);
        self.outcome()
    }

    fn outcome(&self) -> Result<AcceptResult, AcceptParserError> {
        let context = self.machine.context();
        match context.error {
            AcceptParserError::None => Ok(context.accept_result),
            error => Err(error),
        }
    }


    /// Returns generated state inspection data.
    #[must_use]
    pub fn state(&self) -> &GbnfSamplerAcceptParserStates { self.machine.state() }

    /// Reports whether the generated machine is in `state`.
    #[must_use]
    pub fn is(&self, state: &GbnfSamplerAcceptParserStates) -> bool { self.machine.is(state) }

    /// Returns the actor context for result inspection.
    #[must_use]
    pub fn context(&self) -> &GbnfSamplerAcceptParserContext { self.machine.context() }
}

/// Short actor alias for accept-parser callers.
pub type AcceptParser = GbnfSamplerAcceptParserActor;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_token_in_bounded_grammar_range() {
        let mut parser = AcceptParser::new();
        assert_eq!(parser.process_event(AcceptInput::new(4, 3)), Ok(AcceptResult::Accepted));
        assert!(parser.is(&GbnfSamplerAcceptParserStates::X));
        assert_eq!(parser.context().accept_result, AcceptResult::Accepted);
    }

    #[test]
    fn rejects_negative_and_out_of_range_token_ids() {
        let mut parser = AcceptParser::new();
        assert_eq!(parser.process_event(AcceptInput::new(4, -1)), Ok(AcceptResult::Rejected));
        let mut parser = AcceptParser::new();
        assert_eq!(parser.process_event(AcceptInput::new(4, 4)), Ok(AcceptResult::Rejected));
        assert!(parser.is(&GbnfSamplerAcceptParserStates::X));
    }

    #[test]
    fn failed_and_invalid_inputs_are_parse_failed() {
        for input in [AcceptInput::failed(AcceptParserError::ParseFailed), AcceptInput::invalid()] {
            let mut parser = AcceptParser::new();
            assert_eq!(parser.process_event(input), Err(AcceptParserError::ParseFailed));
            assert!(parser.is(&GbnfSamplerAcceptParserStates::X));
            assert_eq!(parser.context().accept_result, AcceptResult::Unknown);
        }
    }

    #[test]
    fn explicit_unexpected_event_is_internal_error() {
        let mut parser = AcceptParser::new();
        assert_eq!(parser.process_unexpected(), Err(AcceptParserError::InternalError));
        assert!(parser.is(&GbnfSamplerAcceptParserStates::ParseFailed));
    }
}
