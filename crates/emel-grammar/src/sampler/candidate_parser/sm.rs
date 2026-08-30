//! Source-aligned synchronous GBNF sampler candidate parser.

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
)]

use sml::sml;

/// Candidate classifications from the pinned `candidate_kind` enum.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum CandidateKind {
    /// No candidate classification is available.
    #[default]
    Unknown = 0,
    /// The candidate contains text to apply.
    Text = 1,
    /// The candidate is empty.
    Empty = 2,
}

/// Error values carried by the sampler runtime context.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum CandidateParserError {
    /// No prior sampler error.
    #[default]
    None = 0,
    /// The candidate parser rejected the input.
    ParseFailed = 1 << 1,
    /// An event was not valid for the current machine state.
    InternalError = 1 << 2,
}

/// Owned result produced by one candidate-parser dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CandidateParserOutcome {
    /// The candidate was classified.
    Parsed(CandidateKind),
    /// The candidate kind could not be classified.
    ParseFailed,
    /// An event arrived after classification or was otherwise unexpected.
    Unexpected,
}

/// Copied runtime input corresponding to `sampler::event::sample_runtime`.
///
/// The C++ event carries references into the parent sampler context. This
/// bounded port copies only the fields read by candidate-parser guards.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SamplerEventSampleRuntime {
    /// Prior sampler error observed by the candidate parser.
    pub err: CandidateParserError,
    /// Candidate classification supplied by the sampler.
    pub candidate_kind: CandidateKind,
}

impl SamplerEventSampleRuntime {
    /// Constructs a no-error runtime input with the supplied candidate kind.
    #[must_use]
    pub const fn new(candidate_kind: CandidateKind) -> Self {
        Self {
            err: CandidateParserError::None,
            candidate_kind,
        }
    }

    /// Constructs a runtime input carrying an existing sampler error.
    #[must_use]
    pub const fn with_error(err: CandidateParserError) -> Self {
        Self {
            err,
            candidate_kind: CandidateKind::Unknown,
        }
    }
}

// Source mapping: candidate_parser/sm.hpp, with explicit destination-first
// rows matching the pinned deciding/parsed/parse_failed topology.
sml! {
    GbnfSamplerCandidateParser {
        *"deciding"_s + event<SamplerEventSampleRuntime>,
        "parsed"_s <= "deciding"_s + completion<SamplerEventSampleRuntime> [has_apply_text] / consume_text,
        "parsed"_s <= "deciding"_s + completion<SamplerEventSampleRuntime> [has_empty_apply_text] / consume_empty,
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
#[derive(Debug)]
pub struct GbnfSamplerCandidateParserContext {
    input: SamplerEventSampleRuntime,
    result: CandidateParserOutcome,
}

impl Default for GbnfSamplerCandidateParserContext {
    fn default() -> Self {
        Self {
            input: SamplerEventSampleRuntime::default(),
            result: CandidateParserOutcome::Parsed(CandidateKind::Unknown),
        }
    }
}

impl GbnfSamplerCandidateParserContext {
    fn set_input(&mut self, input: SamplerEventSampleRuntime) {
        self.input = input;
        self.result = CandidateParserOutcome::Parsed(CandidateKind::Unknown);
    }

    fn consume(&mut self, kind: CandidateKind) -> Result<(), ()> {
        self.result = CandidateParserOutcome::Parsed(kind);
        Ok(())
    }

    fn dispatch_parse_failed_result(&mut self) -> Result<(), ()> {
        self.result = CandidateParserOutcome::ParseFailed;
        Ok(())
    }

    fn unexpected(&mut self) -> Result<(), ()> {
        self.result = CandidateParserOutcome::Unexpected;
        Ok(())
    }
}

impl GbnfSamplerCandidateParserStateMachineContext for GbnfSamplerCandidateParserContext {
    // Source mapping: candidate_parser/actions.hpp::consume_empty.
    fn consume_empty(&mut self, _event_data: &SamplerEventSampleRuntime) -> Result<(), ()> {
        self.consume(CandidateKind::Empty)
    }

    // Source mapping: candidate_parser/actions.hpp::consume_text.
    fn consume_text(&mut self, _event_data: &SamplerEventSampleRuntime) -> Result<(), ()> {
        self.consume(CandidateKind::Text)
    }

    // Source mapping: candidate_parser/actions.hpp::dispatch_parse_failed.
    fn dispatch_parse_failed(&mut self, _event_data: &SamplerEventSampleRuntime) -> Result<(), ()> {
        self.dispatch_parse_failed_result()
    }

    // Source mapping: candidate_parser/actions.hpp::on_unexpected.
    fn on_unexpected_from_deciding(&mut self) -> Result<(), ()> {
        self.unexpected()
    }

    fn on_unexpected_from_parse_failed(&mut self) -> Result<(), ()> {
        self.unexpected()
    }

    fn on_unexpected_from_parsed(&mut self) -> Result<(), ()> {
        self.unexpected()
    }

    fn on_unexpected_from_unexpected_event(&mut self) -> Result<(), ()> {
        self.unexpected()
    }

    // Source mapping: candidate_parser/guards.hpp::has_apply_text.
    fn has_apply_text(&self, event_data: &SamplerEventSampleRuntime) -> Result<bool, ()> {
        Ok(event_data.err == CandidateParserError::None
            && event_data.candidate_kind == CandidateKind::Text)
    }

    // Source mapping: candidate_parser/guards.hpp::has_empty_apply_text.
    fn has_empty_apply_text(&self, event_data: &SamplerEventSampleRuntime) -> Result<bool, ()> {
        Ok(event_data.err == CandidateParserError::None
            && event_data.candidate_kind == CandidateKind::Empty)
    }

    // Source mapping: candidate_parser/guards.hpp::parse_failed.
    fn parse_failed(&self, event_data: &SamplerEventSampleRuntime) -> Result<bool, ()> {
        Ok(event_data.err == CandidateParserError::None
            && !self.has_apply_text(event_data)?
            && !self.has_empty_apply_text(event_data)?)
    }
}

/// Synchronous actor around the generated candidate-parser machine.
pub struct GbnfSamplerCandidateParserActor {
    machine: GbnfSamplerCandidateParserStateMachine<GbnfSamplerCandidateParserContext>,
}

impl Default for GbnfSamplerCandidateParserActor {
    fn default() -> Self {
        Self::new()
    }
}

impl GbnfSamplerCandidateParserActor {
    /// Creates an actor in the generated initial state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: GbnfSamplerCandidateParserStateMachine::new(Default::default()),
        }
    }

    /// Dispatches one copied runtime input to completion.
    pub fn process_event(&mut self, input: SamplerEventSampleRuntime) -> CandidateParserOutcome {
        if !self.machine.is(&GbnfSamplerCandidateParserStates::Deciding) {
            return self.process_unexpected();
        }
        self.machine.context_mut().set_input(input);
        let _ = self.machine.process_event(
            GbnfSamplerCandidateParserEvents::SamplerEventSampleRuntime(input),
        );
        self.machine.context().result
    }

    /// Classifies a copied candidate kind.
    pub fn classify(&mut self, candidate_kind: CandidateKind) -> CandidateParserOutcome {
        self.process_event(SamplerEventSampleRuntime::new(candidate_kind))
    }
    pub fn process_unexpected(&mut self) -> CandidateParserOutcome {
        let _ = self.machine.context_mut().unexpected();
        self.machine
            .set_state(GbnfSamplerCandidateParserStates::UnexpectedEvent);
        self.machine.context().result
    }

    /// Returns generated state inspection data.
    #[must_use]
    pub fn state(&self) -> &GbnfSamplerCandidateParserStates {
        self.machine.state()
    }

    /// Reports whether the generated machine is in `state`.
    #[must_use]
    pub fn is(&self, state: &GbnfSamplerCandidateParserStates) -> bool {
        self.machine.is(state)
    }

    /// Returns the retained copied input and result context.
    #[must_use]
    pub fn context(&self) -> &GbnfSamplerCandidateParserContext {
        self.machine.context()
    }
}

/// Short actor alias for candidate-parser callers.
pub type CandidateParser = GbnfSamplerCandidateParserActor;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_text_and_empty_candidates() {
        let mut parser = CandidateParser::new();
        assert_eq!(parser.classify(CandidateKind::Text), CandidateParserOutcome::Parsed(CandidateKind::Text));
        assert!(parser.is(&GbnfSamplerCandidateParserStates::Parsed));

        let mut parser = CandidateParser::new();
        assert_eq!(parser.classify(CandidateKind::Empty), CandidateParserOutcome::Parsed(CandidateKind::Empty));
        assert!(parser.is(&GbnfSamplerCandidateParserStates::Parsed));
    }

    #[test]
    fn unknown_candidate_is_parse_failed() {
        let mut parser = CandidateParser::new();
        assert_eq!(parser.classify(CandidateKind::Unknown), CandidateParserOutcome::ParseFailed);
        assert!(parser.is(&GbnfSamplerCandidateParserStates::ParseFailed));
    }

    #[test]
    fn explicit_unexpected_event_is_reported() {
        let mut parser = CandidateParser::new();
        assert_eq!(parser.process_unexpected(), CandidateParserOutcome::Unexpected);
        assert!(parser.is(&GbnfSamplerCandidateParserStates::UnexpectedEvent));
    }
}
