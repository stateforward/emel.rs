//! Source-aligned, bounded GBNF sampler actor.
//!
//! The transition table mirrors the pinned sampler machine. Candidate buffers
//! remain caller-owned, and the dispatch runs to completion synchronously.

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

use crate::gbnf::{grammar, k_max_gbnf_rule_elements};
use super::accept_parser::sm::{AcceptInput, AcceptParser, AcceptParserError, AcceptResult};
use super::candidate_parser::sm::{CandidateKind, CandidateParser, CandidateParserOutcome};
use super::matcher_parser::sm::{MatchResult, MatcherInput, MatcherParser, MatcherParserError, TokenKind as MatcherTokenKind};
use super::token_parser::sm::{CandidateKind as TokenCandidateKind, TokenParser, TokenParserError, TokenParserInput, TokenKind};

/// Errors represented by the pinned GBNF sampler boundary.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum SamplerError {
    /// No error was reported.
    #[default]
    None = 0,
    /// The caller supplied an invalid request or output capacity.
    InvalidRequest = 1,
    /// No candidate survived grammar filtering.
    ParseFailed = 2,
    /// An event or child-machine dispatch violated the contract.
    InternalError = 4,
    /// An error not classified by this boundary was observed.
    Untracked = 8,
}

/// Successful result published by one sampler dispatch.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SampleDone {
    /// Number of candidates retained at the front of the caller's buffers.
    pub candidate_count: i32,
    /// Selected token output. The GBNF filter leaves this unchanged.
    pub selected_token: i32,
}

/// Bounded sampler request over caller-owned candidate buffers.
#[derive(Debug)]
pub struct Sample<'a> {
    /// Candidate token identifiers.
    pub candidate_ids: &'a mut [i32],
    /// Candidate scores paired with `candidate_ids`.
    pub candidate_scores: &'a mut [f32],
    /// Number of input candidates.
    pub candidate_count: i32,
    /// Number of candidates written by the sampler.
    pub candidate_count_out: &'a mut i32,
    /// Selected token output, left unchanged by this filter.
    pub selected_token_out: &'a mut i32,
    /// Error output, reset and published during dispatch.
    pub error_out: &'a mut SamplerError,
}

impl<'a> Sample<'a> {
    /// Creates a request over caller-owned candidate buffers.
    #[must_use]
    pub const fn new(
        candidate_ids: &'a mut [i32],
        candidate_scores: &'a mut [f32],
        candidate_count: i32,
        candidate_count_out: &'a mut i32,
        selected_token_out: &'a mut i32,
        error_out: &'a mut SamplerError,
    ) -> Self {
        Self { candidate_ids, candidate_scores, candidate_count, candidate_count_out, selected_token_out, error_out }
    }
}

/// Runtime event corresponding to the pinned `sample_runtime` event.
#[derive(Debug)]
pub struct EventSampleRuntime<'a> {
    /// Caller-owned request and outputs for this dispatch.
    pub request: Sample<'a>,
}

impl<'a> EventSampleRuntime<'a> {
    /// Wraps a sampler request as a runtime event.
    #[must_use]
    pub const fn new(request: Sample<'a>) -> Self { Self { request } }
}

sml! {
    GbnfSampler {
        "request_decision"_s <= *"ready"_s + Sample(&'dispatch EventSampleRuntime<'dispatch>) / begin_sample,
        "filter_candidates"_s <= "request_decision"_s + completion<Sample>(&'dispatch EventSampleRuntime<'dispatch>) [valid_sample_request],
        "errored"_s <= "request_decision"_s + completion<Sample>(&'dispatch EventSampleRuntime<'dispatch>) [invalid_sample_request] / mark_invalid_request,
        "finalize_decision"_s <= "filter_candidates"_s + completion<Sample>(&'dispatch EventSampleRuntime<'dispatch>) / filter_candidates,
        "done"_s <= "finalize_decision"_s + completion<Sample>(&'dispatch EventSampleRuntime<'dispatch>) [filtered_candidates_available],
        "errored"_s <= "finalize_decision"_s + completion<Sample>(&'dispatch EventSampleRuntime<'dispatch>) [no_filtered_candidates] / mark_parse_failed,
        "ready"_s <= "done"_s + completion<Sample>(&'dispatch EventSampleRuntime<'dispatch>) / publish_done,
        "ready"_s <= "errored"_s + completion<Sample>(&'dispatch EventSampleRuntime<'dispatch>) / publish_error,
        "ready"_s <= "ready"_s + unexpected_event<_> / on_unexpected_from_ready,
        "ready"_s <= "request_decision"_s + unexpected_event<_> / on_unexpected_from_request_decision,
        "ready"_s <= "filter_candidates"_s + unexpected_event<_> / on_unexpected_from_filter_candidates,
        "ready"_s <= "finalize_decision"_s + unexpected_event<_> / on_unexpected_from_finalize_decision,
        "ready"_s <= "done"_s + unexpected_event<_> / on_unexpected_from_done,
        "ready"_s <= "errored"_s + unexpected_event<_> / on_unexpected_from_errored,
    }
}

/// Persistent bounded context for the generated sampler machine.
#[derive(Debug)]
pub struct GbnfSamplerContext {
    /// Number of grammar rules available to acceptance.
    pub grammar_rule_count: u32,
    /// Starting grammar rule selected by the caller.
    pub start_rule_id: u32,
    /// Bounded frontier storage matching the pinned context layout.
    pub frontier: [u32; k_max_gbnf_rule_elements],
    /// Bounded scratch storage matching the pinned context layout.
    pub scratch: [u32; k_max_gbnf_rule_elements],
    /// Number of active frontier entries.
    pub frontier_size: u32,
    /// Active grammar rule.
    pub active_rule_id: u32,
    /// Current sampler error.
    pub err: SamplerError,
    /// Number of candidates inspected.
    pub read_index: i32,
    /// Number of candidates retained.
    pub write_index: i32,
    /// Candidate currently being inspected.
    pub current_token_id: i32,
    /// Last candidate-parser classification.
    pub candidate_kind: CandidateKind,
    /// Last token-parser classification.
    pub token_kind: TokenKind,
    /// Last matcher result.
    pub match_result: MatchResult,
    /// Whether the current candidate passed matching.
    pub candidate_allowed: bool,
    /// Last accept-parser result.
    pub accept_result: AcceptResult,
}

impl Default for GbnfSamplerContext {
    fn default() -> Self {
        Self {
            grammar_rule_count: 0,
            start_rule_id: 0,
            frontier: [0; k_max_gbnf_rule_elements],
            scratch: [0; k_max_gbnf_rule_elements],
            frontier_size: 0,
            active_rule_id: 0,
            err: SamplerError::None,
            read_index: 0,
            write_index: 0,
            current_token_id: -1,
            candidate_kind: CandidateKind::Unknown,
            token_kind: TokenKind::Unknown,
            match_result: MatchResult::Unknown,
            candidate_allowed: false,
            accept_result: AcceptResult::Unknown,
        }
    }
}

impl GbnfSamplerContext {
    fn reset_runtime(&mut self) {
        self.err = SamplerError::None;
        self.read_index = 0;
        self.write_index = 0;
        self.current_token_id = -1;
        self.candidate_kind = CandidateKind::Unknown;
        self.token_kind = TokenKind::Unknown;
        self.match_result = MatchResult::Unknown;
        self.candidate_allowed = false;
        self.accept_result = AcceptResult::Unknown;
    }

    fn unexpected(&mut self) -> Result<(), ()> {
        self.err = SamplerError::InternalError;
        Ok(())
    }
}

impl GbnfSamplerStateMachineContext for GbnfSamplerContext {
    // Source mapping: sampler/actions.hpp::begin_sample.
    fn begin_sample(&mut self, event: &EventSampleRuntime<'_>) -> Result<(), ()> {
        self.reset_runtime();
        *event.request.error_out = SamplerError::None;
        Ok(())
    }

    // Source mapping: sampler/actions.hpp::filter_candidates. Child actors
    // are run to completion per candidate; no deferred work is retained.
    fn filter_candidates(&mut self, event: &EventSampleRuntime<'_>) -> Result<(), ()> {
        let count = usize::try_from(event.request.candidate_count).map_err(|_| ())?;
        let mut write = 0usize;
        for index in 0..count {
            let token_id = event.request.candidate_ids[index];
            self.read_index = i32::try_from(index + 1).map_err(|_| ())?;
            self.current_token_id = token_id;

            let candidate_kind = if token_id >= 0 { CandidateKind::Text } else { CandidateKind::Empty };
            let candidate_kind = match CandidateParser::new().classify(candidate_kind) {
                CandidateParserOutcome::Parsed(kind) => kind,
                CandidateParserOutcome::ParseFailed => { self.err = SamplerError::ParseFailed; break; }
                CandidateParserOutcome::Unexpected => { self.err = SamplerError::InternalError; break; }
            };
            self.candidate_kind = candidate_kind;

            let token_kind = match TokenParser::new().process_event(TokenParserInput {
                candidate_kind: match candidate_kind {
                    CandidateKind::Text => TokenCandidateKind::Text,
                    CandidateKind::Empty => TokenCandidateKind::Empty,
                    CandidateKind::Unknown => TokenCandidateKind::Unknown,
                },
                error: TokenParserError::None,
            }) {
                Ok(kind) => kind,
                Err(TokenParserError::ParseFailed) => { self.err = SamplerError::ParseFailed; break; }
                Err(_) => { self.err = SamplerError::InternalError; break; }
            };
            self.token_kind = token_kind;

            let match_result = match MatcherParser::new().process_event(MatcherInput {
                token_kind: match token_kind {
                    TokenKind::TextToken => MatcherTokenKind::Text,
                    TokenKind::EmptyToken => MatcherTokenKind::Empty,
                    TokenKind::Unknown => MatcherTokenKind::Unknown,
                },
                error: MatcherParserError::None,
            }) {
                Ok(result) => result,
                Err(MatcherParserError::ParseFailed) => { self.err = SamplerError::ParseFailed; break; }
                Err(_) => { self.err = SamplerError::InternalError; break; }
            };
            self.match_result = match_result;
            self.candidate_allowed = match_result == MatchResult::Accepted;

            let accept_result = match AcceptParser::new().process_event(AcceptInput::new(self.grammar_rule_count, token_id)) {
                Ok(result) => result,
                Err(AcceptParserError::ParseFailed) => { self.err = SamplerError::ParseFailed; break; }
                Err(_) => { self.err = SamplerError::InternalError; break; }
            };
            self.accept_result = accept_result;
            if self.candidate_allowed && accept_result == AcceptResult::Accepted {
                event.request.candidate_ids[write] = token_id;
                event.request.candidate_scores[write] = event.request.candidate_scores[index];
                write += 1;
            }
        }
        self.write_index = i32::try_from(write).map_err(|_| ())?;
        Ok(())
    }

    fn filtered_candidates_available(&self, _event: &EventSampleRuntime<'_>) -> Result<bool, ()> {
        Ok(self.err == SamplerError::None && self.write_index > 0)
    }

    fn invalid_sample_request(&self, event: &EventSampleRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.valid_sample_request(event)?)
    }

    // Source mapping: sampler/actions.hpp::mark_invalid_request.
    fn mark_invalid_request(&mut self, event: &EventSampleRuntime<'_>) -> Result<(), ()> {
        self.err = SamplerError::InvalidRequest;
        self.write_index = 0;
        *event.request.candidate_count_out = 0;
        *event.request.error_out = self.err;
        Ok(())
    }

    // Source mapping: sampler/actions.hpp::mark_parse_failed.
    fn mark_parse_failed(&mut self, event: &EventSampleRuntime<'_>) -> Result<(), ()> {
        self.err = SamplerError::ParseFailed;
        *event.request.candidate_count_out = self.write_index;
        *event.request.error_out = self.err;
        Ok(())
    }

    fn no_filtered_candidates(&self, _event: &EventSampleRuntime<'_>) -> Result<bool, ()> {
        Ok(self.err == SamplerError::None && self.write_index == 0)
    }

    fn on_unexpected_from_done(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_errored(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_filter_candidates(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_finalize_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_ready(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_request_decision(&mut self) -> Result<(), ()> { self.unexpected() }

    // Source mapping: sampler/actions.hpp::publish_done.
    fn publish_done(&mut self, event: &EventSampleRuntime<'_>) -> Result<(), ()> {
        self.err = SamplerError::None;
        *event.request.candidate_count_out = self.write_index;
        *event.request.error_out = SamplerError::None;
        Ok(())
    }

    // Source mapping: sampler/actions.hpp::publish_error.
    fn publish_error(&mut self, event: &EventSampleRuntime<'_>) -> Result<(), ()> {
        *event.request.candidate_count_out = self.write_index;
        *event.request.error_out = self.err;
        Ok(())
    }

    fn valid_sample_request(&self, event: &EventSampleRuntime<'_>) -> Result<bool, ()> {
        let count = usize::try_from(event.request.candidate_count).ok();
        Ok(event.request.candidate_count > 0
            && self.grammar_rule_count > 0
            && self.start_rule_id < self.grammar_rule_count
            && count.is_some_and(|n| n <= event.request.candidate_ids.len() && n <= event.request.candidate_scores.len()))
    }
}

/// Single-writer, synchronous sampler actor.
pub struct GbnfSampler {
    machine: GbnfSamplerStateMachine<GbnfSamplerContext>,
}

impl core::fmt::Debug for GbnfSampler {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.debug_struct("GbnfSampler").field("state", self.machine.state()).finish_non_exhaustive()
    }
}

impl Default for GbnfSampler {
    fn default() -> Self { Self::new() }
}

impl GbnfSampler {
    /// Creates an unconfigured sampler in generated `ready` state.
    #[must_use]
    pub fn new() -> Self {
        Self { machine: GbnfSamplerStateMachine::new(GbnfSamplerContext::default()) }
    }

    /// Creates a sampler using the grammar's bounded rule count.
    #[must_use]
    pub fn with_grammar(grammar: &grammar, start_rule_id: u32) -> Self {
        let mut context = GbnfSamplerContext::default();
        context.grammar_rule_count = grammar.rule_count;
        context.start_rule_id = start_rule_id;
        context.active_rule_id = start_rule_id;
        Self { machine: GbnfSamplerStateMachine::new(context) }
    }

    /// Dispatches one runtime request synchronously to completion.
    pub fn process_event(&mut self, request: EventSampleRuntime<'_>) -> Result<SampleDone, SamplerError> {
        if !self.machine.is(&GbnfSamplerStates::Ready) {
            self.machine.context_mut().err = SamplerError::InternalError;
            *request.request.candidate_count_out = 0;
            *request.request.error_out = SamplerError::InternalError;
            self.machine.set_state(GbnfSamplerStates::Ready);
            return Err(SamplerError::InternalError);
        }
        if self.machine.process_event(GbnfSamplerEvents::Sample(&request)).is_err() {
            self.machine.context_mut().err = SamplerError::InternalError;
            *request.request.candidate_count_out = 0;
            *request.request.error_out = SamplerError::InternalError;
            self.machine.set_state(GbnfSamplerStates::Ready);
            return Err(SamplerError::InternalError);
        }
        let error = *request.request.error_out;
        if error != SamplerError::None { return Err(error); }
        Ok(SampleDone { candidate_count: *request.request.candidate_count_out, selected_token: *request.request.selected_token_out })
    }

    /// Dispatches a request without requiring a runtime wrapper.
    pub fn sample(&mut self, request: Sample<'_>) -> Result<SampleDone, SamplerError> {
        self.process_event(EventSampleRuntime::new(request))
    }

    /// Dispatches an explicit unexpected event and returns to `ready`.
    pub fn process_unexpected(&mut self) -> Result<SampleDone, SamplerError> {
        self.machine.context_mut().err = SamplerError::InternalError;
        self.machine.set_state(GbnfSamplerStates::Ready);
        Err(SamplerError::InternalError)
    }

    /// Returns generated state inspection data.
    #[must_use]
    pub fn state(&self) -> &GbnfSamplerStates { self.machine.state() }

    /// Reports whether the generated machine is in `state`.
    #[must_use]
    pub fn is(&self, state: &GbnfSamplerStates) -> bool { self.machine.is(state) }

    /// Returns the actor context for bounded result inspection.
    #[must_use]
    pub fn context(&self) -> &GbnfSamplerContext { self.machine.context() }
}

/// Short actor alias matching the pinned sampler naming.
pub type Sampler = GbnfSampler;

/// Constructs a sampler actor from a grammar and start rule.
#[must_use]
pub fn make_sampler(grammar: &grammar, start_rule_id: u32) -> GbnfSampler {
    GbnfSampler::with_grammar(grammar, start_rule_id)
}
