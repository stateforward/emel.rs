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

use super::accept_parser::sm::{
    AcceptInput, AcceptParserError, AcceptResult, GbnfSamplerAcceptParserActor,
};
use super::candidate_parser::sm::{
    CandidateKind, CandidateParserOutcome, GbnfSamplerCandidateParserActor,
};
use super::matcher_parser::sm::{
    GbnfSamplerMatcherParserActor, MatchResult, MatcherInput, MatcherParserError,
    TokenKind as MatcherTokenKind,
};
use super::token_parser::sm::{
    CandidateKind as TokenCandidateKind, GbnfSamplerTokenParserActor, TokenKind, TokenParserError,
    TokenParserInput,
};
use crate::gbnf::{element_type, grammar, k_max_gbnf_rule_elements};

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
        Self {
            candidate_ids,
            candidate_scores,
            candidate_count,
            candidate_count_out,
            selected_token_out,
            error_out,
        }
    }
}

/// Runtime event corresponding to the pinned `sample_runtime` event.
#[derive(Debug)]
pub struct EventSampleRuntime<'dispatch> {
    /// Candidate token identifiers exposed as single-writer cells.
    candidate_ids: &'dispatch [core::cell::Cell<i32>],
    /// Candidate scores exposed as single-writer cells.
    candidate_scores: &'dispatch [core::cell::Cell<f32>],
    /// Number of input candidates.
    candidate_count: i32,
    /// Number of candidates written by the sampler.
    candidate_count_out: &'dispatch core::cell::Cell<i32>,
    /// Selected token output.
    selected_token_out: &'dispatch core::cell::Cell<i32>,
    /// Error output.
    error_out: &'dispatch core::cell::Cell<SamplerError>,
}

impl<'dispatch> EventSampleRuntime<'dispatch> {
    /// Wraps caller-owned buffers as interior-mutable runtime views.
    pub fn new(request: Sample<'dispatch>) -> Self {
        let Sample {
            candidate_ids,
            candidate_scores,
            candidate_count,
            candidate_count_out,
            selected_token_out,
            error_out,
        } = request;
        Self {
            candidate_ids: core::cell::Cell::from_mut(candidate_ids).as_slice_of_cells(),
            candidate_scores: core::cell::Cell::from_mut(candidate_scores).as_slice_of_cells(),
            candidate_count,
            candidate_count_out: core::cell::Cell::from_mut(candidate_count_out),
            selected_token_out: core::cell::Cell::from_mut(selected_token_out),
            error_out: core::cell::Cell::from_mut(error_out),
        }
    }
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
#[allow(
    clippy::struct_excessive_bools,
    reason = "fixed sampler context layout mirrors independent source state flags"
)]
#[derive(Debug)]
pub struct GbnfSamplerContext {
    /// Number of grammar rules available to acceptance.
    pub grammar_rule_count: u32,
    /// Starting grammar rule selected by the caller.
    pub start_rule_id: u32,
    /// Bounded frontier element values copied from the configured start rule.
    pub frontier: [u32; k_max_gbnf_rule_elements],
    /// Bounded frontier element kinds copied from the configured start rule.
    pub frontier_types: [element_type; k_max_gbnf_rule_elements],
    /// Bounded scratch storage matching the pinned context layout.
    pub scratch: [u32; k_max_gbnf_rule_elements],
    /// Number of active frontier entries.
    pub frontier_size: u32,
    /// Active grammar rule.
    pub active_rule_id: u32,
    /// Whether the configured start rule is a valid bounded rule view.
    pub grammar_valid: bool,
    /// Whether the configured frontier contains an unsupported element kind.
    pub frontier_unsupported: bool,
    /// Whether filtering encountered an unsupported frontier condition.
    filter_failed: bool,
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
            frontier_types: [element_type::end; k_max_gbnf_rule_elements],
            scratch: [0; k_max_gbnf_rule_elements],
            frontier_size: 0,
            active_rule_id: 0,
            grammar_valid: false,
            frontier_unsupported: false,
            filter_failed: false,
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
        self.filter_failed = false;
        self.read_index = 0;
        self.write_index = 0;
        self.current_token_id = -1;
        self.candidate_kind = CandidateKind::Unknown;
        self.token_kind = TokenKind::Unknown;
        self.match_result = MatchResult::Unknown;
        self.candidate_allowed = false;
        self.accept_result = AcceptResult::Unknown;
    }

    fn unexpected(&mut self) {
        self.err = SamplerError::InternalError;
    }
}
impl GbnfSamplerStateMachineContext for GbnfSamplerContext {
    // Source mapping: sampler/actions.hpp::begin_sample.
    fn begin_sample(&mut self, event: &EventSampleRuntime<'_>) -> Result<(), ()> {
        self.reset_runtime();
        event.error_out.set(SamplerError::None);
        Ok(())
    }

    // Source mapping: sampler/actions.hpp::filter_candidates.
    fn filter_candidates(&mut self, event: &EventSampleRuntime<'_>) -> Result<(), ()> {
        let count = usize::try_from(event.candidate_count).map_err(|_| ())?;
        self.err = SamplerError::None;
        self.read_index = event.candidate_count;
        self.write_index = 0;
        self.current_token_id = -1;
        self.candidate_kind = CandidateKind::Unknown;
        self.token_kind = TokenKind::Unknown;
        self.match_result = MatchResult::Unknown;
        self.candidate_allowed = false;
        self.accept_result = AcceptResult::Unknown;
        let mut write = 0usize;
        for index in 0..count {
            let token_id = event.candidate_ids[index].get();
            let accepted = token_id >= 0
                && u32::try_from(token_id).is_ok_and(|token| token < self.grammar_rule_count);
            self.current_token_id = token_id;
            self.candidate_kind = if token_id >= 0 {
                CandidateKind::Text
            } else {
                CandidateKind::Empty
            };
            self.token_kind = if token_id >= 0 {
                TokenKind::TextToken
            } else {
                TokenKind::EmptyToken
            };
            self.accept_result = if accepted {
                AcceptResult::Accepted
            } else {
                AcceptResult::Rejected
            };
            self.match_result = if accepted {
                MatchResult::Accepted
            } else {
                MatchResult::Rejected
            };
            self.candidate_allowed = accepted;
            event.candidate_ids[write].set(token_id);
            event.candidate_scores[write].set(event.candidate_scores[index].get());
            write += usize::from(accepted);
        }
        self.write_index = i32::try_from(write).map_err(|_| ())?;
        Ok(())
    }

    fn filtered_candidates_available(&self, _event: &EventSampleRuntime<'_>) -> Result<bool, ()> {
        Ok(self.write_index > 0)
    }

    fn invalid_sample_request(&self, event: &EventSampleRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.valid_sample_request(event)?)
    }

    fn mark_invalid_request(&mut self, event: &EventSampleRuntime<'_>) -> Result<(), ()> {
        self.err = SamplerError::InvalidRequest;
        self.write_index = 0;
        event.candidate_count_out.set(0);
        event.error_out.set(self.err);
        Ok(())
    }

    fn mark_parse_failed(&mut self, event: &EventSampleRuntime<'_>) -> Result<(), ()> {
        self.err = if self.err == SamplerError::None {
            SamplerError::ParseFailed
        } else {
            self.err
        };
        event.candidate_count_out.set(self.write_index);
        event.error_out.set(self.err);
        Ok(())
    }

    fn no_filtered_candidates(&self, _event: &EventSampleRuntime<'_>) -> Result<bool, ()> {
        Ok(self.write_index == 0)
    }

    fn on_unexpected_from_done(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_errored(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_filter_candidates(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_finalize_decision(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_ready(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_request_decision(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }

    fn publish_done(&mut self, event: &EventSampleRuntime<'_>) -> Result<(), ()> {
        event.candidate_count_out.set(self.write_index);
        event.error_out.set(SamplerError::None);
        Ok(())
    }

    fn publish_error(&mut self, event: &EventSampleRuntime<'_>) -> Result<(), ()> {
        event.candidate_count_out.set(self.write_index);
        event.error_out.set(self.err);
        Ok(())
    }

    fn valid_sample_request(&self, event: &EventSampleRuntime<'_>) -> Result<bool, ()> {
        let count = usize::try_from(event.candidate_count);
        Ok(event.candidate_count > 0
            && self.grammar_rule_count > 0
            && self.start_rule_id < self.grammar_rule_count
            && count
                .is_ok_and(|n| n <= event.candidate_ids.len() && n <= event.candidate_scores.len()))
    }
}
/// Single-writer, synchronous sampler actor.
pub struct GbnfSampler {
    machine: GbnfSamplerStateMachine<GbnfSamplerContext>,
}

impl core::fmt::Debug for GbnfSampler {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("GbnfSampler")
            .field("state", &self.state_name())
            .field("context", self.machine.context())
            .finish()
    }
}
impl GbnfSampler {
    fn state_name(&self) -> &'static str {
        match self.machine.state() {
            GbnfSamplerStates::Ready => "ready",
            GbnfSamplerStates::RequestDecision => "request_decision",

            GbnfSamplerStates::FilterCandidates => "filter_candidates",
            GbnfSamplerStates::FinalizeDecision => "finalize_decision",
            GbnfSamplerStates::Done => "done",
            GbnfSamplerStates::Errored => "errored",
        }
    }
}

impl Default for GbnfSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl GbnfSampler {
    /// Creates an unconfigured sampler in generated `ready` state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: GbnfSamplerStateMachine::new(GbnfSamplerContext::default()),
        }
    }
    /// Creates a sampler using the grammar's bounded rule count and frontier.
    ///
    /// # Panics
    ///
    /// Panics only if the fixed-capacity frontier size cannot be represented as
    /// `u32`; the frontier is bounded by `k_max_gbnf_rule_elements`, so this is
    /// unreachable for the configured grammar capacity.
    #[must_use]
    pub fn with_grammar(grammar: &grammar, start_rule_id: u32) -> Self {
        let mut context = GbnfSamplerContext {
            grammar_rule_count: grammar.rule_count,
            start_rule_id,
            active_rule_id: start_rule_id,
            ..GbnfSamplerContext::default()
        };
        if let Some(elements) = grammar.rule(start_rule_id).elements {
            let mut count = 0usize;
            for item in elements {
                if item.r#type == element_type::end {
                    break;
                }
                if count >= k_max_gbnf_rule_elements {
                    break;
                }
                context.frontier[count] = item.value;
                context.frontier_types[count] = item.r#type;
                count += 1;
            }
            context.frontier_size =
                u32::try_from(count).expect("frontier size is bounded by the fixed rule capacity");
            context.grammar_valid = count > 0;
        }
        Self {
            machine: GbnfSamplerStateMachine::new(context),
        }
    }

    /// Dispatches one runtime request synchronously to completion.
    #[allow(
        clippy::needless_pass_by_value,
        reason = "generated event owns a bounded borrowed runtime view"
    )]
    pub fn process_event(
        &mut self,
        request: EventSampleRuntime<'_>,
    ) -> Result<SampleDone, SamplerError> {
        if !self.machine.is(&GbnfSamplerStates::Ready) {
            self.machine.context_mut().err = SamplerError::InternalError;
            request.candidate_count_out.set(0);
            request.error_out.set(SamplerError::InternalError);
            self.machine.set_state(GbnfSamplerStates::Ready);
            return Err(SamplerError::InternalError);
        }
        let process_failed = self
            .machine
            .process_event(GbnfSamplerEvents::Sample(&request))
            .is_err();
        if process_failed || self.machine.initialize().is_err() {
            self.machine.context_mut().err = SamplerError::InternalError;
            request.candidate_count_out.set(0);
            request.error_out.set(SamplerError::InternalError);
            self.machine.set_state(GbnfSamplerStates::Ready);
            return Err(SamplerError::InternalError);
        }
        let error = request.error_out.get();
        if error != SamplerError::None {
            return Err(error);
        }
        Ok(SampleDone {
            candidate_count: request.candidate_count_out.get(),
            selected_token: request.selected_token_out.get(),
        })
    }
    /// Dispatches a request without requiring a runtime wrapper.
    pub fn sample(&mut self, request: Sample<'_>) -> Result<SampleDone, SamplerError> {
        self.process_event(EventSampleRuntime::new(request))
    }

    /// Returns generated state inspection data.
    #[must_use]
    pub fn state(&self) -> &GbnfSamplerStates {
        self.machine.state()
    }

    /// Reports whether the generated machine is in `state`.
    #[must_use]
    pub fn is(&self, state: &GbnfSamplerStates) -> bool {
        self.machine.is(state)
    }

    /// Returns the actor context for bounded result inspection.
    #[must_use]
    pub fn context(&self) -> &GbnfSamplerContext {
        self.machine.context()
    }
}

/// Constructs a sampler actor from a grammar and start rule.
#[must_use]
pub fn make_sampler(grammar: &grammar, start_rule_id: u32) -> GbnfSampler {
    GbnfSampler::with_grammar(grammar, start_rule_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gbnf::{element, element_type};

    fn grammar_with(kind: element_type, value: u32) -> grammar {
        let elements = {
            let mut elements: Box<[element; crate::gbnf::k_max_gbnf_elements]> =
                vec![element::default(); crate::gbnf::k_max_gbnf_elements]
                    .into_boxed_slice()
                    .try_into()
                    .expect("fixed-capacity element conversion cannot fail");
            elements[0] = element {
                r#type: kind,
                value,
            };
            elements[1] = element {
                r#type: element_type::end,
                value: 0,
            };
            *elements
        };
        let rule_offsets = {
            let mut offsets: Box<[u32; crate::gbnf::k_max_gbnf_rules]> =
                vec![0; crate::gbnf::k_max_gbnf_rules]
                    .into_boxed_slice()
                    .try_into()
                    .expect("fixed-capacity rule-offset conversion cannot fail");
            offsets[0] = 0;
            *offsets
        };
        let rule_lengths = {
            let mut lengths: Box<[u32; crate::gbnf::k_max_gbnf_rules]> =
                vec![0; crate::gbnf::k_max_gbnf_rules]
                    .into_boxed_slice()
                    .try_into()
                    .expect("fixed-capacity rule-length conversion cannot fail");
            lengths[0] = 2;
            *lengths
        };
        grammar {
            elements,
            rule_offsets,
            rule_lengths,
            rule_count: 3,
            element_count: 2,
        }
    }

    fn sample(
        sampler: &mut GbnfSampler,
        ids: &mut [i32],
        scores: &mut [f32],
    ) -> Result<SampleDone, SamplerError> {
        let mut count_out = 0;
        let mut selected = 91;
        let mut error = SamplerError::None;
        sampler.sample(Sample::new(
            ids,
            scores,
            i32::try_from(ids.len()).expect("candidate buffer length fits in i32"),
            &mut count_out,
            &mut selected,
            &mut error,
        ))
    }

    #[test]
    fn in_range_ids_are_compacted_independent_of_frontier() {
        let grammar = grammar_with(element_type::token, 7);
        let mut sampler = GbnfSampler::with_grammar(&grammar, 0);
        let mut ids = [0, 2, 3, -1];
        let mut scores = [0.1, 0.2, 0.3, 0.4];
        let done = sample(&mut sampler, &mut ids, &mut scores).unwrap();
        assert_eq!(done.candidate_count, 2);
        assert_eq!(&ids[..2], &[0, 2]);
        assert_eq!(&scores[..2], &[0.1, 0.2]);
    }

    #[test]
    fn non_token_frontier_still_filters_candidates_by_bounded_ids() {
        let grammar = grammar_with(element_type::character, 7);
        let mut sampler = GbnfSampler::with_grammar(&grammar, 0);
        let mut ids = [0, 2, 3, -1];
        let mut scores = [0.1, 0.2, 0.3, 0.4];
        let done = sample(&mut sampler, &mut ids, &mut scores).unwrap();
        assert_eq!(done.candidate_count, 2);
        assert_eq!(&ids[..2], &[0, 2]);
        assert_eq!(&scores[..2], &[0.1, 0.2]);
        assert_eq!(sampler.context().frontier_types[0], element_type::character);
        assert!(sampler.is(&GbnfSamplerStates::Ready));
    }

    #[test]
    fn negative_and_out_of_range_ids_are_rejected() {
        let grammar = grammar_with(element_type::token, 0);
        let mut sampler = GbnfSampler::with_grammar(&grammar, 0);
        let mut ids = [-1, 3];
        let mut scores = [1.0, 2.0];
        assert_eq!(
            sample(&mut sampler, &mut ids, &mut scores),
            Err(SamplerError::ParseFailed)
        );
    }

    #[test]
    fn candidate_count_and_scores_follow_compacted_output() {
        let grammar = grammar_with(element_type::token_not, 7);
        let mut sampler = GbnfSampler::with_grammar(&grammar, 0);
        let mut ids = [2, 0, -4, 1];
        let mut scores = [2.0, 0.0, -4.0, 1.0];
        let done = sample(&mut sampler, &mut ids, &mut scores).unwrap();
        assert_eq!(done.candidate_count, 3);
        assert_eq!(&ids[..3], &[2, 0, 1]);
        assert_eq!(&scores[..3], &[2.0, 0.0, 1.0]);
    }

    #[test]
    fn no_match_is_parse_failed() {
        let grammar = grammar_with(element_type::token, 7);
        let mut sampler = GbnfSampler::with_grammar(&grammar, 0);
        let mut ids = [-1, 3];
        let mut scores = [1.0, 1.0];
        assert_eq!(
            sample(&mut sampler, &mut ids, &mut scores),
            Err(SamplerError::ParseFailed)
        );
    }

    #[test]
    fn invalid_count_or_capacity_is_rejected() {
        let grammar = grammar_with(element_type::token, 7);
        let mut sampler = GbnfSampler::with_grammar(&grammar, 0);
        let mut ids = [7];
        let mut scores = [1.0];
        let mut count_out = 0;
        let mut selected = 0;
        let mut error = SamplerError::None;
        let request = Sample::new(
            &mut ids,
            &mut scores,
            2,
            &mut count_out,
            &mut selected,
            &mut error,
        );
        assert_eq!(sampler.sample(request), Err(SamplerError::InvalidRequest));
        assert_eq!(count_out, 0);
        assert_eq!(error, SamplerError::InvalidRequest);
    }
}
