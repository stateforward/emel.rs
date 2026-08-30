//! Run-to-completion batch planner.
//!
//! The public request/result types mirror the pinned C++ planner contract while
//! keeping all planner scratch storage bounded.  Runtime mode selection remains
//! in guarded SML transitions; planning actions only execute the selected path.

#![allow(
    clippy::module_name_repetitions,
    clippy::derive_partial_eq_without_eq,
    clippy::redundant_pub_crate,
    clippy::cast_sign_loss,
    clippy::manual_div_ceil,
    clippy::missing_const_for_fn,
    clippy::collapsible_if,
    clippy::len_zero,
    private_interfaces
)]

use core::fmt;
use std::cell::RefCell;
use std::rc::Rc;

use sml::sml;

pub const MAX_PLAN_STEPS: usize = 4096;
pub const MAX_SEQ: usize = 256;
pub const SEQ_WORDS: usize = (MAX_SEQ + 63) / 64;
pub type SharedScratch = Rc<RefCell<PlanScratch>>;

/// A bounded, allocation-free sequence used for planner scratch output.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct FixedVec<T, const N: usize>
where
    T: Copy + Default,
{
    values: [T; N],
    len: usize,
}

impl<T, const N: usize> Default for FixedVec<T, N>
where
    T: Copy + Default,
{
    fn default() -> Self {
        Self {
            values: [T::default(); N],
            len: 0,
        }
    }
}

impl<T, const N: usize> FixedVec<T, N>
where
    T: Copy + Default,
{
    pub(crate) fn clear(&mut self) {
        self.len = 0;
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(crate) fn push(&mut self, value: T) -> bool {
        if self.len >= N {
            return false;
        }
        self.values[self.len] = value;
        self.len += 1;
        true
    }

    pub(crate) fn as_slice(&self) -> &[T] {
        &self.values[..self.len]
    }

    pub(crate) fn iter(&self) -> core::slice::Iter<'_, T> {
        self.as_slice().iter()
    }
}

impl<T, const N: usize> Extend<T> for FixedVec<T, N>
where
    T: Copy + Default,
{
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for value in iter {
            if !self.push(value) {
                break;
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum PlanMode {
    Simple,
    Equal,
    Sequential,
    Invalid,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanRequest {
    pub token_ids: Vec<i32>,
    pub n_steps: usize,
    pub mode: PlanMode,
    pub seq_masks: Option<Vec<[u64; SEQ_WORDS]>>,
    pub seq_primary_ids: Option<Vec<i32>>,
    pub equal_sequential: bool,
    pub seq_mask_words: usize,
    pub output_mask: Option<Vec<bool>>,
    pub output_all: bool,
}

impl PlanRequest {
    #[must_use]
    pub fn new(token_ids: Vec<i32>, n_steps: usize, mode: PlanMode) -> Self {
        Self {
            token_ids,
            n_steps,
            mode,
            seq_masks: None,
            seq_primary_ids: None,
            equal_sequential: true,
            seq_mask_words: 1,
            output_mask: None,
            output_all: false,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanResult {
    pub step_sizes: Vec<usize>,
    pub step_token_indices: Vec<usize>,
    pub step_token_offsets: Vec<usize>,
    pub total_outputs: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlannerError {
    InvalidRequest,
    InvalidTokenData,
    InvalidStepSize,
    InvalidSequenceMetadata,
    InvalidSequenceId,
    InvalidSequenceMask,
    MultipleBitsInMask,
    MissingMode,
    InvalidMode,
    OutputPlanFull,
    OutputIndicesFull,
    OutputStepsFull,
    PlanningProgressStalled,
    AlgorithmFailed,
    UnsupportedLayout,
    Internal,
    Untracked,
    UnexpectedEvent,
}

impl fmt::Display for PlannerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::InvalidRequest => "invalid planner request",
            Self::InvalidTokenData => "invalid token data",
            Self::InvalidStepSize => "invalid step size",
            Self::InvalidSequenceMetadata => "invalid sequence metadata",
            Self::InvalidSequenceId => "invalid sequence id",
            Self::InvalidSequenceMask => "invalid sequence mask",
            Self::MultipleBitsInMask => "multiple bits in sequence mask",
            Self::MissingMode => "missing planning mode",
            Self::InvalidMode => "invalid planning mode",
            Self::OutputPlanFull => "plan output is full",
            Self::OutputIndicesFull => "plan index output is full",
            Self::OutputStepsFull => "plan step output is full",
            Self::PlanningProgressStalled => "planning progress stalled",
            Self::AlgorithmFailed => "planning algorithm failed",
            Self::UnsupportedLayout => "unsupported planner layout",
            Self::Internal => "internal planner error",
            Self::Untracked => "untracked planner event",
            Self::UnexpectedEvent => "unexpected planner event",
        })
    }
}

impl std::error::Error for PlannerError {}

#[derive(Clone, Debug)]
pub(crate) struct PlanScratch {
    pub(crate) error: Option<PlannerError>,
    pub(crate) effective_step_size: usize,
    pub(crate) step_sizes: FixedVec<usize, MAX_PLAN_STEPS>,
    pub(crate) step_token_indices: FixedVec<usize, MAX_PLAN_STEPS>,
    pub(crate) step_token_offsets: FixedVec<usize, { MAX_PLAN_STEPS + 1 }>,
    pub(crate) total_outputs: usize,
}

impl Default for PlanScratch {
    fn default() -> Self {
        Self {
            error: None,
            effective_step_size: 0,
            step_sizes: FixedVec::default(),
            step_token_indices: FixedVec::default(),
            step_token_offsets: FixedVec::default(),
            total_outputs: 0,
        }
    }
}

impl PlanScratch {
    pub(crate) fn reset(&mut self) {
        self.error = None;
        self.effective_step_size = 0;
        self.step_sizes.clear();
        self.step_token_indices.clear();
        self.step_token_offsets.clear();
        self.total_outputs = 0;
    }

    pub(crate) fn fail(&mut self, error: PlannerError) {
        self.error = Some(error);
        self.step_sizes.clear();
        self.step_token_indices.clear();
        self.step_token_offsets.clear();
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PlanRuntime {
    pub(crate) request: PlanRequest,
    pub(crate) scratch: SharedScratch,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct SeqMask(pub(crate) [u64; SEQ_WORDS]);

pub(crate) fn normalized_seq_mask(request: &PlanRequest, index: usize) -> SeqMask {
    let mut mask = [0; SEQ_WORDS];
    if let Some(masks) = &request.seq_masks {
        if let Some(row) = masks.get(index) {
            let words = request.seq_mask_words.min(SEQ_WORDS);
            mask[..words].copy_from_slice(&row[..words]);
        }
        return SeqMask(mask);
    }
    if let Some(ids) = &request.seq_primary_ids {
        if let Some(&id) = ids.get(index) {
            if id >= 0 && (id as usize) < request.seq_mask_words.saturating_mul(64) {
                mask[id as usize / 64] = 1u64 << (id as usize % 64);
            }
        }
        return SeqMask(mask);
    }
    mask[0] = 1;
    SeqMask(mask)
}

pub(crate) fn mask_any(mask: SeqMask) -> bool {
    mask.0.iter().any(|word| *word != 0)
}

pub(crate) fn mask_overlaps(left: SeqMask, right: SeqMask) -> bool {
    left.0.iter().zip(right.0).any(|(a, b)| a & b != 0)
}

pub(crate) fn mask_equal(left: SeqMask, right: SeqMask) -> bool {
    left == right
}

pub(crate) fn mask_subset(superset: SeqMask, subset: SeqMask) -> bool {
    superset.0.iter().zip(subset.0).all(|(a, b)| a & b == b)
}

pub(crate) fn mask_multiple_bits(mask: SeqMask) -> bool {
    let mut seen = false;
    mask.0.iter().any(|word| {
        let has = *word != 0;
        let multiple = has && ((*word & (*word - 1)) != 0 || seen);
        seen |= has;
        multiple
    })
}

pub(crate) fn validate_request(request: &PlanRequest) -> Result<(), PlannerError> {
    let n_tokens = request.token_ids.len();
    if n_tokens == 0 {
        return Err(PlannerError::InvalidRequest);
    }
    if n_tokens > MAX_PLAN_STEPS {
        return Err(PlannerError::OutputPlanFull);
    }
    if request.seq_mask_words == 0 || request.seq_mask_words > SEQ_WORDS {
        return Err(PlannerError::InvalidSequenceMetadata);
    }
    if let Some(mask) = &request.output_mask {
        if mask.len() < n_tokens {
            return Err(PlannerError::InvalidSequenceMetadata);
        }
    }
    if let Some(masks) = &request.seq_masks {
        if masks.len() < n_tokens {
            return Err(PlannerError::InvalidSequenceMetadata);
        }
    }
    if let Some(ids) = &request.seq_primary_ids {
        if ids.len() < n_tokens {
            return Err(PlannerError::InvalidSequenceId);
        }
        let max_seq = request.seq_mask_words * 64;
        if ids[..n_tokens]
            .iter()
            .any(|id| *id < 0 || (*id as usize) >= max_seq)
        {
            return Err(PlannerError::InvalidSequenceId);
        }
    }
    if request.mode == PlanMode::Equal && request.equal_sequential {
        if request.seq_masks.is_some() && request.seq_primary_ids.is_none() {
            return Err(PlannerError::InvalidSequenceMetadata);
        }
        if let Some(_masks) = &request.seq_masks {
            if (0..n_tokens).any(|index| !mask_any(normalized_seq_mask(request, index))) {
                return Err(PlannerError::InvalidSequenceMask);
            }
            if (0..n_tokens)
                .any(|index| mask_multiple_bits(normalized_seq_mask(request, index)))
            {
                return Err(PlannerError::MultipleBitsInMask);
            }
        }
    }
    Ok(())
}

pub(crate) fn normalize_step(request: &PlanRequest) -> usize {
    let requested = if request.n_steps > 0 {
        request.n_steps
    } else {
        request.token_ids.len()
    };
    requested.max(1).min(request.token_ids.len())
}

pub(crate) fn total_outputs(request: &PlanRequest) -> usize {
    if request.output_all {
        request.token_ids.len()
    } else if let Some(mask) = &request.output_mask {
        mask[..request.token_ids.len()]
            .iter()
            .filter(|value| **value)
            .count()
    } else {
        usize::from(!request.token_ids.is_empty())
    }
}

#[derive(Clone, Debug)]
pub enum Event {
    Plan(PlanRequest),
    Unexpected,
}

struct Plan;

sml! {
    BatchPlanner {
        "input_validation"_s <= *"idle"_s + Plan(PlanRuntime) / effect_begin,
        "step_normalization"_s <= "input_validation"_s + completion<Plan>(PlanRuntime) [guard_valid],
        "request_rejected"_s <= "input_validation"_s + completion<Plan>(PlanRuntime) [guard_invalid] / effect_reject,
        "mode_selection"_s <= "step_normalization"_s + completion<Plan>(PlanRuntime) / effect_normalize,
        "simple_planning"_s <= "mode_selection"_s + completion<Plan>(PlanRuntime) [guard_simple] / effect_simple,
        "equal_planning"_s <= "mode_selection"_s + completion<Plan>(PlanRuntime) [guard_equal] / effect_equal,
        "sequential_planning"_s <= "mode_selection"_s + completion<Plan>(PlanRuntime) [guard_sequential] / effect_sequential,
        "request_rejected"_s <= "mode_selection"_s + completion<Plan>(PlanRuntime) [guard_invalid_mode] / effect_invalid_mode,
        "result_publish"_s <= "simple_planning"_s + completion<Plan>(PlanRuntime) [guard_success],
        "result_publish"_s <= "equal_planning"_s + completion<Plan>(PlanRuntime) [guard_success],
        "result_publish"_s <= "sequential_planning"_s + completion<Plan>(PlanRuntime) [guard_success],
        "completed"_s <= "simple_planning"_s + completion<Plan>(PlanRuntime) [guard_failure] / effect_failure,
        "completed"_s <= "equal_planning"_s + completion<Plan>(PlanRuntime) [guard_failure] / effect_failure,
        "completed"_s <= "sequential_planning"_s + completion<Plan>(PlanRuntime) [guard_failure] / effect_failure,
        "idle"_s <= "result_publish"_s + completion<Plan>(PlanRuntime),
        "idle"_s <= "completed"_s + completion<Plan>(PlanRuntime),
        "idle"_s <= "request_rejected"_s + completion<Plan>(PlanRuntime),
        "idle"_s <= "idle"_s + unexpected_event<_> / effect_unexpected,
        "idle"_s <= "input_validation"_s + unexpected_event<_> / effect_unexpected,
        "idle"_s <= "step_normalization"_s + unexpected_event<_> / effect_unexpected,
        "idle"_s <= "mode_selection"_s + unexpected_event<_> / effect_unexpected,
        "idle"_s <= "simple_planning"_s + unexpected_event<_> / effect_unexpected,
        "idle"_s <= "equal_planning"_s + unexpected_event<_> / effect_unexpected,
        "idle"_s <= "sequential_planning"_s + unexpected_event<_> / effect_unexpected,
        "idle"_s <= "result_publish"_s + unexpected_event<_> / effect_unexpected,
        "idle"_s <= "completed"_s + unexpected_event<_> / effect_unexpected,
        "idle"_s <= "request_rejected"_s + unexpected_event<_> / effect_unexpected,
    }
}

#[derive(Debug, Default)]
struct Context;

impl BatchPlannerStateMachineContext for Context {
    fn effect_begin(&mut self, event: PlanRuntime) -> Result<(), ()> {
        event.scratch.borrow_mut().reset();
        Ok(())
    }
    fn guard_valid(&self, event: &PlanRuntime) -> Result<bool, ()> {
        Ok(validate_request(&event.request).is_ok())
    }
    fn guard_invalid(&self, event: &PlanRuntime) -> Result<bool, ()> {
        Ok(!self.guard_valid(event)?)
    }
    fn effect_reject(&mut self, event: PlanRuntime) -> Result<(), ()> {
        let error = validate_request(&event.request).err().unwrap_or(PlannerError::InvalidRequest);
        event.scratch.borrow_mut().fail(error);
        Ok(())
    }
    fn effect_normalize(&mut self, event: PlanRuntime) -> Result<(), ()> {
        let mut scratch = event.scratch.borrow_mut();
        scratch.effective_step_size = normalize_step(&event.request);
        scratch.total_outputs = total_outputs(&event.request);
        Ok(())
    }
    fn guard_simple(&self, event: &PlanRuntime) -> Result<bool, ()> {
        Ok(matches!(event.request.mode, PlanMode::Simple))
    }
    fn guard_equal(&self, event: &PlanRuntime) -> Result<bool, ()> {
        Ok(matches!(event.request.mode, PlanMode::Equal))
    }
    fn guard_sequential(&self, event: &PlanRuntime) -> Result<bool, ()> {
        Ok(matches!(event.request.mode, PlanMode::Sequential))
    }
    fn guard_invalid_mode(&self, event: &PlanRuntime) -> Result<bool, ()> {
        Ok(matches!(event.request.mode, PlanMode::Invalid))
    }
    fn effect_simple(&mut self, event: PlanRuntime) -> Result<(), ()> {
        crate::planner::modes::simple::sm::run(event);
        Ok(())
    }
    fn effect_equal(&mut self, event: PlanRuntime) -> Result<(), ()> {
        crate::planner::modes::equal::sm::run(event);
        Ok(())
    }
    fn effect_sequential(&mut self, event: PlanRuntime) -> Result<(), ()> {
        crate::planner::modes::sequential::sm::run(event);
        Ok(())
    }
    fn guard_success(&self, event: &PlanRuntime) -> Result<bool, ()> {
        let scratch = event.scratch.borrow();
        Ok(scratch.error.is_none()
            && !scratch.step_sizes.is_empty()
            && scratch.step_token_indices.len() == event.request.token_ids.len()
            && scratch.step_token_offsets.len() == scratch.step_sizes.len() + 1)
    }
    fn guard_failure(&self, event: &PlanRuntime) -> Result<bool, ()> {
        Ok(!self.guard_success(event)?)
    }
    fn effect_failure(&mut self, event: PlanRuntime) -> Result<(), ()> {
        let mut scratch = event.scratch.borrow_mut();
        if scratch.error.is_none() {
            scratch.fail(PlannerError::AlgorithmFailed);
        }
        Ok(())
    }
    fn effect_invalid_mode(&mut self, event: PlanRuntime) -> Result<(), ()> {
        event.scratch.borrow_mut().fail(PlannerError::InvalidMode);
        Ok(())
    }
    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Ok(())
    }
}

/// Reusable batch planner state machine.
pub struct Planner {
    machine: BatchPlannerStateMachine<Context>,
    scratch: SharedScratch,
}

impl Default for Planner {
    fn default() -> Self {
        Self::new()
    }
}

impl Planner {
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: BatchPlannerStateMachine::new(Context),
            scratch: Rc::new(RefCell::new(PlanScratch::default())),
        }
    }

    pub fn process_event(&mut self, event: Event) -> Result<PlanResult, PlannerError> {
        match event {
            Event::Unexpected => self.process_unexpected(),
            Event::Plan(request) => self.plan(request),
        }
    }

    pub fn plan(&mut self, request: PlanRequest) -> Result<PlanResult, PlannerError> {
        let runtime = PlanRuntime {
            request,
            scratch: self.scratch.clone(),
        };
        if self
            .machine
            .process_event(BatchPlannerEvents::Plan(runtime))
            .is_err()
        {
            return Err(PlannerError::Internal);
        }
        let scratch = self.scratch.borrow();
        if let Some(error) = scratch.error {
            return Err(error);
        }
        Ok(PlanResult {
            step_sizes: scratch.step_sizes.as_slice().to_vec(),
            step_token_indices: scratch.step_token_indices.as_slice().to_vec(),
            step_token_offsets: scratch.step_token_offsets.as_slice().to_vec(),
            total_outputs: scratch.total_outputs,
        })
    }

    #[allow(clippy::needless_pass_by_ref_mut)]
    pub fn process_unexpected(&mut self) -> Result<PlanResult, PlannerError> {
        Err(PlannerError::UnexpectedEvent)
    }

    #[must_use]
    pub fn is_idle(&self) -> bool {
        self.machine.is(&BatchPlannerStates::Idle)
    }
}

pub type BatchPlanner = Planner;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_plan_normalizes_and_chunks() {
        let mut planner = Planner::new();
        let result = planner
            .plan(PlanRequest::new((0..5).collect(), 2, PlanMode::Simple))
            .unwrap();
        assert_eq!(result.step_sizes, vec![2, 2, 1]);
        assert_eq!(result.step_token_indices, vec![0, 1, 2, 3, 4]);
        assert_eq!(result.step_token_offsets, vec![0, 2, 4, 5]);
        assert_eq!(result.total_outputs, 1);
    }

    #[test]
    fn invalid_request_and_unexpected_are_typed() {
        let mut planner = Planner::new();
        assert_eq!(
            planner.plan(PlanRequest::new(Vec::new(), 1, PlanMode::Simple)),
            Err(PlannerError::InvalidRequest)
        );
        assert_eq!(planner.process_unexpected(), Err(PlannerError::UnexpectedEvent));
        assert!(planner.is_idle());
    }
}
