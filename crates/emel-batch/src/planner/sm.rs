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
use opentelemetry::trace::{self, Status, Tracer};
use opentelemetry::{KeyValue, global};
use sml::sml;

const PLANNER_TRACER: &str = "emel-batch";
const PLANNER_OPERATION: &str = "plan";

fn planner_mode(mode: PlanMode) -> &'static str {
    match mode {
        PlanMode::Simple => "simple",
        PlanMode::Equal => "equal",
        PlanMode::Sequential | PlanMode::Seq => "sequential",
        PlanMode::Invalid => "invalid",
    }
}

fn planner_failure(error: PlannerError) -> &'static str {
    match error {
        PlannerError::InvalidRequest
        | PlannerError::InvalidRequestAndMode
        | PlannerError::InvalidTokenData
        | PlannerError::InvalidStepSize
        | PlannerError::InvalidSequenceMetadata
        | PlannerError::InvalidSequenceId
        | PlannerError::InvalidSequenceMask
        | PlannerError::MultipleBitsInMask
        | PlannerError::MissingMode
        | PlannerError::InvalidMode => "invalid",
        PlannerError::OutputPlanFull
        | PlannerError::OutputIndicesFull
        | PlannerError::OutputStepsFull => "capacity",
        PlannerError::PlanningProgressStalled | PlannerError::AlgorithmFailed => "algorithm",
        PlannerError::UnsupportedLayout | PlannerError::Internal | PlannerError::Untracked => {
            "internal"
        }
        PlannerError::UnexpectedEvent => "unexpected",
    }
}

fn planner_event(name: &'static str, key: &'static str, value: &'static str) {
    trace::get_active_span(|span| {
        if span.is_recording() {
            span.add_event(
                name,
                vec![
                    KeyValue::new("operation", PLANNER_OPERATION),
                    KeyValue::new(key, value),
                ],
            );
        }
    });
}

fn planner_result(result: Result<(), PlannerError>) {
    match result {
        Ok(()) => planner_event("emel.planner.result", "result", "success"),
        Err(error) => {
            planner_event("emel.planner.failure", "failure", planner_failure(error));
            trace::get_active_span(|span| span.set_status(Status::error("planner failure")));
        }
    }
}
fn with_planner_span<T>(
    mode: PlanMode,
    operation: &'static str,
    f: impl FnOnce() -> Result<T, PlannerError>,
) -> Result<T, PlannerError> {
    global::tracer(PLANNER_TRACER).in_span(operation, |_cx| {
        trace::get_active_span(|span| {
            if span.is_recording() {
                span.set_attribute(KeyValue::new("operation", PLANNER_OPERATION));
                span.set_attribute(KeyValue::new("mode", planner_mode(mode)));
            }
        });
        planner_event("emel.planner.control", "control", "dispatch_start");
        let result = f();
        planner_result(result.as_ref().map(|_| ()).map_err(|error| *error));
        result
    })
}

pub const MAX_PLAN_STEPS: usize = 4096;
pub const MAX_SEQ: usize = 256;
pub const SEQ_WORDS: usize = (MAX_SEQ + 63) / 64;

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
    /// Alias for the pinned C++ `plan_mode::seq` enumerator.
    Seq,
    Invalid,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StepSize {
    Automatic,
    Fixed(core::num::NonZeroUsize),
}

impl StepSize {
    #[must_use]
    pub const fn automatic() -> Self {
        Self::Automatic
    }

    #[must_use]
    pub const fn fixed(value: core::num::NonZeroUsize) -> Self {
        Self::Fixed(value)
    }

    pub(crate) const fn requested(self) -> Option<usize> {
        match self {
            Self::Automatic => None,
            Self::Fixed(value) => Some(value.get()),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MaskWordCount(core::num::NonZeroUsize);

impl MaskWordCount {
    /// Creates a mask-word count after validating its bounds.
    ///
    /// # Errors
    ///
    /// Returns [`PlannerError::InvalidSequenceMetadata`] when `value` is zero
    /// or exceeds [`SEQ_WORDS`].
    pub fn new(value: usize) -> Result<Self, PlannerError> {
        let Some(value) = core::num::NonZeroUsize::new(value) else {
            return Err(PlannerError::InvalidSequenceMetadata);
        };
        if value.get() > SEQ_WORDS {
            return Err(PlannerError::InvalidSequenceMetadata);
        }
        Ok(Self(value))
    }

    #[must_use]
    pub const fn get(self) -> usize {
        self.0.get()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EqualStrategy {
    General,
    Sequential,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum SequenceData {
    None,
    Masks(Vec<[u64; SEQ_WORDS]>),
    PrimaryIds(Vec<i32>),
    Both {
        masks: Vec<[u64; SEQ_WORDS]>,
        primary_ids: Vec<i32>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SequenceConfig {
    data: SequenceData,
    mask_words: MaskWordCount,
}

impl Default for SequenceConfig {
    fn default() -> Self {
        Self {
            data: SequenceData::None,
            mask_words: MaskWordCount(core::num::NonZeroUsize::new(1).unwrap()),
        }
    }
}

impl SequenceConfig {
    #[must_use]
    pub fn new(mask_words: MaskWordCount) -> Self {
        Self {
            mask_words,
            ..Self::default()
        }
    }

    /// Supplies per-token sequence masks.
    ///
    /// If primary IDs were already supplied, this creates an explicit paired
    /// sequence source rather than silently keeping two unrelated options.
    #[must_use]
    pub fn with_masks(mut self, masks: Vec<[u64; SEQ_WORDS]>) -> Self {
        self.data = match core::mem::replace(&mut self.data, SequenceData::None) {
            SequenceData::PrimaryIds(primary_ids) | SequenceData::Both { primary_ids, .. } => {
                SequenceData::Both { masks, primary_ids }
            }
            SequenceData::None | SequenceData::Masks(_) => SequenceData::Masks(masks),
        };
        self
    }

    /// Supplies one primary sequence ID per token.
    ///
    /// If masks were already supplied, this creates an explicit paired
    /// sequence source rather than silently keeping two unrelated options.
    #[must_use]
    pub fn with_primary_ids(mut self, primary_ids: Vec<i32>) -> Self {
        self.data = match core::mem::replace(&mut self.data, SequenceData::None) {
            SequenceData::Masks(masks) | SequenceData::Both { masks, .. } => {
                SequenceData::Both { masks, primary_ids }
            }
            SequenceData::None | SequenceData::PrimaryIds(_) => {
                SequenceData::PrimaryIds(primary_ids)
            }
        };
        self
    }

    pub(crate) fn masks(&self) -> Option<&[[u64; SEQ_WORDS]]> {
        match &self.data {
            SequenceData::Masks(masks) | SequenceData::Both { masks, .. } => Some(masks),
            SequenceData::None | SequenceData::PrimaryIds(_) => None,
        }
    }
    pub(crate) fn primary_ids(&self) -> Option<&[i32]> {
        match &self.data {
            SequenceData::PrimaryIds(primary_ids) | SequenceData::Both { primary_ids, .. } => {
                Some(primary_ids)
            }
            SequenceData::None | SequenceData::Masks(_) => None,
        }
    }
    pub(crate) const fn mask_words(&self) -> usize {
        self.mask_words.get()
    }
}

/// A typed output-selection mask.
///
/// The planner validates that the mask covers the request's token count when
/// the request is dispatched. Extra entries remain permitted for compatibility
/// with the prior bounded planner contract.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OutputMask {
    values: Vec<bool>,
}

impl OutputMask {
    #[must_use]
    pub fn new(values: Vec<bool>) -> Self {
        Self { values }
    }

    #[must_use]
    pub fn as_slice(&self) -> &[bool] {
        &self.values
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

impl From<Vec<bool>> for OutputMask {
    fn from(values: Vec<bool>) -> Self {
        Self::new(values)
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub enum OutputSelection {
    #[default]
    First,
    All,
    Mask(OutputMask),
}

impl OutputSelection {
    #[must_use]
    pub fn masked(mask: impl Into<OutputMask>) -> Self {
        Self::Mask(mask.into())
    }
}
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanConfig {
    mode: PlanMode,
    step_size: StepSize,
    sequence: SequenceConfig,
    outputs: OutputSelection,
    equal_strategy: EqualStrategy,
}

impl PlanConfig {
    #[must_use]
    pub fn new(mode: PlanMode) -> Self {
        Self {
            mode,
            step_size: StepSize::Automatic,
            sequence: SequenceConfig::default(),
            outputs: OutputSelection::default(),
            equal_strategy: EqualStrategy::Sequential,
        }
    }

    #[must_use]
    pub fn with_step_size(mut self, step_size: StepSize) -> Self {
        self.step_size = step_size;
        self
    }

    #[must_use]
    pub fn with_sequence(mut self, sequence: SequenceConfig) -> Self {
        self.sequence = sequence;
        self
    }

    #[must_use]
    pub fn with_output_selection(mut self, outputs: OutputSelection) -> Self {
        self.outputs = outputs;
        self
    }

    #[must_use]
    pub fn with_equal_strategy(mut self, strategy: EqualStrategy) -> Self {
        self.equal_strategy = strategy;
        self
    }

    pub(crate) const fn mode(&self) -> PlanMode {
        self.mode
    }
    pub(crate) const fn step_size(&self) -> StepSize {
        self.step_size
    }
    pub(crate) fn sequence(&self) -> &SequenceConfig {
        &self.sequence
    }
    pub(crate) fn outputs(&self) -> &OutputSelection {
        &self.outputs
    }
    pub(crate) const fn equal_strategy(&self) -> EqualStrategy {
        self.equal_strategy
    }
}

#[allow(unpredictable_function_pointer_comparisons)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlanRequest {
    token_ids: Vec<i32>,
    config: PlanConfig,
    on_done: DoneCallback,
    on_error: ErrorCallback,
}

impl PlanRequest {
    #[must_use]
    pub fn new(token_ids: Vec<i32>, config: PlanConfig) -> Self {
        Self {
            token_ids,
            config,
            on_done: ignore_plan_done,
            on_error: ignore_plan_error,
        }
    }

    #[must_use]
    pub fn with_callbacks(mut self, on_done: DoneCallback, on_error: ErrorCallback) -> Self {
        self.on_done = on_done;
        self.on_error = on_error;
        self
    }

    pub(crate) fn token_ids(&self) -> &[i32] {
        &self.token_ids
    }
    pub(crate) const fn config(&self) -> &PlanConfig {
        &self.config
    }
    pub(crate) const fn mode(&self) -> PlanMode {
        self.config.mode()
    }
    pub(crate) const fn on_done(&self) -> DoneCallback {
        self.on_done
    }
    pub(crate) const fn on_error(&self) -> ErrorCallback {
        self.on_error
    }
}
/// Borrowed successful result delivered before planner dispatch returns.
#[derive(Debug)]
pub struct PlanDone<'a> {
    pub request: &'a PlanRequest,
    pub step_sizes: &'a [usize],
    pub step_token_indices: &'a [usize],
    pub step_token_offsets: &'a [usize],
    pub total_outputs: usize,
}

/// Caller-owned storage for a bounded planner result.
#[derive(Debug)]
pub struct PlanOutput<'a> {
    pub step_sizes: &'a mut [usize],
    pub step_token_indices: &'a mut [usize],
    pub step_token_offsets: &'a mut [usize],
    pub step_sizes_len: usize,
    pub step_token_indices_len: usize,
    pub step_token_offsets_len: usize,
    pub total_outputs: usize,
}

impl<'a> PlanOutput<'a> {
    #[must_use]
    pub fn new(
        step_sizes: &'a mut [usize],
        step_token_indices: &'a mut [usize],
        step_token_offsets: &'a mut [usize],
    ) -> Self {
        Self {
            step_sizes,
            step_token_indices,
            step_token_offsets,
            step_sizes_len: 0,
            step_token_indices_len: 0,
            step_token_offsets_len: 0,
            total_outputs: 0,
        }
    }

    fn reset(&mut self) {
        self.step_sizes_len = 0;
        self.step_token_indices_len = 0;
        self.step_token_offsets_len = 0;
        self.total_outputs = 0;
    }

    fn capacities(&self) -> [usize; 3] {
        [
            self.step_sizes.len(),
            self.step_token_indices.len(),
            self.step_token_offsets.len(),
        ]
    }

    fn write_from(&mut self, scratch: &PlanScratch) {
        let step_sizes = scratch.step_sizes.as_slice();
        let step_token_indices = scratch.step_token_indices.as_slice();
        let step_token_offsets = scratch.step_token_offsets.as_slice();
        self.step_sizes[..step_sizes.len()].copy_from_slice(step_sizes);
        self.step_token_indices[..step_token_indices.len()].copy_from_slice(step_token_indices);
        self.step_token_offsets[..step_token_offsets.len()].copy_from_slice(step_token_offsets);
        self.step_sizes_len = step_sizes.len();
        self.step_token_indices_len = step_token_indices.len();
        self.step_token_offsets_len = step_token_offsets.len();
        self.total_outputs = scratch.total_outputs;
    }

    #[must_use]
    pub fn step_sizes_slice(&self) -> &[usize] {
        &self.step_sizes[..self.step_sizes_len]
    }

    #[must_use]
    pub fn step_token_indices_slice(&self) -> &[usize] {
        &self.step_token_indices[..self.step_token_indices_len]
    }

    #[must_use]
    pub fn step_token_offsets_slice(&self) -> &[usize] {
        &self.step_token_offsets[..self.step_token_offsets_len]
    }
}

/// Borrowed failed result delivered before planner dispatch returns.
#[derive(Debug)]
pub struct PlanError<'a> {
    pub request: &'a PlanRequest,
    pub error: PlannerError,
}

/// Synchronous completion callback.
pub type DoneCallback = for<'a> fn(PlanDone<'a>) -> bool;
/// Synchronous failure callback.
pub type ErrorCallback = for<'a> fn(PlanError<'a>) -> bool;

pub type PlanDoneEvent<'a> = PlanDone<'a>;
pub type PlanErrorEvent<'a> = PlanError<'a>;

fn ignore_plan_done(_: PlanDone<'_>) -> bool {
    true
}

fn ignore_plan_error(_: PlanError<'_>) -> bool {
    true
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
    /// Invalid mode rejected as an invalid request, matching the pinned
    /// `invalid_mode | invalid_request` classification.
    InvalidRequestAndMode,
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

impl PlannerError {
    #[must_use]
    pub const fn is_invalid_request(self) -> bool {
        matches!(self, Self::InvalidRequest | Self::InvalidRequestAndMode)
    }

    #[must_use]
    pub const fn is_invalid_mode(self) -> bool {
        matches!(self, Self::InvalidMode | Self::InvalidRequestAndMode)
    }
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
            Self::InvalidRequestAndMode => "invalid planner request and mode",
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

#[derive(Clone, Debug, Default)]
pub(crate) struct PlanScratch {
    pub(crate) error: Option<PlannerError>,
    pub(crate) effective_step_size: usize,
    pub(crate) step_sizes: FixedVec<usize, MAX_PLAN_STEPS>,
    pub(crate) step_token_indices: FixedVec<usize, MAX_PLAN_STEPS>,
    pub(crate) step_token_offsets: FixedVec<usize, { MAX_PLAN_STEPS + 1 }>,
    pub(crate) total_outputs: usize,
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
        self.total_outputs = 0;
    }
}

#[derive(Clone, Debug)]
pub(crate) struct PlanRuntime {
    pub(crate) request: PlanRequest,
    pub(crate) effective_step_size: usize,
    pub(crate) total_outputs: usize,
}

/// Bounded scratch owned exclusively by one planner mode invocation.
#[derive(Clone, Debug, Default)]
pub(crate) struct ModeScratch {
    pub(crate) error: Option<PlannerError>,
    pub(crate) effective_step_size: usize,
    pub(crate) step_sizes: FixedVec<usize, MAX_PLAN_STEPS>,
    pub(crate) step_token_indices: FixedVec<usize, MAX_PLAN_STEPS>,
    pub(crate) step_token_offsets: FixedVec<usize, { MAX_PLAN_STEPS + 1 }>,
    pub(crate) total_outputs: usize,
}

impl ModeScratch {
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
        self.total_outputs = 0;
    }

    pub(crate) fn result(&self) -> ModeResult {
        self.error.map_or(ModeResult::Complete, ModeResult::Error)
    }

    /// Copies bounded child output into caller-owned planner scratch.
    pub(crate) fn copy_into(&self, target: &mut PlanScratch) {
        if let Some(error) = self.error {
            target.fail(error);
            return;
        }
        target.reset();
        target.effective_step_size = self.effective_step_size;
        target.total_outputs = self.total_outputs;
        for &value in self.step_sizes.as_slice() {
            let _ = target.step_sizes.push(value);
        }
        for &value in self.step_token_indices.as_slice() {
            let _ = target.step_token_indices.push(value);
        }
        for &value in self.step_token_offsets.as_slice() {
            let _ = target.step_token_offsets.push(value);
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ModeResult {
    Complete,
    Error(PlannerError),
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct SeqMask(pub(crate) [u64; SEQ_WORDS]);

pub(crate) fn normalized_seq_mask(request: &PlanRequest, index: usize) -> SeqMask {
    let mut mask = [0; SEQ_WORDS];
    if let Some(masks) = request.config().sequence().masks() {
        if let Some(row) = masks.get(index) {
            let words = request.config().sequence().mask_words().min(SEQ_WORDS);
            mask[..words].copy_from_slice(&row[..words]);
        }
        return SeqMask(mask);
    }
    if let Some(ids) = request.config().sequence().primary_ids() {
        if let Some(&id) = ids.get(index) {
            if id >= 0
                && (id as usize) < request.config().sequence().mask_words().saturating_mul(64)
            {
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
    let n_tokens = request.token_ids().len();
    if n_tokens == 0 {
        return Err(PlannerError::InvalidRequest);
    }
    if n_tokens > MAX_PLAN_STEPS {
        return Err(PlannerError::OutputPlanFull);
    }
    if request.config().sequence().mask_words() == 0
        || request.config().sequence().mask_words() > SEQ_WORDS
    {
        return Err(PlannerError::InvalidSequenceMetadata);
    }
    if let OutputSelection::Mask(mask) = request.config().outputs() {
        if mask.len() < n_tokens {
            return Err(PlannerError::InvalidSequenceMetadata);
        }
    }
    if let Some(masks) = &request.config().sequence().masks() {
        if masks.len() < n_tokens {
            return Err(PlannerError::InvalidSequenceMetadata);
        }
    }
    if let Some(ids) = &request.config().sequence().primary_ids() {
        if ids.len() < n_tokens {
            return Err(PlannerError::InvalidSequenceId);
        }
        let max_seq = request.config().sequence().mask_words() * 64;
        if ids[..n_tokens]
            .iter()
            .any(|id| *id < 0 || (*id as usize) >= max_seq)
        {
            return Err(PlannerError::InvalidSequenceId);
        }
    }
    if let Some(_masks) = &request.config().sequence().masks() {
        if (0..n_tokens).any(|index| !mask_any(normalized_seq_mask(request, index))) {
            return Err(PlannerError::InvalidSequenceMask);
        }
        if request.mode() == PlanMode::Equal
            && matches!(request.config().equal_strategy(), EqualStrategy::Sequential)
            && (0..n_tokens).any(|index| mask_multiple_bits(normalized_seq_mask(request, index)))
        {
            return Err(PlannerError::MultipleBitsInMask);
        }
    }
    if request.mode() == PlanMode::Equal
        && matches!(request.config().equal_strategy(), EqualStrategy::Sequential)
    {
        if request.config().sequence().masks().is_some()
            && request.config().sequence().primary_ids().is_none()
        {
            return Err(PlannerError::InvalidSequenceMetadata);
        }
    }
    Ok(())
}

pub(crate) fn normalize_step(request: &PlanRequest) -> usize {
    let requested = request
        .config()
        .step_size()
        .requested()
        .unwrap_or_else(|| request.token_ids().len());
    requested.max(1).min(request.token_ids().len())
}

pub(crate) fn total_outputs(request: &PlanRequest) -> usize {
    match request.config().outputs() {
        OutputSelection::All => request.token_ids().len(),
        OutputSelection::Mask(mask) => mask.as_slice()[..request.token_ids().len()]
            .iter()
            .filter(|value| **value)
            .count(),
        OutputSelection::First => usize::from(!request.token_ids().is_empty()),
    }
}

pub(crate) fn validate_output_capacity(
    request: &PlanRequest,
    output: &PlanOutput<'_>,
) -> Option<PlannerError> {
    let required_steps = request.token_ids().len().div_ceil(normalize_step(request));
    if output.step_sizes.len() < required_steps
        || output.step_token_offsets.len() < required_steps + 1
    {
        Some(PlannerError::OutputStepsFull)
    } else if output.step_token_indices.len() < request.token_ids().len() {
        Some(PlannerError::OutputIndicesFull)
    } else {
        None
    }
}

#[derive(Clone, Debug)]
pub enum Event {
    Plan(PlanRequest),
    Unexpected,
}

struct Plan;

sml! {
    BatchPlanner[temporary_context: &mut PlanScratch] {
        "input_validation"_s <= *"idle"_s + Plan(PlanRuntime) / effect_begin,
        "step_normalization"_s <= "input_validation"_s + completion<Plan>(PlanRuntime) [guard_valid],
        "request_rejected"_s <= "input_validation"_s + completion<Plan>(PlanRuntime) [guard_invalid_request] / effect_reject_invalid_request,
        "request_rejected"_s <= "input_validation"_s + completion<Plan>(PlanRuntime) [guard_output_plan_full] / effect_reject_output_plan_full,
        "request_rejected"_s <= "input_validation"_s + completion<Plan>(PlanRuntime) [guard_invalid_sequence_metadata] / effect_reject_invalid_sequence_metadata,
        "request_rejected"_s <= "input_validation"_s + completion<Plan>(PlanRuntime) [guard_invalid_sequence_id] / effect_reject_invalid_sequence_id,
        "request_rejected"_s <= "input_validation"_s + completion<Plan>(PlanRuntime) [guard_invalid_sequence_mask] / effect_reject_invalid_sequence_mask,
        "request_rejected"_s <= "input_validation"_s + completion<Plan>(PlanRuntime) [guard_multiple_bits_in_mask] / effect_reject_multiple_bits_in_mask,
        "output_classification"_s <= "step_normalization"_s + completion<Plan>(PlanRuntime) [guard_step_normalized],
        "mode_selection"_s <= "output_classification"_s + completion<Plan>(PlanRuntime) [guard_outputs_classified],
        "simple_result_decision"_s(ModeResult) <= "mode_selection"_s + completion<Plan>(PlanRuntime) [guard_simple] / effect_dispatch_simple,
        "equal_result_decision"_s(ModeResult) <= "mode_selection"_s + completion<Plan>(PlanRuntime) [guard_equal] / effect_dispatch_equal,
        "sequential_result_decision"_s(ModeResult) <= "mode_selection"_s + completion<Plan>(PlanRuntime) [guard_sequential] / effect_dispatch_sequential,
        "request_rejected"_s <= "mode_selection"_s + completion<Plan>(PlanRuntime) [guard_invalid_mode] / effect_invalid_mode,
        "result_publish"_s <= "simple_result_decision"_s(ModeResult) + completion<Plan>(PlanRuntime) [guard_simple_succeeded] / effect_publish_done,
        "result_publish"_s <= "equal_result_decision"_s(ModeResult) + completion<Plan>(PlanRuntime) [guard_equal_succeeded] / effect_publish_done,
        "result_publish"_s <= "sequential_result_decision"_s(ModeResult) + completion<Plan>(PlanRuntime) [guard_sequential_succeeded] / effect_publish_done,
        "completed"_s <= "simple_result_decision"_s(ModeResult) + completion<Plan>(PlanRuntime) [guard_simple_failed] / effect_failure,
        "completed"_s <= "equal_result_decision"_s(ModeResult) + completion<Plan>(PlanRuntime) [guard_equal_failed] / effect_failure,
        "completed"_s <= "sequential_result_decision"_s(ModeResult) + completion<Plan>(PlanRuntime) [guard_sequential_failed] / effect_failure,
        "idle"_s <= "result_publish"_s + completion<Plan>(PlanRuntime),
        "idle"_s <= "completed"_s + completion<Plan>(PlanRuntime) / effect_publish_error,
        "idle"_s <= "request_rejected"_s + completion<Plan>(PlanRuntime) / effect_publish_error,
        "idle"_s <= "idle"_s + unexpected_event<_> / effect_unexpected,
        "idle"_s <= "input_validation"_s + unexpected_event<_> / effect_unexpected,
        "idle"_s <= "step_normalization"_s + unexpected_event<_> / effect_unexpected,
        "idle"_s <= "output_classification"_s + unexpected_event<_> / effect_unexpected,
        "idle"_s <= "mode_selection"_s + unexpected_event<_> / effect_unexpected,
        "idle"_s <= "simple_result_decision"_s(ModeResult) + unexpected_event<_> / effect_unexpected_simple_result,
        "idle"_s <= "equal_result_decision"_s(ModeResult) + unexpected_event<_> / effect_unexpected_equal_result,
        "idle"_s <= "sequential_result_decision"_s(ModeResult) + unexpected_event<_> / effect_unexpected_sequential_result,
        "idle"_s <= "result_publish"_s + unexpected_event<_> / effect_unexpected,
        "idle"_s <= "completed"_s + unexpected_event<_> / effect_unexpected,
        "idle"_s <= "request_rejected"_s + unexpected_event<_> / effect_unexpected,
    }
}

#[derive(Debug, Default)]
struct Context {
    simple: Box<crate::planner::modes::simple::sm::Actor>,
    equal: Box<crate::planner::modes::equal::sm::Actor>,
    sequential: Box<crate::planner::modes::sequential::sm::Actor>,
}

impl BatchPlannerStateMachineContext for Context {
    fn effect_begin(&mut self, scratch: &mut PlanScratch, _: PlanRuntime) -> Result<(), ()> {
        scratch.reset();
        self.simple.reset();
        self.equal.reset();
        self.sequential.reset();
        Ok(())
    }
    fn effect_dispatch_simple(
        &mut self,
        scratch: &mut PlanScratch,
        event: PlanRuntime,
    ) -> Result<ModeResult, ()> {
        Ok(self.simple.process_event(event, scratch))
    }
    fn effect_dispatch_equal(
        &mut self,
        scratch: &mut PlanScratch,
        event: PlanRuntime,
    ) -> Result<ModeResult, ()> {
        Ok(self.equal.process_event(event, scratch))
    }
    fn effect_dispatch_sequential(
        &mut self,
        scratch: &mut PlanScratch,
        event: PlanRuntime,
    ) -> Result<ModeResult, ()> {
        Ok(self.sequential.process_event(event, scratch))
    }
    fn guard_valid(&self, _: &mut PlanScratch, event: &PlanRuntime) -> Result<bool, ()> {
        Ok(validate_request(&event.request).is_ok())
    }
    fn guard_invalid_request(&self, _: &mut PlanScratch, event: &PlanRuntime) -> Result<bool, ()> {
        Ok(matches!(
            validate_request(&event.request),
            Err(PlannerError::InvalidRequest)
        ))
    }
    fn guard_output_plan_full(&self, _: &mut PlanScratch, event: &PlanRuntime) -> Result<bool, ()> {
        Ok(matches!(
            validate_request(&event.request),
            Err(PlannerError::OutputPlanFull)
        ))
    }
    fn guard_invalid_sequence_metadata(
        &self,
        _: &mut PlanScratch,
        event: &PlanRuntime,
    ) -> Result<bool, ()> {
        Ok(matches!(
            validate_request(&event.request),
            Err(PlannerError::InvalidSequenceMetadata)
        ))
    }
    fn guard_invalid_sequence_id(
        &self,
        _: &mut PlanScratch,
        event: &PlanRuntime,
    ) -> Result<bool, ()> {
        Ok(matches!(
            validate_request(&event.request),
            Err(PlannerError::InvalidSequenceId)
        ))
    }
    fn guard_invalid_sequence_mask(
        &self,
        _: &mut PlanScratch,
        event: &PlanRuntime,
    ) -> Result<bool, ()> {
        Ok(matches!(
            validate_request(&event.request),
            Err(PlannerError::InvalidSequenceMask)
        ))
    }
    fn guard_multiple_bits_in_mask(
        &self,
        _: &mut PlanScratch,
        event: &PlanRuntime,
    ) -> Result<bool, ()> {
        Ok(matches!(
            validate_request(&event.request),
            Err(PlannerError::MultipleBitsInMask)
        ))
    }
    fn effect_reject_invalid_request(
        &mut self,
        scratch: &mut PlanScratch,
        _: PlanRuntime,
    ) -> Result<(), ()> {
        scratch.fail(PlannerError::InvalidRequest);
        Ok(())
    }
    fn effect_reject_output_plan_full(
        &mut self,
        scratch: &mut PlanScratch,
        _: PlanRuntime,
    ) -> Result<(), ()> {
        scratch.fail(PlannerError::OutputPlanFull);
        Ok(())
    }
    fn effect_reject_invalid_sequence_metadata(
        &mut self,
        scratch: &mut PlanScratch,
        _: PlanRuntime,
    ) -> Result<(), ()> {
        scratch.fail(PlannerError::InvalidSequenceMetadata);
        Ok(())
    }
    fn effect_reject_invalid_sequence_id(
        &mut self,
        scratch: &mut PlanScratch,
        _: PlanRuntime,
    ) -> Result<(), ()> {
        scratch.fail(PlannerError::InvalidSequenceId);
        Ok(())
    }
    fn effect_reject_invalid_sequence_mask(
        &mut self,
        scratch: &mut PlanScratch,
        _: PlanRuntime,
    ) -> Result<(), ()> {
        scratch.fail(PlannerError::InvalidSequenceMask);
        Ok(())
    }
    fn effect_reject_multiple_bits_in_mask(
        &mut self,
        scratch: &mut PlanScratch,
        _: PlanRuntime,
    ) -> Result<(), ()> {
        scratch.fail(PlannerError::MultipleBitsInMask);
        Ok(())
    }
    fn guard_step_normalized(&self, _: &mut PlanScratch, event: &PlanRuntime) -> Result<bool, ()> {
        Ok(event.effective_step_size == normalize_step(&event.request)
            && event.effective_step_size > 0)
    }
    fn guard_outputs_classified(
        &self,
        _: &mut PlanScratch,
        event: &PlanRuntime,
    ) -> Result<bool, ()> {
        Ok(event.total_outputs == total_outputs(&event.request))
    }
    fn guard_simple(&self, _: &mut PlanScratch, event: &PlanRuntime) -> Result<bool, ()> {
        Ok(matches!(event.request.mode(), PlanMode::Simple))
    }
    fn guard_equal(&self, _: &mut PlanScratch, event: &PlanRuntime) -> Result<bool, ()> {
        Ok(matches!(event.request.mode(), PlanMode::Equal))
    }
    fn guard_sequential(&self, _: &mut PlanScratch, event: &PlanRuntime) -> Result<bool, ()> {
        Ok(matches!(
            event.request.mode(),
            PlanMode::Sequential | PlanMode::Seq
        ))
    }
    fn guard_invalid_mode(&self, _: &mut PlanScratch, event: &PlanRuntime) -> Result<bool, ()> {
        Ok(matches!(event.request.mode(), PlanMode::Invalid))
    }
    fn guard_simple_succeeded(
        &self,
        _: &mut PlanScratch,
        result: &ModeResult,
        _: &PlanRuntime,
    ) -> Result<bool, ()> {
        Ok(matches!(result, ModeResult::Complete))
    }
    fn guard_equal_succeeded(
        &self,
        _: &mut PlanScratch,
        result: &ModeResult,
        _: &PlanRuntime,
    ) -> Result<bool, ()> {
        Ok(matches!(result, ModeResult::Complete))
    }
    fn guard_sequential_succeeded(
        &self,
        _: &mut PlanScratch,
        result: &ModeResult,
        _: &PlanRuntime,
    ) -> Result<bool, ()> {
        Ok(matches!(result, ModeResult::Complete))
    }
    fn guard_simple_failed(
        &self,
        _: &mut PlanScratch,
        result: &ModeResult,
        _: &PlanRuntime,
    ) -> Result<bool, ()> {
        Ok(matches!(result, ModeResult::Error(_)))
    }
    fn guard_equal_failed(
        &self,
        _: &mut PlanScratch,
        result: &ModeResult,
        _: &PlanRuntime,
    ) -> Result<bool, ()> {
        Ok(matches!(result, ModeResult::Error(_)))
    }
    fn guard_sequential_failed(
        &self,
        _: &mut PlanScratch,
        result: &ModeResult,
        _: &PlanRuntime,
    ) -> Result<bool, ()> {
        Ok(matches!(result, ModeResult::Error(_)))
    }
    fn effect_failure(
        &mut self,
        scratch: &mut PlanScratch,
        result: &ModeResult,
        _: PlanRuntime,
    ) -> Result<(), ()> {
        if let ModeResult::Error(error) = result {
            scratch.fail(*error);
        }
        Ok(())
    }
    fn effect_publish_done(
        &mut self,
        scratch: &mut PlanScratch,
        result: &ModeResult,
        event: PlanRuntime,
    ) -> Result<(), ()> {
        if matches!(result, ModeResult::Complete) {
            scratch.error = None;
            planner_event("emel.planner.callback", "callback", "done_start");
            let _ = (event.request.on_done())(PlanDone {
                request: &event.request,
                step_sizes: scratch.step_sizes.as_slice(),
                step_token_indices: scratch.step_token_indices.as_slice(),
                step_token_offsets: scratch.step_token_offsets.as_slice(),
                total_outputs: scratch.total_outputs,
            });
            planner_event("emel.planner.callback", "callback", "done_end");
        }
        Ok(())
    }
    fn effect_publish_error(
        &mut self,
        scratch: &mut PlanScratch,
        event: PlanRuntime,
    ) -> Result<(), ()> {
        if let Some(error) = scratch.error {
            planner_event("emel.planner.callback", "callback", "error_start");
            let _ = (event.request.on_error())(PlanError {
                request: &event.request,
                error,
            });
            planner_event("emel.planner.callback", "callback", "error_end");
        }
        Ok(())
    }
    fn effect_invalid_mode(&mut self, scratch: &mut PlanScratch, _: PlanRuntime) -> Result<(), ()> {
        scratch.fail(PlannerError::InvalidRequestAndMode);
        Ok(())
    }
    fn effect_unexpected(&mut self, scratch: &mut PlanScratch) -> Result<(), ()> {
        scratch.fail(PlannerError::UnexpectedEvent);
        Ok(())
    }
    fn effect_unexpected_simple_result(
        &mut self,
        scratch: &mut PlanScratch,
        _: &ModeResult,
    ) -> Result<(), ()> {
        scratch.fail(PlannerError::UnexpectedEvent);
        Ok(())
    }
    fn effect_unexpected_equal_result(
        &mut self,
        scratch: &mut PlanScratch,
        _: &ModeResult,
    ) -> Result<(), ()> {
        scratch.fail(PlannerError::UnexpectedEvent);
        Ok(())
    }
    fn effect_unexpected_sequential_result(
        &mut self,
        scratch: &mut PlanScratch,
        _: &ModeResult,
    ) -> Result<(), ()> {
        scratch.fail(PlannerError::UnexpectedEvent);
        Ok(())
    }
}

/// Reusable batch planner state machine.
pub struct Planner {
    machine: BatchPlannerStateMachine<Context>,
    scratch: Box<PlanScratch>,
}

impl fmt::Debug for Planner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Planner").finish_non_exhaustive()
    }
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
            machine: BatchPlannerStateMachine::new(Context::default()),
            scratch: Box::new(PlanScratch::default()),
        }
    }

    /// Processes a planner event to completion.
    ///
    /// # Errors
    ///
    /// Returns [`PlannerError`] when planning fails or an unexpected event is received.
    pub fn process_event(&mut self, event: Event) -> Result<PlanResult, PlannerError> {
        match event {
            Event::Unexpected => self.process_unexpected(),
            Event::Plan(request) => self.plan(request),
        }
    }

    fn dispatch(&mut self, request: PlanRequest) -> Result<(), PlannerError> {
        let mode = request.mode();
        with_planner_span(mode, PLANNER_OPERATION, || {
            let runtime = PlanRuntime {
                effective_step_size: normalize_step(&request),
                total_outputs: total_outputs(&request),
                request,
            };
            let scratch = &mut *self.scratch;
            if self
                .machine
                .process_event(scratch, BatchPlannerEvents::Plan(runtime))
                .is_err()
            {
                return Err(PlannerError::Internal);
            }
            scratch.error.map_or(Ok(()), Err)
        })
    }

    /// Plans into caller-owned storage without allocating result vectors.
    ///
    /// The supplied slices must have room for the complete result. On success,
    /// their prefixes are described by [`PlanOutput`] length fields. The normal
    /// synchronous completion callback still receives its borrowed scratch view.
    ///
    /// # Errors
    ///
    /// Returns [`PlannerError::OutputStepsFull`] or
    /// [`PlannerError::OutputIndicesFull`] when caller-owned storage is too small,
    /// or another planner error when the request cannot be completed.
    pub fn plan_into(
        &mut self,
        request: PlanRequest,
        output: &mut PlanOutput<'_>,
    ) -> Result<(), PlannerError> {
        output.reset();
        let mode = request.mode();
        if let Some(error) = validate_request(&request)
            .err()
            .or_else(|| validate_output_capacity(&request, output))
        {
            return with_planner_span(mode, PLANNER_OPERATION, || {
                planner_event("emel.planner.callback", "callback", "error_start");
                let _ = (request.on_error())(PlanError {
                    request: &request,
                    error,
                });
                planner_event("emel.planner.callback", "callback", "error_end");
                Err(error)
            });
        }
        self.dispatch(request)?;
        output.write_from(self.scratch.as_ref());
        Ok(())
    }

    /// Plans a batch request using the selected mode.
    ///
    /// # Errors
    ///
    /// Returns a [`PlannerError`] when the request cannot be completed.
    pub fn plan(&mut self, request: PlanRequest) -> Result<PlanResult, PlannerError> {
        self.dispatch(request)?;
        Ok(PlanResult {
            step_sizes: self.scratch.step_sizes.as_slice().to_vec(),
            step_token_indices: self.scratch.step_token_indices.as_slice().to_vec(),
            step_token_offsets: self.scratch.step_token_offsets.as_slice().to_vec(),
            total_outputs: self.scratch.total_outputs,
        })
    }

    /// Always returns [`PlannerError::UnexpectedEvent`].
    ///
    /// # Errors
    ///
    /// Always returns [`PlannerError::UnexpectedEvent`].
    pub fn process_unexpected(&mut self) -> Result<PlanResult, PlannerError> {
        Err(PlannerError::UnexpectedEvent)
    }
    #[must_use]
    pub fn is_idle(&self) -> bool {
        self.machine.is(&BatchPlannerStates::Idle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::num::NonZeroUsize;

    #[test]
    fn simple_plan_normalizes_and_chunks() {
        let mut planner = Planner::new();
        let result = planner
            .plan(PlanRequest::new(
                (0..5).collect(),
                PlanConfig::new(PlanMode::Simple).with_step_size(StepSize::fixed(
                    NonZeroUsize::new(2).expect("step size must be positive"),
                )),
            ))
            .unwrap();
        assert_eq!(result.step_sizes, vec![2, 2, 1]);
        assert_eq!(result.step_token_indices, vec![0, 1, 2, 3, 4]);
        assert_eq!(result.step_token_offsets, vec![0, 2, 4, 5]);
        assert_eq!(result.total_outputs, 1);
    }
    #[test]
    fn callbacks_receive_borrowed_done_and_error_before_dispatch_returns() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static DONE: AtomicUsize = AtomicUsize::new(0);
        static ERR: AtomicUsize = AtomicUsize::new(0);

        #[allow(clippy::needless_pass_by_value)]
        fn on_done(event: PlanDone<'_>) -> bool {
            assert_eq!(event.step_sizes, [2, 1]);
            assert_eq!(event.step_token_indices, [0, 1, 2]);
            assert_eq!(event.step_token_offsets, [0, 2, 3]);
            assert_eq!(event.total_outputs, 1);
            DONE.fetch_add(1, Ordering::SeqCst);
            true
        }
        #[allow(clippy::needless_pass_by_value)]
        fn on_error(event: PlanError<'_>) -> bool {
            assert_eq!(event.error, PlannerError::InvalidRequest);
            ERR.fetch_add(1, Ordering::SeqCst);
            true
        }

        let mut planner = Planner::new();
        let request = PlanRequest::new(
            vec![0, 1, 2],
            PlanConfig::new(PlanMode::Simple).with_step_size(StepSize::fixed(
                NonZeroUsize::new(2).expect("step size must be positive"),
            )),
        )
        .with_callbacks(on_done, on_error);
        let result = planner
            .plan(request)
            .expect("done callback request succeeds");
        assert_eq!(result.step_sizes, [2, 1]);
        assert_eq!(DONE.load(Ordering::SeqCst), 1);

        let invalid = PlanRequest::new(
            Vec::new(),
            PlanConfig::new(PlanMode::Simple).with_step_size(StepSize::fixed(
                NonZeroUsize::new(1).expect("step size must be positive"),
            )),
        )
        .with_callbacks(on_done, on_error);
        assert_eq!(planner.plan(invalid), Err(PlannerError::InvalidRequest));
        assert_eq!(ERR.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn invalid_mode_reports_composite_error_synchronously() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        static CALLBACK_COUNT: AtomicUsize = AtomicUsize::new(0);

        #[allow(clippy::needless_pass_by_value)]
        fn on_error(event: PlanError<'_>) -> bool {
            assert_eq!(event.request.mode(), PlanMode::Invalid);
            assert_eq!(event.error, PlannerError::InvalidRequestAndMode);
            assert!(event.error.is_invalid_request());
            assert!(event.error.is_invalid_mode());
            CALLBACK_COUNT.fetch_add(1, Ordering::SeqCst);
            true
        }

        let mut planner = Planner::new();
        let request = PlanRequest::new(
            vec![0],
            PlanConfig::new(PlanMode::Invalid).with_step_size(StepSize::fixed(
                NonZeroUsize::new(1).expect("step size must be positive"),
            )),
        )
        .with_callbacks(ignore_plan_done, on_error);
        let result = planner.plan(request);

        assert_eq!(result, Err(PlannerError::InvalidRequestAndMode));
        assert_eq!(CALLBACK_COUNT.load(Ordering::SeqCst), 1);
        assert!(planner.is_idle());
    }

    #[test]
    fn invalid_request_and_unexpected_are_typed() {
        let mut planner = Planner::new();
        assert_eq!(
            planner.plan(PlanRequest::new(
                Vec::new(),
                PlanConfig::new(PlanMode::Simple).with_step_size(StepSize::fixed(
                    NonZeroUsize::new(1).expect("step size must be positive")
                )),
            )),
            Err(PlannerError::InvalidRequest)
        );
        assert_eq!(
            planner.process_unexpected(),
            Err(PlannerError::UnexpectedEvent)
        );
        assert!(planner.is_idle());
    }

    #[test]
    fn equal_general_without_masks_chunks_tokens() {
        let mut planner = Planner::new();
        let result = planner
            .plan(PlanRequest::new(
                (0..5).collect(),
                PlanConfig::new(PlanMode::Equal).with_step_size(StepSize::fixed(
                    NonZeroUsize::new(2).expect("step size must be positive"),
                )),
            ))
            .unwrap();
        assert_eq!(result.step_sizes, vec![2, 2, 1]);
        assert_eq!(result.step_token_indices, vec![0, 1, 2, 3, 4]);
        assert_eq!(result.step_token_offsets, vec![0, 2, 4, 5]);
    }

    #[test]
    fn equal_sequential_keeps_nonconsecutive_primary_groups_separate() {
        let mut planner = Planner::new();
        let request = PlanRequest::new(
            (0..3).collect(),
            PlanConfig::new(PlanMode::Equal)
                .with_step_size(StepSize::fixed(
                    NonZeroUsize::new(2).expect("step size must be positive"),
                ))
                .with_sequence(
                    SequenceConfig::new(MaskWordCount::new(1).unwrap())
                        .with_masks(vec![[1, 0, 0, 0], [2, 0, 0, 0], [4, 0, 0, 0]])
                        .with_primary_ids(vec![0, 2, 1]),
                ),
        );
        let result = planner.plan(request).unwrap();
        assert_eq!(result.step_sizes, vec![2, 1]);
        assert_eq!(result.step_token_indices, vec![0, 2, 1]);
        assert_eq!(result.step_token_offsets, vec![0, 2, 3]);
    }

    #[test]
    fn equal_missing_primary_ids_uses_general_path() {
        let mut planner = Planner::new();
        let result = planner
            .plan(PlanRequest::new(
                (0..3).collect(),
                PlanConfig::new(PlanMode::Equal).with_step_size(StepSize::fixed(
                    NonZeroUsize::new(2).expect("step size must be positive"),
                )),
            ))
            .unwrap();
        assert_eq!(result.step_sizes, vec![2, 1]);
        assert_eq!(result.step_token_indices, vec![0, 1, 2]);
    }

    #[test]
    fn equal_rejects_missing_primary_id_entries() {
        let mut planner = Planner::new();
        let request = PlanRequest::new(
            vec![0, 1],
            PlanConfig::new(PlanMode::Equal)
                .with_step_size(StepSize::fixed(
                    NonZeroUsize::new(2).expect("step size must be positive"),
                ))
                .with_sequence(
                    SequenceConfig::new(MaskWordCount::new(1).unwrap()).with_primary_ids(vec![0]),
                ),
        );
        assert_eq!(planner.plan(request), Err(PlannerError::InvalidSequenceId));
    }

    #[test]
    fn equal_rejects_invalid_primary_id() {
        let mut planner = Planner::new();
        let request = PlanRequest::new(
            vec![0, 1],
            PlanConfig::new(PlanMode::Equal)
                .with_step_size(StepSize::fixed(
                    NonZeroUsize::new(2).expect("step size must be positive"),
                ))
                .with_sequence(
                    SequenceConfig::new(MaskWordCount::new(1).unwrap())
                        .with_primary_ids(vec![0, -1]),
                ),
        );
        assert_eq!(planner.plan(request), Err(PlannerError::InvalidSequenceId));
    }

    #[test]
    fn equal_reports_progress_stall_when_groups_exceed_step_capacity() {
        let mut planner = Planner::new();
        let request = PlanRequest::new(
            vec![0, 1],
            PlanConfig::new(PlanMode::Equal)
                .with_step_size(StepSize::fixed(
                    NonZeroUsize::new(1).expect("step size must be positive"),
                ))
                .with_equal_strategy(EqualStrategy::General)
                .with_sequence(
                    SequenceConfig::new(MaskWordCount::new(1).unwrap())
                        .with_primary_ids(vec![0, 1]),
                ),
        );
        assert_eq!(
            planner.plan(request),
            Err(PlannerError::PlanningProgressStalled)
        );
    }

    #[test]
    fn equal_primary_fast_path_preserves_grouped_indices_and_offsets() {
        let mut planner = Planner::new();
        let request = PlanRequest::new(
            (0..6).collect(),
            PlanConfig::new(PlanMode::Equal)
                .with_step_size(StepSize::fixed(
                    NonZeroUsize::new(4).expect("step size must be positive"),
                ))
                .with_sequence(
                    SequenceConfig::new(MaskWordCount::new(1).unwrap())
                        .with_primary_ids(vec![0, 0, 1, 1, 2, 2]),
                ),
        );
        let result = planner.plan(request).unwrap();
        assert_eq!(result.step_sizes, vec![3, 3]);
        assert_eq!(result.step_token_indices, vec![0, 2, 4, 1, 3, 5]);
        assert_eq!(result.step_token_offsets, vec![0, 3, 6]);
    }

    #[test]
    fn equal_zero_requested_step_size_defaults_to_all_tokens() {
        let mut planner = Planner::new();
        let result = planner
            .plan(PlanRequest::new(
                vec![0, 1],
                PlanConfig::new(PlanMode::Equal).with_step_size(StepSize::automatic()),
            ))
            .unwrap();
        assert_eq!(result.step_sizes, vec![2]);
        assert_eq!(result.step_token_indices, vec![0, 1]);
        assert_eq!(result.step_token_offsets, vec![0, 2]);
    }

    #[test]
    fn equal_recovers_after_error() {
        let mut planner = Planner::new();
        let invalid = PlanRequest::new(
            vec![0, 1],
            PlanConfig::new(PlanMode::Equal)
                .with_step_size(StepSize::fixed(
                    NonZeroUsize::new(1).expect("step size must be positive"),
                ))
                .with_sequence(
                    SequenceConfig::new(MaskWordCount::new(1).unwrap())
                        .with_primary_ids(vec![0, -1]),
                ),
        );
        assert_eq!(planner.plan(invalid), Err(PlannerError::InvalidSequenceId));

        let result = planner
            .plan(PlanRequest::new(
                (0..3).collect(),
                PlanConfig::new(PlanMode::Equal).with_step_size(StepSize::fixed(
                    NonZeroUsize::new(2).expect("step size must be positive"),
                )),
            ))
            .unwrap();
        assert_eq!(result.step_sizes, vec![2, 1]);
        assert_eq!(result.step_token_indices, vec![0, 1, 2]);
        assert!(planner.is_idle());
    }

    #[test]
    fn seq_mode_alias_uses_sequential_planner() {
        let mut planner = Planner::new();
        let result = planner
            .plan(PlanRequest::new(
                (0..3).collect(),
                PlanConfig::new(PlanMode::Seq).with_step_size(StepSize::fixed(
                    NonZeroUsize::new(2).expect("step size must be positive"),
                )),
            ))
            .expect("seq alias should select sequential mode");
        assert_eq!(result.step_sizes, [2, 1]);
        assert_eq!(result.step_token_indices, [0, 1, 2]);
        assert_eq!(result.step_token_offsets, [0, 2, 3]);
    }

    #[test]
    fn sequence_masks_are_validated_for_every_mode() {
        let mut planner = Planner::new();
        let request = PlanRequest::new(
            vec![0, 1],
            PlanConfig::new(PlanMode::Simple)
                .with_step_size(StepSize::fixed(
                    NonZeroUsize::new(1).expect("step size must be positive"),
                ))
                .with_sequence(
                    SequenceConfig::new(MaskWordCount::new(1).unwrap())
                        .with_masks(vec![[0; SEQ_WORDS], [1, 0, 0, 0]]),
                ),
        );
        assert_eq!(
            planner.plan(request),
            Err(PlannerError::InvalidSequenceMask)
        );
    }

    #[test]
    fn plan_into_writes_caller_storage_without_replacing_pointers() {
        let mut planner = Planner::new();
        let mut sizes = [usize::MAX; 8];
        let mut indices = [usize::MAX; 8];
        let mut offsets = [usize::MAX; 9];
        let sizes_ptr = sizes.as_mut_ptr();
        let indices_ptr = indices.as_mut_ptr();
        let offsets_ptr = offsets.as_mut_ptr();
        let mut output = PlanOutput::new(&mut sizes, &mut indices, &mut offsets);
        planner
            .plan_into(
                PlanRequest::new(
                    (0..5).collect(),
                    PlanConfig::new(PlanMode::Simple).with_step_size(StepSize::fixed(
                        NonZeroUsize::new(2).expect("step size must be positive"),
                    )),
                ),
                &mut output,
            )
            .unwrap();
        assert_eq!(output.step_sizes_slice(), [2, 2, 1]);
        assert_eq!(output.step_token_indices_slice(), [0, 1, 2, 3, 4]);
        assert_eq!(output.step_token_offsets_slice(), [0, 2, 4, 5]);
        assert_eq!(output.total_outputs, 1);
        assert_eq!(output.step_sizes.as_ptr(), sizes_ptr);
        assert_eq!(output.step_token_indices.as_ptr(), indices_ptr);
        assert_eq!(output.step_token_offsets.as_ptr(), offsets_ptr);
    }

    #[test]
    fn plan_into_reports_capacity_before_dispatch() {
        let mut planner = Planner::new();
        let mut sizes = [0; 2];
        let mut indices = [0; 5];
        let mut offsets = [0; 3];
        let mut output = PlanOutput::new(&mut sizes, &mut indices, &mut offsets);
        assert_eq!(
            planner.plan_into(
                PlanRequest::new(
                    (0..5).collect(),
                    PlanConfig::new(PlanMode::Simple).with_step_size(StepSize::fixed(
                        NonZeroUsize::new(2).expect("step size must be positive")
                    )),
                ),
                &mut output,
            ),
            Err(PlannerError::OutputStepsFull)
        );
        assert_eq!(output.step_sizes_len, 0);
        assert_eq!(output.step_token_indices_len, 0);
        assert_eq!(output.step_token_offsets_len, 0);
    }

    #[test]
    fn plan_into_preserves_partition_semantics_for_all_modes() {
        let cases = [
            (PlanMode::Simple, vec![2, 2, 1], vec![0, 1, 2, 3, 4]),
            (PlanMode::Equal, vec![2, 2, 1], vec![0, 1, 2, 3, 4]),
            (PlanMode::Sequential, vec![2, 2, 1], vec![0, 1, 2, 3, 4]),
            (PlanMode::Seq, vec![2, 2, 1], vec![0, 1, 2, 3, 4]),
        ];
        for (mode, expected_sizes, expected_indices) in cases {
            let mut planner = Planner::new();
            let mut sizes = [0; 5];
            let mut indices = [0; 5];
            let mut offsets = [0; 6];
            let mut output = PlanOutput::new(&mut sizes, &mut indices, &mut offsets);
            planner
                .plan_into(
                    PlanRequest::new(
                        (0..5).collect(),
                        PlanConfig::new(mode).with_step_size(StepSize::fixed(
                            NonZeroUsize::new(2).expect("step size must be positive"),
                        )),
                    ),
                    &mut output,
                )
                .unwrap();
            assert_eq!(output.step_sizes_slice(), expected_sizes);
            assert_eq!(output.step_token_indices_slice(), expected_indices);
            assert_eq!(output.step_token_offsets_slice(), [0, 2, 4, 5]);
        }
    }

    #[test]
    fn multiple_mask_bits_are_restricted_to_equal_sequential_mode() {
        let mut planner = Planner::new();
        let request = PlanRequest::new(
            vec![0, 1],
            PlanConfig::new(PlanMode::Sequential)
                .with_step_size(StepSize::fixed(
                    NonZeroUsize::new(2).expect("step size must be positive"),
                ))
                .with_sequence(
                    SequenceConfig::new(MaskWordCount::new(1).unwrap())
                        .with_masks(vec![[3, 0, 0, 0], [1, 0, 0, 0]]),
                ),
        );
        let result = planner
            .plan(request)
            .expect("sequential mode accepts composite masks");
        assert_eq!(result.step_sizes, [2]);
    }
}
