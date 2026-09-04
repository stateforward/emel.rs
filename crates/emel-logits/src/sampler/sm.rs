//! Run-to-completion logits sampler actor.

#![allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::assigning_clones,
    clippy::cast_sign_loss,
    clippy::derive_partial_eq_without_eq,
    clippy::elidable_lifetime_names,
    clippy::map_unwrap_or,
    clippy::unnecessary_wraps,
    private_interfaces
)]

use core::cell::{Cell, RefCell};
use core::fmt;

use sml::sml;

const RANDOM_MODULUS: u32 = 2_147_483_647;
const RANDOM_MULTIPLIER: u64 = 16_807;
const MAX_SAMPLERS: usize = 32;

/// Error returned by a configured sampler callback.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SamplerCallbackError {
    /// The callback rejected the candidate set.
    InvalidRequest,
    /// The callback backend failed.
    Backend,
    /// The callback encountered an internal failure.
    Internal,
    /// The callback reported an unclassified failure.
    Untracked,
}

/// Error returned by the sampler actor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SamplerError {
    /// The request or workspace is invalid.
    InvalidRequest,
    /// The actor has no configured callback chain.
    MissingSampler,
    /// A configured callback failed.
    Callback(SamplerCallbackError),
    /// An event was received outside the actor's lifecycle.
    UnexpectedEvent,
    /// The generated machine rejected a dispatch.
    Internal,
}

impl fmt::Display for SamplerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRequest => f.write_str("invalid logits sampler request"),
            Self::MissingSampler => f.write_str("sampler chain is empty"),
            Self::Callback(error) => write!(f, "sampler callback failed: {error:?}"),
            Self::UnexpectedEvent => f.write_str("unexpected logits sampler event"),
            Self::Internal => f.write_str("internal logits sampler error"),
        }
    }
}

impl std::error::Error for SamplerError {}

/// Immediate, synchronous sampler callback.
///
/// A callback must not re-enter its owning [`Sampler`] actor.
pub type SamplerFn = fn(
    candidate_ids: &mut [i32],
    candidate_scores: &mut [f32],
    candidate_count: &mut usize,
    selected_token: &mut i32,
) -> Result<(), SamplerCallbackError>;

/// Configuration request for a callback sampler chain.
#[derive(Clone, Copy, Debug)]
pub struct Configure<'event> {
    /// Caller-owned callback chain.
    pub sampler_fns: &'event [SamplerFn],
}

impl<'event> Configure<'event> {
    /// Creates a configuration request over caller-owned callbacks.
    #[must_use]
    pub const fn new(sampler_fns: &'event [SamplerFn]) -> Self {
        Self { sampler_fns }
    }
}

/// Caller-owned logits request and candidate workspace.
pub struct SampleLogits<'event> {
    /// Unnormalized vocabulary logits.
    pub logits: &'event [f32],
    /// Caller-owned candidate token storage.
    pub candidate_ids: RefCell<&'event mut [i32]>,
    /// Caller-owned candidate score storage.
    pub candidate_scores: RefCell<&'event mut [f32]>,
    /// Candidate storage capacity explicitly supplied by the caller.
    pub candidate_capacity: usize,
    /// Caller-owned selected token output.
    pub selected_token_out: RefCell<&'event mut i32>,
}

impl<'event> SampleLogits<'event> {
    /// Creates a request over caller-owned logits, candidates, and output.
    #[must_use]
    pub const fn new(
        logits: &'event [f32],
        candidate_ids: &'event mut [i32],
        candidate_scores: &'event mut [f32],
        candidate_capacity: usize,
        selected_token_out: &'event mut i32,
    ) -> Self {
        Self {
            logits,
            candidate_ids: RefCell::new(candidate_ids),
            candidate_scores: RefCell::new(candidate_scores),
            candidate_capacity,
            selected_token_out: RefCell::new(selected_token_out),
        }
    }
}

/// Request validating a preselected token.
#[derive(Clone, Copy, Debug)]
pub struct SamplePreselected {
    /// Vocabulary cardinality.
    pub vocab_size: usize,
    /// Upstream-selected token.
    pub selected_token: i32,
}

impl SamplePreselected {
    /// Creates a preselected-token request.
    #[must_use]
    pub const fn new(vocab_size: usize, selected_token: i32) -> Self {
        Self {
            vocab_size,
            selected_token,
        }
    }
}

/// Native temperature/top-k request over caller-owned buffers.
pub struct SampleTemperatureTopK<'event> {
    /// Number of logits to process.
    pub card: usize,
    /// Positive temperature divisor.
    pub temperature: f32,
    /// Number of ranked candidates.
    pub top_k: usize,
    /// Caller-owned logits, modified in place by scaling and softmax.
    pub logits: RefCell<&'event mut [f32]>,
    /// Caller-owned sorted-index scratch.
    pub sorted_indices: RefCell<&'event mut [i32]>,
    /// Caller-owned top probability scratch.
    pub top_probabilities: RefCell<&'event mut [f32]>,
    /// Caller-owned top index scratch.
    pub top_indices: RefCell<&'event mut [i32]>,
    /// Caller-owned random state.
    pub random_state: RefCell<&'event mut u32>,
    /// Caller-owned selected token output.
    pub selected_token_out: RefCell<&'event mut i32>,
    /// Caller-owned selected score output.
    pub selected_score_out: RefCell<&'event mut f32>,
}

impl<'event> SampleTemperatureTopK<'event> {
    /// Creates a native sampling request over caller-owned storage.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        card: usize,
        temperature: f32,
        top_k: usize,
        logits: &'event mut [f32],
        sorted_indices: &'event mut [i32],
        top_probabilities: &'event mut [f32],
        top_indices: &'event mut [i32],
        random_state: &'event mut u32,
        selected_token_out: &'event mut i32,
        selected_score_out: &'event mut f32,
    ) -> Self {
        Self {
            card,
            temperature,
            top_k,
            logits: RefCell::new(logits),
            sorted_indices: RefCell::new(sorted_indices),
            top_probabilities: RefCell::new(top_probabilities),
            top_indices: RefCell::new(top_indices),
            random_state: RefCell::new(random_state),
            selected_token_out: RefCell::new(selected_token_out),
            selected_score_out: RefCell::new(selected_score_out),
        }
    }
}

struct ConfigureRuntime<'dispatch, 'event> {
    request: Configure<'event>,
    result: &'dispatch Cell<Result<(), SamplerError>>,
}

struct SampleLogitsRuntime<'dispatch, 'event> {
    request: SampleLogits<'event>,
    candidate_count: &'dispatch Cell<usize>,
    sampler_index: &'dispatch Cell<usize>,
    callback_error: &'dispatch Cell<Option<SamplerCallbackError>>,
    result: &'dispatch Cell<Result<i32, SamplerError>>,
}

struct SamplePreselectedRuntime<'dispatch, 'event> {
    request: SamplePreselected,
    result: &'dispatch Cell<Result<i32, SamplerError>>,
    _event: core::marker::PhantomData<&'event ()>,
}

struct SampleTemperatureTopKRuntime<'dispatch, 'event> {
    request: SampleTemperatureTopK<'event>,
    result: &'dispatch Cell<Result<(i32, f32), SamplerError>>,
}

sml! {
    LogitsSampler<'dispatch, 'event>
    where
        'event: 'dispatch,
    {
        "configure_decision"_s <= *"ready"_s + Configure(&'dispatch ConfigureRuntime<'dispatch, 'event>) / effect_begin_configure,
        "done"_s <= "configure_decision"_s + completion<Configure>(&'dispatch ConfigureRuntime<'dispatch, 'event>) [guard_config_valid] / effect_configure,
        "errored"_s <= "configure_decision"_s + completion<Configure>(&'dispatch ConfigureRuntime<'dispatch, 'event>) [guard_config_invalid] / effect_invalid_configure,

        "logits_decision"_s <= "ready"_s + SampleLogits(&'dispatch SampleLogitsRuntime<'dispatch, 'event>),
        "candidate_prepare"_s <= "logits_decision"_s + completion<SampleLogits>(&'dispatch SampleLogitsRuntime<'dispatch, 'event>) [guard_logits_valid] / effect_begin_logits,
        "errored"_s <= "logits_decision"_s + completion<SampleLogits>(&'dispatch SampleLogitsRuntime<'dispatch, 'event>) [guard_logits_invalid] / effect_invalid_logits,
        "sampler_decision"_s <= "candidate_prepare"_s + completion<SampleLogits>(&'dispatch SampleLogitsRuntime<'dispatch, 'event>) / effect_prepare_candidates,
        "sampler_call"_s <= "sampler_decision"_s + completion<SampleLogits>(&'dispatch SampleLogitsRuntime<'dispatch, 'event>) [guard_sampler_available] / effect_apply_sampler,
        "errored"_s <= "sampler_decision"_s + completion<SampleLogits>(&'dispatch SampleLogitsRuntime<'dispatch, 'event>) [guard_sampler_missing] / effect_missing_sampler,
        "sampler_result"_s <= "sampler_call"_s + completion<SampleLogits>(&'dispatch SampleLogitsRuntime<'dispatch, 'event>),
        "sampler_decision"_s <= "sampler_result"_s + completion<SampleLogits>(&'dispatch SampleLogitsRuntime<'dispatch, 'event>) [guard_callback_valid] / effect_advance_sampler,
        "errored"_s <= "sampler_result"_s + completion<SampleLogits>(&'dispatch SampleLogitsRuntime<'dispatch, 'event>) [guard_callback_invalid] / effect_invalid_logits,
        "errored"_s <= "sampler_result"_s + completion<SampleLogits>(&'dispatch SampleLogitsRuntime<'dispatch, 'event>) [guard_callback_failed] / effect_callback_error,
        "complete_decision"_s <= "sampler_decision"_s + completion<SampleLogits>(&'dispatch SampleLogitsRuntime<'dispatch, 'event>) [guard_chain_complete],
        "done"_s <= "complete_decision"_s + completion<SampleLogits>(&'dispatch SampleLogitsRuntime<'dispatch, 'event>) [guard_selected_valid] / effect_publish_logits,
        "errored"_s <= "complete_decision"_s + completion<SampleLogits>(&'dispatch SampleLogitsRuntime<'dispatch, 'event>) [guard_selected_invalid] / effect_invalid_logits,

        "preselected_decision"_s <= "ready"_s + SamplePreselected(&'dispatch SamplePreselectedRuntime<'dispatch, 'event>),
        "done"_s <= "preselected_decision"_s + completion<SamplePreselected>(&'dispatch SamplePreselectedRuntime<'dispatch, 'event>) [guard_preselected_valid] / effect_publish_preselected,
        "errored"_s <= "preselected_decision"_s + completion<SamplePreselected>(&'dispatch SamplePreselectedRuntime<'dispatch, 'event>) [guard_preselected_invalid] / effect_invalid_preselected,

        "temperature_decision"_s <= "ready"_s + SampleTemperatureTopK(&'dispatch SampleTemperatureTopKRuntime<'dispatch, 'event>),
        "temperature_scale"_s <= "temperature_decision"_s + completion<SampleTemperatureTopK>(&'dispatch SampleTemperatureTopKRuntime<'dispatch, 'event>) [guard_temperature_valid] / effect_scale_temperature,
        "errored"_s <= "temperature_decision"_s + completion<SampleTemperatureTopK>(&'dispatch SampleTemperatureTopKRuntime<'dispatch, 'event>) [guard_temperature_invalid] / effect_invalid_temperature,
        "temperature_softmax"_s <= "temperature_scale"_s + completion<SampleTemperatureTopK>(&'dispatch SampleTemperatureTopKRuntime<'dispatch, 'event>) / effect_softmax,
        "temperature_rank"_s <= "temperature_softmax"_s + completion<SampleTemperatureTopK>(&'dispatch SampleTemperatureTopKRuntime<'dispatch, 'event>) / effect_rank_top_k,
        "temperature_select"_s <= "temperature_rank"_s + completion<SampleTemperatureTopK>(&'dispatch SampleTemperatureTopKRuntime<'dispatch, 'event>) / effect_select_top_k,
        "done"_s <= "temperature_select"_s + completion<SampleTemperatureTopK>(&'dispatch SampleTemperatureTopKRuntime<'dispatch, 'event>) [guard_temperature_result_valid] / effect_publish_temperature,
        "errored"_s <= "temperature_select"_s + completion<SampleTemperatureTopK>(&'dispatch SampleTemperatureTopKRuntime<'dispatch, 'event>) [guard_temperature_result_invalid] / effect_invalid_temperature,

        "ready"_s <= "done"_s + completion<Configure>(&'dispatch ConfigureRuntime<'dispatch, 'event>),
        "ready"_s <= "errored"_s + completion<Configure>(&'dispatch ConfigureRuntime<'dispatch, 'event>),
        "ready"_s <= "done"_s + completion<SampleLogits>(&'dispatch SampleLogitsRuntime<'dispatch, 'event>),
        "ready"_s <= "errored"_s + completion<SampleLogits>(&'dispatch SampleLogitsRuntime<'dispatch, 'event>),
        "ready"_s <= "done"_s + completion<SamplePreselected>(&'dispatch SamplePreselectedRuntime<'dispatch, 'event>),
        "ready"_s <= "errored"_s + completion<SamplePreselected>(&'dispatch SamplePreselectedRuntime<'dispatch, 'event>),
        "ready"_s <= "done"_s + completion<SampleTemperatureTopK>(&'dispatch SampleTemperatureTopKRuntime<'dispatch, 'event>),
        "ready"_s <= "errored"_s + completion<SampleTemperatureTopK>(&'dispatch SampleTemperatureTopKRuntime<'dispatch, 'event>),

        "ready"_s <= "ready"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "configure_decision"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "logits_decision"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "candidate_prepare"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "sampler_decision"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "sampler_call"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "sampler_result"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "complete_decision"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "preselected_decision"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "temperature_decision"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "temperature_scale"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "temperature_softmax"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "temperature_rank"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "temperature_select"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "done"_s + unexpected_event<_> / effect_unexpected,
        "ready"_s <= "errored"_s + unexpected_event<_> / effect_unexpected,
    }
}

struct Context {
    sampler_fns: [Option<SamplerFn>; MAX_SAMPLERS],
    sampler_count: usize,
}

/// Single-writer, synchronous logits sampler actor.
pub struct Sampler {
    machine: LogitsSamplerStateMachine<Context>,
}

impl fmt::Debug for Sampler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Sampler").finish_non_exhaustive()
    }
}

impl Default for Sampler {
    fn default() -> Self {
        Self::new()
    }
}

impl Sampler {
    /// Constructs an unconfigured sampler.
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: LogitsSamplerStateMachine::new(Context {
                sampler_fns: [None; MAX_SAMPLERS],
                sampler_count: 0,
            }),
        }
    }

    /// Returns whether the actor is at its ready boundary.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&LogitsSamplerStates::Ready)
    }

    /// Configures the callback chain synchronously from caller-owned storage.
    pub fn configure<'event>(&mut self, request: Configure<'event>) -> Result<(), SamplerError> {
        let result = Cell::new(Err(SamplerError::Internal));
        let runtime = ConfigureRuntime {
            request,
            result: &result,
        };
        self.machine
            .process_event(LogitsSamplerEvents::Configure(&runtime))
            .map_err(|_| SamplerError::Internal)?;
        result.get()
    }

    /// Runs the configured callback chain over caller-owned storage.
    pub fn sample_logits<'event>(
        &mut self,
        request: SampleLogits<'event>,
    ) -> Result<i32, SamplerError> {
        let candidate_count = Cell::new(0);
        let sampler_index = Cell::new(0);
        let callback_error = Cell::new(None);
        let result = Cell::new(Err(SamplerError::Internal));
        let runtime = SampleLogitsRuntime {
            request,
            candidate_count: &candidate_count,
            sampler_index: &sampler_index,
            callback_error: &callback_error,
            result: &result,
        };
        self.machine
            .process_event(LogitsSamplerEvents::SampleLogits(&runtime))
            .map_err(|_| SamplerError::Internal)?;
        result.get()
    }

    /// Validates and returns a preselected token.
    pub fn sample_preselected(&mut self, request: SamplePreselected) -> Result<i32, SamplerError> {
        let result = Cell::new(Err(SamplerError::Internal));
        let runtime = SamplePreselectedRuntime {
            request,
            result: &result,
            _event: core::marker::PhantomData,
        };
        self.machine
            .process_event(LogitsSamplerEvents::SamplePreselected(&runtime))
            .map_err(|_| SamplerError::Internal)?;
        result.get()
    }

    /// Performs temperature scaling, softmax, top-k ranking, and selection.
    pub fn sample_temperature_top_k<'event>(
        &mut self,
        request: SampleTemperatureTopK<'event>,
    ) -> Result<(i32, f32), SamplerError> {
        let result = Cell::new(Err(SamplerError::Internal));
        let runtime = SampleTemperatureTopKRuntime {
            request,
            result: &result,
        };
        self.machine
            .process_event(LogitsSamplerEvents::SampleTemperatureTopK(&runtime))
            .map_err(|_| SamplerError::Internal)?;
        result.get()
    }
}

impl LogitsSamplerStateMachineContext for Context {
    fn effect_begin_configure<'dispatch, 'event>(
        &mut self,
        _event: &'dispatch ConfigureRuntime<'dispatch, 'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        Ok(())
    }

    fn guard_config_valid<'dispatch, 'event>(
        &self,
        event: &'dispatch ConfigureRuntime<'dispatch, 'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(
            !event.request.sampler_fns.is_empty()
                && event.request.sampler_fns.len() <= MAX_SAMPLERS,
        )
    }

    fn guard_config_invalid<'dispatch, 'event>(
        &self,
        event: &'dispatch ConfigureRuntime<'dispatch, 'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!self.guard_config_valid(event)?)
    }

    fn effect_configure<'dispatch, 'event>(
        &mut self,
        event: &'dispatch ConfigureRuntime<'dispatch, 'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.sampler_fns.fill(None);
        for (slot, sampler) in event.request.sampler_fns.iter().copied().enumerate() {
            self.sampler_fns[slot] = Some(sampler);
        }
        self.sampler_count = event.request.sampler_fns.len();
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_invalid_configure<'dispatch, 'event>(
        &mut self,
        event: &'dispatch ConfigureRuntime<'dispatch, 'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        event.result.set(Err(SamplerError::InvalidRequest));
        Ok(())
    }

    fn guard_logits_valid<'dispatch, 'event>(
        &self,
        event: &'dispatch SampleLogitsRuntime<'dispatch, 'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        let request = &event.request;
        Ok(self.sampler_count > 0
            && !request.logits.is_empty()
            && request.candidate_capacity >= request.logits.len()
            && request.candidate_capacity <= request.candidate_ids.borrow().len()
            && request.candidate_capacity <= request.candidate_scores.borrow().len())
    }

    fn guard_logits_invalid<'dispatch, 'event>(
        &self,
        event: &'dispatch SampleLogitsRuntime<'dispatch, 'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!self.guard_logits_valid(event)?)
    }

    fn effect_begin_logits<'dispatch, 'event>(
        &mut self,
        event: &'dispatch SampleLogitsRuntime<'dispatch, 'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        event.candidate_count.set(event.request.logits.len());
        event.sampler_index.set(0);
        event.callback_error.set(None);
        **event.request.selected_token_out.borrow_mut() = -1;
        Ok(())
    }

    fn effect_prepare_candidates<'dispatch, 'event>(
        &mut self,
        event: &'dispatch SampleLogitsRuntime<'dispatch, 'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let mut ids = event.request.candidate_ids.borrow_mut();
        let mut scores = event.request.candidate_scores.borrow_mut();
        for (index, score) in event.request.logits.iter().copied().enumerate() {
            ids[index] = index as i32;
            scores[index] = score;
        }
        Ok(())
    }

    fn guard_sampler_available<'dispatch, 'event>(
        &self,
        event: &'dispatch SampleLogitsRuntime<'dispatch, 'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.sampler_index.get() < self.sampler_count
            && self.sampler_fns[event.sampler_index.get()].is_some())
    }

    fn guard_sampler_missing<'dispatch, 'event>(
        &self,
        event: &'dispatch SampleLogitsRuntime<'dispatch, 'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.sampler_index.get() < self.sampler_count
            && self.sampler_fns[event.sampler_index.get()].is_none())
    }

    fn effect_apply_sampler<'dispatch, 'event>(
        &mut self,
        event: &'dispatch SampleLogitsRuntime<'dispatch, 'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let sampler = self.sampler_fns[event.sampler_index.get()].ok_or(())?;
        let mut ids = event.request.candidate_ids.borrow_mut();
        let mut scores = event.request.candidate_scores.borrow_mut();
        let mut count = event.candidate_count.get();
        let mut selected = **event.request.selected_token_out.borrow();
        event.callback_error.set(
            sampler(
                &mut ids[..event.request.logits.len()],
                &mut scores[..event.request.logits.len()],
                &mut count,
                &mut selected,
            )
            .err(),
        );
        **event.request.selected_token_out.borrow_mut() = selected;
        Ok(())
    }

    fn guard_callback_valid<'dispatch, 'event>(
        &self,
        event: &'dispatch SampleLogitsRuntime<'dispatch, 'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.callback_error.get().is_none()
            && event.candidate_count.get() > 0
            && event.candidate_count.get() <= event.request.logits.len())
    }

    fn guard_callback_invalid<'dispatch, 'event>(
        &self,
        event: &'dispatch SampleLogitsRuntime<'dispatch, 'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.callback_error.get().is_none()
            && (event.candidate_count.get() == 0
                || event.candidate_count.get() > event.request.logits.len()))
    }

    fn guard_callback_failed<'dispatch, 'event>(
        &self,
        event: &'dispatch SampleLogitsRuntime<'dispatch, 'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.callback_error.get().is_some())
    }

    fn effect_advance_sampler<'dispatch, 'event>(
        &mut self,
        event: &'dispatch SampleLogitsRuntime<'dispatch, 'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        event.sampler_index.set(event.sampler_index.get() + 1);
        Ok(())
    }

    fn guard_chain_complete<'dispatch, 'event>(
        &self,
        event: &'dispatch SampleLogitsRuntime<'dispatch, 'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.sampler_index.get() >= self.sampler_count)
    }

    fn guard_selected_valid<'dispatch, 'event>(
        &self,
        event: &'dispatch SampleLogitsRuntime<'dispatch, 'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        let selected = **event.request.selected_token_out.borrow();
        Ok(selected >= 0 && (selected as usize) < event.request.logits.len())
    }

    fn guard_selected_invalid<'dispatch, 'event>(
        &self,
        event: &'dispatch SampleLogitsRuntime<'dispatch, 'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!self.guard_selected_valid(event)?)
    }

    fn effect_publish_logits<'dispatch, 'event>(
        &mut self,
        event: &'dispatch SampleLogitsRuntime<'dispatch, 'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        event
            .result
            .set(Ok(**event.request.selected_token_out.borrow()));
        Ok(())
    }

    fn effect_invalid_logits<'dispatch, 'event>(
        &mut self,
        event: &'dispatch SampleLogitsRuntime<'dispatch, 'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        event.result.set(Err(SamplerError::InvalidRequest));
        Ok(())
    }

    fn effect_missing_sampler<'dispatch, 'event>(
        &mut self,
        event: &'dispatch SampleLogitsRuntime<'dispatch, 'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        event.result.set(Err(SamplerError::MissingSampler));
        Ok(())
    }

    fn effect_callback_error<'dispatch, 'event>(
        &mut self,
        event: &'dispatch SampleLogitsRuntime<'dispatch, 'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        event.result.set(Err(event
            .callback_error
            .get()
            .map(SamplerError::Callback)
            .unwrap_or(SamplerError::Internal)));
        Ok(())
    }

    fn guard_preselected_valid<'dispatch, 'event>(
        &self,
        event: &'dispatch SamplePreselectedRuntime<'dispatch, 'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.request.vocab_size > 0
            && event.request.selected_token >= 0
            && (event.request.selected_token as usize) < event.request.vocab_size)
    }

    fn guard_preselected_invalid<'dispatch, 'event>(
        &self,
        event: &'dispatch SamplePreselectedRuntime<'dispatch, 'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!self.guard_preselected_valid(event)?)
    }

    fn effect_publish_preselected<'dispatch, 'event>(
        &mut self,
        event: &'dispatch SamplePreselectedRuntime<'dispatch, 'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        event.result.set(Ok(event.request.selected_token));
        Ok(())
    }

    fn effect_invalid_preselected<'dispatch, 'event>(
        &mut self,
        event: &'dispatch SamplePreselectedRuntime<'dispatch, 'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        event.result.set(Err(SamplerError::InvalidRequest));
        Ok(())
    }

    fn guard_temperature_valid<'dispatch, 'event>(
        &self,
        event: &'dispatch SampleTemperatureTopKRuntime<'dispatch, 'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        let request = &event.request;
        Ok(request.card > 0
            && request.top_k > 0
            && request.top_k <= request.card
            && request.temperature.is_finite()
            && request.temperature > 0.0
            && request.logits.borrow().len() >= request.card
            && request.sorted_indices.borrow().len() >= request.card
            && request.top_probabilities.borrow().len() >= request.top_k
            && request.top_indices.borrow().len() >= request.top_k
            && !(*request.random_state.borrow()).is_multiple_of(RANDOM_MODULUS))
    }

    fn guard_temperature_invalid<'dispatch, 'event>(
        &self,
        event: &'dispatch SampleTemperatureTopKRuntime<'dispatch, 'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!self.guard_temperature_valid(event)?)
    }

    fn effect_scale_temperature<'dispatch, 'event>(
        &mut self,
        event: &'dispatch SampleTemperatureTopKRuntime<'dispatch, 'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let scale = 1.0 / event.request.temperature;
        for value in &mut event.request.logits.borrow_mut()[..event.request.card] {
            *value *= scale;
        }
        Ok(())
    }

    fn effect_softmax<'dispatch, 'event>(
        &mut self,
        event: &'dispatch SampleTemperatureTopKRuntime<'dispatch, 'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let card = event.request.card;
        let mut values = event.request.logits.borrow_mut();
        let maximum = values[..card]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut total = 0.0;
        for value in &mut values[..card] {
            *value = (*value - maximum).exp();
            total += *value;
        }
        for value in &mut values[..card] {
            *value /= total;
        }
        Ok(())
    }

    fn effect_rank_top_k<'dispatch, 'event>(
        &mut self,
        event: &'dispatch SampleTemperatureTopKRuntime<'dispatch, 'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let card = event.request.card;
        let mut sorted = event.request.sorted_indices.borrow_mut();
        for (index, value) in sorted[..card].iter_mut().enumerate() {
            *value = index as i32;
        }
        let values = event.request.logits.borrow();
        sorted[..card].sort_unstable_by(|a, b| values[*b as usize].total_cmp(&values[*a as usize]));
        let mut probabilities = event.request.top_probabilities.borrow_mut();
        let mut indices = event.request.top_indices.borrow_mut();
        for slot in 0..event.request.top_k {
            let index = sorted[slot] as usize;
            probabilities[slot] = values[index];
            indices[slot] = index as i32;
        }
        Ok(())
    }

    fn effect_select_top_k<'dispatch, 'event>(
        &mut self,
        event: &'dispatch SampleTemperatureTopKRuntime<'dispatch, 'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let mut state = **event.request.random_state.borrow();
        let probabilities = event.request.top_probabilities.borrow();
        let indices = event.request.top_indices.borrow();
        let mut selected = -1;
        let mut score = f32::NEG_INFINITY;
        for slot in 0..event.request.top_k {
            state = ((state as u64 * RANDOM_MULTIPLIER) % RANDOM_MODULUS as u64) as u32;
            let uniform = state as f32 / RANDOM_MODULUS as f32;
            let race = probabilities[slot] / -uniform.ln();
            if race > score {
                score = race;
                selected = indices[slot];
            }
        }
        **event.request.random_state.borrow_mut() = state;
        **event.request.selected_token_out.borrow_mut() = selected;
        **event.request.selected_score_out.borrow_mut() = score;
        event.result.set(Ok((selected, score)));
        Ok(())
    }

    fn guard_temperature_result_valid<'dispatch, 'event>(
        &self,
        event: &'dispatch SampleTemperatureTopKRuntime<'dispatch, 'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(matches!(event.result.get(), Ok((token, score))
            if token >= 0 && (token as usize) < event.request.card && score.is_finite()))
    }

    fn guard_temperature_result_invalid<'dispatch, 'event>(
        &self,
        event: &'dispatch SampleTemperatureTopKRuntime<'dispatch, 'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!self.guard_temperature_result_valid(event)?)
    }

    fn effect_publish_temperature<'dispatch, 'event>(
        &mut self,
        _event: &'dispatch SampleTemperatureTopKRuntime<'dispatch, 'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        Ok(())
    }

    fn effect_invalid_temperature<'dispatch, 'event>(
        &mut self,
        event: &'dispatch SampleTemperatureTopKRuntime<'dispatch, 'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        event.result.set(Err(SamplerError::InvalidRequest));
        Ok(())
    }

    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn choose_first(
        _ids: &mut [i32],
        _scores: &mut [f32],
        count: &mut usize,
        selected: &mut i32,
    ) -> Result<(), SamplerCallbackError> {
        *count = (*count).min(2);
        *selected = 0;
        Ok(())
    }

    fn choose_last(
        ids: &mut [i32],
        _scores: &mut [f32],
        count: &mut usize,
        selected: &mut i32,
    ) -> Result<(), SamplerCallbackError> {
        *count = (*count).min(ids.len());
        *selected = ids[*count - 1];
        Ok(())
    }

    #[test]
    fn configured_chain_and_preselected_token() {
        let callbacks: [SamplerFn; 1] = [choose_first];
        let mut sampler = Sampler::new();
        assert_eq!(sampler.configure(Configure::new(&callbacks)), Ok(()));
        assert_eq!(
            sampler.sample_preselected(SamplePreselected::new(3, 2)),
            Ok(2)
        );
        assert_eq!(
            sampler.sample_preselected(SamplePreselected::new(3, 3)),
            Err(SamplerError::InvalidRequest)
        );
        assert!(sampler.is_ready());
    }

    #[test]
    fn configured_chain_selects_last_callback_result_with_caller_storage() {
        let callbacks: [SamplerFn; 2] = [choose_first, choose_last];
        let mut sampler = Sampler::new();
        sampler.configure(Configure::new(&callbacks)).unwrap();
        let logits = [0.5, 2.0, -1.0];
        let mut ids = [0; 3];
        let mut scores = [0.0; 3];
        let mut selected = -1;
        assert_eq!(
            sampler.sample_logits(SampleLogits::new(
                &logits,
                &mut ids,
                &mut scores,
                3,
                &mut selected
            )),
            Ok(2)
        );
        assert_eq!(selected, 2);
        assert_eq!(&ids[..2], &[0, 1]);
        assert!(sampler.is_ready());
    }
    #[test]
    fn invalid_candidate_capacity_fails_before_writes() {
        let callbacks: [SamplerFn; 1] = [choose_first];
        let mut sampler = Sampler::new();
        sampler.configure(Configure::new(&callbacks)).unwrap();
        let logits = [0.5, 2.0, -1.0];
        let mut ids = [9; 3];
        let mut scores = [8.0; 3];
        let mut selected = 7;
        assert_eq!(
            sampler.sample_logits(SampleLogits::new(
                &logits,
                &mut ids,
                &mut scores,
                2,
                &mut selected
            )),
            Err(SamplerError::InvalidRequest)
        );
        assert_eq!(ids, [9; 3]);
        assert!(
            scores
                .iter()
                .zip([8.0f32; 3])
                .all(|(actual, expected)| actual.to_bits() == expected.to_bits())
        );
        assert_eq!(selected, 7);
    }

    #[test]
    fn native_temperature_top_k_returns_finite_result() {
        let mut sampler = Sampler::new();
        let mut logits = [1.0, 3.0, 2.0];
        let mut sorted = [0; 3];
        let mut probabilities = [0.0; 2];
        let mut indices = [0; 2];
        let mut random = 7;
        let mut selected = -1;
        let mut score = f32::NEG_INFINITY;
        let result = sampler
            .sample_temperature_top_k(SampleTemperatureTopK::new(
                3,
                1.0,
                2,
                &mut logits,
                &mut sorted,
                &mut probabilities,
                &mut indices,
                &mut random,
                &mut selected,
                &mut score,
            ))
            .unwrap();
        assert_eq!(result.0, selected);
        assert_eq!(result.1.to_bits(), score.to_bits());
        assert!(result.0 >= 0 && result.0 < 3);
        assert!(result.1.is_finite());
        assert!(sampler.is_ready());
    }

    #[test]
    fn invalid_temperature_capacity_fails_before_writes() {
        let mut sampler = Sampler::new();
        let mut logits = [1.0, 3.0, 2.0];
        let mut sorted = [0; 2];
        let mut probabilities = [0.0; 2];
        let mut indices = [0; 2];
        let mut random = 7;
        let mut selected = 11;
        let mut score = 12.0;
        assert_eq!(
            sampler.sample_temperature_top_k(SampleTemperatureTopK::new(
                3,
                1.0,
                2,
                &mut logits,
                &mut sorted,
                &mut probabilities,
                &mut indices,
                &mut random,
                &mut selected,
                &mut score
            )),
            Err(SamplerError::InvalidRequest)
        );
        assert!(
            logits
                .iter()
                .zip([1.0f32, 3.0, 2.0])
                .all(|(actual, expected)| actual.to_bits() == expected.to_bits())
        );
        assert_eq!(random, 7);
        assert_eq!(selected, 11);
        assert_eq!(score.to_bits(), 12.0f32.to_bits());
    }
}
