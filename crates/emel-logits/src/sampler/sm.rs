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
use std::rc::Rc;

use sml::sml;

const RANDOM_MODULUS: u32 = 2_147_483_647;
const RANDOM_MULTIPLIER: u64 = 16_807;

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
    /// Native sampling produced no representable result.
    InvalidResult,
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
            Self::InvalidResult => f.write_str("invalid logits sampler result"),
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
#[derive(Clone, Debug)]
pub struct Configure {
    /// Callback chain, copied into actor-owned storage before dispatch.
    pub sampler_fns: Vec<SamplerFn>,
}

impl Configure {
    /// Creates a configuration request.
    #[must_use]
    pub fn new(sampler_fns: &[SamplerFn]) -> Self {
        Self {
            sampler_fns: sampler_fns.to_vec(),
        }
    }
}

/// Immutable logits request.
#[derive(Clone, Debug)]
pub struct SampleLogits {
    /// Unnormalized vocabulary logits.
    pub logits: Vec<f32>,
}

impl SampleLogits {
    /// Creates a logits request, copying input before dispatch.
    #[must_use]
    pub fn new(logits: &[f32]) -> Self {
        Self {
            logits: logits.to_vec(),
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

/// Native temperature/top-k parameters.
#[derive(Clone, Copy, Debug)]
pub struct SampleTemperatureTopK {
    /// Number of logits to process.
    pub card: usize,
    /// Positive temperature divisor.
    pub temperature: f32,
    /// Number of ranked candidates.
    pub top_k: usize,
}

impl SampleTemperatureTopK {
    /// Creates a native sampling request.
    #[must_use]
    pub const fn new(card: usize, temperature: f32, top_k: usize) -> Self {
        Self {
            card,
            temperature,
            top_k,
        }
    }
}

type Shared<T> = Rc<RefCell<T>>;

#[derive(Clone)]
struct ConfigureRuntime<'dispatch> {
    request: Configure,
    result: &'dispatch Cell<Result<(), SamplerError>>,
}

#[derive(Clone)]
struct SampleLogitsRuntime<'dispatch> {
    request: SampleLogits,
    sampler_fns: Shared<Vec<SamplerFn>>,
    candidate_ids: Shared<Vec<i32>>,
    candidate_scores: Shared<Vec<f32>>,
    candidate_count: &'dispatch Cell<usize>,
    sampler_index: &'dispatch Cell<usize>,
    callback_error: &'dispatch Cell<Option<SamplerCallbackError>>,
    selected_token: &'dispatch Cell<i32>,
    result: &'dispatch Cell<Result<i32, SamplerError>>,
}

#[derive(Clone)]
struct SamplePreselectedRuntime<'dispatch> {
    request: SamplePreselected,
    result: &'dispatch Cell<Result<i32, SamplerError>>,
}

#[derive(Clone)]
struct SampleTemperatureTopKRuntime<'dispatch> {
    request: SampleTemperatureTopK,
    logits: Shared<Vec<f32>>,
    sorted_indices: Shared<Vec<i32>>,
    top_probabilities: Shared<Vec<f32>>,
    top_indices: Shared<Vec<i32>>,
    random_state: &'dispatch Cell<u32>,
    result: &'dispatch Cell<Result<(i32, f32), SamplerError>>,
}

sml! {
    LogitsSampler<'dispatch> {
        "configure_decision"_s <= *"ready"_s + Configure(ConfigureRuntime<'dispatch>) / effect_begin_configure,
        "done"_s <= "configure_decision"_s + completion<Configure>(ConfigureRuntime<'dispatch>) [guard_config_valid] / effect_configure,
        "errored"_s <= "configure_decision"_s + completion<Configure>(ConfigureRuntime<'dispatch>) [guard_config_invalid] / effect_invalid_configure,

        "logits_decision"_s <= "ready"_s + SampleLogits(SampleLogitsRuntime<'dispatch>),
        "candidate_prepare"_s <= "logits_decision"_s + completion<SampleLogits>(SampleLogitsRuntime<'dispatch>) [guard_logits_valid] / effect_begin_logits,
        "errored"_s <= "logits_decision"_s + completion<SampleLogits>(SampleLogitsRuntime<'dispatch>) [guard_logits_invalid] / effect_invalid_logits,
        "sampler_decision"_s <= "candidate_prepare"_s + completion<SampleLogits>(SampleLogitsRuntime<'dispatch>) / effect_prepare_candidates,
        "sampler_call"_s <= "sampler_decision"_s + completion<SampleLogits>(SampleLogitsRuntime<'dispatch>) [guard_sampler_available] / effect_apply_sampler,
        "errored"_s <= "sampler_decision"_s + completion<SampleLogits>(SampleLogitsRuntime<'dispatch>) [guard_sampler_missing] / effect_missing_sampler,
        "sampler_result"_s <= "sampler_call"_s + completion<SampleLogits>(SampleLogitsRuntime<'dispatch>),
        "sampler_decision"_s <= "sampler_result"_s + completion<SampleLogits>(SampleLogitsRuntime<'dispatch>) [guard_callback_valid] / effect_advance_sampler,
        "errored"_s <= "sampler_result"_s + completion<SampleLogits>(SampleLogitsRuntime<'dispatch>) [guard_callback_invalid] / effect_invalid_logits,
        "errored"_s <= "sampler_result"_s + completion<SampleLogits>(SampleLogitsRuntime<'dispatch>) [guard_callback_failed] / effect_callback_error,
        "complete_decision"_s <= "sampler_decision"_s + completion<SampleLogits>(SampleLogitsRuntime<'dispatch>) [guard_chain_complete],
        "done"_s <= "complete_decision"_s + completion<SampleLogits>(SampleLogitsRuntime<'dispatch>) [guard_selected_valid] / effect_publish_logits,
        "errored"_s <= "complete_decision"_s + completion<SampleLogits>(SampleLogitsRuntime<'dispatch>) [guard_selected_invalid] / effect_invalid_logits,

        "preselected_decision"_s <= "ready"_s + SamplePreselected(SamplePreselectedRuntime<'dispatch>),
        "done"_s <= "preselected_decision"_s + completion<SamplePreselected>(SamplePreselectedRuntime<'dispatch>) [guard_preselected_valid] / effect_publish_preselected,
        "errored"_s <= "preselected_decision"_s + completion<SamplePreselected>(SamplePreselectedRuntime<'dispatch>) [guard_preselected_invalid] / effect_invalid_preselected,

        "temperature_decision"_s <= "ready"_s + SampleTemperatureTopK(SampleTemperatureTopKRuntime<'dispatch>),
        "temperature_scale"_s <= "temperature_decision"_s + completion<SampleTemperatureTopK>(SampleTemperatureTopKRuntime<'dispatch>) [guard_temperature_valid] / effect_scale_temperature,
        "errored"_s <= "temperature_decision"_s + completion<SampleTemperatureTopK>(SampleTemperatureTopKRuntime<'dispatch>) [guard_temperature_invalid] / effect_invalid_temperature,
        "temperature_softmax"_s <= "temperature_scale"_s + completion<SampleTemperatureTopK>(SampleTemperatureTopKRuntime<'dispatch>) / effect_softmax,
        "temperature_rank"_s <= "temperature_softmax"_s + completion<SampleTemperatureTopK>(SampleTemperatureTopKRuntime<'dispatch>) / effect_rank_top_k,
        "temperature_select"_s <= "temperature_rank"_s + completion<SampleTemperatureTopK>(SampleTemperatureTopKRuntime<'dispatch>) / effect_select_top_k,
        "done"_s <= "temperature_select"_s + completion<SampleTemperatureTopK>(SampleTemperatureTopKRuntime<'dispatch>) [guard_temperature_result_valid] / effect_publish_temperature,
        "errored"_s <= "temperature_select"_s + completion<SampleTemperatureTopK>(SampleTemperatureTopKRuntime<'dispatch>) [guard_temperature_result_invalid] / effect_invalid_temperature,

        "ready"_s <= "done"_s + completion<Configure>(ConfigureRuntime<'dispatch>),
        "ready"_s <= "errored"_s + completion<Configure>(ConfigureRuntime<'dispatch>),
        "ready"_s <= "done"_s + completion<SampleLogits>(SampleLogitsRuntime<'dispatch>),
        "ready"_s <= "errored"_s + completion<SampleLogits>(SampleLogitsRuntime<'dispatch>),
        "ready"_s <= "done"_s + completion<SamplePreselected>(SamplePreselectedRuntime<'dispatch>),
        "ready"_s <= "errored"_s + completion<SamplePreselected>(SamplePreselectedRuntime<'dispatch>),
        "ready"_s <= "done"_s + completion<SampleTemperatureTopK>(SampleTemperatureTopKRuntime<'dispatch>),
        "ready"_s <= "errored"_s + completion<SampleTemperatureTopK>(SampleTemperatureTopKRuntime<'dispatch>),

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
    sampler_fns: Shared<Vec<SamplerFn>>,
}

/// Single-writer, synchronous logits sampler actor.
pub struct Sampler {
    machine: LogitsSamplerStateMachine<Context>,
    sampler_fns: Shared<Vec<SamplerFn>>,
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
        let sampler_fns = Rc::new(RefCell::new(Vec::new()));
        Self {
            machine: LogitsSamplerStateMachine::new(Context {
                sampler_fns: Rc::clone(&sampler_fns),
            }),
            sampler_fns,
        }
    }

    /// Returns whether the actor is at its ready boundary.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.machine.is(&LogitsSamplerStates::Ready)
    }

    /// Configures the callback chain synchronously.
    pub fn configure(&mut self, request: Configure) -> Result<(), SamplerError> {
        let result = Cell::new(Err(SamplerError::Internal));
        *self.sampler_fns.borrow_mut() = request.sampler_fns.clone();
        self.machine
            .process_event(LogitsSamplerEvents::Configure(ConfigureRuntime {
                request,
                result: &result,
            }))
            .map_err(|_| SamplerError::Internal)?;
        result.get()
    }

    /// Runs the configured callback chain synchronously.
    pub fn sample_logits(&mut self, request: SampleLogits) -> Result<i32, SamplerError> {
        let result = Cell::new(Err(SamplerError::Internal));
        let count = Cell::new(0);
        let index = Cell::new(0);
        let error = Cell::new(None);
        let selected = Cell::new(-1);
        let ids = Rc::new(RefCell::new(vec![0; request.logits.len()]));
        let scores = Rc::new(RefCell::new(vec![0.0; request.logits.len()]));
        self.machine
            .process_event(LogitsSamplerEvents::SampleLogits(SampleLogitsRuntime {
                request,
                sampler_fns: Rc::clone(&self.sampler_fns),
                candidate_ids: ids,
                candidate_scores: scores,
                candidate_count: &count,
                sampler_index: &index,
                callback_error: &error,
                selected_token: &selected,
                result: &result,
            }))
            .map_err(|_| SamplerError::Internal)?;
        result.get()
    }

    /// Validates and returns a preselected token.
    pub fn sample_preselected(&mut self, request: SamplePreselected) -> Result<i32, SamplerError> {
        let result = Cell::new(Err(SamplerError::Internal));
        self.machine
            .process_event(LogitsSamplerEvents::SamplePreselected(
                SamplePreselectedRuntime {
                    request,
                    result: &result,
                },
            ))
            .map_err(|_| SamplerError::Internal)?;
        result.get()
    }

    /// Performs temperature scaling, softmax, top-k ranking, and selection.
    pub fn sample_temperature_top_k(
        &mut self,
        request: SampleTemperatureTopK,
        logits: &[f32],
        random_state: &mut u32,
    ) -> Result<(i32, f32), SamplerError> {
        let result = Cell::new(Err(SamplerError::Internal));
        let values = Rc::new(RefCell::new(logits.to_vec()));
        let sorted = Rc::new(RefCell::new(vec![0; request.card]));
        let probabilities = Rc::new(RefCell::new(vec![0.0; request.top_k]));
        let indices = Rc::new(RefCell::new(vec![0; request.top_k]));
        let random = Cell::new(*random_state);
        self.machine
            .process_event(LogitsSamplerEvents::SampleTemperatureTopK(
                SampleTemperatureTopKRuntime {
                    request,
                    logits: values,
                    sorted_indices: sorted,
                    top_probabilities: probabilities,
                    top_indices: indices,
                    random_state: &random,
                    result: &result,
                },
            ))
            .map_err(|_| SamplerError::Internal)?;
        *random_state = random.get();
        result.get()
    }
}

impl LogitsSamplerStateMachineContext for Context {
    fn effect_begin_configure<'dispatch>(
        &mut self,
        event: ConfigureRuntime<'dispatch>,
    ) -> Result<(), ()> {
        *self.sampler_fns.borrow_mut() = event.request.sampler_fns;
        Ok(())
    }
    fn guard_config_valid<'dispatch>(
        &self,
        event: &ConfigureRuntime<'dispatch>,
    ) -> Result<bool, ()> {
        Ok(!event.request.sampler_fns.is_empty())
    }
    fn guard_config_invalid<'dispatch>(
        &self,
        event: &ConfigureRuntime<'dispatch>,
    ) -> Result<bool, ()> {
        Ok(event.request.sampler_fns.is_empty())
    }
    fn effect_configure<'dispatch>(
        &mut self,
        event: ConfigureRuntime<'dispatch>,
    ) -> Result<(), ()> {
        event.result.set(Ok(()));
        Ok(())
    }
    fn effect_invalid_configure<'dispatch>(
        &mut self,
        event: ConfigureRuntime<'dispatch>,
    ) -> Result<(), ()> {
        event.result.set(Err(SamplerError::InvalidRequest));
        Ok(())
    }

    fn guard_logits_valid<'dispatch>(
        &self,
        event: &SampleLogitsRuntime<'dispatch>,
    ) -> Result<bool, ()> {
        Ok(!self.sampler_fns.borrow().is_empty() && !event.request.logits.is_empty())
    }
    fn guard_logits_invalid<'dispatch>(
        &self,
        event: &SampleLogitsRuntime<'dispatch>,
    ) -> Result<bool, ()> {
        Ok(self.sampler_fns.borrow().is_empty() || event.request.logits.is_empty())
    }
    fn effect_begin_logits<'dispatch>(
        &mut self,
        event: SampleLogitsRuntime<'dispatch>,
    ) -> Result<(), ()> {
        event.candidate_count.set(event.request.logits.len());
        event.sampler_index.set(0);
        event.callback_error.set(None);
        event.selected_token.set(-1);
        Ok(())
    }
    fn effect_prepare_candidates<'dispatch>(
        &mut self,
        event: SampleLogitsRuntime<'dispatch>,
    ) -> Result<(), ()> {
        let mut ids = event.candidate_ids.borrow_mut();
        let mut scores = event.candidate_scores.borrow_mut();
        for (index, score) in event.request.logits.iter().copied().enumerate() {
            ids[index] = index as i32;
            scores[index] = score;
        }
        Ok(())
    }
    fn guard_sampler_available<'dispatch>(
        &self,
        event: &SampleLogitsRuntime<'dispatch>,
    ) -> Result<bool, ()> {
        Ok(event.sampler_index.get() < self.sampler_fns.borrow().len())
    }
    fn guard_sampler_missing<'dispatch>(
        &self,
        _event: &SampleLogitsRuntime<'dispatch>,
    ) -> Result<bool, ()> {
        Ok(self.sampler_fns.borrow().is_empty())
    }
    fn effect_apply_sampler<'dispatch>(
        &mut self,
        event: SampleLogitsRuntime<'dispatch>,
    ) -> Result<(), ()> {
        let sampler = self.sampler_fns.borrow()[event.sampler_index.get()];
        let mut ids = event.candidate_ids.borrow_mut();
        let mut scores = event.candidate_scores.borrow_mut();
        let mut count = event.candidate_count.get();
        let mut selected = event.selected_token.get();
        event
            .callback_error
            .set(sampler(&mut ids, &mut scores, &mut count, &mut selected).err());
        event.candidate_count.set(count);
        event.selected_token.set(selected);
        Ok(())
    }
    fn guard_callback_valid<'dispatch>(
        &self,
        event: &SampleLogitsRuntime<'dispatch>,
    ) -> Result<bool, ()> {
        Ok(event.callback_error.get().is_none()
            && event.candidate_count.get() > 0
            && event.candidate_count.get() <= event.request.logits.len())
    }
    fn guard_callback_invalid<'dispatch>(
        &self,
        event: &SampleLogitsRuntime<'dispatch>,
    ) -> Result<bool, ()> {
        Ok(event.callback_error.get().is_none()
            && (event.candidate_count.get() == 0
                || event.candidate_count.get() > event.request.logits.len()))
    }
    fn guard_callback_failed<'dispatch>(
        &self,
        event: &SampleLogitsRuntime<'dispatch>,
    ) -> Result<bool, ()> {
        Ok(event.callback_error.get().is_some())
    }
    fn effect_advance_sampler<'dispatch>(
        &mut self,
        event: SampleLogitsRuntime<'dispatch>,
    ) -> Result<(), ()> {
        event.sampler_index.set(event.sampler_index.get() + 1);
        Ok(())
    }
    fn guard_chain_complete<'dispatch>(
        &self,
        event: &SampleLogitsRuntime<'dispatch>,
    ) -> Result<bool, ()> {
        Ok(event.sampler_index.get() >= self.sampler_fns.borrow().len())
    }
    fn guard_selected_valid<'dispatch>(
        &self,
        event: &SampleLogitsRuntime<'dispatch>,
    ) -> Result<bool, ()> {
        Ok(event.selected_token.get() >= 0
            && (event.selected_token.get() as usize) < event.request.logits.len())
    }
    fn guard_selected_invalid<'dispatch>(
        &self,
        event: &SampleLogitsRuntime<'dispatch>,
    ) -> Result<bool, ()> {
        Ok(!self.guard_selected_valid(event)?)
    }
    fn effect_publish_logits<'dispatch>(
        &mut self,
        event: SampleLogitsRuntime<'dispatch>,
    ) -> Result<(), ()> {
        event.result.set(Ok(event.selected_token.get()));
        Ok(())
    }
    fn effect_invalid_logits<'dispatch>(
        &mut self,
        event: SampleLogitsRuntime<'dispatch>,
    ) -> Result<(), ()> {
        event.result.set(Err(SamplerError::InvalidRequest));
        Ok(())
    }
    fn effect_missing_sampler<'dispatch>(
        &mut self,
        event: SampleLogitsRuntime<'dispatch>,
    ) -> Result<(), ()> {
        event.result.set(Err(SamplerError::MissingSampler));
        Ok(())
    }
    fn effect_callback_error<'dispatch>(
        &mut self,
        event: SampleLogitsRuntime<'dispatch>,
    ) -> Result<(), ()> {
        event.result.set(Err(event
            .callback_error
            .get()
            .map(SamplerError::Callback)
            .unwrap_or(SamplerError::Internal)));
        Ok(())
    }

    fn guard_preselected_valid<'dispatch>(
        &self,
        event: &SamplePreselectedRuntime<'dispatch>,
    ) -> Result<bool, ()> {
        Ok(event.request.vocab_size > 0
            && event.request.selected_token >= 0
            && (event.request.selected_token as usize) < event.request.vocab_size)
    }
    fn guard_preselected_invalid<'dispatch>(
        &self,
        event: &SamplePreselectedRuntime<'dispatch>,
    ) -> Result<bool, ()> {
        Ok(!self.guard_preselected_valid(event)?)
    }
    fn effect_publish_preselected<'dispatch>(
        &mut self,
        event: SamplePreselectedRuntime<'dispatch>,
    ) -> Result<(), ()> {
        event.result.set(Ok(event.request.selected_token));
        Ok(())
    }
    fn effect_invalid_preselected<'dispatch>(
        &mut self,
        event: SamplePreselectedRuntime<'dispatch>,
    ) -> Result<(), ()> {
        event.result.set(Err(SamplerError::InvalidRequest));
        Ok(())
    }

    fn guard_temperature_valid<'dispatch>(
        &self,
        event: &SampleTemperatureTopKRuntime<'dispatch>,
    ) -> Result<bool, ()> {
        let r = event.request;
        Ok(r.card > 0
            && r.top_k > 0
            && r.top_k <= r.card
            && r.temperature.is_finite()
            && r.temperature > 0.0
            && event.logits.borrow().len() >= r.card
            && event.random_state.get() != 0)
    }
    fn guard_temperature_invalid<'dispatch>(
        &self,
        event: &SampleTemperatureTopKRuntime<'dispatch>,
    ) -> Result<bool, ()> {
        Ok(!self.guard_temperature_valid(event)?)
    }
    fn effect_scale_temperature<'dispatch>(
        &mut self,
        event: SampleTemperatureTopKRuntime<'dispatch>,
    ) -> Result<(), ()> {
        let scale = 1.0 / event.request.temperature;
        for value in &mut event.logits.borrow_mut()[..event.request.card] {
            *value *= scale;
        }
        Ok(())
    }
    fn effect_softmax<'dispatch>(
        &mut self,
        event: SampleTemperatureTopKRuntime<'dispatch>,
    ) -> Result<(), ()> {
        let card = event.request.card;
        let mut values = event.logits.borrow_mut();
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
    fn effect_rank_top_k<'dispatch>(
        &mut self,
        event: SampleTemperatureTopKRuntime<'dispatch>,
    ) -> Result<(), ()> {
        let card = event.request.card;
        let mut sorted = event.sorted_indices.borrow_mut();
        for (index, value) in sorted[..card].iter_mut().enumerate() {
            *value = index as i32;
        }
        let values = event.logits.borrow();
        sorted[..card].sort_unstable_by(|a, b| values[*b as usize].total_cmp(&values[*a as usize]));
        let mut probabilities = event.top_probabilities.borrow_mut();
        let mut indices = event.top_indices.borrow_mut();
        for slot in 0..event.request.top_k {
            let index = sorted[slot] as usize;
            probabilities[slot] = values[index];
            indices[slot] = index as i32;
        }
        Ok(())
    }
    fn effect_select_top_k<'dispatch>(
        &mut self,
        event: SampleTemperatureTopKRuntime<'dispatch>,
    ) -> Result<(), ()> {
        let mut state = event.random_state.get();
        let probabilities = event.top_probabilities.borrow();
        let indices = event.top_indices.borrow();
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
        event.random_state.set(state);
        event.result.set(Ok((selected, score)));
        Ok(())
    }
    fn guard_temperature_result_valid<'dispatch>(
        &self,
        event: &SampleTemperatureTopKRuntime<'dispatch>,
    ) -> Result<bool, ()> {
        Ok(matches!(event.result.get(), Ok((token, score)) if token >= 0 && score.is_finite()))
    }
    fn guard_temperature_result_invalid<'dispatch>(
        &self,
        event: &SampleTemperatureTopKRuntime<'dispatch>,
    ) -> Result<bool, ()> {
        Ok(!self.guard_temperature_result_valid(event)?)
    }
    fn effect_publish_temperature<'dispatch>(
        &mut self,
        _event: SampleTemperatureTopKRuntime<'dispatch>,
    ) -> Result<(), ()> {
        Ok(())
    }
    fn effect_invalid_temperature<'dispatch>(
        &mut self,
        event: SampleTemperatureTopKRuntime<'dispatch>,
    ) -> Result<(), ()> {
        event.result.set(Err(SamplerError::InvalidResult));
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

    #[test]
    fn configured_chain_and_preselected_token() {
        let mut sampler = Sampler::new();
        assert_eq!(sampler.configure(Configure::new(&[choose_first])), Ok(()));
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
    fn configured_logits_chain_selects_token() {
        let mut sampler = Sampler::new();
        sampler.configure(Configure::new(&[choose_first])).unwrap();
        assert_eq!(
            sampler.sample_logits(SampleLogits::new(&[0.5, 2.0, -1.0])),
            Ok(0)
        );
        assert!(sampler.is_ready());
    }

    #[test]
    fn native_temperature_top_k_returns_finite_result() {
        let mut sampler = Sampler::new();
        let mut random = 7;
        let result = sampler
            .sample_temperature_top_k(
                SampleTemperatureTopK::new(3, 1.0, 2),
                &[1.0, 3.0, 2.0],
                &mut random,
            )
            .unwrap();
        assert!(result.0 >= 0 && result.0 < 3);
        assert!(result.1.is_finite());
        assert!(sampler.is_ready());
    }
}
