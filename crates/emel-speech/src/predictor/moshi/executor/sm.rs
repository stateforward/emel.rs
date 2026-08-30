//! Run-to-completion Moshi executor actor.
//!
//! The executor owns the control-plane state for one Moshi predictor.  Each
//! request is consumed by one SML dispatch; no operation reads another actor's
//! state and no dispatch allocates.  Graph and memory work is represented by
//! explicit completion decisions so a backend can be attached without changing
//! the lifecycle contract.

#![allow(
    clippy::enum_variant_names,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::module_name_repetitions,
    clippy::missing_const_for_fn,
    clippy::struct_excessive_bools,
    clippy::derive_partial_eq_without_eq,
    dead_code,
    missing_docs
)]

use core::fmt;

use sml::sml;

/// Errors published by the executor control plane.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ExecutorError {
    /// No error has been recorded.
    #[default]
    None,
    /// The executor has not completed initialization.
    NotInitialized,
    /// Model or tensor binding failed.
    BindFailed,
    /// A request has an invalid shape or phase.
    RequestShape,
    /// A step was submitted for another model.
    ModelMismatch,
    /// The selected graph operation is unavailable.
    GraphExecutionUnsupported,
    /// The event is not valid in the current state.
    UnexpectedEvent,
    /// Reset of a bound resource failed.
    ResetFailed,
}

impl fmt::Display for ExecutorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Self::None => "no error",
            Self::NotInitialized => "executor is not initialized",
            Self::BindFailed => "model binding failed",
            Self::RequestShape => "request shape is invalid",
            Self::ModelMismatch => "step model does not match the initialized model",
            Self::GraphExecutionUnsupported => "graph execution is unsupported",
            Self::UnexpectedEvent => "unexpected executor event",
            Self::ResetFailed => "reset failed",
        };
        formatter.write_str(text)
    }
}

/// Initialization request.  The booleans are the results of the independent
/// model-contract checks performed by the parent actor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InitRun {
    /// Whether model and required operands can be bound.
    pub contract_valid: bool,
    /// Whether the temporal projection layout is supported.
    pub temporal_projection_layout_supported: bool,
    /// Whether the depformer projection layout is supported.
    pub depformer_projection_layout_supported: bool,
    /// Sampling seed supplied by the caller; zero is replaced by one.
    pub sampling_seed: u32,
}

impl Default for InitRun {
    fn default() -> Self {
        Self {
            contract_valid: true,
            temporal_projection_layout_supported: true,
            depformer_projection_layout_supported: true,
            sampling_seed: 1,
        }
    }
}

/// One predictor graph-step request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StepRun {
    /// Whether the step targets the initialized model.
    pub model_matches: bool,
    /// Whether request dimensions and cache spans are valid.
    pub shape_valid: bool,
    /// Whether the selected input embedding operation is available.
    pub input_embedding_supported: bool,
    /// Result reported by the graph backend.
    pub execution_succeeded: bool,
}

impl Default for StepRun {
    fn default() -> Self {
        Self {
            model_matches: true,
            shape_valid: true,
            input_embedding_supported: true,
            execution_succeeded: true,
        }
    }
}

/// Reset request and the two independently resettable cache resources.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResetRun {
    /// Whether temporal memory reset succeeded.
    pub temporal_succeeded: bool,
    /// Whether depformer memory reset succeeded.
    pub depformer_succeeded: bool,
}

impl Default for ResetRun {
    fn default() -> Self {
        Self {
            temporal_succeeded: true,
            depformer_succeeded: true,
        }
    }
}

sml! {
    SpeechPredictorMoshiExecutor {
        "state_bind_contract_decision"_s <= *"state_uninitialized"_s + event<InitRun>,
        "state_temporal_projection_layout_decision"_s <= "state_bind_contract_decision"_s + completion<InitRun> [guard_bind_contract_valid] / effect_bind_contract,
        "state_init_failed_error_out_decision"_s <= "state_bind_contract_decision"_s + completion<InitRun> [guard_bind_contract_invalid] / effect_mark_bind_failed,
        "state_depformer_projection_layout_decision"_s <= "state_temporal_projection_layout_decision"_s + completion<InitRun> [guard_temporal_projection_layout_supported] / effect_bind_temporal_projection_layout,
        "state_init_failed_error_out_decision"_s <= "state_temporal_projection_layout_decision"_s + completion<InitRun> [guard_temporal_projection_layout_unsupported] / effect_mark_bind_failed,
        "state_sampling_seed_decision"_s <= "state_depformer_projection_layout_decision"_s + completion<InitRun> [guard_depformer_projection_layout_supported] / effect_bind_depformer_projection_layout,
        "state_init_failed_error_out_decision"_s <= "state_depformer_projection_layout_decision"_s + completion<InitRun> [guard_depformer_projection_layout_unsupported] / effect_mark_bind_failed,
        "state_ready"_s <= "state_sampling_seed_decision"_s + completion<InitRun> / effect_complete_initialization,
        "state_uninitialized"_s <= "state_init_failed_error_out_decision"_s + completion<InitRun> / effect_complete_initialization_error,

        "state_step_model_decision"_s <= "state_ready"_s + event<StepRun>,
        "state_step_shape_decision"_s <= "state_step_model_decision"_s + completion<StepRun> [guard_step_model_matches],
        "state_step_error_out_decision"_s <= "state_step_model_decision"_s + completion<StepRun> [guard_step_model_mismatch] / effect_mark_model_mismatch,
        "state_step_execution_decision"_s <= "state_step_shape_decision"_s + completion<StepRun> [guard_step_shape_valid],
        "state_step_error_out_decision"_s <= "state_step_shape_decision"_s + completion<StepRun> [guard_step_shape_invalid] / effect_mark_request_shape,
        "state_input_embedding_decision"_s <= "state_step_execution_decision"_s + completion<StepRun> [guard_input_embedding_supported],
        "state_step_error_out_decision"_s <= "state_step_execution_decision"_s + completion<StepRun> [guard_input_embedding_unsupported] / effect_mark_graph_execution_unsupported,
        "state_step_result_decision"_s <= "state_input_embedding_decision"_s + completion<StepRun> [guard_execution_succeeded] / effect_publish_step,
        "state_step_error_out_decision"_s <= "state_input_embedding_decision"_s + completion<StepRun> [guard_execution_failed] / effect_mark_graph_execution_unsupported,
        "state_ready"_s <= "state_step_result_decision"_s + completion<StepRun>,
        "state_ready"_s <= "state_step_error_out_decision"_s + completion<StepRun> / effect_finish_step_error,

        "state_reset_decision"_s <= "state_ready"_s + event<ResetRun>,
        "state_reset_temporal_decision"_s <= "state_reset_decision"_s + completion<ResetRun> [guard_temporal_reset_present] / effect_reset_temporal,
        "state_reset_depformer_decision"_s <= "state_reset_decision"_s + completion<ResetRun> [guard_temporal_reset_missing],
        "state_reset_depformer_result_decision"_s <= "state_reset_temporal_decision"_s + completion<ResetRun> [guard_temporal_reset_succeeded],
        "state_reset_failed"_s <= "state_reset_temporal_decision"_s + completion<ResetRun> [guard_temporal_reset_failed] / effect_mark_reset_failed,
        "state_uninitialized"_s <= "state_reset_depformer_decision"_s + completion<ResetRun> [guard_depformer_reset_succeeded] / effect_reset_session,
        "state_reset_failed"_s <= "state_reset_depformer_decision"_s + completion<ResetRun> [guard_depformer_reset_failed] / effect_mark_reset_failed,
        "state_uninitialized"_s <= "state_reset_depformer_result_decision"_s + completion<ResetRun> / effect_reset_session,
        "state_uninitialized"_s <= "state_reset_failed"_s + completion<ResetRun> / effect_reset_session,

        "state_uninitialized"_s <= "state_uninitialized"_s + unexpected_event<_> / effect_mark_unexpected,
        "state_ready"_s <= "state_ready"_s + unexpected_event<_> / effect_mark_unexpected,
    }
}

/// Mutable, single-writer state owned by the executor actor.
#[derive(Debug, Default)]
pub struct SpeechPredictorMoshiExecutorContext {
    initialized: bool,
    sampling_seed: u32,
    last_error: ExecutorError,
    completed_steps: u32,
    reset_count: u32,
    reset_failed: bool,
}

impl SpeechPredictorMoshiExecutorContext {
    /// Returns the most recently published control-plane error.
    #[must_use]
    pub const fn last_error(&self) -> ExecutorError {
        self.last_error
    }

    /// Returns whether initialization completed successfully.
    #[must_use]
    pub const fn initialized(&self) -> bool {
        self.initialized
    }

    /// Returns the number of successful graph steps.
    #[must_use]
    pub const fn completed_steps(&self) -> u32 {
        self.completed_steps
    }

    /// Returns the normalized nonzero sampling seed.
    #[must_use]
    pub const fn sampling_seed(&self) -> u32 {
        self.sampling_seed
    }

    fn clear_error(&mut self) {
        self.last_error = ExecutorError::None;
    }

    fn mark_error(&mut self, error: ExecutorError) {
        self.last_error = error;
    }
}

impl SpeechPredictorMoshiExecutorStateMachineContext for SpeechPredictorMoshiExecutorContext {
    fn effect_bind_contract(&mut self, _event: &InitRun) -> Result<(), ()> {
        self.clear_error();
        Ok(())
    }

    fn effect_bind_temporal_projection_layout(&mut self, _event: &InitRun) -> Result<(), ()> {
        Ok(())
    }

    fn effect_bind_depformer_projection_layout(&mut self, event: &InitRun) -> Result<(), ()> {
        self.sampling_seed = event.sampling_seed.max(1);
        Ok(())
    }

    fn effect_mark_bind_failed(&mut self, _event: &InitRun) -> Result<(), ()> {
        self.initialized = false;
        self.mark_error(ExecutorError::BindFailed);
        Ok(())
    }

    fn effect_complete_initialization(&mut self, _event: &InitRun) -> Result<(), ()> {
        self.initialized = true;
        self.clear_error();
        Ok(())
    }

    fn effect_complete_initialization_error(&mut self, _event: &InitRun) -> Result<(), ()> {
        self.initialized = false;
        if self.last_error == ExecutorError::None {
            self.mark_error(ExecutorError::BindFailed);
        }
        Ok(())
    }

    fn effect_mark_model_mismatch(&mut self, _event: &StepRun) -> Result<(), ()> {
        self.mark_error(ExecutorError::ModelMismatch);
        Ok(())
    }

    fn effect_mark_request_shape(&mut self, _event: &StepRun) -> Result<(), ()> {
        self.mark_error(ExecutorError::RequestShape);
        Ok(())
    }

    fn effect_mark_graph_execution_unsupported(&mut self, _event: &StepRun) -> Result<(), ()> {
        self.mark_error(ExecutorError::GraphExecutionUnsupported);
        Ok(())
    }

    fn effect_publish_step(&mut self, _event: &StepRun) -> Result<(), ()> {
        self.completed_steps = self.completed_steps.saturating_add(1);
        self.clear_error();
        Ok(())
    }

    fn effect_finish_step_error(&mut self, _event: &StepRun) -> Result<(), ()> {
        Ok(())
    }

    fn effect_reset_temporal(&mut self, _event: &ResetRun) -> Result<(), ()> {
        Ok(())
    }

    fn effect_mark_reset_failed(&mut self, _event: &ResetRun) -> Result<(), ()> {
        self.reset_failed = true;
        self.mark_error(ExecutorError::ResetFailed);
        Ok(())
    }

    fn effect_reset_session(&mut self, event: &ResetRun) -> Result<(), ()> {
        self.initialized = false;
        self.reset_count = self.reset_count.saturating_add(1);
        if event.temporal_succeeded && event.depformer_succeeded && !self.reset_failed {
            self.clear_error();
        } else {
            self.mark_error(ExecutorError::ResetFailed);
        }
        self.reset_failed = false;
        Ok(())
    }

    fn effect_mark_unexpected(&mut self) -> Result<(), ()> {
        self.mark_error(ExecutorError::UnexpectedEvent);
        Ok(())
    }

    fn guard_bind_contract_valid(&self, event: &InitRun) -> Result<bool, ()> {
        Ok(event.contract_valid)
    }

    fn guard_bind_contract_invalid(&self, event: &InitRun) -> Result<bool, ()> {
        Ok(!event.contract_valid)
    }

    fn guard_temporal_projection_layout_supported(&self, event: &InitRun) -> Result<bool, ()> {
        Ok(event.temporal_projection_layout_supported)
    }

    fn guard_temporal_projection_layout_unsupported(&self, event: &InitRun) -> Result<bool, ()> {
        Ok(!event.temporal_projection_layout_supported)
    }

    fn guard_depformer_projection_layout_supported(&self, event: &InitRun) -> Result<bool, ()> {
        Ok(event.depformer_projection_layout_supported)
    }

    fn guard_depformer_projection_layout_unsupported(&self, event: &InitRun) -> Result<bool, ()> {
        Ok(!event.depformer_projection_layout_supported)
    }

    fn guard_step_model_matches(&self, event: &StepRun) -> Result<bool, ()> {
        Ok(event.model_matches)
    }

    fn guard_step_model_mismatch(&self, event: &StepRun) -> Result<bool, ()> {
        Ok(!event.model_matches)
    }

    fn guard_step_shape_valid(&self, event: &StepRun) -> Result<bool, ()> {
        Ok(event.shape_valid)
    }

    fn guard_step_shape_invalid(&self, event: &StepRun) -> Result<bool, ()> {
        Ok(!event.shape_valid)
    }

    fn guard_input_embedding_supported(&self, event: &StepRun) -> Result<bool, ()> {
        Ok(event.input_embedding_supported)
    }

    fn guard_input_embedding_unsupported(&self, event: &StepRun) -> Result<bool, ()> {
        Ok(!event.input_embedding_supported)
    }

    fn guard_execution_succeeded(&self, event: &StepRun) -> Result<bool, ()> {
        Ok(event.execution_succeeded)
    }

    fn guard_execution_failed(&self, event: &StepRun) -> Result<bool, ()> {
        Ok(!event.execution_succeeded)
    }

    fn guard_temporal_reset_present(&self, event: &ResetRun) -> Result<bool, ()> {
        Ok(event.temporal_succeeded)
    }

    fn guard_temporal_reset_missing(&self, event: &ResetRun) -> Result<bool, ()> {
        Ok(!event.temporal_succeeded)
    }

    fn guard_temporal_reset_succeeded(&self, event: &ResetRun) -> Result<bool, ()> {
        Ok(event.temporal_succeeded)
    }

    fn guard_temporal_reset_failed(&self, event: &ResetRun) -> Result<bool, ()> {
        Ok(!event.temporal_succeeded)
    }

    fn guard_depformer_reset_succeeded(&self, event: &ResetRun) -> Result<bool, ()> {
        Ok(event.depformer_succeeded)
    }

    fn guard_depformer_reset_failed(&self, event: &ResetRun) -> Result<bool, ()> {
        Ok(!event.depformer_succeeded)
    }
}

/// Public single-writer Moshi executor actor.
pub struct SpeechPredictorMoshiExecutor {
    machine: SpeechPredictorMoshiExecutorStateMachine<SpeechPredictorMoshiExecutorContext>,
}

impl Default for SpeechPredictorMoshiExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl SpeechPredictorMoshiExecutor {
    /// Constructs an uninitialized executor without allocating.
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: SpeechPredictorMoshiExecutorStateMachine::new(
                SpeechPredictorMoshiExecutorContext::default(),
            ),
        }
    }

    /// Dispatches one typed event synchronously.
    pub fn process_event(
        &mut self,
        event: SpeechPredictorMoshiExecutorEvents,
    ) -> Result<(), ExecutorError> {
        self.dispatch(event)
    }

    /// Dispatches initialization synchronously.
    pub fn process_init(&mut self, event: InitRun) -> Result<(), ExecutorError> {
        self.process_event(SpeechPredictorMoshiExecutorEvents::InitRun(event))
    }

    /// Dispatches one graph step synchronously.
    pub fn process_step(&mut self, event: StepRun) -> Result<(), ExecutorError> {
        self.process_event(SpeechPredictorMoshiExecutorEvents::StepRun(event))
    }

    /// Dispatches reset synchronously.
    pub fn process_reset(&mut self, event: ResetRun) -> Result<(), ExecutorError> {
        self.process_event(SpeechPredictorMoshiExecutorEvents::ResetRun(event))
    }

    /// Returns the generated SML state for inspection by parent actors.
    #[must_use]
    pub fn state(&self) -> &SpeechPredictorMoshiExecutorStates {
        self.machine.state()
    }

    /// Returns the actor's mutable context for public observers.
    #[must_use]
    pub fn context(&self) -> &SpeechPredictorMoshiExecutorContext {
        self.machine.context()
    }

    fn dispatch(&mut self, event: SpeechPredictorMoshiExecutorEvents) -> Result<(), ExecutorError> {
        if self.machine.process_event(event).is_err() {
            return Err(ExecutorError::UnexpectedEvent);
        }
        let error = self.machine.context().last_error();
        if error == ExecutorError::None {
            Ok(())
        } else {
            Err(error)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initialization_and_successful_step_are_explicit() {
        let mut executor = SpeechPredictorMoshiExecutor::new();
        assert!(matches!(
            executor.state(),
            &SpeechPredictorMoshiExecutorStates::StateUninitialized
        ));
        assert_eq!(
            executor.process_init(InitRun {
                sampling_seed: 0,
                ..InitRun::default()
            }),
            Ok(())
        );
        assert!(executor.context().initialized());
        assert_eq!(executor.context().sampling_seed(), 1);
        assert!(matches!(
            executor.state(),
            &SpeechPredictorMoshiExecutorStates::StateReady
        ));
        assert_eq!(executor.process_step(StepRun::default()), Ok(()));
        assert_eq!(executor.context().completed_steps(), 1);
    }

    #[test]
    fn invalid_initialization_and_step_paths_publish_typed_errors() {
        let mut executor = SpeechPredictorMoshiExecutor::new();
        assert_eq!(
            executor.process_init(InitRun {
                contract_valid: false,
                ..InitRun::default()
            }),
            Err(ExecutorError::BindFailed)
        );
        assert_eq!(
            executor.process_step(StepRun::default()),
            Err(ExecutorError::UnexpectedEvent)
        );

        assert_eq!(executor.process_init(InitRun::default()), Ok(()));
        assert_eq!(
            executor.process_step(StepRun {
                model_matches: false,
                ..StepRun::default()
            }),
            Err(ExecutorError::ModelMismatch)
        );
        assert_eq!(
            executor.process_step(StepRun {
                shape_valid: false,
                ..StepRun::default()
            }),
            Err(ExecutorError::RequestShape)
        );
        assert_eq!(
            executor.process_step(StepRun {
                execution_succeeded: false,
                ..StepRun::default()
            }),
            Err(ExecutorError::GraphExecutionUnsupported)
        );
    }

    #[test]
    fn reset_success_and_failure_follow_separate_transitions() {
        let mut executor = SpeechPredictorMoshiExecutor::new();
        executor.process_init(InitRun::default()).unwrap();
        assert_eq!(executor.process_reset(ResetRun::default()), Ok(()));
        assert!(matches!(
            executor.state(),
            &SpeechPredictorMoshiExecutorStates::StateUninitialized
        ));
        assert_eq!(executor.process_init(InitRun::default()), Ok(()));
        assert_eq!(
            executor.process_reset(ResetRun {
                depformer_succeeded: false,
                ..ResetRun::default()
            }),
            Err(ExecutorError::ResetFailed)
        );
        assert!(matches!(
            executor.state(),
            &SpeechPredictorMoshiExecutorStates::StateUninitialized
        ));
    }
}
