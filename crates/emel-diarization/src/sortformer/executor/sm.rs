//! Source-aligned bounded Sortformer executor actor.
//!
//! The actor consumes one fixed `188 x 512` encoder chunk and publishes one
//! fixed `188 x 192` hidden chunk. Model execution is supplied through typed,
//! synchronous callbacks; missing callbacks are a model-contract error rather
//! than an implicit no-op.

#![allow(
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::empty_structs_with_brackets,
    clippy::missing_const_for_fn,
    clippy::large_stack_arrays,
    dead_code,
    unused_imports,
    missing_docs
)]

use core::cell::RefCell;
use sml::sml;

pub const FRAME_COUNT: usize = 188;
pub const ENCODER_DIM: usize = 512;
pub const HIDDEN_DIM: usize = 192;
pub const SPEAKER_COUNT: usize = 4;
pub const TRANSFORMER_LAYER_COUNT: usize = 18;
pub const REQUIRED_ENCODER_VALUE_COUNT: usize = FRAME_COUNT * ENCODER_DIM;
pub const REQUIRED_HIDDEN_VALUE_COUNT: usize = FRAME_COUNT * HIDDEN_DIM;

/// Executor errors from the pinned C++ contract.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u32)]
pub enum Error {
    #[default]
    None = 0,
    ModelInvalid = 1 << 0,
    TensorContract = 1 << 1,
    InputShape = 1 << 2,
    OutputCapacity = 1 << 3,
    Kernel = 1 << 4,
    Unexpected = 1 << 5,
}

/// Bounded tensor-family summary used by model validation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct FamilyContract {
    pub tensor_count: u32,
}

/// Synchronous encoder-projection callback.
///
/// The callback must consume exactly `188 x 512` values and write exactly
/// `188 x 192` values. It must not allocate, retain either borrow, or re-enter
/// this actor.
pub type ProjectEncoderFn = fn(&[f32], &ExecutionContract, &mut [f32]) -> bool;

/// Synchronous transformer-layer callback.
///
/// `layer` is in `0..18`; the callback must execute that layer into the
/// supplied output buffer and return `false` when its lower kernel fails.
pub type TransformerLayerFn =
    fn(usize, &[f32], &ExecutionContract, &mut [f32]) -> bool;

/// Caller-owned bounded execution model contract.
#[derive(Clone, Copy, Debug, Default)]
pub struct ExecutionContract {
    pub model_present: bool,
    pub chunk_len: i32,
    pub speaker_count: i32,
    pub encoder: FamilyContract,
    pub modules: FamilyContract,
    pub transformer_encoder: FamilyContract,
    pub project_encoder: Option<ProjectEncoderFn>,
    pub transformer_layer: Option<TransformerLayerFn>,
}

impl ExecutionContract {
    #[must_use]
    pub const fn pinned() -> Self {
        Self {
            model_present: true,
            chunk_len: FRAME_COUNT as i32,
            speaker_count: SPEAKER_COUNT as i32,
            encoder: FamilyContract { tensor_count: 1 },
            modules: FamilyContract { tensor_count: 1 },
            transformer_encoder: FamilyContract { tensor_count: 1 },
            project_encoder: None,
            transformer_layer: None,
        }
    }

    #[must_use]
    pub const fn model_contract_valid(self) -> bool {
        self.model_present
            && self.chunk_len == FRAME_COUNT as i32
            && self.speaker_count == SPEAKER_COUNT as i32
            && self.encoder.tensor_count != 0
            && self.modules.tensor_count != 0
            && self.transformer_encoder.tensor_count != 0
    }
}

/// Synchronous completion callback.
pub type DoneCallback = fn(ExecuteDone) -> bool;
/// Synchronous failure callback.
pub type ErrorCallback = fn(ExecuteError) -> bool;

/// Borrowed runtime execution request.
pub struct EventExecuteRun<'a> {
    pub contract: &'a ExecutionContract,
    pub encoder_frames: &'a [f32],
    pub hidden_out: RefCell<&'a mut [f32]>,
    pub frame_count_out: RefCell<&'a mut i32>,
    pub hidden_dim_out: RefCell<&'a mut i32>,
    pub error_out: RefCell<Option<&'a mut Error>>,
    pub on_done: Option<DoneCallback>,
    pub on_error: Option<ErrorCallback>,
}

impl<'a> EventExecuteRun<'a> {
    #[must_use]
    pub fn new(
        contract: &'a ExecutionContract,
        encoder_frames: &'a [f32],
        hidden_out: &'a mut [f32],
        frame_count_out: &'a mut i32,
        hidden_dim_out: &'a mut i32,
    ) -> Self {
        Self {
            contract,
            encoder_frames,
            hidden_out: RefCell::new(hidden_out),
            frame_count_out: RefCell::new(frame_count_out),
            hidden_dim_out: RefCell::new(hidden_dim_out),
            error_out: RefCell::new(None),
            on_done: None,
            on_error: None,
        }
    }

    #[must_use]
    pub fn with_callbacks(
        mut self,
        on_done: Option<DoneCallback>,
        on_error: Option<ErrorCallback>,
    ) -> Self {
        self.on_done = on_done;
        self.on_error = on_error;
        self
    }

    #[must_use]
    pub fn with_error_out(mut self, error_out: &'a mut Error) -> Self {
        *self.error_out.get_mut() = Some(error_out);
        self
    }
}

sml! {
    DiarizationSortformerExecutor {
        "state_model_contract_decision"_s <= *"state_ready"_s + EventExecuteRun(&'dispatch EventExecuteRun<'dispatch>) / effect_begin_execute,
        "state_tensor_contract_decision"_s <= "state_model_contract_decision"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'dispatch>) [guard_model_contract_valid],
        "state_error_error_out_decision"_s <= "state_model_contract_decision"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_model_contract_invalid] / effect_mark_model_invalid,
        "state_input_shape_decision"_s <= "state_tensor_contract_decision"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_tensor_contract_valid],
        "state_error_error_out_decision"_s <= "state_tensor_contract_decision"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_tensor_contract_invalid] / effect_mark_tensor_contract_invalid,
        "state_output_capacity_decision"_s <= "state_input_shape_decision"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_input_shape_valid],
        "state_error_error_out_decision"_s <= "state_input_shape_decision"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_input_shape_invalid] / effect_mark_input_shape_invalid,
        "state_binding"_s <= "state_output_capacity_decision"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_output_capacity_valid] / effect_bind_contracts,
        "state_error_error_out_decision"_s <= "state_output_capacity_decision"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_output_capacity_invalid] / effect_mark_output_capacity_invalid,
        "state_projecting"_s <= "state_binding"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) / effect_project_encoder,
        "state_transformer_cache"_s <= "state_projecting"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_execution_ok] / effect_write_projected_frames_to_cache,
        "state_error_error_out_decision"_s <= "state_projecting"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_execution_failed],
        "state_transformer_layer_00"_s <= "state_transformer_cache"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) / effect_execute_transformer_layer_00,
        "state_transformer_layer_01"_s <= "state_transformer_layer_00"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_execution_ok] / effect_execute_transformer_layer_01,
        "state_error_error_out_decision"_s <= "state_transformer_layer_00"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_execution_failed],
        "state_transformer_layer_02"_s <= "state_transformer_layer_01"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_execution_ok] / effect_execute_transformer_layer_02,
        "state_error_error_out_decision"_s <= "state_transformer_layer_01"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_execution_failed],
        "state_transformer_layer_03"_s <= "state_transformer_layer_02"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_execution_ok] / effect_execute_transformer_layer_03,
        "state_error_error_out_decision"_s <= "state_transformer_layer_02"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_execution_failed],
        "state_transformer_layer_04"_s <= "state_transformer_layer_03"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_execution_ok] / effect_execute_transformer_layer_04,
        "state_error_error_out_decision"_s <= "state_transformer_layer_03"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_execution_failed],
        "state_transformer_layer_05"_s <= "state_transformer_layer_04"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_execution_ok] / effect_execute_transformer_layer_05,
        "state_error_error_out_decision"_s <= "state_transformer_layer_04"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_execution_failed],
        "state_transformer_layer_06"_s <= "state_transformer_layer_05"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_execution_ok] / effect_execute_transformer_layer_06,
        "state_error_error_out_decision"_s <= "state_transformer_layer_05"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_execution_failed],
        "state_transformer_layer_07"_s <= "state_transformer_layer_06"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_execution_ok] / effect_execute_transformer_layer_07,
        "state_error_error_out_decision"_s <= "state_transformer_layer_06"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_execution_failed],
        "state_transformer_layer_08"_s <= "state_transformer_layer_07"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_execution_ok] / effect_execute_transformer_layer_08,
        "state_error_error_out_decision"_s <= "state_transformer_layer_07"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_execution_failed],
        "state_transformer_layer_09"_s <= "state_transformer_layer_08"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_execution_ok] / effect_execute_transformer_layer_09,
        "state_error_error_out_decision"_s <= "state_transformer_layer_08"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_execution_failed],
        "state_transformer_layer_10"_s <= "state_transformer_layer_09"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_execution_ok] / effect_execute_transformer_layer_10,
        "state_error_error_out_decision"_s <= "state_transformer_layer_09"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_execution_failed],
        "state_transformer_layer_11"_s <= "state_transformer_layer_10"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_execution_ok] / effect_execute_transformer_layer_11,
        "state_error_error_out_decision"_s <= "state_transformer_layer_10"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_execution_failed],
        "state_transformer_layer_12"_s <= "state_transformer_layer_11"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_execution_ok] / effect_execute_transformer_layer_12,
        "state_error_error_out_decision"_s <= "state_transformer_layer_11"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_execution_failed],
        "state_transformer_layer_13"_s <= "state_transformer_layer_12"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_execution_ok] / effect_execute_transformer_layer_13,
        "state_error_error_out_decision"_s <= "state_transformer_layer_12"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_execution_failed],
        "state_transformer_layer_14"_s <= "state_transformer_layer_13"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_execution_ok] / effect_execute_transformer_layer_14,
        "state_error_error_out_decision"_s <= "state_transformer_layer_13"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_execution_failed],
        "state_transformer_layer_15"_s <= "state_transformer_layer_14"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_execution_ok] / effect_execute_transformer_layer_15,
        "state_error_error_out_decision"_s <= "state_transformer_layer_14"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_execution_failed],
        "state_transformer_layer_16"_s <= "state_transformer_layer_15"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_execution_ok] / effect_execute_transformer_layer_16,
        "state_error_error_out_decision"_s <= "state_transformer_layer_15"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_execution_failed],
        "state_transformer_layer_17"_s <= "state_transformer_layer_16"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_execution_ok] / effect_execute_transformer_layer_17,
        "state_error_error_out_decision"_s <= "state_transformer_layer_16"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_execution_failed],
        "state_publishing_hidden"_s <= "state_transformer_layer_17"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_execution_ok] / effect_publish_hidden,
        "state_error_error_out_decision"_s <= "state_transformer_layer_17"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_execution_failed],
        "state_success_error_out_decision"_s <= "state_publishing_hidden"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_execution_ok],
        "state_success_callback_decision"_s <= "state_success_error_out_decision"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_has_error_out] / effect_store_success_error,
        "state_success_callback_decision"_s <= "state_success_error_out_decision"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_no_error_out],
        "state_error_callback_decision"_s <= "state_error_error_out_decision"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_has_error_out] / effect_store_error_error,
        "state_error_callback_decision"_s <= "state_error_error_out_decision"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_no_error_out],
        "state_done"_s <= "state_success_callback_decision"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_has_done_callback] / effect_emit_done,
        "state_done"_s <= "state_success_callback_decision"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_no_done_callback],
        "state_errored"_s <= "state_error_callback_decision"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_has_error_callback] / effect_emit_error,
        "state_errored"_s <= "state_error_callback_decision"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>) [guard_no_error_callback],
        "state_ready"_s <= "state_done"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>),
        "state_ready"_s <= "state_errored"_s + completion<EventExecuteRun<'dispatch>>(&'dispatch EventExecuteRun<'dispatch>),
        "state_ready"_s <= "state_ready"_s + unexpected_event<_> / effect_on_unexpected_from_state_ready,
        "state_ready"_s <= "state_model_contract_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_model_contract_decision,
        "state_ready"_s <= "state_tensor_contract_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_tensor_contract_decision,
        "state_ready"_s <= "state_input_shape_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_input_shape_decision,
        "state_ready"_s <= "state_output_capacity_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_output_capacity_decision,
        "state_ready"_s <= "state_binding"_s + unexpected_event<_> / effect_on_unexpected_from_state_binding,
        "state_ready"_s <= "state_projecting"_s + unexpected_event<_> / effect_on_unexpected_from_state_projecting,
        "state_ready"_s <= "state_transformer_cache"_s + unexpected_event<_> / effect_on_unexpected_from_state_transformer_cache,
        "state_ready"_s <= "state_transformer_layer_00"_s + unexpected_event<_> / effect_on_unexpected_from_state_transformer_layer_00,
        "state_ready"_s <= "state_transformer_layer_01"_s + unexpected_event<_> / effect_on_unexpected_from_state_transformer_layer_01,
        "state_ready"_s <= "state_transformer_layer_02"_s + unexpected_event<_> / effect_on_unexpected_from_state_transformer_layer_02,
        "state_ready"_s <= "state_transformer_layer_03"_s + unexpected_event<_> / effect_on_unexpected_from_state_transformer_layer_03,
        "state_ready"_s <= "state_transformer_layer_04"_s + unexpected_event<_> / effect_on_unexpected_from_state_transformer_layer_04,
        "state_ready"_s <= "state_transformer_layer_05"_s + unexpected_event<_> / effect_on_unexpected_from_state_transformer_layer_05,
        "state_ready"_s <= "state_transformer_layer_06"_s + unexpected_event<_> / effect_on_unexpected_from_state_transformer_layer_06,
        "state_ready"_s <= "state_transformer_layer_07"_s + unexpected_event<_> / effect_on_unexpected_from_state_transformer_layer_07,
        "state_ready"_s <= "state_transformer_layer_08"_s + unexpected_event<_> / effect_on_unexpected_from_state_transformer_layer_08,
        "state_ready"_s <= "state_transformer_layer_09"_s + unexpected_event<_> / effect_on_unexpected_from_state_transformer_layer_09,
        "state_ready"_s <= "state_transformer_layer_10"_s + unexpected_event<_> / effect_on_unexpected_from_state_transformer_layer_10,
        "state_ready"_s <= "state_transformer_layer_11"_s + unexpected_event<_> / effect_on_unexpected_from_state_transformer_layer_11,
        "state_ready"_s <= "state_transformer_layer_12"_s + unexpected_event<_> / effect_on_unexpected_from_state_transformer_layer_12,
        "state_ready"_s <= "state_transformer_layer_13"_s + unexpected_event<_> / effect_on_unexpected_from_state_transformer_layer_13,
        "state_ready"_s <= "state_transformer_layer_14"_s + unexpected_event<_> / effect_on_unexpected_from_state_transformer_layer_14,
        "state_ready"_s <= "state_transformer_layer_15"_s + unexpected_event<_> / effect_on_unexpected_from_state_transformer_layer_15,
        "state_ready"_s <= "state_transformer_layer_16"_s + unexpected_event<_> / effect_on_unexpected_from_state_transformer_layer_16,
        "state_ready"_s <= "state_transformer_layer_17"_s + unexpected_event<_> / effect_on_unexpected_from_state_transformer_layer_17,
        "state_ready"_s <= "state_publishing_hidden"_s + unexpected_event<_> / effect_on_unexpected_from_state_publishing_hidden,
        "state_ready"_s <= "state_success_error_out_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_success_error_out_decision,
        "state_ready"_s <= "state_success_callback_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_success_callback_decision,
        "state_ready"_s <= "state_error_error_out_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_error_error_out_decision,
        "state_ready"_s <= "state_error_callback_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_error_callback_decision,
        "state_ready"_s <= "state_done"_s + unexpected_event<_> / effect_on_unexpected_from_state_done,
        "state_ready"_s <= "state_errored"_s + unexpected_event<_> / effect_on_unexpected_from_state_errored,
    }
}

#[derive(Debug)]
pub struct DiarizationSortformerExecutorContext {
    pub err: Error,
    pub hidden_a: [f32; REQUIRED_HIDDEN_VALUE_COUNT],
    pub hidden_b: [f32; REQUIRED_HIDDEN_VALUE_COUNT],
}

impl Default for DiarizationSortformerExecutorContext {
    fn default() -> Self {
        Self { err: Error::None, hidden_a: [0.0; REQUIRED_HIDDEN_VALUE_COUNT], hidden_b: [0.0; REQUIRED_HIDDEN_VALUE_COUNT] }
    }
}

impl DiarizationSortformerExecutorStateMachineContext for DiarizationSortformerExecutorContext {
    fn effect_begin_execute(&mut self, event: &EventExecuteRun<'_>) -> Result<(), ()> {
        self.err = Error::None;
        *event.frame_count_out.borrow_mut() = 0;
        *event.hidden_dim_out.borrow_mut() = 0;
        Ok(())
    }
    fn effect_bind_contracts(&mut self, _: &EventExecuteRun<'_>) -> Result<(), ()> { Ok(()) }
    fn effect_project_encoder(&mut self, event: &EventExecuteRun<'_>) -> Result<(), ()> {
        self.hidden_a.fill(0.0);
        let Some(project) = event.contract.project_encoder else { self.err = Error::Kernel; return Ok(()); };
        if !project(event.encoder_frames, event.contract, &mut self.hidden_a) { self.err = Error::Kernel; }
        Ok(())
    }
    fn effect_write_projected_frames_to_cache(&mut self, _: &EventExecuteRun<'_>) -> Result<(), ()> { Ok(()) }
    fn execute_layer(&mut self, event: &EventExecuteRun<'_>, layer: usize) {
        let Some(transformer) = event.contract.transformer_layer else { self.err = Error::Kernel; return; };
        let (input, output) = if layer % 2 == 0 { (&self.hidden_a[..], &mut self.hidden_b[..]) } else { (&self.hidden_b[..], &mut self.hidden_a[..]) };
        output.fill(0.0);
        if !transformer(layer, input, event.contract, output) { self.err = Error::Kernel; }
    }
    fn effect_execute_transformer_layer_00(&mut self, e: &EventExecuteRun<'_>) -> Result<(), ()> { self.execute_layer(e, 0); Ok(()) }
    fn effect_execute_transformer_layer_01(&mut self, e: &EventExecuteRun<'_>) -> Result<(), ()> { self.execute_layer(e, 1); Ok(()) }
    fn effect_execute_transformer_layer_02(&mut self, e: &EventExecuteRun<'_>) -> Result<(), ()> { self.execute_layer(e, 2); Ok(()) }
    fn effect_execute_transformer_layer_03(&mut self, e: &EventExecuteRun<'_>) -> Result<(), ()> { self.execute_layer(e, 3); Ok(()) }
    fn effect_execute_transformer_layer_04(&mut self, e: &EventExecuteRun<'_>) -> Result<(), ()> { self.execute_layer(e, 4); Ok(()) }
    fn effect_execute_transformer_layer_05(&mut self, e: &EventExecuteRun<'_>) -> Result<(), ()> { self.execute_layer(e, 5); Ok(()) }
    fn effect_execute_transformer_layer_06(&mut self, e: &EventExecuteRun<'_>) -> Result<(), ()> { self.execute_layer(e, 6); Ok(()) }
    fn effect_execute_transformer_layer_07(&mut self, e: &EventExecuteRun<'_>) -> Result<(), ()> { self.execute_layer(e, 7); Ok(()) }
    fn effect_execute_transformer_layer_08(&mut self, e: &EventExecuteRun<'_>) -> Result<(), ()> { self.execute_layer(e, 8); Ok(()) }
    fn effect_execute_transformer_layer_09(&mut self, e: &EventExecuteRun<'_>) -> Result<(), ()> { self.execute_layer(e, 9); Ok(()) }
    fn effect_execute_transformer_layer_10(&mut self, e: &EventExecuteRun<'_>) -> Result<(), ()> { self.execute_layer(e, 10); Ok(()) }
    fn effect_execute_transformer_layer_11(&mut self, e: &EventExecuteRun<'_>) -> Result<(), ()> { self.execute_layer(e, 11); Ok(()) }
    fn effect_execute_transformer_layer_12(&mut self, e: &EventExecuteRun<'_>) -> Result<(), ()> { self.execute_layer(e, 12); Ok(()) }
    fn effect_execute_transformer_layer_13(&mut self, e: &EventExecuteRun<'_>) -> Result<(), ()> { self.execute_layer(e, 13); Ok(()) }
    fn effect_execute_transformer_layer_14(&mut self, e: &EventExecuteRun<'_>) -> Result<(), ()> { self.execute_layer(e, 14); Ok(()) }
    fn effect_execute_transformer_layer_15(&mut self, e: &EventExecuteRun<'_>) -> Result<(), ()> { self.execute_layer(e, 15); Ok(()) }
    fn effect_execute_transformer_layer_16(&mut self, e: &EventExecuteRun<'_>) -> Result<(), ()> { self.execute_layer(e, 16); Ok(()) }
    fn effect_execute_transformer_layer_17(&mut self, e: &EventExecuteRun<'_>) -> Result<(), ()> { self.execute_layer(e, 17); Ok(()) }
    fn effect_publish_hidden(&mut self, event: &EventExecuteRun<'_>) -> Result<(), ()> {
        let final_hidden = if TRANSFORMER_LAYER_COUNT % 2 == 0 { &self.hidden_a[..] } else { &self.hidden_b[..] };
        event.hidden_out.borrow_mut()[..REQUIRED_HIDDEN_VALUE_COUNT].copy_from_slice(final_hidden);
        *event.frame_count_out.borrow_mut() = FRAME_COUNT as i32;
        *event.hidden_dim_out.borrow_mut() = HIDDEN_DIM as i32;
        Ok(())
    }
    fn effect_store_success_error(&mut self, event: &EventExecuteRun<'_>) -> Result<(), ()> { if let Some(out) = event.error_out.borrow_mut().as_deref_mut() { *out = self.err; } Ok(()) }
    fn effect_store_error_error(&mut self, event: &EventExecuteRun<'_>) -> Result<(), ()> { if let Some(out) = event.error_out.borrow_mut().as_deref_mut() { *out = self.err; } Ok(()) }
    fn effect_emit_done(&mut self, event: &EventExecuteRun<'_>) -> Result<(), ()> { if let Some(callback) = event.on_done { let _ = callback(ExecuteDone { frame_count: *event.frame_count_out.borrow(), hidden_dim: *event.hidden_dim_out.borrow() }); } Ok(()) }
    fn effect_emit_error(&mut self, event: &EventExecuteRun<'_>) -> Result<(), ()> { if let Some(callback) = event.on_error { let _ = callback(ExecuteError { error: self.err }); } Ok(()) }
    fn effect_mark_model_invalid(&mut self, _: &EventExecuteRun<'_>) -> Result<(), ()> { self.err = Error::ModelInvalid; Ok(()) }
    fn effect_mark_tensor_contract_invalid(&mut self, _: &EventExecuteRun<'_>) -> Result<(), ()> { self.err = Error::TensorContract; Ok(()) }
    fn effect_mark_input_shape_invalid(&mut self, _: &EventExecuteRun<'_>) -> Result<(), ()> { self.err = Error::InputShape; Ok(()) }
    fn effect_mark_output_capacity_invalid(&mut self, _: &EventExecuteRun<'_>) -> Result<(), ()> { self.err = Error::OutputCapacity; Ok(()) }
    fn guard_model_contract_valid(&self, e: &EventExecuteRun<'_>) -> Result<bool, ()> { Ok(e.contract.model_contract_valid()) }
    fn guard_model_contract_invalid(&self, e: &EventExecuteRun<'_>) -> Result<bool, ()> { Ok(!e.contract.model_contract_valid()) }
    fn guard_tensor_contract_valid(&self, e: &EventExecuteRun<'_>) -> Result<bool, ()> { Ok(e.contract.encoder.tensor_count != 0 && e.contract.modules.tensor_count != 0 && e.contract.transformer_encoder.tensor_count != 0) }
    fn guard_tensor_contract_invalid(&self, e: &EventExecuteRun<'_>) -> Result<bool, ()> { Ok(!self.guard_tensor_contract_valid(e)?) }
    fn guard_input_shape_valid(&self, e: &EventExecuteRun<'_>) -> Result<bool, ()> { Ok(!e.encoder_frames.is_empty() && e.encoder_frames.len() == REQUIRED_ENCODER_VALUE_COUNT) }
    fn guard_input_shape_invalid(&self, e: &EventExecuteRun<'_>) -> Result<bool, ()> { Ok(!self.guard_input_shape_valid(e)?) }
    fn guard_output_capacity_valid(&self, e: &EventExecuteRun<'_>) -> Result<bool, ()> { Ok(!e.hidden_out.borrow().is_empty() && e.hidden_out.borrow().len() >= REQUIRED_HIDDEN_VALUE_COUNT) }
    fn guard_output_capacity_invalid(&self, e: &EventExecuteRun<'_>) -> Result<bool, ()> { Ok(!self.guard_output_capacity_valid(e)?) }
    fn guard_execution_ok(&self, _: &EventExecuteRun<'_>) -> Result<bool, ()> { Ok(self.err == Error::None) }
    fn guard_execution_failed(&self, e: &EventExecuteRun<'_>) -> Result<bool, ()> { Ok(!self.guard_execution_ok(e)?) }
    fn guard_has_error_out(&self, e: &EventExecuteRun<'_>) -> Result<bool, ()> { Ok(e.error_out.borrow().is_some()) }
    fn guard_no_error_out(&self, e: &EventExecuteRun<'_>) -> Result<bool, ()> { Ok(!self.guard_has_error_out(e)?) }
    fn guard_has_done_callback(&self, e: &EventExecuteRun<'_>) -> Result<bool, ()> { Ok(e.on_done.is_some()) }
    fn guard_no_done_callback(&self, e: &EventExecuteRun<'_>) -> Result<bool, ()> { Ok(e.on_done.is_none()) }
    fn guard_has_error_callback(&self, e: &EventExecuteRun<'_>) -> Result<bool, ()> { Ok(e.on_error.is_some()) }
    fn guard_no_error_callback(&self, e: &EventExecuteRun<'_>) -> Result<bool, ()> { Ok(e.on_error.is_none()) }
    fn unexpected(&mut self) -> Result<(), ()> { self.err = Error::Unexpected; Ok(()) }
    fn effect_on_unexpected_from_state_ready(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_model_contract_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_tensor_contract_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_input_shape_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_output_capacity_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_binding(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_projecting(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_transformer_cache(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_transformer_layer_00(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_transformer_layer_01(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_transformer_layer_02(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_transformer_layer_03(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_transformer_layer_04(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_transformer_layer_05(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_transformer_layer_06(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_transformer_layer_07(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_transformer_layer_08(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_transformer_layer_09(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_transformer_layer_10(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_transformer_layer_11(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_transformer_layer_12(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_transformer_layer_13(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_transformer_layer_14(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_transformer_layer_15(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_transformer_layer_16(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_transformer_layer_17(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_publishing_hidden(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_success_error_out_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_success_callback_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_error_error_out_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_error_callback_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_done(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_errored(&mut self) -> Result<(), ()> { self.unexpected() }
}

/// Generated state-machine alias matching the pinned actor.
pub type ExecutorStateMachine = DiarizationSortformerExecutorStateMachine;

/// Single-writer synchronous Sortformer executor actor.
pub struct Executor {
    machine: ExecutorStateMachine,
}

impl Default for Executor {
    fn default() -> Self { Self::new() }
}

impl Executor {
    /// Constructs an executor in generated `state_ready`.
    #[must_use]
    pub fn new() -> Self {
        Self { machine: ExecutorStateMachine::new(DiarizationSortformerExecutorContext::default()) }
    }

    /// Dispatches one borrowed request synchronously through every phase.
    pub fn execute<'a>(&mut self, event: EventExecuteRun<'a>) -> bool {
        self.process_event(event)
    }

    /// Source-compatible process-event spelling for the pinned actor.
    pub fn process_event<'a>(&mut self, event: EventExecuteRun<'a>) -> bool {
        if self.machine.process_event(DiarizationSortformerExecutorEvents::EventExecuteRun(&event)).is_err() {
            self.machine.context_mut().err = Error::Unexpected;
            return false;
        }
        self.machine.context().err == Error::None
    }

    /// Explicit unexpected-event path used by lifecycle recovery tests.
    pub fn process_unexpected_event(&mut self) -> bool {
        self.machine.process_event(DiarizationSortformerExecutorEvents::UnexpectedEvent).is_ok()
    }

    /// Returns generated state inspection data.
    #[must_use]
    pub fn state(&self) -> &DiarizationSortformerExecutorStates { self.machine.state() }

    /// Reports whether the generated machine is in `state`.
    #[must_use]
    pub fn is(&self, state: &DiarizationSortformerExecutorStates) -> bool { self.machine.is(state) }

    /// Returns the actor-owned bounded context.
    #[must_use]
    pub fn context(&self) -> &DiarizationSortformerExecutorContext { self.machine.context() }

    /// Returns the most recent execution error.
    #[must_use]
    pub fn error(&self) -> Error { self.machine.context().err }
}
