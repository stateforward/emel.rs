//! Source-aligned bounded Sortformer executor actor.
//!
//! The actor consumes one fixed `188 x 512` encoder chunk and publishes one
//! fixed `188 x 192` hidden chunk. Model execution is supplied through typed,
//! synchronous callbacks or a model-bound native encoder projection. Missing
//! execution routes are explicit model-contract errors rather than no-ops.

#![allow(
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::empty_structs_with_brackets,
    clippy::missing_const_for_fn,
    clippy::unnecessary_wraps,
    clippy::large_types_passed_by_value,
    missing_debug_implementations,
    dead_code,
    unused_imports,
    missing_docs
)]

use super::native_projection::{
    Binding as NativeProjectionBinding, Error as NativeProjectionError,
    Route as NativeProjectionRoute,
};
use super::native_transformer::{Error as NativeTransformerError, Route as NativeTransformerRoute};
use core::cell::RefCell;
use opentelemetry::trace::{self, Span, Status, Tracer};
use opentelemetry::{KeyValue, global};
use sml::sml;

const SORTFORMER_TRACER: &str = "emel-diarization";
const SORTFORMER_OPERATION: &str = "sortformer_execute";

fn sortformer_event(
    name: &'static str,
    route: &'static str,
    key: &'static str,
    value: &'static str,
) {
    trace::get_active_span(|span| {
        if span.is_recording() {
            span.add_event(
                name,
                vec![
                    KeyValue::new("operation", SORTFORMER_OPERATION),
                    KeyValue::new("route", route),
                    KeyValue::new(key, value),
                ],
            );
        }
    });
}

fn sortformer_result(success: bool) {
    trace::get_active_span(|span| {
        if span.is_recording() {
            if success {
                span.add_event(
                    "emel.sortformer.result",
                    vec![
                        KeyValue::new("operation", SORTFORMER_OPERATION),
                        KeyValue::new("result", "success"),
                    ],
                );
            } else {
                span.add_event(
                    "emel.sortformer.failure",
                    vec![
                        KeyValue::new("operation", SORTFORMER_OPERATION),
                        KeyValue::new("failure", "error"),
                    ],
                );
                span.set_status(Status::error("sortformer failure"));
            }
        }
    });
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
enum PhaseOutcome {
    #[default]
    Pending,
    Succeeded,
    Failed(Error),
}

impl From<Result<(), NativeProjectionError>> for PhaseOutcome {
    fn from(result: Result<(), NativeProjectionError>) -> Self {
        match result {
            Ok(()) => Self::Succeeded,
            Err(_) => Self::Failed(Error::Kernel),
        }
    }
}

impl From<Result<(), NativeTransformerError>> for PhaseOutcome {
    fn from(result: Result<(), NativeTransformerError>) -> Self {
        match result {
            Ok(()) => Self::Succeeded,
            Err(_) => Self::Failed(Error::Kernel),
        }
    }
}

impl From<bool> for PhaseOutcome {
    fn from(succeeded: bool) -> Self {
        if succeeded {
            Self::Succeeded
        } else {
            Self::Failed(Error::Kernel)
        }
    }
}

const FRAME_COUNT_I32: i32 = 188;
const SPEAKER_COUNT_I32: i32 = 4;
const HIDDEN_DIM_I32: i32 = 192;

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
    ProjectionRoute = 1 << 6,
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
pub type TransformerLayerFn = fn(usize, &[f32], &ExecutionContract, &mut [f32]) -> bool;
#[derive(Clone, Copy, Debug, Default)]
pub struct ExecutionContract<'model> {
    pub model_present: bool,
    pub chunk_len: i32,
    pub speaker_count: i32,
    pub encoder: FamilyContract,
    pub modules: FamilyContract,
    pub transformer_encoder: FamilyContract,
    pub native_projection: Option<&'model NativeProjectionBinding>,
    pub project_encoder: Option<ProjectEncoderFn>,
    pub transformer_layer: Option<TransformerLayerFn>,
}

impl<'model> ExecutionContract<'model> {
    #[must_use]
    pub const fn pinned() -> Self {
        Self {
            model_present: true,
            chunk_len: FRAME_COUNT_I32,
            speaker_count: SPEAKER_COUNT_I32,
            encoder: FamilyContract { tensor_count: 1 },
            modules: FamilyContract { tensor_count: 1 },
            transformer_encoder: FamilyContract { tensor_count: 1 },
            native_projection: None,
            project_encoder: None,
            transformer_layer: None,
        }
    }

    #[must_use]
    pub fn with_native_projection(mut self, binding: &'model NativeProjectionBinding) -> Self {
        self.native_projection = Some(binding);
        self
    }

    #[must_use]
    pub const fn model_contract_valid(self) -> bool {
        self.model_present
            && self.chunk_len == FRAME_COUNT_I32
            && self.speaker_count == SPEAKER_COUNT_I32
            && self.encoder.tensor_count != 0
            && self.modules.tensor_count != 0
            && self.transformer_encoder.tensor_count != 0
    }
}

/// Successful executor completion payload.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExecuteDone {
    pub frame_count: i32,
    pub hidden_dim: i32,
}

/// Failed executor completion payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExecuteError {
    pub error: Error,
}

/// Synchronous completion callback. The callback must not retain or re-enter
/// the executor.
pub type DoneCallback = fn(ExecuteDone) -> bool;
/// Synchronous error callback. The callback must not retain or re-enter the
/// executor.
pub type ErrorCallback = fn(ExecuteError) -> bool;

/// Borrowed executor request for one synchronous, single-writer dispatch.
///
/// The caller owns the contract, input/output buffers, output slots, and
/// optional native routes. Every borrow is valid only while `execute` runs;
/// the event must not be shared across dispatches or retained by the actor.
/// `RefCell` supplies temporary interior access for generated state-machine
/// effects; it is not a cross-dispatch synchronization primitive and does not
/// make this event thread-safe.
///
/// Native route slots are caller-owned. A selected route may be temporarily
/// taken for an effect and is replaced into its slot before that effect
/// returns, so route ownership remains with the caller. Completion callbacks
/// receive copied outcome data synchronously and must not retain event borrows
/// or re-enter the executor.
pub struct EventExecuteRun<'a> {
    /// Caller-owned execution contract, immutably borrowed for this dispatch.
    pub contract: &'a ExecutionContract<'a>,
    /// Caller-owned encoder frames, immutably borrowed for this dispatch.
    pub encoder_frames: &'a [f32],
    /// Caller-owned hidden output buffer, written only during this dispatch.
    pub hidden_out: RefCell<&'a mut [f32]>,
    /// Caller-owned output slot for the number of produced frames.
    pub frame_count_out: RefCell<&'a mut i32>,
    /// Caller-owned output slot for the hidden dimension.
    pub hidden_dim_out: RefCell<&'a mut i32>,
    projection_outcome: RefCell<PhaseOutcome>,
    transformer_outcome: RefCell<PhaseOutcome>,
    contracts_bound: RefCell<bool>,
    /// Optional caller-owned error output slot for the immediate result.
    pub error_out: RefCell<Option<&'a mut Error>>,
    /// Optional caller-owned native projection route. Dispatch temporarily
    /// takes and replaces the slot value; the actor never retains the route.
    pub native_projection_route: RefCell<Option<&'a mut NativeProjectionRoute<'a>>>,
    /// Optional caller-owned native transformer route. Dispatch temporarily
    /// takes and replaces the slot value; the actor never retains the route.
    pub native_transformer_route: RefCell<Option<&'a mut NativeTransformerRoute<'a>>>,
    /// Optional synchronous completion callback; it must not retain event data
    /// or re-enter the executor.
    pub on_done: Option<DoneCallback>,
    /// Optional synchronous error callback; it must not retain event data or
    /// re-enter the executor.
    pub on_error: Option<ErrorCallback>,
}

impl<'a> EventExecuteRun<'a> {
    #[must_use]
    pub fn new(
        contract: &'a ExecutionContract<'a>,
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
            projection_outcome: RefCell::new(PhaseOutcome::Pending),
            transformer_outcome: RefCell::new(PhaseOutcome::Pending),
            contracts_bound: RefCell::new(false),
            error_out: RefCell::new(None),
            native_projection_route: RefCell::new(None),
            native_transformer_route: RefCell::new(None),
            on_done: None,
            on_error: None,
        }
    }

    #[must_use]
    pub fn with_native_projection_route(
        mut self,
        route: &'a mut NativeProjectionRoute<'a>,
    ) -> Self {
        *self.native_projection_route.get_mut() = Some(route);
        self
    }
    #[must_use]
    pub fn with_native_transformer_route(
        mut self,
        route: &'a mut NativeTransformerRoute<'a>,
    ) -> Self {
        *self.native_transformer_route.get_mut() = Some(route);
        self
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
    DiarizationSortformerExecutor<'dispatch, 'event>
    where
        'event: 'dispatch,
    {
        "state_model_contract_decision"_s <= *"state_ready"_s + EventExecuteRun(&'dispatch EventExecuteRun<'event>) / effect_begin_execute,
        "state_tensor_contract_decision"_s <= "state_model_contract_decision"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_model_contract_valid],
        "state_error_error_out_decision"_s <= "state_model_contract_decision"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_model_contract_invalid] / effect_mark_model_invalid,
        "state_input_shape_decision"_s <= "state_tensor_contract_decision"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_tensor_contract_valid],
        "state_error_error_out_decision"_s <= "state_tensor_contract_decision"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_tensor_contract_invalid] / effect_mark_tensor_contract_invalid,
        "state_output_capacity_decision"_s <= "state_input_shape_decision"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_input_shape_valid],
        "state_error_error_out_decision"_s <= "state_input_shape_decision"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_input_shape_invalid] / effect_mark_input_shape_invalid,
        "state_binding"_s <= "state_output_capacity_decision"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_output_capacity_valid] / effect_bind_contracts,
        "state_error_error_out_decision"_s <= "state_output_capacity_decision"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_output_capacity_invalid] / effect_mark_output_capacity_invalid,
        "state_native_projecting"_s <= "state_binding"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_native_projection_route] / effect_project_encoder_native,
        "state_callback_projecting"_s <= "state_binding"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_callback_projection_route] / effect_project_encoder_callback,
        "state_error_error_out_decision"_s <= "state_binding"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_projection_route_missing] / effect_mark_projection_route_missing,
        "state_transformer_cache"_s <= "state_native_projecting"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_projection_succeeded] / effect_write_projected_frames_to_cache,
        "state_error_error_out_decision"_s <= "state_native_projecting"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_projection_failed],
        "state_transformer_cache"_s <= "state_callback_projecting"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_projection_succeeded] / effect_write_projected_frames_to_cache,
        "state_error_error_out_decision"_s <= "state_callback_projecting"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_projection_failed],
        "state_transformer_layer_00"_s <= "state_transformer_cache"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_native_transformer_route] / effect_execute_transformer_layer_00_native,
        "state_transformer_layer_00"_s <= "state_transformer_cache"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_callback_transformer_route] / effect_execute_transformer_layer_00_callback,
        "state_error_error_out_decision"_s <= "state_transformer_cache"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_transformer_route_missing] / effect_mark_transformer_route_missing,
        "state_transformer_layer_01"_s <= "state_transformer_layer_00"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_native_transformer_route_after_success] / effect_execute_transformer_layer_01_native,
        "state_transformer_layer_01"_s <= "state_transformer_layer_00"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_callback_transformer_route_after_success] / effect_execute_transformer_layer_01_callback,
        "state_error_error_out_decision"_s <= "state_transformer_layer_00"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_transformer_failed],
        "state_error_error_out_decision"_s <= "state_transformer_layer_00"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_transformer_route_missing] / effect_mark_transformer_route_missing,
        "state_transformer_layer_02"_s <= "state_transformer_layer_01"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_native_transformer_route_after_success] / effect_execute_transformer_layer_02_native,
        "state_transformer_layer_02"_s <= "state_transformer_layer_01"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_callback_transformer_route_after_success] / effect_execute_transformer_layer_02_callback,
        "state_error_error_out_decision"_s <= "state_transformer_layer_01"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_transformer_failed],
        "state_error_error_out_decision"_s <= "state_transformer_layer_01"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_transformer_route_missing] / effect_mark_transformer_route_missing,
        "state_transformer_layer_03"_s <= "state_transformer_layer_02"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_native_transformer_route_after_success] / effect_execute_transformer_layer_03_native,
        "state_transformer_layer_03"_s <= "state_transformer_layer_02"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_callback_transformer_route_after_success] / effect_execute_transformer_layer_03_callback,
        "state_error_error_out_decision"_s <= "state_transformer_layer_02"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_transformer_failed],
        "state_error_error_out_decision"_s <= "state_transformer_layer_02"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_transformer_route_missing] / effect_mark_transformer_route_missing,
        "state_transformer_layer_04"_s <= "state_transformer_layer_03"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_native_transformer_route_after_success] / effect_execute_transformer_layer_04_native,
        "state_transformer_layer_04"_s <= "state_transformer_layer_03"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_callback_transformer_route_after_success] / effect_execute_transformer_layer_04_callback,
        "state_error_error_out_decision"_s <= "state_transformer_layer_03"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_transformer_failed],
        "state_error_error_out_decision"_s <= "state_transformer_layer_03"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_transformer_route_missing] / effect_mark_transformer_route_missing,
        "state_transformer_layer_05"_s <= "state_transformer_layer_04"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_native_transformer_route_after_success] / effect_execute_transformer_layer_05_native,
        "state_transformer_layer_05"_s <= "state_transformer_layer_04"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_callback_transformer_route_after_success] / effect_execute_transformer_layer_05_callback,
        "state_error_error_out_decision"_s <= "state_transformer_layer_04"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_transformer_failed],
        "state_error_error_out_decision"_s <= "state_transformer_layer_04"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_transformer_route_missing] / effect_mark_transformer_route_missing,
        "state_transformer_layer_06"_s <= "state_transformer_layer_05"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_native_transformer_route_after_success] / effect_execute_transformer_layer_06_native,
        "state_transformer_layer_06"_s <= "state_transformer_layer_05"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_callback_transformer_route_after_success] / effect_execute_transformer_layer_06_callback,
        "state_error_error_out_decision"_s <= "state_transformer_layer_05"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_transformer_failed],
        "state_error_error_out_decision"_s <= "state_transformer_layer_05"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_transformer_route_missing] / effect_mark_transformer_route_missing,
        "state_transformer_layer_07"_s <= "state_transformer_layer_06"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_native_transformer_route_after_success] / effect_execute_transformer_layer_07_native,
        "state_transformer_layer_07"_s <= "state_transformer_layer_06"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_callback_transformer_route_after_success] / effect_execute_transformer_layer_07_callback,
        "state_error_error_out_decision"_s <= "state_transformer_layer_06"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_transformer_failed],
        "state_error_error_out_decision"_s <= "state_transformer_layer_06"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_transformer_route_missing] / effect_mark_transformer_route_missing,
        "state_transformer_layer_08"_s <= "state_transformer_layer_07"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_native_transformer_route_after_success] / effect_execute_transformer_layer_08_native,
        "state_transformer_layer_08"_s <= "state_transformer_layer_07"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_callback_transformer_route_after_success] / effect_execute_transformer_layer_08_callback,
        "state_error_error_out_decision"_s <= "state_transformer_layer_07"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_transformer_failed],
        "state_error_error_out_decision"_s <= "state_transformer_layer_07"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_transformer_route_missing] / effect_mark_transformer_route_missing,
        "state_transformer_layer_09"_s <= "state_transformer_layer_08"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_native_transformer_route_after_success] / effect_execute_transformer_layer_09_native,
        "state_transformer_layer_09"_s <= "state_transformer_layer_08"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_callback_transformer_route_after_success] / effect_execute_transformer_layer_09_callback,
        "state_error_error_out_decision"_s <= "state_transformer_layer_08"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_transformer_failed],
        "state_error_error_out_decision"_s <= "state_transformer_layer_08"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_transformer_route_missing] / effect_mark_transformer_route_missing,
        "state_transformer_layer_10"_s <= "state_transformer_layer_09"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_native_transformer_route_after_success] / effect_execute_transformer_layer_10_native,
        "state_transformer_layer_10"_s <= "state_transformer_layer_09"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_callback_transformer_route_after_success] / effect_execute_transformer_layer_10_callback,
        "state_error_error_out_decision"_s <= "state_transformer_layer_09"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_transformer_failed],
        "state_error_error_out_decision"_s <= "state_transformer_layer_09"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_transformer_route_missing] / effect_mark_transformer_route_missing,
        "state_transformer_layer_11"_s <= "state_transformer_layer_10"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_native_transformer_route_after_success] / effect_execute_transformer_layer_11_native,
        "state_transformer_layer_11"_s <= "state_transformer_layer_10"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_callback_transformer_route_after_success] / effect_execute_transformer_layer_11_callback,
        "state_error_error_out_decision"_s <= "state_transformer_layer_10"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_transformer_failed],
        "state_error_error_out_decision"_s <= "state_transformer_layer_10"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_transformer_route_missing] / effect_mark_transformer_route_missing,
        "state_transformer_layer_12"_s <= "state_transformer_layer_11"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_native_transformer_route_after_success] / effect_execute_transformer_layer_12_native,
        "state_transformer_layer_12"_s <= "state_transformer_layer_11"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_callback_transformer_route_after_success] / effect_execute_transformer_layer_12_callback,
        "state_error_error_out_decision"_s <= "state_transformer_layer_11"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_transformer_failed],
        "state_error_error_out_decision"_s <= "state_transformer_layer_11"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_transformer_route_missing] / effect_mark_transformer_route_missing,
        "state_transformer_layer_13"_s <= "state_transformer_layer_12"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_native_transformer_route_after_success] / effect_execute_transformer_layer_13_native,
        "state_transformer_layer_13"_s <= "state_transformer_layer_12"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_callback_transformer_route_after_success] / effect_execute_transformer_layer_13_callback,
        "state_error_error_out_decision"_s <= "state_transformer_layer_12"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_transformer_failed],
        "state_error_error_out_decision"_s <= "state_transformer_layer_12"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_transformer_route_missing] / effect_mark_transformer_route_missing,
        "state_transformer_layer_14"_s <= "state_transformer_layer_13"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_native_transformer_route_after_success] / effect_execute_transformer_layer_14_native,
        "state_transformer_layer_14"_s <= "state_transformer_layer_13"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_callback_transformer_route_after_success] / effect_execute_transformer_layer_14_callback,
        "state_error_error_out_decision"_s <= "state_transformer_layer_13"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_transformer_failed],
        "state_error_error_out_decision"_s <= "state_transformer_layer_13"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_transformer_route_missing] / effect_mark_transformer_route_missing,
        "state_transformer_layer_15"_s <= "state_transformer_layer_14"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_native_transformer_route_after_success] / effect_execute_transformer_layer_15_native,
        "state_transformer_layer_15"_s <= "state_transformer_layer_14"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_callback_transformer_route_after_success] / effect_execute_transformer_layer_15_callback,
        "state_error_error_out_decision"_s <= "state_transformer_layer_14"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_transformer_failed],
        "state_error_error_out_decision"_s <= "state_transformer_layer_14"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_transformer_route_missing] / effect_mark_transformer_route_missing,
        "state_transformer_layer_16"_s <= "state_transformer_layer_15"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_native_transformer_route_after_success] / effect_execute_transformer_layer_16_native,
        "state_transformer_layer_16"_s <= "state_transformer_layer_15"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_callback_transformer_route_after_success] / effect_execute_transformer_layer_16_callback,
        "state_error_error_out_decision"_s <= "state_transformer_layer_15"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_transformer_failed],
        "state_error_error_out_decision"_s <= "state_transformer_layer_15"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_transformer_route_missing] / effect_mark_transformer_route_missing,
        "state_transformer_layer_17"_s <= "state_transformer_layer_16"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_native_transformer_route_after_success] / effect_execute_transformer_layer_17_native,
        "state_transformer_layer_17"_s <= "state_transformer_layer_16"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_callback_transformer_route_after_success] / effect_execute_transformer_layer_17_callback,
        "state_error_error_out_decision"_s <= "state_transformer_layer_16"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_transformer_failed],
        "state_error_error_out_decision"_s <= "state_transformer_layer_16"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_transformer_route_missing] / effect_mark_transformer_route_missing,
        "state_publishing_hidden"_s <= "state_transformer_layer_17"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_transformer_succeeded] / effect_publish_hidden,
        "state_error_error_out_decision"_s <= "state_transformer_layer_17"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_transformer_failed],
        "state_success_error_out_decision"_s <= "state_publishing_hidden"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>),
        "state_success_callback_decision"_s <= "state_success_error_out_decision"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_has_error_out] / effect_store_success_error,
        "state_success_callback_decision"_s <= "state_success_error_out_decision"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_no_error_out],
        "state_error_callback_decision"_s <= "state_error_error_out_decision"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_has_error_out] / effect_store_error_error,
        "state_error_callback_decision"_s <= "state_error_error_out_decision"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_no_error_out],
        "state_done"_s <= "state_success_callback_decision"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_has_done_callback] / effect_emit_done,
        "state_done"_s <= "state_success_callback_decision"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_no_done_callback],
        "state_errored"_s <= "state_error_callback_decision"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_has_error_callback] / effect_emit_error,
        "state_errored"_s <= "state_error_callback_decision"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>) [guard_no_error_callback],
        "state_ready"_s <= "state_done"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>),
        "state_ready"_s <= "state_errored"_s + completion<EventExecuteRun>(&'dispatch EventExecuteRun<'event>),
        "state_ready"_s <= "state_ready"_s + unexpected_event<_> / effect_on_unexpected_from_state_ready,
        "state_ready"_s <= "state_model_contract_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_model_contract_decision,
        "state_ready"_s <= "state_tensor_contract_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_tensor_contract_decision,
        "state_ready"_s <= "state_input_shape_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_input_shape_decision,
        "state_ready"_s <= "state_output_capacity_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_output_capacity_decision,
        "state_ready"_s <= "state_binding"_s + unexpected_event<_> / effect_on_unexpected_from_state_binding,
        "state_ready"_s <= "state_native_projecting"_s + unexpected_event<_> / effect_on_unexpected_from_state_native_projecting,
        "state_ready"_s <= "state_callback_projecting"_s + unexpected_event<_> / effect_on_unexpected_from_state_callback_projecting,
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
        "state_ready"_s <= "state_transformer_layer_16"_s + unexpected_event<_> / effect_on_unexpected_state_transformer_layer_16,
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
impl DiarizationSortformerExecutorContext {
    fn execute_layer_a_to_b_native(
        &mut self,
        event: &EventExecuteRun<'_>,
        layer: usize,
    ) -> Result<(), NativeTransformerError> {
        self.hidden_b.fill(0.0);
        let route = event
            .native_transformer_route
            .borrow_mut()
            .take()
            .expect("native transformer route selected by guard");
        let result = route.execute(layer, &self.hidden_a, &mut self.hidden_b);
        event.native_transformer_route.borrow_mut().replace(route);
        result
    }
    fn execute_layer_a_to_b_callback(&mut self, event: &EventExecuteRun<'_>, layer: usize) -> bool {
        self.hidden_b.fill(0.0);
        (event
            .contract
            .transformer_layer
            .expect("callback transformer route selected by guard"))(
            layer,
            &self.hidden_a,
            event.contract,
            &mut self.hidden_b,
        )
    }
    fn execute_layer_b_to_a_native(
        &mut self,
        event: &EventExecuteRun<'_>,
        layer: usize,
    ) -> Result<(), NativeTransformerError> {
        self.hidden_a.fill(0.0);
        let route = event
            .native_transformer_route
            .borrow_mut()
            .take()
            .expect("native transformer route selected by guard");
        let result = route.execute(layer, &self.hidden_b, &mut self.hidden_a);
        event.native_transformer_route.borrow_mut().replace(route);
        result
    }
    fn execute_layer_b_to_a_callback(&mut self, event: &EventExecuteRun<'_>, layer: usize) -> bool {
        self.hidden_a.fill(0.0);
        (event
            .contract
            .transformer_layer
            .expect("callback transformer route selected by guard"))(
            layer,
            &self.hidden_b,
            event.contract,
            &mut self.hidden_a,
        )
    }
    fn effect_execute_transformer_layer_00_native<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_a_to_b_native(e, 0).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_00_callback<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_a_to_b_callback(e, 0).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_01_native<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_b_to_a_native(e, 1).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_01_callback<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_b_to_a_callback(e, 1).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_02_native<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_a_to_b_native(e, 2).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_02_callback<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_a_to_b_callback(e, 2).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_03_native<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_b_to_a_native(e, 3).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_03_callback<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_b_to_a_callback(e, 3).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_04_native<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_a_to_b_native(e, 4).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_04_callback<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_a_to_b_callback(e, 4).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_05_native<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_b_to_a_native(e, 5).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_05_callback<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_b_to_a_callback(e, 5).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_06_native<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_a_to_b_native(e, 6).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_06_callback<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_a_to_b_callback(e, 6).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_07_native<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_b_to_a_native(e, 7).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_07_callback<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_b_to_a_callback(e, 7).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_08_native<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_a_to_b_native(e, 8).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_08_callback<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_a_to_b_callback(e, 8).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_09_native<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_b_to_a_native(e, 9).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_09_callback<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_b_to_a_callback(e, 9).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_10_native<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_a_to_b_native(e, 10).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_10_callback<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_a_to_b_callback(e, 10).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_11_native<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_b_to_a_native(e, 11).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_11_callback<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_b_to_a_callback(e, 11).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_12_native<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_a_to_b_native(e, 12).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_12_callback<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_a_to_b_callback(e, 12).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_13_native<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_b_to_a_native(e, 13).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_13_callback<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_b_to_a_callback(e, 13).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_14_native<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_a_to_b_native(e, 14).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_14_callback<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_a_to_b_callback(e, 14).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_15_native<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_b_to_a_native(e, 15).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_15_callback<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_b_to_a_callback(e, 15).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_16_native<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_a_to_b_native(e, 16).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_16_callback<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_a_to_b_callback(e, 16).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_17_native<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_b_to_a_native(e, 17).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_17_callback<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_b_to_a_callback(e, 17).into();
        Ok(())
    }

    #[allow(clippy::unnecessary_wraps)]
    fn unexpected(&mut self) -> Result<(), ()> {
        self.err = Error::Unexpected;
        Ok(())
    }
}

/// Actor-owned persistent executor state.
///
/// This context is not a caller event and must not be shared across executor
/// instances or dispatches. Its fixed-capacity work buffers are retained by the
/// single-writer actor between calls. Dispatch-local phase outcomes are
/// carried by `EventExecuteRun` and are never stored on this context.
#[derive(Debug)]
pub struct DiarizationSortformerExecutorContext {
    err: Error,
    hidden_a: [f32; REQUIRED_HIDDEN_VALUE_COUNT],
    hidden_b: [f32; REQUIRED_HIDDEN_VALUE_COUNT],
    cache: [f32; REQUIRED_HIDDEN_VALUE_COUNT],
    cache_frame_count: i32,
}

impl Default for DiarizationSortformerExecutorContext {
    #[allow(
        clippy::large_stack_arrays,
        clippy::large_stack_frames,
        reason = "fixed-capacity hidden and cache buffers preserve allocation-free bounded dispatch"
    )]
    fn default() -> Self {
        Self {
            err: Error::None,
            hidden_a: [0.0; REQUIRED_HIDDEN_VALUE_COUNT],
            hidden_b: [0.0; REQUIRED_HIDDEN_VALUE_COUNT],
            cache: [0.0; REQUIRED_HIDDEN_VALUE_COUNT],
            cache_frame_count: 0,
        }
    }
}

impl DiarizationSortformerExecutorStateMachineContext for DiarizationSortformerExecutorContext {
    fn effect_begin_execute<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *event.projection_outcome.borrow_mut() = PhaseOutcome::Pending;
        *event.transformer_outcome.borrow_mut() = PhaseOutcome::Pending;
        self.cache.fill(0.0);
        self.cache_frame_count = 0;
        **event.frame_count_out.borrow_mut() = 0;
        **event.hidden_dim_out.borrow_mut() = 0;
        Ok(())
    }
    fn effect_bind_contracts<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *event.contracts_bound.borrow_mut() = event.contract.model_contract_valid()
            && (event.native_projection_route.borrow().is_some()
                || event.contract.project_encoder.is_some());
        Ok(())
    }
    fn effect_project_encoder_native<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.hidden_a.fill(0.0);
        let route = event
            .native_projection_route
            .borrow_mut()
            .take()
            .expect("native projection route selected by guard");
        let result = route.project(event.encoder_frames, &mut self.hidden_a);
        event.native_projection_route.borrow_mut().replace(route);
        *event.projection_outcome.borrow_mut() = result.into();
        Ok(())
    }
    fn effect_project_encoder_callback<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.hidden_a.fill(0.0);
        let project = event
            .contract
            .project_encoder
            .expect("callback projection route selected by guard");
        *event.projection_outcome.borrow_mut() =
            project(event.encoder_frames, event.contract, &mut self.hidden_a).into();
        Ok(())
    }
    fn effect_mark_projection_route_missing<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.err = Error::ProjectionRoute;
        Ok(())
    }
    fn effect_write_projected_frames_to_cache<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        for frame in 0..FRAME_COUNT {
            let offset = frame * HIDDEN_DIM;
            self.cache[offset..offset + HIDDEN_DIM]
                .copy_from_slice(&self.hidden_a[offset..offset + HIDDEN_DIM]);
            self.cache_frame_count = self
                .cache_frame_count
                .max(i32::try_from(frame + 1).expect("fixed frame count fits i32"));
        }
        Ok(())
    }
    fn effect_execute_transformer_layer_00_native<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_a_to_b_native(e, 0).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_00_callback<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_a_to_b_callback(e, 0).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_01_native<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_b_to_a_native(e, 1).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_01_callback<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_b_to_a_callback(e, 1).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_02_native<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_a_to_b_native(e, 2).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_02_callback<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_a_to_b_callback(e, 2).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_03_native<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_b_to_a_native(e, 3).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_03_callback<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_b_to_a_callback(e, 3).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_04_native<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_a_to_b_native(e, 4).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_04_callback<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_a_to_b_callback(e, 4).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_05_native<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_b_to_a_native(e, 5).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_05_callback<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_b_to_a_callback(e, 5).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_06_native<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_a_to_b_native(e, 6).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_06_callback<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_a_to_b_callback(e, 6).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_07_native<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_b_to_a_native(e, 7).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_07_callback<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_b_to_a_callback(e, 7).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_08_native<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_a_to_b_native(e, 8).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_08_callback<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_a_to_b_callback(e, 8).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_09_native<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_b_to_a_native(e, 9).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_09_callback<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_b_to_a_callback(e, 9).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_10_native<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_a_to_b_native(e, 10).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_10_callback<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_a_to_b_callback(e, 10).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_11_native<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_b_to_a_native(e, 11).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_11_callback<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_b_to_a_callback(e, 11).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_12_native<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_a_to_b_native(e, 12).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_12_callback<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_a_to_b_callback(e, 12).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_13_native<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_b_to_a_native(e, 13).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_13_callback<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_b_to_a_callback(e, 13).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_14_native<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_a_to_b_native(e, 14).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_14_callback<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_a_to_b_callback(e, 14).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_15_native<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_b_to_a_native(e, 15).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_15_callback<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_b_to_a_callback(e, 15).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_16_native<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_a_to_b_native(e, 16).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_16_callback<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_a_to_b_callback(e, 16).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_17_native<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_b_to_a_native(e, 17).into();
        Ok(())
    }
    fn effect_execute_transformer_layer_17_callback<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        *e.transformer_outcome.borrow_mut() = self.execute_layer_b_to_a_callback(e, 17).into();
        Ok(())
    }
    fn effect_publish_hidden<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let final_hidden = &self.hidden_a;
        let mut hidden_out = event.hidden_out.borrow_mut();
        (**hidden_out)[..REQUIRED_HIDDEN_VALUE_COUNT].copy_from_slice(final_hidden);
        **event.frame_count_out.borrow_mut() = FRAME_COUNT_I32;
        **event.hidden_dim_out.borrow_mut() = HIDDEN_DIM_I32;
        Ok(())
    }
    fn effect_store_success_error<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        if let Some(out) = event.error_out.borrow_mut().as_deref_mut() {
            *out = self.err;
        }
        Ok(())
    }
    fn effect_store_error_error<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        if let PhaseOutcome::Failed(error) = *event.projection_outcome.borrow() {
            self.err = error;
        }
        if let PhaseOutcome::Failed(error) = *event.transformer_outcome.borrow() {
            self.err = error;
        }
        if let Some(out) = event.error_out.borrow_mut().as_deref_mut() {
            *out = self.err;
        }
        Ok(())
    }
    fn effect_emit_done<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        if let Some(callback) = event.on_done {
            sortformer_event(
                "emel.sortformer.callback",
                "callback",
                "callback",
                "done_start",
            );
            let _ = callback(ExecuteDone {
                frame_count: **event.frame_count_out.borrow(),
                hidden_dim: **event.hidden_dim_out.borrow(),
            });
            sortformer_event(
                "emel.sortformer.callback",
                "callback",
                "callback",
                "done_end",
            );
        }
        Ok(())
    }
    fn effect_emit_error<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        if let Some(callback) = event.on_error {
            sortformer_event(
                "emel.sortformer.callback",
                "callback",
                "callback",
                "error_start",
            );
            let _ = callback(ExecuteError { error: self.err });
            sortformer_event(
                "emel.sortformer.callback",
                "callback",
                "callback",
                "error_end",
            );
        }
        Ok(())
    }
    fn effect_mark_model_invalid<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.err = Error::ModelInvalid;
        Ok(())
    }
    fn effect_mark_tensor_contract_invalid<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.err = Error::TensorContract;
        Ok(())
    }
    fn effect_mark_input_shape_invalid<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.err = Error::InputShape;
        Ok(())
    }
    fn effect_mark_output_capacity_invalid<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.err = Error::OutputCapacity;
        Ok(())
    }
    fn guard_model_contract_valid<'dispatch, 'event>(
        &self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(e.contract.model_contract_valid())
    }
    fn guard_model_contract_invalid<'dispatch, 'event>(
        &self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!e.contract.model_contract_valid())
    }
    fn guard_tensor_contract_valid<'dispatch, 'event>(
        &self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(e.contract.encoder.tensor_count != 0
            && e.contract.modules.tensor_count != 0
            && e.contract.transformer_encoder.tensor_count != 0)
    }
    fn guard_tensor_contract_invalid<'dispatch, 'event>(
        &self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!self.guard_tensor_contract_valid(e)?)
    }
    fn guard_native_projection_route<'dispatch, 'event>(
        &self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(*e.contracts_bound.borrow() && e.native_projection_route.borrow().is_some())
    }
    fn guard_callback_projection_route<'dispatch, 'event>(
        &self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(*e.contracts_bound.borrow()
            && e.native_projection_route.borrow().is_none()
            && e.contract.project_encoder.is_some())
    }
    fn guard_projection_route_missing<'dispatch, 'event>(
        &self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!*e.contracts_bound.borrow()
            || (e.native_projection_route.borrow().is_none()
                && e.contract.project_encoder.is_none()))
    }
    fn guard_input_shape_valid<'dispatch, 'event>(
        &self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!e.encoder_frames.is_empty() && e.encoder_frames.len() == REQUIRED_ENCODER_VALUE_COUNT)
    }
    fn guard_input_shape_invalid<'dispatch, 'event>(
        &self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!self.guard_input_shape_valid(e)?)
    }
    fn guard_output_capacity_valid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventExecuteRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok((**event.hidden_out.borrow()).len() >= REQUIRED_HIDDEN_VALUE_COUNT)
    }
    fn guard_native_transformer_route<'dispatch, 'event>(
        &self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(
            matches!(*e.transformer_outcome.borrow(), PhaseOutcome::Pending)
                && e.native_transformer_route.borrow().is_some(),
        )
    }
    fn guard_callback_transformer_route<'dispatch, 'event>(
        &self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(
            matches!(*e.transformer_outcome.borrow(), PhaseOutcome::Pending)
                && e.native_transformer_route.borrow().is_none()
                && e.contract.transformer_layer.is_some(),
        )
    }
    fn guard_transformer_route_missing<'dispatch, 'event>(
        &self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(
            matches!(*e.transformer_outcome.borrow(), PhaseOutcome::Pending)
                && e.native_transformer_route.borrow().is_none()
                && e.contract.transformer_layer.is_none(),
        )
    }
    fn effect_mark_transformer_route_missing<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventExecuteRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.err = Error::Kernel;
        Ok(())
    }
    fn guard_projection_succeeded<'dispatch, 'event>(
        &self,
        event: &'dispatch EventExecuteRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(matches!(
            *event.projection_outcome.borrow(),
            PhaseOutcome::Succeeded
        ))
    }
    fn guard_projection_failed<'dispatch, 'event>(
        &self,
        event: &'dispatch EventExecuteRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(matches!(
            *event.projection_outcome.borrow(),
            PhaseOutcome::Failed(_)
        ))
    }
    fn guard_transformer_succeeded<'dispatch, 'event>(
        &self,
        event: &'dispatch EventExecuteRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(matches!(
            *event.transformer_outcome.borrow(),
            PhaseOutcome::Succeeded
        ))
    }
    fn guard_transformer_failed<'dispatch, 'event>(
        &self,
        event: &'dispatch EventExecuteRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(matches!(
            *event.transformer_outcome.borrow(),
            PhaseOutcome::Failed(_)
        ))
    }
    fn guard_native_transformer_route_after_success<'dispatch, 'event>(
        &self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(self.guard_transformer_succeeded(e)? && e.native_transformer_route.borrow().is_some())
    }
    fn guard_callback_transformer_route_after_success<'dispatch, 'event>(
        &self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(self.guard_transformer_succeeded(e)?
            && e.native_transformer_route.borrow().is_none()
            && e.contract.transformer_layer.is_some())
    }
    fn guard_output_capacity_invalid<'dispatch, 'event>(
        &self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!self.guard_output_capacity_valid(e)?)
    }
    fn guard_has_error_out<'dispatch, 'event>(
        &self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(e.error_out.borrow().is_some())
    }
    fn guard_no_error_out<'dispatch, 'event>(
        &self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!self.guard_has_error_out(e)?)
    }
    fn guard_has_done_callback<'dispatch, 'event>(
        &self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(e.on_done.is_some())
    }
    fn guard_no_done_callback<'dispatch, 'event>(
        &self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(e.on_done.is_none())
    }
    fn guard_has_error_callback<'dispatch, 'event>(
        &self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(e.on_error.is_some())
    }
    fn guard_no_error_callback<'dispatch, 'event>(
        &self,
        e: &'dispatch EventExecuteRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(e.on_error.is_none())
    }
    fn effect_on_unexpected_from_state_ready(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_model_contract_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_tensor_contract_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_input_shape_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_output_capacity_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_binding(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_native_projecting(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_callback_projecting(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_transformer_cache(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_transformer_layer_00(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_transformer_layer_01(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_transformer_layer_02(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_transformer_layer_03(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_transformer_layer_04(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_transformer_layer_05(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_transformer_layer_06(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_transformer_layer_07(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_transformer_layer_08(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_transformer_layer_09(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_transformer_layer_10(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_transformer_layer_11(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_transformer_layer_12(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_transformer_layer_13(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_transformer_layer_14(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_transformer_layer_15(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_state_transformer_layer_16(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_transformer_layer_17(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_publishing_hidden(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_success_error_out_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_success_callback_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_error_error_out_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_error_callback_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_done(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_errored(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
}

/// Generated state-machine alias matching the pinned actor.
pub type ExecutorStateMachine =
    DiarizationSortformerExecutorStateMachine<DiarizationSortformerExecutorContext>;

/// Single-writer synchronous Sortformer executor actor.
pub struct Executor {
    machine: ExecutorStateMachine,
}

impl Default for Executor {
    fn default() -> Self {
        Self::new()
    }
}

impl Executor {
    /// Constructs an executor in generated `state_ready`.
    #[allow(
        clippy::large_stack_arrays,
        clippy::large_stack_frames,
        reason = "the executor context is intentionally fixed-capacity and allocation-free"
    )]
    pub fn new() -> Self {
        Self {
            machine: ExecutorStateMachine::new(DiarizationSortformerExecutorContext::default()),
        }
    }

    /// Dispatches one borrowed request synchronously through every phase.
    pub fn execute(&mut self, event: &EventExecuteRun<'_>) -> bool {
        global::tracer(SORTFORMER_TRACER).in_span(SORTFORMER_OPERATION, |_cx| {
            let projection_route = if event.native_projection_route.borrow().is_some() {
                "native_projection"
            } else if event.contract.project_encoder.is_some() {
                "callback_projection"
            } else {
                "none"
            };
            let transformer_route = if event.native_transformer_route.borrow().is_some() {
                "native_transformer"
            } else if event.contract.transformer_layer.is_some() {
                "callback_transformer"
            } else {
                "none"
            };
            sortformer_event(
                "emel.sortformer.control",
                "none",
                "control",
                "dispatch_start",
            );
            sortformer_event(
                "emel.sortformer.boundary",
                projection_route,
                "boundary",
                "projection_start",
            );
            sortformer_event(
                "emel.sortformer.boundary",
                transformer_route,
                "boundary",
                "transformer_start",
            );
            let success = self.process_event(event);
            sortformer_result(success);
            success
        })
    }

    /// Source-compatible process-event spelling for the pinned actor.
    pub fn process_event(&mut self, event: &EventExecuteRun<'_>) -> bool {
        if self
            .machine
            .process_event(DiarizationSortformerExecutorEvents::EventExecuteRun(event))
            .is_err()
        {
            return false;
        }
        self.machine.context().err == Error::None
    }

    /// Explicit unexpected-event path used by lifecycle recovery tests.
    pub fn process_unexpected_event(&mut self) -> bool {
        self.machine.context_mut().unexpected().is_ok()
    }

    /// Returns generated state inspection data.
    #[must_use]
    pub fn state(&self) -> &DiarizationSortformerExecutorStates {
        self.machine.state()
    }

    /// Returns the most recent execution error.
    #[must_use]
    pub fn error(&self) -> Error {
        self.machine.context().err
    }
}
