//! State machine scaffold port — not a stable public API.
//! Bodies are stubs (`todo!`) until contexts/guards/actions are ported from C++.

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
    missing_docs
)]

use sml::sml;

// --- machine GraphTensor from emel.cpp/src/emel/graph/tensor/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct DetailCaptureTensorStateRuntime;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct DetailPublishFilledTensorRuntime;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct DetailReleaseTensorRefRuntime;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct DetailReserveTensorRuntime;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct DetailResetTensorEpochRuntime;

sml! {
    GraphTensor {
        "reserve_tensor_request_decision"_s <= *"ready"_s + event<DetailReserveTensorRuntime> / begin_reserve_tensor,
        "reserve_tensor_exec"_s <= "reserve_tensor_request_decision"_s + completion<DetailReserveTensorRuntime> [reserve_tensor_request_valid],
        "errored"_s <= "reserve_tensor_request_decision"_s + completion<DetailReserveTensorRuntime> [reserve_tensor_request_invalid] / mark_invalid_request_detail_reserve_tensor_runtime,
        "reserve_tensor_result_decision"_s <= "reserve_tensor_exec"_s + completion<DetailReserveTensorRuntime> / exec_reserve_tensor,
        "done"_s <= "reserve_tensor_result_decision"_s + completion<DetailReserveTensorRuntime> [operation_succeeded_detail_reserve_tensor_runtime],
        "errored"_s <= "reserve_tensor_result_decision"_s + completion<DetailReserveTensorRuntime> [operation_failed_internal_detail_reserve_tensor_runtime] / mark_internal_error_detail_reserve_tensor_runtime,
        "errored"_s <= "reserve_tensor_result_decision"_s + completion<DetailReserveTensorRuntime> [operation_not_dispatched_detail_reserve_tensor_runtime] / mark_internal_error_detail_reserve_tensor_runtime,
        "publish_filled_tensor_request_decision"_s <= "ready"_s + event<DetailPublishFilledTensorRuntime> / begin_publish_filled_tensor,
        "publish_filled_tensor_exec"_s <= "publish_filled_tensor_request_decision"_s + completion<DetailPublishFilledTensorRuntime> [publish_filled_tensor_request_valid],
        "errored"_s <= "publish_filled_tensor_request_decision"_s + completion<DetailPublishFilledTensorRuntime> [publish_filled_tensor_request_invalid] / mark_invalid_request_detail_publish_filled_tensor_runtime,
        "publish_filled_tensor_result_decision"_s <= "publish_filled_tensor_exec"_s + completion<DetailPublishFilledTensorRuntime> / exec_publish_filled_tensor,
        "done"_s <= "publish_filled_tensor_result_decision"_s + completion<DetailPublishFilledTensorRuntime> [operation_succeeded_detail_publish_filled_tensor_runtime],
        "errored"_s <= "publish_filled_tensor_result_decision"_s + completion<DetailPublishFilledTensorRuntime> [operation_failed_internal_detail_publish_filled_tensor_runtime] / mark_internal_error_detail_publish_filled_tensor_runtime,
        "errored"_s <= "publish_filled_tensor_result_decision"_s + completion<DetailPublishFilledTensorRuntime> [operation_not_dispatched_detail_publish_filled_tensor_runtime] / mark_internal_error_detail_publish_filled_tensor_runtime,
        "release_tensor_ref_request_decision"_s <= "ready"_s + event<DetailReleaseTensorRefRuntime> / begin_release_tensor_ref,
        "release_tensor_ref_exec"_s <= "release_tensor_ref_request_decision"_s + completion<DetailReleaseTensorRefRuntime> [release_tensor_ref_request_valid],
        "errored"_s <= "release_tensor_ref_request_decision"_s + completion<DetailReleaseTensorRefRuntime> [release_tensor_ref_request_invalid] / mark_invalid_request_detail_release_tensor_ref_runtime,
        "release_tensor_ref_result_decision"_s <= "release_tensor_ref_exec"_s + completion<DetailReleaseTensorRefRuntime> / exec_release_tensor_ref,
        "done"_s <= "release_tensor_ref_result_decision"_s + completion<DetailReleaseTensorRefRuntime> [operation_succeeded_detail_release_tensor_ref_runtime],
        "errored"_s <= "release_tensor_ref_result_decision"_s + completion<DetailReleaseTensorRefRuntime> [operation_failed_internal_detail_release_tensor_ref_runtime] / mark_internal_error_detail_release_tensor_ref_runtime,
        "errored"_s <= "release_tensor_ref_result_decision"_s + completion<DetailReleaseTensorRefRuntime> [operation_not_dispatched_detail_release_tensor_ref_runtime] / mark_internal_error_detail_release_tensor_ref_runtime,
        "reset_tensor_epoch_request_decision"_s <= "ready"_s + event<DetailResetTensorEpochRuntime> / begin_reset_tensor_epoch,
        "reset_tensor_epoch_exec"_s <= "reset_tensor_epoch_request_decision"_s + completion<DetailResetTensorEpochRuntime> [reset_tensor_epoch_request_valid],
        "errored"_s <= "reset_tensor_epoch_request_decision"_s + completion<DetailResetTensorEpochRuntime> [reset_tensor_epoch_request_invalid] / mark_invalid_request_detail_reset_tensor_epoch_runtime,
        "reset_tensor_epoch_result_decision"_s <= "reset_tensor_epoch_exec"_s + completion<DetailResetTensorEpochRuntime> / exec_reset_tensor_epoch,
        "done"_s <= "reset_tensor_epoch_result_decision"_s + completion<DetailResetTensorEpochRuntime> [operation_succeeded_detail_reset_tensor_epoch_runtime],
        "errored"_s <= "reset_tensor_epoch_result_decision"_s + completion<DetailResetTensorEpochRuntime> [operation_failed_internal_detail_reset_tensor_epoch_runtime] / mark_internal_error_detail_reset_tensor_epoch_runtime,
        "errored"_s <= "reset_tensor_epoch_result_decision"_s + completion<DetailResetTensorEpochRuntime> [operation_not_dispatched_detail_reset_tensor_epoch_runtime] / mark_internal_error_detail_reset_tensor_epoch_runtime,
        "capture_tensor_state_request_decision"_s <= "ready"_s + event<DetailCaptureTensorStateRuntime> / begin_capture_tensor_state,
        "capture_tensor_state_exec"_s <= "capture_tensor_state_request_decision"_s + completion<DetailCaptureTensorStateRuntime> [capture_tensor_state_request_valid],
        "errored"_s <= "capture_tensor_state_request_decision"_s + completion<DetailCaptureTensorStateRuntime> [capture_tensor_state_request_invalid] / mark_invalid_request_detail_capture_tensor_state_runtime,
        "capture_tensor_state_result_decision"_s <= "capture_tensor_state_exec"_s + completion<DetailCaptureTensorStateRuntime> / exec_capture_tensor_state,
        "done"_s <= "capture_tensor_state_result_decision"_s + completion<DetailCaptureTensorStateRuntime> [capture_operation_succeeded],
        "errored"_s <= "capture_tensor_state_result_decision"_s + completion<DetailCaptureTensorStateRuntime> [capture_operation_not_dispatched] / mark_internal_error_detail_capture_tensor_state_runtime,
        "ready"_s <= "done"_s + completion<DetailReserveTensorRuntime> / publish_done_detail_reserve_tensor_runtime,
        "ready"_s <= "errored"_s + completion<DetailReserveTensorRuntime> / publish_error_detail_reserve_tensor_runtime,
        "ready"_s <= "done"_s + completion<DetailPublishFilledTensorRuntime> / publish_done_detail_publish_filled_tensor_runtime,
        "ready"_s <= "errored"_s + completion<DetailPublishFilledTensorRuntime> / publish_error_detail_publish_filled_tensor_runtime,
        "ready"_s <= "done"_s + completion<DetailReleaseTensorRefRuntime> / publish_done_detail_release_tensor_ref_runtime,
        "ready"_s <= "errored"_s + completion<DetailReleaseTensorRefRuntime> / publish_error_detail_release_tensor_ref_runtime,
        "ready"_s <= "done"_s + completion<DetailResetTensorEpochRuntime> / publish_done_detail_reset_tensor_epoch_runtime,
        "ready"_s <= "errored"_s + completion<DetailResetTensorEpochRuntime> / publish_error_detail_reset_tensor_epoch_runtime,
        "ready"_s <= "done"_s + completion<DetailCaptureTensorStateRuntime> / publish_done_detail_capture_tensor_state_runtime,
        "ready"_s <= "errored"_s + completion<DetailCaptureTensorStateRuntime> / publish_error_detail_capture_tensor_state_runtime,
        "ready"_s <= "ready"_s + unexpected_event<_> / on_unexpected_from_ready,
        "ready"_s <= "reserve_tensor_request_decision"_s + unexpected_event<_> / on_unexpected_from_reserve_tensor_request_decision,
        "ready"_s <= "reserve_tensor_exec"_s + unexpected_event<_> / on_unexpected_from_reserve_tensor_exec,
        "ready"_s <= "reserve_tensor_result_decision"_s + unexpected_event<_> / on_unexpected_from_reserve_tensor_result_decision,
        "ready"_s <= "publish_filled_tensor_request_decision"_s + unexpected_event<_> / on_unexpected_from_publish_filled_tensor_request_decision,
        "ready"_s <= "publish_filled_tensor_exec"_s + unexpected_event<_> / on_unexpected_from_publish_filled_tensor_exec,
        "ready"_s <= "publish_filled_tensor_result_decision"_s + unexpected_event<_> / on_unexpected_from_publish_filled_tensor_result_decision,
        "ready"_s <= "release_tensor_ref_request_decision"_s + unexpected_event<_> / on_unexpected_from_release_tensor_ref_request_decision,
        "ready"_s <= "release_tensor_ref_exec"_s + unexpected_event<_> / on_unexpected_from_release_tensor_ref_exec,
        "ready"_s <= "release_tensor_ref_result_decision"_s + unexpected_event<_> / on_unexpected_from_release_tensor_ref_result_decision,
        "ready"_s <= "reset_tensor_epoch_request_decision"_s + unexpected_event<_> / on_unexpected_from_reset_tensor_epoch_request_decision,
        "ready"_s <= "reset_tensor_epoch_exec"_s + unexpected_event<_> / on_unexpected_from_reset_tensor_epoch_exec,
        "ready"_s <= "reset_tensor_epoch_result_decision"_s + unexpected_event<_> / on_unexpected_from_reset_tensor_epoch_result_decision,
        "ready"_s <= "capture_tensor_state_request_decision"_s + unexpected_event<_> / on_unexpected_from_capture_tensor_state_request_decision,
        "ready"_s <= "capture_tensor_state_exec"_s + unexpected_event<_> / on_unexpected_from_capture_tensor_state_exec,
        "ready"_s <= "capture_tensor_state_result_decision"_s + unexpected_event<_> / on_unexpected_from_capture_tensor_state_result_decision,
        "ready"_s <= "done"_s + unexpected_event<_> / on_unexpected_from_done,
        "ready"_s <= "errored"_s + unexpected_event<_> / on_unexpected_from_errored,
    }
}

/// Context for `GraphTensor` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct GraphTensorContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl GraphTensorStateMachineContext for GraphTensorContext {
    fn begin_capture_tensor_state(
        &mut self,
        _event: &DetailCaptureTensorStateRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::begin_capture_tensor_state
        todo!(
            "TODO: port action `begin_capture_tensor_state` from emel.cpp/src/emel/graph/tensor/actions.hpp"
        )
    }
    fn begin_publish_filled_tensor(
        &mut self,
        _event: &DetailPublishFilledTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::begin_publish_filled_tensor
        todo!(
            "TODO: port action `begin_publish_filled_tensor` from emel.cpp/src/emel/graph/tensor/actions.hpp"
        )
    }
    fn begin_release_tensor_ref(
        &mut self,
        _event: &DetailReleaseTensorRefRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::begin_release_tensor_ref
        todo!(
            "TODO: port action `begin_release_tensor_ref` from emel.cpp/src/emel/graph/tensor/actions.hpp"
        )
    }
    fn begin_reserve_tensor(&mut self, _event: &DetailReserveTensorRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::begin_reserve_tensor
        todo!(
            "TODO: port action `begin_reserve_tensor` from emel.cpp/src/emel/graph/tensor/actions.hpp"
        )
    }
    fn begin_reset_tensor_epoch(
        &mut self,
        _event: &DetailResetTensorEpochRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::begin_reset_tensor_epoch
        todo!(
            "TODO: port action `begin_reset_tensor_epoch` from emel.cpp/src/emel/graph/tensor/actions.hpp"
        )
    }
    fn capture_operation_not_dispatched(
        &self,
        _event: &DetailCaptureTensorStateRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/guards.hpp::capture_operation_not_dispatched
        todo!(
            "TODO: port guard `capture_operation_not_dispatched` from emel.cpp/src/emel/graph/tensor/guards.hpp"
        )
    }
    fn capture_operation_succeeded(
        &self,
        _event: &DetailCaptureTensorStateRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/guards.hpp::capture_operation_succeeded
        todo!(
            "TODO: port guard `capture_operation_succeeded` from emel.cpp/src/emel/graph/tensor/guards.hpp"
        )
    }
    fn capture_tensor_state_request_invalid(
        &self,
        _event: &DetailCaptureTensorStateRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/guards.hpp::capture_tensor_state_request_invalid
        todo!(
            "TODO: port guard `capture_tensor_state_request_invalid` from emel.cpp/src/emel/graph/tensor/guards.hpp"
        )
    }
    fn capture_tensor_state_request_valid(
        &self,
        _event: &DetailCaptureTensorStateRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/guards.hpp::capture_tensor_state_request_valid
        todo!(
            "TODO: port guard `capture_tensor_state_request_valid` from emel.cpp/src/emel/graph/tensor/guards.hpp"
        )
    }
    fn exec_capture_tensor_state(
        &mut self,
        _event: &DetailCaptureTensorStateRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::exec_capture_tensor_state
        todo!(
            "TODO: port action `exec_capture_tensor_state` from emel.cpp/src/emel/graph/tensor/actions.hpp"
        )
    }
    fn exec_publish_filled_tensor(
        &mut self,
        _event: &DetailPublishFilledTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::exec_publish_filled_tensor
        todo!(
            "TODO: port action `exec_publish_filled_tensor` from emel.cpp/src/emel/graph/tensor/actions.hpp"
        )
    }
    fn exec_release_tensor_ref(
        &mut self,
        _event: &DetailReleaseTensorRefRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::exec_release_tensor_ref
        todo!(
            "TODO: port action `exec_release_tensor_ref` from emel.cpp/src/emel/graph/tensor/actions.hpp"
        )
    }
    fn exec_reserve_tensor(&mut self, _event: &DetailReserveTensorRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::exec_reserve_tensor
        todo!(
            "TODO: port action `exec_reserve_tensor` from emel.cpp/src/emel/graph/tensor/actions.hpp"
        )
    }
    fn exec_reset_tensor_epoch(
        &mut self,
        _event: &DetailResetTensorEpochRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::exec_reset_tensor_epoch
        todo!(
            "TODO: port action `exec_reset_tensor_epoch` from emel.cpp/src/emel/graph/tensor/actions.hpp"
        )
    }
    fn mark_internal_error_detail_capture_tensor_state_runtime(
        &mut self,
        _event: &DetailCaptureTensorStateRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::mark_internal_error
        todo!(
            "TODO: port action `mark_internal_error` from emel.cpp/src/emel/graph/tensor/actions.hpp"
        )
    }
    fn mark_internal_error_detail_publish_filled_tensor_runtime(
        &mut self,
        _event: &DetailPublishFilledTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::mark_internal_error
        todo!(
            "TODO: port action `mark_internal_error` from emel.cpp/src/emel/graph/tensor/actions.hpp"
        )
    }
    fn mark_internal_error_detail_release_tensor_ref_runtime(
        &mut self,
        _event: &DetailReleaseTensorRefRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::mark_internal_error
        todo!(
            "TODO: port action `mark_internal_error` from emel.cpp/src/emel/graph/tensor/actions.hpp"
        )
    }
    fn mark_internal_error_detail_reserve_tensor_runtime(
        &mut self,
        _event: &DetailReserveTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::mark_internal_error
        todo!(
            "TODO: port action `mark_internal_error` from emel.cpp/src/emel/graph/tensor/actions.hpp"
        )
    }
    fn mark_internal_error_detail_reset_tensor_epoch_runtime(
        &mut self,
        _event: &DetailResetTensorEpochRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::mark_internal_error
        todo!(
            "TODO: port action `mark_internal_error` from emel.cpp/src/emel/graph/tensor/actions.hpp"
        )
    }
    fn mark_invalid_request_detail_capture_tensor_state_runtime(
        &mut self,
        _event: &DetailCaptureTensorStateRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/graph/tensor/actions.hpp"
        )
    }
    fn mark_invalid_request_detail_publish_filled_tensor_runtime(
        &mut self,
        _event: &DetailPublishFilledTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/graph/tensor/actions.hpp"
        )
    }
    fn mark_invalid_request_detail_release_tensor_ref_runtime(
        &mut self,
        _event: &DetailReleaseTensorRefRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/graph/tensor/actions.hpp"
        )
    }
    fn mark_invalid_request_detail_reserve_tensor_runtime(
        &mut self,
        _event: &DetailReserveTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/graph/tensor/actions.hpp"
        )
    }
    fn mark_invalid_request_detail_reset_tensor_epoch_runtime(
        &mut self,
        _event: &DetailResetTensorEpochRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/graph/tensor/actions.hpp"
        )
    }
    fn on_unexpected_from_capture_tensor_state_exec(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/tensor/actions.hpp")
    }
    fn on_unexpected_from_capture_tensor_state_request_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/tensor/actions.hpp")
    }
    fn on_unexpected_from_capture_tensor_state_result_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/tensor/actions.hpp")
    }
    fn on_unexpected_from_done(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/tensor/actions.hpp")
    }
    fn on_unexpected_from_errored(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/tensor/actions.hpp")
    }
    fn on_unexpected_from_publish_filled_tensor_exec(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/tensor/actions.hpp")
    }
    fn on_unexpected_from_publish_filled_tensor_request_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/tensor/actions.hpp")
    }
    fn on_unexpected_from_publish_filled_tensor_result_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/tensor/actions.hpp")
    }
    fn on_unexpected_from_ready(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/tensor/actions.hpp")
    }
    fn on_unexpected_from_release_tensor_ref_exec(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/tensor/actions.hpp")
    }
    fn on_unexpected_from_release_tensor_ref_request_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/tensor/actions.hpp")
    }
    fn on_unexpected_from_release_tensor_ref_result_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/tensor/actions.hpp")
    }
    fn on_unexpected_from_reserve_tensor_exec(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/tensor/actions.hpp")
    }
    fn on_unexpected_from_reserve_tensor_request_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/tensor/actions.hpp")
    }
    fn on_unexpected_from_reserve_tensor_result_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/tensor/actions.hpp")
    }
    fn on_unexpected_from_reset_tensor_epoch_exec(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/tensor/actions.hpp")
    }
    fn on_unexpected_from_reset_tensor_epoch_request_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/tensor/actions.hpp")
    }
    fn on_unexpected_from_reset_tensor_epoch_result_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/tensor/actions.hpp")
    }
    fn operation_failed_internal_detail_publish_filled_tensor_runtime(
        &self,
        _event: &DetailPublishFilledTensorRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/guards.hpp::operation_failed_internal
        todo!(
            "TODO: port guard `operation_failed_internal` from emel.cpp/src/emel/graph/tensor/guards.hpp"
        )
    }
    fn operation_failed_internal_detail_release_tensor_ref_runtime(
        &self,
        _event: &DetailReleaseTensorRefRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/guards.hpp::operation_failed_internal
        todo!(
            "TODO: port guard `operation_failed_internal` from emel.cpp/src/emel/graph/tensor/guards.hpp"
        )
    }
    fn operation_failed_internal_detail_reserve_tensor_runtime(
        &self,
        _event: &DetailReserveTensorRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/guards.hpp::operation_failed_internal
        todo!(
            "TODO: port guard `operation_failed_internal` from emel.cpp/src/emel/graph/tensor/guards.hpp"
        )
    }
    fn operation_failed_internal_detail_reset_tensor_epoch_runtime(
        &self,
        _event: &DetailResetTensorEpochRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/guards.hpp::operation_failed_internal
        todo!(
            "TODO: port guard `operation_failed_internal` from emel.cpp/src/emel/graph/tensor/guards.hpp"
        )
    }
    fn operation_not_dispatched_detail_publish_filled_tensor_runtime(
        &self,
        _event: &DetailPublishFilledTensorRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/guards.hpp::operation_not_dispatched
        todo!(
            "TODO: port guard `operation_not_dispatched` from emel.cpp/src/emel/graph/tensor/guards.hpp"
        )
    }
    fn operation_not_dispatched_detail_release_tensor_ref_runtime(
        &self,
        _event: &DetailReleaseTensorRefRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/guards.hpp::operation_not_dispatched
        todo!(
            "TODO: port guard `operation_not_dispatched` from emel.cpp/src/emel/graph/tensor/guards.hpp"
        )
    }
    fn operation_not_dispatched_detail_reserve_tensor_runtime(
        &self,
        _event: &DetailReserveTensorRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/guards.hpp::operation_not_dispatched
        todo!(
            "TODO: port guard `operation_not_dispatched` from emel.cpp/src/emel/graph/tensor/guards.hpp"
        )
    }
    fn operation_not_dispatched_detail_reset_tensor_epoch_runtime(
        &self,
        _event: &DetailResetTensorEpochRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/guards.hpp::operation_not_dispatched
        todo!(
            "TODO: port guard `operation_not_dispatched` from emel.cpp/src/emel/graph/tensor/guards.hpp"
        )
    }
    fn operation_succeeded_detail_publish_filled_tensor_runtime(
        &self,
        _event: &DetailPublishFilledTensorRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/guards.hpp::operation_succeeded
        todo!(
            "TODO: port guard `operation_succeeded` from emel.cpp/src/emel/graph/tensor/guards.hpp"
        )
    }
    fn operation_succeeded_detail_release_tensor_ref_runtime(
        &self,
        _event: &DetailReleaseTensorRefRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/guards.hpp::operation_succeeded
        todo!(
            "TODO: port guard `operation_succeeded` from emel.cpp/src/emel/graph/tensor/guards.hpp"
        )
    }
    fn operation_succeeded_detail_reserve_tensor_runtime(
        &self,
        _event: &DetailReserveTensorRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/guards.hpp::operation_succeeded
        todo!(
            "TODO: port guard `operation_succeeded` from emel.cpp/src/emel/graph/tensor/guards.hpp"
        )
    }
    fn operation_succeeded_detail_reset_tensor_epoch_runtime(
        &self,
        _event: &DetailResetTensorEpochRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/guards.hpp::operation_succeeded
        todo!(
            "TODO: port guard `operation_succeeded` from emel.cpp/src/emel/graph/tensor/guards.hpp"
        )
    }
    fn publish_done_detail_capture_tensor_state_runtime(
        &mut self,
        _event: &DetailCaptureTensorStateRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::publish_done
        todo!("TODO: port action `publish_done` from emel.cpp/src/emel/graph/tensor/actions.hpp")
    }
    fn publish_done_detail_publish_filled_tensor_runtime(
        &mut self,
        _event: &DetailPublishFilledTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::publish_done
        todo!("TODO: port action `publish_done` from emel.cpp/src/emel/graph/tensor/actions.hpp")
    }
    fn publish_done_detail_release_tensor_ref_runtime(
        &mut self,
        _event: &DetailReleaseTensorRefRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::publish_done
        todo!("TODO: port action `publish_done` from emel.cpp/src/emel/graph/tensor/actions.hpp")
    }
    fn publish_done_detail_reserve_tensor_runtime(
        &mut self,
        _event: &DetailReserveTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::publish_done
        todo!("TODO: port action `publish_done` from emel.cpp/src/emel/graph/tensor/actions.hpp")
    }
    fn publish_done_detail_reset_tensor_epoch_runtime(
        &mut self,
        _event: &DetailResetTensorEpochRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::publish_done
        todo!("TODO: port action `publish_done` from emel.cpp/src/emel/graph/tensor/actions.hpp")
    }
    fn publish_error_detail_capture_tensor_state_runtime(
        &mut self,
        _event: &DetailCaptureTensorStateRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::publish_error
        todo!("TODO: port action `publish_error` from emel.cpp/src/emel/graph/tensor/actions.hpp")
    }
    fn publish_error_detail_publish_filled_tensor_runtime(
        &mut self,
        _event: &DetailPublishFilledTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::publish_error
        todo!("TODO: port action `publish_error` from emel.cpp/src/emel/graph/tensor/actions.hpp")
    }
    fn publish_error_detail_release_tensor_ref_runtime(
        &mut self,
        _event: &DetailReleaseTensorRefRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::publish_error
        todo!("TODO: port action `publish_error` from emel.cpp/src/emel/graph/tensor/actions.hpp")
    }
    fn publish_error_detail_reserve_tensor_runtime(
        &mut self,
        _event: &DetailReserveTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::publish_error
        todo!("TODO: port action `publish_error` from emel.cpp/src/emel/graph/tensor/actions.hpp")
    }
    fn publish_error_detail_reset_tensor_epoch_runtime(
        &mut self,
        _event: &DetailResetTensorEpochRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/actions.hpp::publish_error
        todo!("TODO: port action `publish_error` from emel.cpp/src/emel/graph/tensor/actions.hpp")
    }
    fn publish_filled_tensor_request_invalid(
        &self,
        _event: &DetailPublishFilledTensorRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/guards.hpp::publish_filled_tensor_request_invalid
        todo!(
            "TODO: port guard `publish_filled_tensor_request_invalid` from emel.cpp/src/emel/graph/tensor/guards.hpp"
        )
    }
    fn publish_filled_tensor_request_valid(
        &self,
        _event: &DetailPublishFilledTensorRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/guards.hpp::publish_filled_tensor_request_valid
        todo!(
            "TODO: port guard `publish_filled_tensor_request_valid` from emel.cpp/src/emel/graph/tensor/guards.hpp"
        )
    }
    fn release_tensor_ref_request_invalid(
        &self,
        _event: &DetailReleaseTensorRefRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/guards.hpp::release_tensor_ref_request_invalid
        todo!(
            "TODO: port guard `release_tensor_ref_request_invalid` from emel.cpp/src/emel/graph/tensor/guards.hpp"
        )
    }
    fn release_tensor_ref_request_valid(
        &self,
        _event: &DetailReleaseTensorRefRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/guards.hpp::release_tensor_ref_request_valid
        todo!(
            "TODO: port guard `release_tensor_ref_request_valid` from emel.cpp/src/emel/graph/tensor/guards.hpp"
        )
    }
    fn reserve_tensor_request_invalid(
        &self,
        _event: &DetailReserveTensorRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/guards.hpp::reserve_tensor_request_invalid
        todo!(
            "TODO: port guard `reserve_tensor_request_invalid` from emel.cpp/src/emel/graph/tensor/guards.hpp"
        )
    }
    fn reserve_tensor_request_valid(
        &self,
        _event: &DetailReserveTensorRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/guards.hpp::reserve_tensor_request_valid
        todo!(
            "TODO: port guard `reserve_tensor_request_valid` from emel.cpp/src/emel/graph/tensor/guards.hpp"
        )
    }
    fn reset_tensor_epoch_request_invalid(
        &self,
        _event: &DetailResetTensorEpochRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/guards.hpp::reset_tensor_epoch_request_invalid
        todo!(
            "TODO: port guard `reset_tensor_epoch_request_invalid` from emel.cpp/src/emel/graph/tensor/guards.hpp"
        )
    }
    fn reset_tensor_epoch_request_valid(
        &self,
        _event: &DetailResetTensorEpochRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/tensor/guards.hpp::reset_tensor_epoch_request_valid
        todo!(
            "TODO: port guard `reset_tensor_epoch_request_valid` from emel.cpp/src/emel/graph/tensor/guards.hpp"
        )
    }
}
