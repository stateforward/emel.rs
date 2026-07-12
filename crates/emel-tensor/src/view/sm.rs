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

// --- machine TensorView from emel.cpp/src/emel/tensor/view/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct DetailCaptureTensorViewRuntime;

sml! {
    TensorView {
        "capture_tensor_view_request_decision"_s <= *"ready"_s + event<DetailCaptureTensorViewRuntime> / begin_capture_tensor_view,
        "capture_tensor_view_exec"_s <= "capture_tensor_view_request_decision"_s + completion<DetailCaptureTensorViewRuntime> [capture_tensor_view_request_valid],
        "errored"_s <= "capture_tensor_view_request_decision"_s + completion<DetailCaptureTensorViewRuntime> [capture_tensor_view_request_invalid] / mark_invalid_request,
        "capture_tensor_view_result_decision"_s <= "capture_tensor_view_exec"_s + completion<DetailCaptureTensorViewRuntime> / exec_capture_tensor_view,
        "done"_s <= "capture_tensor_view_result_decision"_s + completion<DetailCaptureTensorViewRuntime> [operation_succeeded],
        "errored"_s <= "capture_tensor_view_result_decision"_s + completion<DetailCaptureTensorViewRuntime> [operation_failed_with_error] / mark_error_from_operation,
        "errored"_s <= "capture_tensor_view_result_decision"_s + completion<DetailCaptureTensorViewRuntime> [operation_failed_without_error] / mark_internal_error,
        "ready"_s <= "done"_s + completion<DetailCaptureTensorViewRuntime> / publish_done,
        "ready"_s <= "errored"_s + completion<DetailCaptureTensorViewRuntime> / publish_error,
        "ready"_s <= "ready"_s + unexpected_event<_> / on_unexpected_from_ready,
        "ready"_s <= "capture_tensor_view_request_decision"_s + unexpected_event<_> / on_unexpected_from_capture_tensor_view_request_decision,
        "ready"_s <= "capture_tensor_view_exec"_s + unexpected_event<_> / on_unexpected_from_capture_tensor_view_exec,
        "ready"_s <= "capture_tensor_view_result_decision"_s + unexpected_event<_> / on_unexpected_from_capture_tensor_view_result_decision,
        "ready"_s <= "done"_s + unexpected_event<_> / on_unexpected_from_done,
        "ready"_s <= "errored"_s + unexpected_event<_> / on_unexpected_from_errored,
    }
}

/// Context for `TensorView` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct TensorViewContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl TensorViewStateMachineContext for TensorViewContext {
    fn begin_capture_tensor_view(
        &mut self,
        _event: &DetailCaptureTensorViewRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/tensor/view/actions.hpp::begin_capture_tensor_view
        todo!(
            "TODO: port action `begin_capture_tensor_view` from emel.cpp/src/emel/tensor/view/actions.hpp"
        )
    }
    fn capture_tensor_view_request_invalid(
        &self,
        _event: &DetailCaptureTensorViewRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/tensor/view/guards.hpp::capture_tensor_view_request_invalid
        todo!(
            "TODO: port guard `capture_tensor_view_request_invalid` from emel.cpp/src/emel/tensor/view/guards.hpp"
        )
    }
    fn capture_tensor_view_request_valid(
        &self,
        _event: &DetailCaptureTensorViewRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/tensor/view/guards.hpp::capture_tensor_view_request_valid
        todo!(
            "TODO: port guard `capture_tensor_view_request_valid` from emel.cpp/src/emel/tensor/view/guards.hpp"
        )
    }
    fn exec_capture_tensor_view(
        &mut self,
        _event: &DetailCaptureTensorViewRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/tensor/view/actions.hpp::exec_capture_tensor_view
        todo!(
            "TODO: port action `exec_capture_tensor_view` from emel.cpp/src/emel/tensor/view/actions.hpp"
        )
    }
    fn mark_error_from_operation(
        &mut self,
        _event: &DetailCaptureTensorViewRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/tensor/view/actions.hpp::mark_error_from_operation
        todo!(
            "TODO: port action `mark_error_from_operation` from emel.cpp/src/emel/tensor/view/actions.hpp"
        )
    }
    fn mark_internal_error(&mut self, _event: &DetailCaptureTensorViewRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/tensor/view/actions.hpp::mark_internal_error
        todo!(
            "TODO: port action `mark_internal_error` from emel.cpp/src/emel/tensor/view/actions.hpp"
        )
    }
    fn mark_invalid_request(&mut self, _event: &DetailCaptureTensorViewRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/tensor/view/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/tensor/view/actions.hpp"
        )
    }
    fn on_unexpected_from_capture_tensor_view_exec(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/tensor/view/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/tensor/view/actions.hpp")
    }
    fn on_unexpected_from_capture_tensor_view_request_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/tensor/view/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/tensor/view/actions.hpp")
    }
    fn on_unexpected_from_capture_tensor_view_result_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/tensor/view/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/tensor/view/actions.hpp")
    }
    fn on_unexpected_from_done(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/tensor/view/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/tensor/view/actions.hpp")
    }
    fn on_unexpected_from_errored(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/tensor/view/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/tensor/view/actions.hpp")
    }
    fn on_unexpected_from_ready(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/tensor/view/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/tensor/view/actions.hpp")
    }
    fn operation_failed_with_error(
        &self,
        _event: &DetailCaptureTensorViewRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/tensor/view/guards.hpp::operation_failed_with_error
        todo!(
            "TODO: port guard `operation_failed_with_error` from emel.cpp/src/emel/tensor/view/guards.hpp"
        )
    }
    fn operation_failed_without_error(
        &self,
        _event: &DetailCaptureTensorViewRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/tensor/view/guards.hpp::operation_failed_without_error
        todo!(
            "TODO: port guard `operation_failed_without_error` from emel.cpp/src/emel/tensor/view/guards.hpp"
        )
    }
    fn operation_succeeded(&self, _event: &DetailCaptureTensorViewRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/tensor/view/guards.hpp::operation_succeeded
        todo!(
            "TODO: port guard `operation_succeeded` from emel.cpp/src/emel/tensor/view/guards.hpp"
        )
    }
    fn publish_done(&mut self, _event: &DetailCaptureTensorViewRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/tensor/view/actions.hpp::publish_done
        todo!("TODO: port action `publish_done` from emel.cpp/src/emel/tensor/view/actions.hpp")
    }
    fn publish_error(&mut self, _event: &DetailCaptureTensorViewRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/tensor/view/actions.hpp::publish_error
        todo!("TODO: port action `publish_error` from emel.cpp/src/emel/tensor/view/actions.hpp")
    }
}
