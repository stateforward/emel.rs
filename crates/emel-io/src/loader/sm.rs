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

// --- machine IoLoader from emel.cpp/src/emel/io/loader/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct DetailLoadTensorBatchRuntime;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct DetailLoadTensorRuntime;

sml! {
    IoLoader {
        "state_request_decision"_s <= *"state_ready"_s + event<DetailLoadTensorRuntime> [tensor_span_valid] / effect_begin_load_tensor,
        "state_no_strategy_error_decision"_s <= "state_ready"_s + event<DetailLoadTensorRuntime> [tensor_span_invalid] / effect_mark_invalid_request_from_state_ready,
        "state_no_strategy_error_decision"_s <= "state_request_decision"_s + completion<DetailLoadTensorRuntime> [strategy_none] / effect_mark_unsupported_strategy_from_state_request_decision,
        "state_unsupported_strategy_error_decision"_s <= "state_request_decision"_s + completion<DetailLoadTensorRuntime> [strategy_mapped_file] / effect_mark_unsupported_strategy_from_state_request_decision,
        "state_read_dispatch_decision"_s <= "state_request_decision"_s + completion<DetailLoadTensorRuntime> [strategy_read_copy_with_actor] / effect_dispatch_read_tensor,
        "state_staged_read_dispatch_decision"_s <= "state_request_decision"_s + completion<DetailLoadTensorRuntime> [strategy_staged_read_with_actor_and_source_span_valid] / effect_dispatch_staged_read_tensor,
        "state_no_strategy_error_decision"_s <= "state_request_decision"_s + completion<DetailLoadTensorRuntime> [strategy_staged_read_with_actor_and_source_span_invalid] / effect_mark_invalid_request_from_state_request_decision,
        "state_unsupported_strategy_error_decision"_s <= "state_request_decision"_s + completion<DetailLoadTensorRuntime> [strategy_read_copy_without_actor] / effect_mark_unsupported_strategy_from_state_request_decision,
        "state_unsupported_strategy_error_decision"_s <= "state_request_decision"_s + completion<DetailLoadTensorRuntime> [strategy_staged_read_without_actor] / effect_mark_unsupported_strategy_from_state_request_decision,
        "state_unsupported_strategy_error_decision"_s <= "state_request_decision"_s + completion<DetailLoadTensorRuntime> [strategy_external_buffer] / effect_mark_unsupported_strategy_from_state_request_decision,
        "state_unsupported_strategy_error_decision"_s <= "state_request_decision"_s + completion<DetailLoadTensorRuntime> / effect_mark_unsupported_strategy_from_state_request_decision,
        "state_done_decision"_s <= "state_read_dispatch_decision"_s + completion<DetailLoadTensorRuntime> [read_load_succeeded],
        "state_unsupported_strategy_error_decision"_s <= "state_read_dispatch_decision"_s + completion<DetailLoadTensorRuntime> [read_load_failed],
        "state_done_decision"_s <= "state_staged_read_dispatch_decision"_s + completion<DetailLoadTensorRuntime> [read_load_succeeded],
        "state_unsupported_strategy_error_decision"_s <= "state_staged_read_dispatch_decision"_s + completion<DetailLoadTensorRuntime> [read_load_failed],
        "state_done_callback"_s <= "state_done_decision"_s + completion<DetailLoadTensorRuntime> [done_callback_present] / effect_publish_load_tensor_done,
        "state_ready"_s <= "state_done_decision"_s + completion<DetailLoadTensorRuntime> [done_callback_absent] / effect_record_load_tensor_done_from_state_done_decision,
        "state_ready"_s <= "state_done_callback"_s + completion<DetailLoadTensorRuntime> / effect_record_load_tensor_done_from_state_done_callback,
        "state_error_callback"_s <= "state_no_strategy_error_decision"_s + completion<DetailLoadTensorRuntime> [error_callback_present] / effect_publish_load_tensor_error_from_state_no_strategy_error_decision,
        "state_ready"_s <= "state_no_strategy_error_decision"_s + completion<DetailLoadTensorRuntime> [error_callback_absent] / effect_record_load_tensor_error_from_state_no_strategy_error_decision,
        "state_error_callback"_s <= "state_unsupported_strategy_error_decision"_s + completion<DetailLoadTensorRuntime> [error_callback_present] / effect_publish_load_tensor_error_from_state_unsupported_strategy_error_decision,
        "state_ready"_s <= "state_unsupported_strategy_error_decision"_s + completion<DetailLoadTensorRuntime> [error_callback_absent] / effect_record_load_tensor_error_from_state_unsupported_strategy_error_decision,
        "state_ready"_s <= "state_error_callback"_s + completion<DetailLoadTensorRuntime> / effect_record_load_tensor_error_from_state_error_callback,
        "state_batch_request_decision"_s <= "state_ready"_s + event<DetailLoadTensorBatchRuntime> [batch_span_valid] / effect_begin_load_tensor_batch,
        "state_batch_unsupported_strategy_error_decision"_s <= "state_ready"_s + event<DetailLoadTensorBatchRuntime> [batch_span_invalid] / effect_mark_load_tensor_batch_invalid_request_from_state_ready,
        "state_batch_read_dispatch_decision"_s <= "state_batch_request_decision"_s + completion<DetailLoadTensorBatchRuntime> [strategy_read_copy_batch_with_actor] / effect_dispatch_read_tensor_batch,
        "state_batch_staged_read_dispatch_decision"_s <= "state_batch_request_decision"_s + completion<DetailLoadTensorBatchRuntime> [strategy_staged_read_batch_with_actor_and_source_span_valid] / effect_dispatch_staged_read_tensor_batch,
        "state_batch_unsupported_strategy_error_decision"_s <= "state_batch_request_decision"_s + completion<DetailLoadTensorBatchRuntime> [strategy_staged_read_batch_with_actor_and_source_span_invalid] / effect_mark_load_tensor_batch_invalid_request_from_state_batch_request_decision,
        "state_batch_unsupported_strategy_error_decision"_s <= "state_batch_request_decision"_s + completion<DetailLoadTensorBatchRuntime> [strategy_read_copy_batch_without_actor] / effect_mark_load_tensor_batch_unsupported_strategy_from_state_batch_request_decision,
        "state_batch_unsupported_strategy_error_decision"_s <= "state_batch_request_decision"_s + completion<DetailLoadTensorBatchRuntime> [strategy_staged_read_batch_without_actor] / effect_mark_load_tensor_batch_unsupported_strategy_from_state_batch_request_decision,
        "state_batch_unsupported_strategy_error_decision"_s <= "state_batch_request_decision"_s + completion<DetailLoadTensorBatchRuntime> / effect_mark_load_tensor_batch_unsupported_strategy_from_state_batch_request_decision,
        "state_batch_done_decision"_s <= "state_batch_read_dispatch_decision"_s + completion<DetailLoadTensorBatchRuntime> [read_batch_succeeded],
        "state_batch_unsupported_strategy_error_decision"_s <= "state_batch_read_dispatch_decision"_s + completion<DetailLoadTensorBatchRuntime> [read_batch_failed] / effect_record_read_tensor_batch_failed_from_state_batch_read_dispatch_decision,
        "state_batch_done_decision"_s <= "state_batch_staged_read_dispatch_decision"_s + completion<DetailLoadTensorBatchRuntime> [read_batch_succeeded],
        "state_batch_unsupported_strategy_error_decision"_s <= "state_batch_staged_read_dispatch_decision"_s + completion<DetailLoadTensorBatchRuntime> [read_batch_failed] / effect_record_read_tensor_batch_failed_from_state_batch_staged_read_dispatch_decision,
        "state_batch_done_callback"_s <= "state_batch_done_decision"_s + completion<DetailLoadTensorBatchRuntime> [batch_done_callback_present] / effect_publish_load_tensor_batch_done,
        "state_ready"_s <= "state_batch_done_decision"_s + completion<DetailLoadTensorBatchRuntime> [batch_done_callback_absent] / effect_record_load_tensor_batch_done_from_state_batch_done_decision,
        "state_ready"_s <= "state_batch_done_callback"_s + completion<DetailLoadTensorBatchRuntime> / effect_record_load_tensor_batch_done_from_state_batch_done_callback,
        "state_batch_error_callback"_s <= "state_batch_unsupported_strategy_error_decision"_s + completion<DetailLoadTensorBatchRuntime> [batch_error_callback_present] / effect_publish_load_tensor_batch_error,
        "state_ready"_s <= "state_batch_unsupported_strategy_error_decision"_s + completion<DetailLoadTensorBatchRuntime> [batch_error_callback_absent] / effect_record_load_tensor_batch_error_from_state_batch_unsupported_strategy_error_decision,
        "state_ready"_s <= "state_batch_error_callback"_s + completion<DetailLoadTensorBatchRuntime> / effect_record_load_tensor_batch_error_from_state_batch_error_callback,
        "state_ready"_s <= "state_ready"_s + unexpected_event<_> / effect_on_unexpected_from_state_ready,
        "state_ready"_s <= "state_request_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_request_decision,
        "state_ready"_s <= "state_no_strategy_error_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_no_strategy_error_decision,
        "state_ready"_s <= "state_unsupported_strategy_error_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_unsupported_strategy_error_decision,
        "state_ready"_s <= "state_error_callback"_s + unexpected_event<_> / effect_on_unexpected_from_state_error_callback,
        "state_ready"_s <= "state_read_dispatch_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_read_dispatch_decision,
        "state_ready"_s <= "state_staged_read_dispatch_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_staged_read_dispatch_decision,
        "state_ready"_s <= "state_done_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_done_decision,
        "state_ready"_s <= "state_done_callback"_s + unexpected_event<_> / effect_on_unexpected_from_state_done_callback,
        "state_ready"_s <= "state_batch_request_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_batch_request_decision,
        "state_ready"_s <= "state_batch_unsupported_strategy_error_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_batch_unsupported_strategy_error_decision,
        "state_ready"_s <= "state_batch_error_callback"_s + unexpected_event<_> / effect_on_unexpected_from_state_batch_error_callback,
        "state_ready"_s <= "state_batch_read_dispatch_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_batch_read_dispatch_decision,
        "state_ready"_s <= "state_batch_staged_read_dispatch_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_batch_staged_read_dispatch_decision,
        "state_ready"_s <= "state_batch_done_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_batch_done_decision,
        "state_ready"_s <= "state_batch_done_callback"_s + unexpected_event<_> / effect_on_unexpected_from_state_batch_done_callback,
    }
}

/// Context for `IoLoader` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct IoLoaderContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl IoLoaderStateMachineContext for IoLoaderContext {
    fn batch_done_callback_absent(
        &self,
        _event: &DetailLoadTensorBatchRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/guards.hpp::batch_done_callback_absent
        todo!(
            "TODO: port guard `batch_done_callback_absent` from emel.cpp/src/emel/io/loader/guards.hpp"
        )
    }
    fn batch_done_callback_present(
        &self,
        _event: &DetailLoadTensorBatchRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/guards.hpp::batch_done_callback_present
        todo!(
            "TODO: port guard `batch_done_callback_present` from emel.cpp/src/emel/io/loader/guards.hpp"
        )
    }
    fn batch_error_callback_absent(
        &self,
        _event: &DetailLoadTensorBatchRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/guards.hpp::batch_error_callback_absent
        todo!(
            "TODO: port guard `batch_error_callback_absent` from emel.cpp/src/emel/io/loader/guards.hpp"
        )
    }
    fn batch_error_callback_present(
        &self,
        _event: &DetailLoadTensorBatchRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/guards.hpp::batch_error_callback_present
        todo!(
            "TODO: port guard `batch_error_callback_present` from emel.cpp/src/emel/io/loader/guards.hpp"
        )
    }
    fn batch_span_invalid(&self, _event: &DetailLoadTensorBatchRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/guards.hpp::batch_span_invalid
        todo!("TODO: port guard `batch_span_invalid` from emel.cpp/src/emel/io/loader/guards.hpp")
    }
    fn batch_span_valid(&self, _event: &DetailLoadTensorBatchRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/guards.hpp::batch_span_valid
        todo!("TODO: port guard `batch_span_valid` from emel.cpp/src/emel/io/loader/guards.hpp")
    }
    fn done_callback_absent(&self, _event: &DetailLoadTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/guards.hpp::done_callback_absent
        todo!("TODO: port guard `done_callback_absent` from emel.cpp/src/emel/io/loader/guards.hpp")
    }
    fn done_callback_present(&self, _event: &DetailLoadTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/guards.hpp::done_callback_present
        todo!(
            "TODO: port guard `done_callback_present` from emel.cpp/src/emel/io/loader/guards.hpp"
        )
    }
    fn effect_begin_load_tensor(&mut self, _event: &DetailLoadTensorRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/actions.hpp::effect_begin_load_tensor
        todo!(
            "TODO: port action `effect_begin_load_tensor` from emel.cpp/src/emel/io/loader/actions.hpp"
        )
    }
    fn effect_begin_load_tensor_batch(
        &mut self,
        _event: &DetailLoadTensorBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/actions.hpp::effect_begin_load_tensor_batch
        todo!(
            "TODO: port action `effect_begin_load_tensor_batch` from emel.cpp/src/emel/io/loader/actions.hpp"
        )
    }
    fn effect_dispatch_read_tensor(&mut self, _event: &DetailLoadTensorRuntime) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/actions.hpp::effect_dispatch_read_tensor
        todo!(
            "TODO: port action `effect_dispatch_read_tensor` from emel.cpp/src/emel/io/loader/actions.hpp"
        )
    }
    fn effect_dispatch_read_tensor_batch(
        &mut self,
        _event: &DetailLoadTensorBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/actions.hpp::effect_dispatch_read_tensor_batch
        todo!(
            "TODO: port action `effect_dispatch_read_tensor_batch` from emel.cpp/src/emel/io/loader/actions.hpp"
        )
    }
    fn effect_dispatch_staged_read_tensor(
        &mut self,
        _event: &DetailLoadTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/actions.hpp::effect_dispatch_staged_read_tensor
        todo!(
            "TODO: port action `effect_dispatch_staged_read_tensor` from emel.cpp/src/emel/io/loader/actions.hpp"
        )
    }
    fn effect_dispatch_staged_read_tensor_batch(
        &mut self,
        _event: &DetailLoadTensorBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/actions.hpp::effect_dispatch_staged_read_tensor_batch
        todo!(
            "TODO: port action `effect_dispatch_staged_read_tensor_batch` from emel.cpp/src/emel/io/loader/actions.hpp"
        )
    }
    fn effect_mark_invalid_request_from_state_ready(
        &mut self,
        _event: &DetailLoadTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/actions.hpp::effect_mark_invalid_request
        todo!(
            "TODO: port action `effect_mark_invalid_request` from emel.cpp/src/emel/io/loader/actions.hpp"
        )
    }
    fn effect_mark_invalid_request_from_state_request_decision(
        &mut self,
        _event: &DetailLoadTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/actions.hpp::effect_mark_invalid_request
        todo!(
            "TODO: port action `effect_mark_invalid_request` from emel.cpp/src/emel/io/loader/actions.hpp"
        )
    }
    fn effect_mark_load_tensor_batch_invalid_request_from_state_batch_request_decision(
        &mut self,
        _event: &DetailLoadTensorBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/actions.hpp::effect_mark_load_tensor_batch_invalid_request
        todo!(
            "TODO: port action `effect_mark_load_tensor_batch_invalid_request` from emel.cpp/src/emel/io/loader/actions.hpp"
        )
    }
    fn effect_mark_load_tensor_batch_invalid_request_from_state_ready(
        &mut self,
        _event: &DetailLoadTensorBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/actions.hpp::effect_mark_load_tensor_batch_invalid_request
        todo!(
            "TODO: port action `effect_mark_load_tensor_batch_invalid_request` from emel.cpp/src/emel/io/loader/actions.hpp"
        )
    }
    fn effect_mark_load_tensor_batch_unsupported_strategy_from_state_batch_request_decision(
        &mut self,
        _event: &DetailLoadTensorBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/actions.hpp::effect_mark_load_tensor_batch_unsupported_strategy
        todo!(
            "TODO: port action `effect_mark_load_tensor_batch_unsupported_strategy` from emel.cpp/src/emel/io/loader/actions.hpp"
        )
    }
    fn effect_mark_unsupported_strategy_from_state_request_decision(
        &mut self,
        _event: &DetailLoadTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/actions.hpp::effect_mark_unsupported_strategy
        todo!(
            "TODO: port action `effect_mark_unsupported_strategy` from emel.cpp/src/emel/io/loader/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_batch_done_callback(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/loader/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_batch_done_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/loader/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_batch_error_callback(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/loader/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_batch_read_dispatch_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/loader/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_batch_request_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/loader/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_batch_staged_read_dispatch_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/loader/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_batch_unsupported_strategy_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/loader/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_done_callback(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/loader/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_done_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/loader/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_error_callback(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/loader/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_no_strategy_error_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/loader/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_read_dispatch_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/loader/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_ready(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/loader/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_request_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/loader/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_staged_read_dispatch_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/loader/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_unsupported_strategy_error_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/io/loader/actions.hpp"
        )
    }
    fn effect_publish_load_tensor_batch_done(
        &mut self,
        _event: &DetailLoadTensorBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/actions.hpp::effect_publish_load_tensor_batch_done
        todo!(
            "TODO: port action `effect_publish_load_tensor_batch_done` from emel.cpp/src/emel/io/loader/actions.hpp"
        )
    }
    fn effect_publish_load_tensor_batch_error(
        &mut self,
        _event: &DetailLoadTensorBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/actions.hpp::effect_publish_load_tensor_batch_error
        todo!(
            "TODO: port action `effect_publish_load_tensor_batch_error` from emel.cpp/src/emel/io/loader/actions.hpp"
        )
    }
    fn effect_publish_load_tensor_done(
        &mut self,
        _event: &DetailLoadTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/actions.hpp::effect_publish_load_tensor_done
        todo!(
            "TODO: port action `effect_publish_load_tensor_done` from emel.cpp/src/emel/io/loader/actions.hpp"
        )
    }
    fn effect_publish_load_tensor_error_from_state_no_strategy_error_decision(
        &mut self,
        _event: &DetailLoadTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/actions.hpp::effect_publish_load_tensor_error
        todo!(
            "TODO: port action `effect_publish_load_tensor_error` from emel.cpp/src/emel/io/loader/actions.hpp"
        )
    }
    fn effect_publish_load_tensor_error_from_state_unsupported_strategy_error_decision(
        &mut self,
        _event: &DetailLoadTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/actions.hpp::effect_publish_load_tensor_error
        todo!(
            "TODO: port action `effect_publish_load_tensor_error` from emel.cpp/src/emel/io/loader/actions.hpp"
        )
    }
    fn effect_record_load_tensor_batch_done_from_state_batch_done_callback(
        &mut self,
        _event: &DetailLoadTensorBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/actions.hpp::effect_record_load_tensor_batch_done
        todo!(
            "TODO: port action `effect_record_load_tensor_batch_done` from emel.cpp/src/emel/io/loader/actions.hpp"
        )
    }
    fn effect_record_load_tensor_batch_done_from_state_batch_done_decision(
        &mut self,
        _event: &DetailLoadTensorBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/actions.hpp::effect_record_load_tensor_batch_done
        todo!(
            "TODO: port action `effect_record_load_tensor_batch_done` from emel.cpp/src/emel/io/loader/actions.hpp"
        )
    }
    fn effect_record_load_tensor_batch_error_from_state_batch_error_callback(
        &mut self,
        _event: &DetailLoadTensorBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/actions.hpp::effect_record_load_tensor_batch_error
        todo!(
            "TODO: port action `effect_record_load_tensor_batch_error` from emel.cpp/src/emel/io/loader/actions.hpp"
        )
    }
    fn effect_record_load_tensor_batch_error_from_state_batch_unsupported_strategy_error_decision(
        &mut self,
        _event: &DetailLoadTensorBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/actions.hpp::effect_record_load_tensor_batch_error
        todo!(
            "TODO: port action `effect_record_load_tensor_batch_error` from emel.cpp/src/emel/io/loader/actions.hpp"
        )
    }
    fn effect_record_load_tensor_done_from_state_done_callback(
        &mut self,
        _event: &DetailLoadTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/actions.hpp::effect_record_load_tensor_done
        todo!(
            "TODO: port action `effect_record_load_tensor_done` from emel.cpp/src/emel/io/loader/actions.hpp"
        )
    }
    fn effect_record_load_tensor_done_from_state_done_decision(
        &mut self,
        _event: &DetailLoadTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/actions.hpp::effect_record_load_tensor_done
        todo!(
            "TODO: port action `effect_record_load_tensor_done` from emel.cpp/src/emel/io/loader/actions.hpp"
        )
    }
    fn effect_record_load_tensor_error_from_state_error_callback(
        &mut self,
        _event: &DetailLoadTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/actions.hpp::effect_record_load_tensor_error
        todo!(
            "TODO: port action `effect_record_load_tensor_error` from emel.cpp/src/emel/io/loader/actions.hpp"
        )
    }
    fn effect_record_load_tensor_error_from_state_no_strategy_error_decision(
        &mut self,
        _event: &DetailLoadTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/actions.hpp::effect_record_load_tensor_error
        todo!(
            "TODO: port action `effect_record_load_tensor_error` from emel.cpp/src/emel/io/loader/actions.hpp"
        )
    }
    fn effect_record_load_tensor_error_from_state_unsupported_strategy_error_decision(
        &mut self,
        _event: &DetailLoadTensorRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/actions.hpp::effect_record_load_tensor_error
        todo!(
            "TODO: port action `effect_record_load_tensor_error` from emel.cpp/src/emel/io/loader/actions.hpp"
        )
    }
    fn effect_record_read_tensor_batch_failed_from_state_batch_read_dispatch_decision(
        &mut self,
        _event: &DetailLoadTensorBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/actions.hpp::effect_record_read_tensor_batch_failed
        todo!(
            "TODO: port action `effect_record_read_tensor_batch_failed` from emel.cpp/src/emel/io/loader/actions.hpp"
        )
    }
    fn effect_record_read_tensor_batch_failed_from_state_batch_staged_read_dispatch_decision(
        &mut self,
        _event: &DetailLoadTensorBatchRuntime,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/actions.hpp::effect_record_read_tensor_batch_failed
        todo!(
            "TODO: port action `effect_record_read_tensor_batch_failed` from emel.cpp/src/emel/io/loader/actions.hpp"
        )
    }
    fn error_callback_absent(&self, _event: &DetailLoadTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/guards.hpp::error_callback_absent
        todo!(
            "TODO: port guard `error_callback_absent` from emel.cpp/src/emel/io/loader/guards.hpp"
        )
    }
    fn error_callback_present(&self, _event: &DetailLoadTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/guards.hpp::error_callback_present
        todo!(
            "TODO: port guard `error_callback_present` from emel.cpp/src/emel/io/loader/guards.hpp"
        )
    }
    fn read_batch_failed(&self, _event: &DetailLoadTensorBatchRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/guards.hpp::read_batch_failed
        todo!("TODO: port guard `read_batch_failed` from emel.cpp/src/emel/io/loader/guards.hpp")
    }
    fn read_batch_succeeded(&self, _event: &DetailLoadTensorBatchRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/guards.hpp::read_batch_succeeded
        todo!("TODO: port guard `read_batch_succeeded` from emel.cpp/src/emel/io/loader/guards.hpp")
    }
    fn read_load_failed(&self, _event: &DetailLoadTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/guards.hpp::read_load_failed
        todo!("TODO: port guard `read_load_failed` from emel.cpp/src/emel/io/loader/guards.hpp")
    }
    fn read_load_succeeded(&self, _event: &DetailLoadTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/guards.hpp::read_load_succeeded
        todo!("TODO: port guard `read_load_succeeded` from emel.cpp/src/emel/io/loader/guards.hpp")
    }
    fn strategy_external_buffer(&self, _event: &DetailLoadTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/guards.hpp::strategy_external_buffer
        todo!(
            "TODO: port guard `strategy_external_buffer` from emel.cpp/src/emel/io/loader/guards.hpp"
        )
    }
    fn strategy_mapped_file(&self, _event: &DetailLoadTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/guards.hpp::strategy_mapped_file
        todo!("TODO: port guard `strategy_mapped_file` from emel.cpp/src/emel/io/loader/guards.hpp")
    }
    fn strategy_none(&self, _event: &DetailLoadTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/guards.hpp::strategy_none
        todo!("TODO: port guard `strategy_none` from emel.cpp/src/emel/io/loader/guards.hpp")
    }
    fn strategy_read_copy_batch_with_actor(
        &self,
        _event: &DetailLoadTensorBatchRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/guards.hpp::strategy_read_copy_batch_with_actor
        todo!(
            "TODO: port guard `strategy_read_copy_batch_with_actor` from emel.cpp/src/emel/io/loader/guards.hpp"
        )
    }
    fn strategy_read_copy_batch_without_actor(
        &self,
        _event: &DetailLoadTensorBatchRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/guards.hpp::strategy_read_copy_batch_without_actor
        todo!(
            "TODO: port guard `strategy_read_copy_batch_without_actor` from emel.cpp/src/emel/io/loader/guards.hpp"
        )
    }
    fn strategy_read_copy_with_actor(&self, _event: &DetailLoadTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/guards.hpp::strategy_read_copy_with_actor
        todo!(
            "TODO: port guard `strategy_read_copy_with_actor` from emel.cpp/src/emel/io/loader/guards.hpp"
        )
    }
    fn strategy_read_copy_without_actor(
        &self,
        _event: &DetailLoadTensorRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/guards.hpp::strategy_read_copy_without_actor
        todo!(
            "TODO: port guard `strategy_read_copy_without_actor` from emel.cpp/src/emel/io/loader/guards.hpp"
        )
    }
    fn strategy_staged_read_batch_with_actor_and_source_span_invalid(
        &self,
        _event: &DetailLoadTensorBatchRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/guards.hpp::strategy_staged_read_batch_with_actor_and_source_span_invalid
        todo!(
            "TODO: port guard `strategy_staged_read_batch_with_actor_and_source_span_invalid` from emel.cpp/src/emel/io/loader/guards.hpp"
        )
    }
    fn strategy_staged_read_batch_with_actor_and_source_span_valid(
        &self,
        _event: &DetailLoadTensorBatchRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/guards.hpp::strategy_staged_read_batch_with_actor_and_source_span_valid
        todo!(
            "TODO: port guard `strategy_staged_read_batch_with_actor_and_source_span_valid` from emel.cpp/src/emel/io/loader/guards.hpp"
        )
    }
    fn strategy_staged_read_batch_without_actor(
        &self,
        _event: &DetailLoadTensorBatchRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/guards.hpp::strategy_staged_read_batch_without_actor
        todo!(
            "TODO: port guard `strategy_staged_read_batch_without_actor` from emel.cpp/src/emel/io/loader/guards.hpp"
        )
    }
    fn strategy_staged_read_with_actor_and_source_span_invalid(
        &self,
        _event: &DetailLoadTensorRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/guards.hpp::strategy_staged_read_with_actor_and_source_span_invalid
        todo!(
            "TODO: port guard `strategy_staged_read_with_actor_and_source_span_invalid` from emel.cpp/src/emel/io/loader/guards.hpp"
        )
    }
    fn strategy_staged_read_with_actor_and_source_span_valid(
        &self,
        _event: &DetailLoadTensorRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/guards.hpp::strategy_staged_read_with_actor_and_source_span_valid
        todo!(
            "TODO: port guard `strategy_staged_read_with_actor_and_source_span_valid` from emel.cpp/src/emel/io/loader/guards.hpp"
        )
    }
    fn strategy_staged_read_without_actor(
        &self,
        _event: &DetailLoadTensorRuntime,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/guards.hpp::strategy_staged_read_without_actor
        todo!(
            "TODO: port guard `strategy_staged_read_without_actor` from emel.cpp/src/emel/io/loader/guards.hpp"
        )
    }
    fn tensor_span_invalid(&self, _event: &DetailLoadTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/guards.hpp::tensor_span_invalid
        todo!("TODO: port guard `tensor_span_invalid` from emel.cpp/src/emel/io/loader/guards.hpp")
    }
    fn tensor_span_valid(&self, _event: &DetailLoadTensorRuntime) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/io/loader/guards.hpp::tensor_span_valid
        todo!("TODO: port guard `tensor_span_valid` from emel.cpp/src/emel/io/loader/guards.hpp")
    }
}
