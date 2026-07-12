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

// --- machine TextGeneratorPrefill from emel.cpp/src/emel/text/generator/prefill/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventRun;

sml! {
    TextGeneratorPrefill {
        "slots"_s <= *"idle"_s + event<EventRun>,
        "slots_decision"_s <= "slots"_s + completion<EventRun> / request_slots,
        "snapshot"_s <= "slots_decision"_s + completion<EventRun> [slots_ok],
        "idle"_s <= "slots_decision"_s + completion<EventRun> [slots_invalid_request] / mark_invalid_request_from_slots_decision,
        "idle"_s <= "slots_decision"_s + completion<EventRun> [slots_backend_error] / mark_backend_error_from_slots_decision,
        "snapshot_decision"_s <= "snapshot"_s + completion<EventRun> / request_memory_snapshot,
        "contract_runtime_decision"_s <= "snapshot_decision"_s + completion<EventRun> [snapshot_ok],
        "idle"_s <= "snapshot_decision"_s + completion<EventRun> [snapshot_invalid_request] / mark_invalid_request_from_snapshot_decision,
        "idle"_s <= "snapshot_decision"_s + completion<EventRun> [snapshot_backend_error] / mark_backend_error_from_snapshot_decision,
        "contract_flash_decision"_s <= "contract_runtime_decision"_s + completion<EventRun> [flash_runtime_supported],
        "contract_nonflash_decision"_s <= "contract_runtime_decision"_s + completion<EventRun> [nonflash_runtime_required],
        "idle"_s <= "contract_flash_decision"_s + completion<EventRun> [guard_compute_invalid_request] / mark_invalid_request_from_contract_flash_decision,
        "idle"_s <= "contract_flash_decision"_s + completion<EventRun> [guard_compute_backend_unavailable] / mark_backend_error_from_contract_flash_decision,
        "compute_result_decision"_s <= "contract_flash_decision"_s + completion<EventRun> [guard_materialized_logits_with_parallel_chunk8_q8_k_ready] / request_contract_flash_materialized_parallel_chunk8_q8_k,
        "compute_result_decision"_s <= "contract_flash_decision"_s + completion<EventRun> [guard_materialized_logits_with_chunk8_q8_k_ready] / request_contract_flash_materialized_chunk8_q8_k,
        "compute_result_decision"_s <= "contract_flash_decision"_s + completion<EventRun> [guard_materialized_logits_with_parallel_chunk4_packed_q8_0_ready] / request_contract_flash_materialized_parallel_chunk4_packed_q8_0,
        "compute_result_decision"_s <= "contract_flash_decision"_s + completion<EventRun> [guard_materialized_logits_with_chunk4_packed_q8_0_ready] / request_contract_flash_materialized_chunk4_packed_q8_0,
        "compute_result_decision"_s <= "contract_flash_decision"_s + completion<EventRun> [guard_materialized_logits_with_parallel_chunk4_q8_k_ready] / request_contract_flash_materialized_parallel_chunk4_q8_k,
        "compute_result_decision"_s <= "contract_flash_decision"_s + completion<EventRun> [guard_materialized_logits_with_chunk4_q8_k_ready] / request_contract_flash_materialized_chunk4_q8_k,
        "compute_result_decision"_s <= "contract_flash_decision"_s + completion<EventRun> [guard_materialized_logits_with_scalar_packed_q8_0_ready] / request_contract_flash_materialized_scalar_packed_q8_0,
        "compute_result_decision"_s <= "contract_flash_decision"_s + completion<EventRun> [guard_materialized_logits_with_scalar_q8_k_ready] / request_contract_flash_materialized_scalar_q8_k,
        "compute_result_decision"_s <= "contract_flash_decision"_s + completion<EventRun> [guard_materialized_logits_with_scalar_native_quantized_q8_k_ready] / request_contract_flash_materialized_scalar_native_quantized_q8_k,
        "compute_result_decision"_s <= "contract_flash_decision"_s + completion<EventRun> [guard_materialized_logits_with_scalar_native_quantized_kernel_ready] / request_contract_flash_materialized_scalar_native_quantized,
        "compute_result_decision"_s <= "contract_flash_decision"_s + completion<EventRun> [guard_materialized_logits_with_scalar_kernel_ready] / request_contract_flash_materialized_scalar_kernel,
        "compute_result_decision"_s <= "contract_flash_decision"_s + completion<EventRun> [guard_preselected_argmax_with_parallel_chunk8_q8_k_ready] / request_contract_flash_preselected_parallel_chunk8_q8_k,
        "compute_result_decision"_s <= "contract_flash_decision"_s + completion<EventRun> [guard_preselected_argmax_with_chunk8_q8_k_ready] / request_contract_flash_preselected_chunk8_q8_k,
        "compute_result_decision"_s <= "contract_flash_decision"_s + completion<EventRun> [guard_preselected_argmax_with_parallel_chunk4_packed_q8_0_ready] / request_contract_flash_preselected_parallel_chunk4_packed_q8_0,
        "compute_result_decision"_s <= "contract_flash_decision"_s + completion<EventRun> [guard_preselected_argmax_with_chunk4_packed_q8_0_ready] / request_contract_flash_preselected_chunk4_packed_q8_0,
        "compute_result_decision"_s <= "contract_flash_decision"_s + completion<EventRun> [guard_preselected_argmax_with_parallel_chunk4_q8_k_ready] / request_contract_flash_preselected_parallel_chunk4_q8_k,
        "compute_result_decision"_s <= "contract_flash_decision"_s + completion<EventRun> [guard_preselected_argmax_with_chunk4_q8_k_ready] / request_contract_flash_preselected_chunk4_q8_k,
        "compute_result_decision"_s <= "contract_flash_decision"_s + completion<EventRun> [guard_preselected_argmax_with_scalar_q8_k_ready] / request_contract_flash_preselected_scalar_q8_k,
        "compute_result_decision"_s <= "contract_flash_decision"_s + completion<EventRun> [guard_preselected_argmax_with_scalar_native_quantized_q8_k_ready] / request_contract_flash_preselected_scalar_native_quantized_q8_k,
        "compute_result_decision"_s <= "contract_flash_decision"_s + completion<EventRun> [guard_preselected_argmax_with_scalar_native_quantized_kernel_ready] / request_contract_flash_preselected_scalar_native_quantized_kernel,
        "compute_result_decision"_s <= "contract_flash_decision"_s + completion<EventRun> [guard_preselected_argmax_with_scalar_kernel_ready] / request_contract_flash_preselected_scalar_kernel,
        "idle"_s <= "contract_nonflash_decision"_s + completion<EventRun> [guard_compute_invalid_request] / mark_invalid_request_from_contract_nonflash_decision,
        "idle"_s <= "contract_nonflash_decision"_s + completion<EventRun> [guard_compute_backend_unavailable] / mark_backend_error_from_contract_nonflash_decision,
        "compute_result_decision"_s <= "contract_nonflash_decision"_s + completion<EventRun> [guard_materialized_logits_with_chunk8_q8_k_ready] / request_contract_nonflash_materialized_chunk8_q8_k,
        "compute_result_decision"_s <= "contract_nonflash_decision"_s + completion<EventRun> [guard_materialized_logits_with_chunk4_packed_q8_0_ready] / request_contract_nonflash_materialized_chunk4_packed_q8_0,
        "compute_result_decision"_s <= "contract_nonflash_decision"_s + completion<EventRun> [guard_materialized_logits_with_chunk4_q8_k_ready] / request_contract_nonflash_materialized_chunk4_q8_k,
        "compute_result_decision"_s <= "contract_nonflash_decision"_s + completion<EventRun> [guard_materialized_logits_with_scalar_packed_q8_0_ready] / request_contract_nonflash_materialized_scalar_packed_q8_0,
        "compute_result_decision"_s <= "contract_nonflash_decision"_s + completion<EventRun> [guard_materialized_logits_with_scalar_q8_k_ready] / request_contract_nonflash_materialized_scalar_q8_k,
        "compute_result_decision"_s <= "contract_nonflash_decision"_s + completion<EventRun> [guard_materialized_logits_with_scalar_native_quantized_q8_k_ready] / request_contract_nonflash_materialized_scalar_native_quantized_q8_k,
        "compute_result_decision"_s <= "contract_nonflash_decision"_s + completion<EventRun> [guard_materialized_logits_with_scalar_native_quantized_kernel_ready] / request_contract_nonflash_materialized_scalar_native_quantized,
        "compute_result_decision"_s <= "contract_nonflash_decision"_s + completion<EventRun> [guard_materialized_logits_with_scalar_kernel_ready] / request_contract_nonflash_materialized_scalar_kernel,
        "compute_result_decision"_s <= "contract_nonflash_decision"_s + completion<EventRun> [guard_preselected_argmax_with_chunk8_q8_k_ready] / request_contract_nonflash_preselected_chunk8_q8_k,
        "compute_result_decision"_s <= "contract_nonflash_decision"_s + completion<EventRun> [guard_preselected_argmax_with_chunk4_packed_q8_0_ready] / request_contract_nonflash_preselected_chunk4_packed_q8_0,
        "compute_result_decision"_s <= "contract_nonflash_decision"_s + completion<EventRun> [guard_preselected_argmax_with_chunk4_q8_k_ready] / request_contract_nonflash_preselected_chunk4_q8_k,
        "compute_result_decision"_s <= "contract_nonflash_decision"_s + completion<EventRun> [guard_preselected_argmax_with_scalar_q8_k_ready] / request_contract_nonflash_preselected_scalar_q8_k,
        "compute_result_decision"_s <= "contract_nonflash_decision"_s + completion<EventRun> [guard_preselected_argmax_with_scalar_native_quantized_q8_k_ready] / request_contract_nonflash_preselected_scalar_native_quantized_q8_k,
        "compute_result_decision"_s <= "contract_nonflash_decision"_s + completion<EventRun> [guard_preselected_argmax_with_scalar_native_quantized_kernel_ready] / request_contract_nonflash_preselected_scalar_native_quantized_kernel,
        "compute_result_decision"_s <= "contract_nonflash_decision"_s + completion<EventRun> [guard_preselected_argmax_with_scalar_kernel_ready] / request_contract_nonflash_preselected_scalar_kernel,
        "idle"_s <= "compute_result_decision"_s + completion<EventRun> [compute_ok] / mark_prefill_cached,
        "idle"_s <= "compute_result_decision"_s + completion<EventRun> [compute_invalid_request] / mark_invalid_request_from_compute_result_decision,
        "idle"_s <= "compute_result_decision"_s + completion<EventRun> [compute_backend_error] / mark_backend_error_from_compute_result_decision,
        "idle"_s <= "idle"_s + unexpected_event<_> / on_unexpected_from_idle,
        "idle"_s <= "slots"_s + unexpected_event<_> / on_unexpected_from_slots,
        "idle"_s <= "slots_decision"_s + unexpected_event<_> / on_unexpected_from_slots_decision,
        "idle"_s <= "snapshot"_s + unexpected_event<_> / on_unexpected_from_snapshot,
        "idle"_s <= "snapshot_decision"_s + unexpected_event<_> / on_unexpected_from_snapshot_decision,
        "idle"_s <= "contract_runtime_decision"_s + unexpected_event<_> / on_unexpected_from_contract_runtime_decision,
        "idle"_s <= "contract_flash_decision"_s + unexpected_event<_> / on_unexpected_from_contract_flash_decision,
        "idle"_s <= "contract_nonflash_decision"_s + unexpected_event<_> / on_unexpected_from_contract_nonflash_decision,
        "idle"_s <= "compute_result_decision"_s + unexpected_event<_> / on_unexpected_from_compute_result_decision,
    }
}

/// Context for `TextGeneratorPrefill` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct TextGeneratorPrefillContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl TextGeneratorPrefillStateMachineContext for TextGeneratorPrefillContext {
    fn compute_backend_error(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/guards.hpp::compute_backend_error
        todo!(
            "TODO: port guard `compute_backend_error` from emel.cpp/src/emel/text/generator/prefill/guards.hpp"
        )
    }
    fn compute_invalid_request(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/guards.hpp::compute_invalid_request
        todo!(
            "TODO: port guard `compute_invalid_request` from emel.cpp/src/emel/text/generator/prefill/guards.hpp"
        )
    }
    fn compute_ok(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/guards.hpp::compute_ok
        todo!(
            "TODO: port guard `compute_ok` from emel.cpp/src/emel/text/generator/prefill/guards.hpp"
        )
    }
    fn flash_runtime_supported(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/guards.hpp::flash_runtime_supported
        todo!(
            "TODO: port guard `flash_runtime_supported` from emel.cpp/src/emel/text/generator/prefill/guards.hpp"
        )
    }
    fn guard_compute_backend_unavailable(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/guards.hpp::guard_compute_backend_unavailable
        todo!(
            "TODO: port guard `guard_compute_backend_unavailable` from emel.cpp/src/emel/text/generator/prefill/guards.hpp"
        )
    }
    fn guard_compute_invalid_request(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/guards.hpp::guard_compute_invalid_request
        todo!(
            "TODO: port guard `guard_compute_invalid_request` from emel.cpp/src/emel/text/generator/prefill/guards.hpp"
        )
    }
    fn guard_materialized_logits_with_chunk4_packed_q8_0_ready(
        &self,
        _event: &EventRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/guards.hpp::guard_materialized_logits_with_chunk4_packed_q8_0_ready
        todo!(
            "TODO: port guard `guard_materialized_logits_with_chunk4_packed_q8_0_ready` from emel.cpp/src/emel/text/generator/prefill/guards.hpp"
        )
    }
    fn guard_materialized_logits_with_chunk4_q8_k_ready(
        &self,
        _event: &EventRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/guards.hpp::guard_materialized_logits_with_chunk4_q8_k_ready
        todo!(
            "TODO: port guard `guard_materialized_logits_with_chunk4_q8_k_ready` from emel.cpp/src/emel/text/generator/prefill/guards.hpp"
        )
    }
    fn guard_materialized_logits_with_chunk8_q8_k_ready(
        &self,
        _event: &EventRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/guards.hpp::guard_materialized_logits_with_chunk8_q8_k_ready
        todo!(
            "TODO: port guard `guard_materialized_logits_with_chunk8_q8_k_ready` from emel.cpp/src/emel/text/generator/prefill/guards.hpp"
        )
    }
    fn guard_materialized_logits_with_parallel_chunk4_packed_q8_0_ready(
        &self,
        _event: &EventRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/guards.hpp::guard_materialized_logits_with_parallel_chunk4_packed_q8_0_ready
        todo!(
            "TODO: port guard `guard_materialized_logits_with_parallel_chunk4_packed_q8_0_ready` from emel.cpp/src/emel/text/generator/prefill/guards.hpp"
        )
    }
    fn guard_materialized_logits_with_parallel_chunk4_q8_k_ready(
        &self,
        _event: &EventRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/guards.hpp::guard_materialized_logits_with_parallel_chunk4_q8_k_ready
        todo!(
            "TODO: port guard `guard_materialized_logits_with_parallel_chunk4_q8_k_ready` from emel.cpp/src/emel/text/generator/prefill/guards.hpp"
        )
    }
    fn guard_materialized_logits_with_parallel_chunk8_q8_k_ready(
        &self,
        _event: &EventRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/guards.hpp::guard_materialized_logits_with_parallel_chunk8_q8_k_ready
        todo!(
            "TODO: port guard `guard_materialized_logits_with_parallel_chunk8_q8_k_ready` from emel.cpp/src/emel/text/generator/prefill/guards.hpp"
        )
    }
    fn guard_materialized_logits_with_scalar_kernel_ready(
        &self,
        _event: &EventRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/guards.hpp::guard_materialized_logits_with_scalar_kernel_ready
        todo!(
            "TODO: port guard `guard_materialized_logits_with_scalar_kernel_ready` from emel.cpp/src/emel/text/generator/prefill/guards.hpp"
        )
    }
    fn guard_materialized_logits_with_scalar_native_quantized_kernel_ready(
        &self,
        _event: &EventRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/guards.hpp::guard_materialized_logits_with_scalar_native_quantized_kernel_ready
        todo!(
            "TODO: port guard `guard_materialized_logits_with_scalar_native_quantized_kernel_ready` from emel.cpp/src/emel/text/generator/prefill/guards.hpp"
        )
    }
    fn guard_materialized_logits_with_scalar_native_quantized_q8_k_ready(
        &self,
        _event: &EventRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/guards.hpp::guard_materialized_logits_with_scalar_native_quantized_q8_k_ready
        todo!(
            "TODO: port guard `guard_materialized_logits_with_scalar_native_quantized_q8_k_ready` from emel.cpp/src/emel/text/generator/prefill/guards.hpp"
        )
    }
    fn guard_materialized_logits_with_scalar_packed_q8_0_ready(
        &self,
        _event: &EventRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/guards.hpp::guard_materialized_logits_with_scalar_packed_q8_0_ready
        todo!(
            "TODO: port guard `guard_materialized_logits_with_scalar_packed_q8_0_ready` from emel.cpp/src/emel/text/generator/prefill/guards.hpp"
        )
    }
    fn guard_materialized_logits_with_scalar_q8_k_ready(
        &self,
        _event: &EventRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/guards.hpp::guard_materialized_logits_with_scalar_q8_k_ready
        todo!(
            "TODO: port guard `guard_materialized_logits_with_scalar_q8_k_ready` from emel.cpp/src/emel/text/generator/prefill/guards.hpp"
        )
    }
    fn guard_preselected_argmax_with_chunk4_packed_q8_0_ready(
        &self,
        _event: &EventRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/guards.hpp::guard_preselected_argmax_with_chunk4_packed_q8_0_ready
        todo!(
            "TODO: port guard `guard_preselected_argmax_with_chunk4_packed_q8_0_ready` from emel.cpp/src/emel/text/generator/prefill/guards.hpp"
        )
    }
    fn guard_preselected_argmax_with_chunk4_q8_k_ready(
        &self,
        _event: &EventRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/guards.hpp::guard_preselected_argmax_with_chunk4_q8_k_ready
        todo!(
            "TODO: port guard `guard_preselected_argmax_with_chunk4_q8_k_ready` from emel.cpp/src/emel/text/generator/prefill/guards.hpp"
        )
    }
    fn guard_preselected_argmax_with_chunk8_q8_k_ready(
        &self,
        _event: &EventRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/guards.hpp::guard_preselected_argmax_with_chunk8_q8_k_ready
        todo!(
            "TODO: port guard `guard_preselected_argmax_with_chunk8_q8_k_ready` from emel.cpp/src/emel/text/generator/prefill/guards.hpp"
        )
    }
    fn guard_preselected_argmax_with_parallel_chunk4_packed_q8_0_ready(
        &self,
        _event: &EventRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/guards.hpp::guard_preselected_argmax_with_parallel_chunk4_packed_q8_0_ready
        todo!(
            "TODO: port guard `guard_preselected_argmax_with_parallel_chunk4_packed_q8_0_ready` from emel.cpp/src/emel/text/generator/prefill/guards.hpp"
        )
    }
    fn guard_preselected_argmax_with_parallel_chunk4_q8_k_ready(
        &self,
        _event: &EventRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/guards.hpp::guard_preselected_argmax_with_parallel_chunk4_q8_k_ready
        todo!(
            "TODO: port guard `guard_preselected_argmax_with_parallel_chunk4_q8_k_ready` from emel.cpp/src/emel/text/generator/prefill/guards.hpp"
        )
    }
    fn guard_preselected_argmax_with_parallel_chunk8_q8_k_ready(
        &self,
        _event: &EventRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/guards.hpp::guard_preselected_argmax_with_parallel_chunk8_q8_k_ready
        todo!(
            "TODO: port guard `guard_preselected_argmax_with_parallel_chunk8_q8_k_ready` from emel.cpp/src/emel/text/generator/prefill/guards.hpp"
        )
    }
    fn guard_preselected_argmax_with_scalar_kernel_ready(
        &self,
        _event: &EventRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/guards.hpp::guard_preselected_argmax_with_scalar_kernel_ready
        todo!(
            "TODO: port guard `guard_preselected_argmax_with_scalar_kernel_ready` from emel.cpp/src/emel/text/generator/prefill/guards.hpp"
        )
    }
    fn guard_preselected_argmax_with_scalar_native_quantized_kernel_ready(
        &self,
        _event: &EventRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/guards.hpp::guard_preselected_argmax_with_scalar_native_quantized_kernel_ready
        todo!(
            "TODO: port guard `guard_preselected_argmax_with_scalar_native_quantized_kernel_ready` from emel.cpp/src/emel/text/generator/prefill/guards.hpp"
        )
    }
    fn guard_preselected_argmax_with_scalar_native_quantized_q8_k_ready(
        &self,
        _event: &EventRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/guards.hpp::guard_preselected_argmax_with_scalar_native_quantized_q8_k_ready
        todo!(
            "TODO: port guard `guard_preselected_argmax_with_scalar_native_quantized_q8_k_ready` from emel.cpp/src/emel/text/generator/prefill/guards.hpp"
        )
    }
    fn guard_preselected_argmax_with_scalar_q8_k_ready(
        &self,
        _event: &EventRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/guards.hpp::guard_preselected_argmax_with_scalar_q8_k_ready
        todo!(
            "TODO: port guard `guard_preselected_argmax_with_scalar_q8_k_ready` from emel.cpp/src/emel/text/generator/prefill/guards.hpp"
        )
    }
    fn mark_backend_error_from_compute_result_decision(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::mark_backend_error
        todo!(
            "TODO: port action `mark_backend_error` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn mark_backend_error_from_contract_flash_decision(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::mark_backend_error
        todo!(
            "TODO: port action `mark_backend_error` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn mark_backend_error_from_contract_nonflash_decision(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::mark_backend_error
        todo!(
            "TODO: port action `mark_backend_error` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn mark_backend_error_from_slots_decision(&mut self, _event: &EventRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::mark_backend_error
        todo!(
            "TODO: port action `mark_backend_error` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn mark_backend_error_from_snapshot_decision(&mut self, _event: &EventRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::mark_backend_error
        todo!(
            "TODO: port action `mark_backend_error` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn mark_invalid_request_from_compute_result_decision(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn mark_invalid_request_from_contract_flash_decision(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn mark_invalid_request_from_contract_nonflash_decision(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn mark_invalid_request_from_slots_decision(&mut self, _event: &EventRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn mark_invalid_request_from_snapshot_decision(&mut self, _event: &EventRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn mark_prefill_cached(&mut self, _event: &EventRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::mark_prefill_cached
        todo!(
            "TODO: port action `mark_prefill_cached` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn nonflash_runtime_required(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/guards.hpp::nonflash_runtime_required
        todo!(
            "TODO: port guard `nonflash_runtime_required` from emel.cpp/src/emel/text/generator/prefill/guards.hpp"
        )
    }
    fn on_unexpected_from_compute_result_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn on_unexpected_from_contract_flash_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn on_unexpected_from_contract_nonflash_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn on_unexpected_from_contract_runtime_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn on_unexpected_from_idle(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn on_unexpected_from_slots(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn on_unexpected_from_slots_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn on_unexpected_from_snapshot(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn on_unexpected_from_snapshot_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn request_contract_flash_materialized_chunk4_packed_q8_0(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::request_contract_flash_materialized_chunk4_packed_q8_0
        todo!(
            "TODO: port action `request_contract_flash_materialized_chunk4_packed_q8_0` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn request_contract_flash_materialized_chunk4_q8_k(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::request_contract_flash_materialized_chunk4_q8_k
        todo!(
            "TODO: port action `request_contract_flash_materialized_chunk4_q8_k` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn request_contract_flash_materialized_chunk8_q8_k(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::request_contract_flash_materialized_chunk8_q8_k
        todo!(
            "TODO: port action `request_contract_flash_materialized_chunk8_q8_k` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn request_contract_flash_materialized_parallel_chunk4_packed_q8_0(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::request_contract_flash_materialized_parallel_chunk4_packed_q8_0
        todo!(
            "TODO: port action `request_contract_flash_materialized_parallel_chunk4_packed_q8_0` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn request_contract_flash_materialized_parallel_chunk4_q8_k(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::request_contract_flash_materialized_parallel_chunk4_q8_k
        todo!(
            "TODO: port action `request_contract_flash_materialized_parallel_chunk4_q8_k` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn request_contract_flash_materialized_parallel_chunk8_q8_k(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::request_contract_flash_materialized_parallel_chunk8_q8_k
        todo!(
            "TODO: port action `request_contract_flash_materialized_parallel_chunk8_q8_k` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn request_contract_flash_materialized_scalar_kernel(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::request_contract_flash_materialized_scalar_kernel
        todo!(
            "TODO: port action `request_contract_flash_materialized_scalar_kernel` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn request_contract_flash_materialized_scalar_native_quantized(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::request_contract_flash_materialized_scalar_native_quantized
        todo!(
            "TODO: port action `request_contract_flash_materialized_scalar_native_quantized` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn request_contract_flash_materialized_scalar_native_quantized_q8_k(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::request_contract_flash_materialized_scalar_native_quantized_q8_k
        todo!(
            "TODO: port action `request_contract_flash_materialized_scalar_native_quantized_q8_k` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn request_contract_flash_materialized_scalar_packed_q8_0(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::request_contract_flash_materialized_scalar_packed_q8_0
        todo!(
            "TODO: port action `request_contract_flash_materialized_scalar_packed_q8_0` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn request_contract_flash_materialized_scalar_q8_k(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::request_contract_flash_materialized_scalar_q8_k
        todo!(
            "TODO: port action `request_contract_flash_materialized_scalar_q8_k` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn request_contract_flash_preselected_chunk4_packed_q8_0(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::request_contract_flash_preselected_chunk4_packed_q8_0
        todo!(
            "TODO: port action `request_contract_flash_preselected_chunk4_packed_q8_0` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn request_contract_flash_preselected_chunk4_q8_k(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::request_contract_flash_preselected_chunk4_q8_k
        todo!(
            "TODO: port action `request_contract_flash_preselected_chunk4_q8_k` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn request_contract_flash_preselected_chunk8_q8_k(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::request_contract_flash_preselected_chunk8_q8_k
        todo!(
            "TODO: port action `request_contract_flash_preselected_chunk8_q8_k` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn request_contract_flash_preselected_parallel_chunk4_packed_q8_0(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::request_contract_flash_preselected_parallel_chunk4_packed_q8_0
        todo!(
            "TODO: port action `request_contract_flash_preselected_parallel_chunk4_packed_q8_0` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn request_contract_flash_preselected_parallel_chunk4_q8_k(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::request_contract_flash_preselected_parallel_chunk4_q8_k
        todo!(
            "TODO: port action `request_contract_flash_preselected_parallel_chunk4_q8_k` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn request_contract_flash_preselected_parallel_chunk8_q8_k(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::request_contract_flash_preselected_parallel_chunk8_q8_k
        todo!(
            "TODO: port action `request_contract_flash_preselected_parallel_chunk8_q8_k` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn request_contract_flash_preselected_scalar_kernel(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::request_contract_flash_preselected_scalar_kernel
        todo!(
            "TODO: port action `request_contract_flash_preselected_scalar_kernel` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn request_contract_flash_preselected_scalar_native_quantized_kernel(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::request_contract_flash_preselected_scalar_native_quantized_kernel
        todo!(
            "TODO: port action `request_contract_flash_preselected_scalar_native_quantized_kernel` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn request_contract_flash_preselected_scalar_native_quantized_q8_k(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::request_contract_flash_preselected_scalar_native_quantized_q8_k
        todo!(
            "TODO: port action `request_contract_flash_preselected_scalar_native_quantized_q8_k` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn request_contract_flash_preselected_scalar_q8_k(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::request_contract_flash_preselected_scalar_q8_k
        todo!(
            "TODO: port action `request_contract_flash_preselected_scalar_q8_k` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn request_contract_nonflash_materialized_chunk4_packed_q8_0(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::request_contract_nonflash_materialized_chunk4_packed_q8_0
        todo!(
            "TODO: port action `request_contract_nonflash_materialized_chunk4_packed_q8_0` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn request_contract_nonflash_materialized_chunk4_q8_k(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::request_contract_nonflash_materialized_chunk4_q8_k
        todo!(
            "TODO: port action `request_contract_nonflash_materialized_chunk4_q8_k` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn request_contract_nonflash_materialized_chunk8_q8_k(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::request_contract_nonflash_materialized_chunk8_q8_k
        todo!(
            "TODO: port action `request_contract_nonflash_materialized_chunk8_q8_k` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn request_contract_nonflash_materialized_scalar_kernel(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::request_contract_nonflash_materialized_scalar_kernel
        todo!(
            "TODO: port action `request_contract_nonflash_materialized_scalar_kernel` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn request_contract_nonflash_materialized_scalar_native_quantized(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::request_contract_nonflash_materialized_scalar_native_quantized
        todo!(
            "TODO: port action `request_contract_nonflash_materialized_scalar_native_quantized` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn request_contract_nonflash_materialized_scalar_native_quantized_q8_k(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::request_contract_nonflash_materialized_scalar_native_quantized_q8_k
        todo!(
            "TODO: port action `request_contract_nonflash_materialized_scalar_native_quantized_q8_k` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn request_contract_nonflash_materialized_scalar_packed_q8_0(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::request_contract_nonflash_materialized_scalar_packed_q8_0
        todo!(
            "TODO: port action `request_contract_nonflash_materialized_scalar_packed_q8_0` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn request_contract_nonflash_materialized_scalar_q8_k(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::request_contract_nonflash_materialized_scalar_q8_k
        todo!(
            "TODO: port action `request_contract_nonflash_materialized_scalar_q8_k` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn request_contract_nonflash_preselected_chunk4_packed_q8_0(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::request_contract_nonflash_preselected_chunk4_packed_q8_0
        todo!(
            "TODO: port action `request_contract_nonflash_preselected_chunk4_packed_q8_0` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn request_contract_nonflash_preselected_chunk4_q8_k(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::request_contract_nonflash_preselected_chunk4_q8_k
        todo!(
            "TODO: port action `request_contract_nonflash_preselected_chunk4_q8_k` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn request_contract_nonflash_preselected_chunk8_q8_k(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::request_contract_nonflash_preselected_chunk8_q8_k
        todo!(
            "TODO: port action `request_contract_nonflash_preselected_chunk8_q8_k` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn request_contract_nonflash_preselected_scalar_kernel(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::request_contract_nonflash_preselected_scalar_kernel
        todo!(
            "TODO: port action `request_contract_nonflash_preselected_scalar_kernel` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn request_contract_nonflash_preselected_scalar_native_quantized_kernel(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::request_contract_nonflash_preselected_scalar_native_quantized_kernel
        todo!(
            "TODO: port action `request_contract_nonflash_preselected_scalar_native_quantized_kernel` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn request_contract_nonflash_preselected_scalar_native_quantized_q8_k(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::request_contract_nonflash_preselected_scalar_native_quantized_q8_k
        todo!(
            "TODO: port action `request_contract_nonflash_preselected_scalar_native_quantized_q8_k` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn request_contract_nonflash_preselected_scalar_q8_k(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::request_contract_nonflash_preselected_scalar_q8_k
        todo!(
            "TODO: port action `request_contract_nonflash_preselected_scalar_q8_k` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn request_memory_snapshot(&mut self, _event: &EventRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::request_memory_snapshot
        todo!(
            "TODO: port action `request_memory_snapshot` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn request_slots(&mut self, _event: &EventRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/actions.hpp::request_slots
        todo!(
            "TODO: port action `request_slots` from emel.cpp/src/emel/text/generator/prefill/actions.hpp"
        )
    }
    fn slots_backend_error(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/guards.hpp::slots_backend_error
        todo!(
            "TODO: port guard `slots_backend_error` from emel.cpp/src/emel/text/generator/prefill/guards.hpp"
        )
    }
    fn slots_invalid_request(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/guards.hpp::slots_invalid_request
        todo!(
            "TODO: port guard `slots_invalid_request` from emel.cpp/src/emel/text/generator/prefill/guards.hpp"
        )
    }
    fn slots_ok(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/guards.hpp::slots_ok
        todo!(
            "TODO: port guard `slots_ok` from emel.cpp/src/emel/text/generator/prefill/guards.hpp"
        )
    }
    fn snapshot_backend_error(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/guards.hpp::snapshot_backend_error
        todo!(
            "TODO: port guard `snapshot_backend_error` from emel.cpp/src/emel/text/generator/prefill/guards.hpp"
        )
    }
    fn snapshot_invalid_request(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/guards.hpp::snapshot_invalid_request
        todo!(
            "TODO: port guard `snapshot_invalid_request` from emel.cpp/src/emel/text/generator/prefill/guards.hpp"
        )
    }
    fn snapshot_ok(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/prefill/guards.hpp::snapshot_ok
        todo!(
            "TODO: port guard `snapshot_ok` from emel.cpp/src/emel/text/generator/prefill/guards.hpp"
        )
    }
}
