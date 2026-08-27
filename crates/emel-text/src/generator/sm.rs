//! State machine scaffold port — not a stable public API.
//! Bodies are stubs (`todo!`) until contexts/guards/actions are ported from C++.

#![allow(
    clippy::enum_variant_names,
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

// --- machine TextGenerator from emel.cpp/src/emel/text/generator/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventCaptureDiagnostics;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventCaptureGraphLifecycle;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventConfigureBenchmarkLane;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventGenerateRun;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventInitializeRun;

sml! {
    TextGenerator {
        "initializing"_s <= *"uninitialized"_s + event<EventInitializeRun> [valid_initialize],
        "initialize_error_channel_decision"_s <= "uninitialized"_s + event<EventInitializeRun> [invalid_initialize] / reject_initialize_from_uninitialized,
        "initializing"_s <= "ready"_s + event<EventInitializeRun> [valid_initialize],
        "initialize_error_channel_decision"_s <= "ready"_s + event<EventInitializeRun> [invalid_initialize] / reject_initialize_from_ready,
        "initializer_result_decision"_s <= "initializing"_s + completion<EventInitializeRun> / request_initializer,
        "initialize_done_channel_decision"_s <= "initializer_result_decision"_s + completion<EventInitializeRun> [initialize_result_none],
        "initialize_error_channel_decision"_s <= "initializer_result_decision"_s + completion<EventInitializeRun> [initialize_result_invalid_request],
        "initialize_error_channel_decision"_s <= "initializer_result_decision"_s + completion<EventInitializeRun> [initialize_result_backend],
        "ready"_s <= "initialize_done_channel_decision"_s + completion<EventInitializeRun> [initialize_done_callback_with_error_out] / dispatch_initialize_done_with_callback_and_error_out,
        "ready"_s <= "initialize_done_channel_decision"_s + completion<EventInitializeRun> [initialize_done_callback_without_error_out] / dispatch_initialize_done_with_callback_only,
        "ready"_s <= "initialize_done_channel_decision"_s + completion<EventInitializeRun> [initialize_no_done_callback_with_error_out] / dispatch_initialize_done_with_error_out_only,
        "ready"_s <= "initialize_done_channel_decision"_s + completion<EventInitializeRun> [initialize_no_done_callback_without_error_out] / dispatch_initialize_done_without_channels,
        "uninitialized"_s <= "initialize_error_channel_decision"_s + completion<EventInitializeRun> [initialize_error_callback_with_error_out] / dispatch_initialize_error_with_callback_and_error_out,
        "uninitialized"_s <= "initialize_error_channel_decision"_s + completion<EventInitializeRun> [initialize_error_callback_without_error_out] / dispatch_initialize_error_with_callback_only,
        "uninitialized"_s <= "initialize_error_channel_decision"_s + completion<EventInitializeRun> [initialize_no_error_callback_with_error_out] / dispatch_initialize_error_with_error_out_only,
        "uninitialized"_s <= "initialize_error_channel_decision"_s + completion<EventInitializeRun> [initialize_no_error_callback_without_error_out] / dispatch_initialize_error_without_channels,
        "generate_uninitialized_error_channel_decision"_s <= "uninitialized"_s + event<EventGenerateRun> / reject_uninitialized_generate,
        "reset_sequence"_s <= "ready"_s + event<EventGenerateRun> [valid_generate_with_reset] / begin_generate_from_ready,
        "conditioning"_s <= "ready"_s + event<EventGenerateRun> [valid_generate_without_reset] / begin_generate_from_ready,
        "generate_ready_error_channel_decision"_s <= "ready"_s + event<EventGenerateRun> [invalid_generate] / reject_invalid_generate,
        "reset_sequence_decision"_s <= "reset_sequence"_s + completion<EventGenerateRun> / request_reset_sequence,
        "conditioning"_s <= "reset_sequence_decision"_s + completion<EventGenerateRun> [reset_sequence_ok] / mark_sequence_clear,
        "generate_ready_error_channel_decision"_s <= "reset_sequence_decision"_s + completion<EventGenerateRun> [reset_sequence_invalid_request] / mark_invalid_request_from_reset_sequence_decision,
        "generate_ready_error_channel_decision"_s <= "reset_sequence_decision"_s + completion<EventGenerateRun> [reset_sequence_backend_error] / mark_backend_error_from_reset_sequence_decision,
        "conditioning_decision"_s <= "conditioning"_s + completion<EventGenerateRun> / request_conditioning,
        "planning_chunk8"_s <= "conditioning_decision"_s + completion<EventGenerateRun> [conditioning_ok_with_chunk8_prefill],
        "planning_chunk4"_s <= "conditioning_decision"_s + completion<EventGenerateRun> [conditioning_ok_with_chunk4_prefill],
        "planning_scalar"_s <= "conditioning_decision"_s + completion<EventGenerateRun> [conditioning_ok_with_scalar_prefill],
        "generate_ready_error_channel_decision"_s <= "conditioning_decision"_s + completion<EventGenerateRun> [conditioning_invalid_request] / mark_invalid_request_from_conditioning_decision,
        "generate_ready_error_channel_decision"_s <= "conditioning_decision"_s + completion<EventGenerateRun> [conditioning_backend_error] / mark_backend_error_from_conditioning_decision,
        "planning_decision"_s <= "planning_chunk8"_s + completion<EventGenerateRun> / request_planning_chunk8,
        "planning_decision"_s <= "planning_chunk4"_s + completion<EventGenerateRun> / request_planning_chunk4,
        "planning_decision"_s <= "planning_scalar"_s + completion<EventGenerateRun> / request_planning_scalar,
        "sequence_allocating"_s <= "planning_decision"_s + completion<EventGenerateRun> [planning_ok],
        "generate_ready_error_channel_decision"_s <= "planning_decision"_s + completion<EventGenerateRun> [planning_invalid_request] / mark_invalid_request_from_planning_decision,
        "generate_ready_error_channel_decision"_s <= "planning_decision"_s + completion<EventGenerateRun> [planning_backend_error] / mark_backend_error_from_planning_decision,
        "sequence_allocating_decision"_s <= "sequence_allocating"_s + completion<EventGenerateRun> / request_allocate_sequence,
        "prefill_running"_s <= "sequence_allocating_decision"_s + completion<EventGenerateRun> [allocate_sequence_ok] / mark_sequence_live,
        "generate_ready_error_channel_decision"_s <= "sequence_allocating_decision"_s + completion<EventGenerateRun> [allocate_sequence_invalid_request] / mark_invalid_request_from_sequence_allocating_decision,
        "generate_ready_error_channel_decision"_s <= "sequence_allocating_decision"_s + completion<EventGenerateRun> [allocate_sequence_backend_error] / mark_backend_error_from_sequence_allocating_decision,
        "prefill_result_decision"_s <= "prefill_running"_s + completion<EventGenerateRun> [prefill_dispatch_available] / request_prefill,
        "generate_ready_error_channel_decision"_s <= "prefill_running"_s + completion<EventGenerateRun> [prefill_dispatch_unavailable] / mark_backend_error_from_prefill_running,
        "decode_selection_mode_decision"_s <= "prefill_result_decision"_s + completion<EventGenerateRun> [prefill_result_ok_with_materialized_logits_contract],
        "decode_sample_preselected"_s <= "prefill_result_decision"_s + completion<EventGenerateRun> [prefill_result_ok_with_preselected_argmax_contract],
        "generate_ready_error_channel_decision"_s <= "prefill_result_decision"_s + completion<EventGenerateRun> [prefill_result_invalid_request],
        "generate_ready_error_channel_decision"_s <= "prefill_result_decision"_s + completion<EventGenerateRun> [prefill_result_backend_error],
        "decode_sample"_s <= "decode_selection_mode_decision"_s + completion<EventGenerateRun> [decode_uses_materialized_logits],
        "decode_preselected_argmax"_s <= "decode_selection_mode_decision"_s + completion<EventGenerateRun> [decode_uses_preselected_argmax],
        "decode_sample_decision"_s <= "decode_sample"_s + completion<EventGenerateRun> / request_decode_sample,
        "decode_preselected_argmax_decision"_s <= "decode_preselected_argmax"_s + completion<EventGenerateRun> [decode_argmax_ready] / request_decode_select_argmax,
        "generate_ready_error_channel_decision"_s <= "decode_preselected_argmax"_s + completion<EventGenerateRun> [decode_argmax_invalid_request] / mark_invalid_request_from_decode_preselected_argmax,
        "decode_sample_preselected"_s <= "decode_preselected_argmax_decision"_s + completion<EventGenerateRun> [decode_sample_ok],
        "generate_ready_error_channel_decision"_s <= "decode_preselected_argmax_decision"_s + completion<EventGenerateRun> [decode_sample_invalid_request] / mark_invalid_request_from_decode_preselected_argmax_decision,
        "generate_ready_error_channel_decision"_s <= "decode_preselected_argmax_decision"_s + completion<EventGenerateRun> [decode_sample_backend_error] / mark_backend_error_from_decode_preselected_argmax_decision,
        "decode_sample_preselected_decision"_s <= "decode_sample_preselected"_s + completion<EventGenerateRun> / request_decode_sample_preselected,
        "decode_render"_s <= "decode_sample_decision"_s + completion<EventGenerateRun> [decode_sample_ok],
        "generate_ready_error_channel_decision"_s <= "decode_sample_decision"_s + completion<EventGenerateRun> [decode_sample_invalid_request] / mark_invalid_request_from_decode_sample_decision,
        "generate_ready_error_channel_decision"_s <= "decode_sample_decision"_s + completion<EventGenerateRun> [decode_sample_backend_error] / mark_backend_error_from_decode_sample_decision,
        "decode_render"_s <= "decode_sample_preselected_decision"_s + completion<EventGenerateRun> [decode_sample_ok],
        "generate_ready_error_channel_decision"_s <= "decode_sample_preselected_decision"_s + completion<EventGenerateRun> [decode_sample_invalid_request] / mark_invalid_request_from_decode_sample_preselected_decision,
        "generate_ready_error_channel_decision"_s <= "decode_sample_preselected_decision"_s + completion<EventGenerateRun> [decode_sample_backend_error] / mark_backend_error_from_decode_sample_preselected_decision,
        "decode_render_decision"_s <= "decode_render"_s + completion<EventGenerateRun> / request_decode_render,
        "decode_loop_decision"_s <= "decode_render_decision"_s + completion<EventGenerateRun> [decode_render_ok] / commit_render_output,
        "generate_ready_error_channel_decision"_s <= "decode_render_decision"_s + completion<EventGenerateRun> [decode_render_invalid_request] / mark_invalid_request_from_decode_render_decision,
        "generate_ready_error_channel_decision"_s <= "decode_render_decision"_s + completion<EventGenerateRun> [decode_render_backend_error] / mark_backend_error_from_decode_render_decision,
        "decode_slots"_s <= "decode_loop_decision"_s + completion<EventGenerateRun> [decode_should_continue],
        "flushing"_s <= "decode_loop_decision"_s + completion<EventGenerateRun> [decode_complete],
        "decode_slots_decision"_s <= "decode_slots"_s + completion<EventGenerateRun> / request_decode_slots,
        "snapshot_decode"_s <= "decode_slots_decision"_s + completion<EventGenerateRun> [decode_slots_ok],
        "generate_ready_error_channel_decision"_s <= "decode_slots_decision"_s + completion<EventGenerateRun> [decode_slots_invalid_request] / mark_invalid_request_from_decode_slots_decision,
        "generate_ready_error_channel_decision"_s <= "decode_slots_decision"_s + completion<EventGenerateRun> [decode_slots_backend_error] / mark_backend_error_from_decode_slots_decision,
        "snapshot_decode_decision"_s <= "snapshot_decode"_s + completion<EventGenerateRun> / request_memory_snapshot,
        "decode_compute_runtime_decision"_s <= "snapshot_decode_decision"_s + completion<EventGenerateRun> [decode_snapshot_ok],
        "generate_ready_error_channel_decision"_s <= "snapshot_decode_decision"_s + completion<EventGenerateRun> [decode_snapshot_invalid_request] / mark_invalid_request_from_snapshot_decode_decision,
        "generate_ready_error_channel_decision"_s <= "snapshot_decode_decision"_s + completion<EventGenerateRun> [decode_snapshot_backend_error] / mark_backend_error_from_snapshot_decode_decision,
        "decode_compute_flash"_s <= "decode_compute_runtime_decision"_s + completion<EventGenerateRun> [decode_flash_runtime_supported],
        "decode_compute_nonflash"_s <= "decode_compute_runtime_decision"_s + completion<EventGenerateRun> [decode_nonflash_runtime_required],
        "generate_ready_error_channel_decision"_s <= "decode_compute_flash"_s + completion<EventGenerateRun> [guard_decode_compute_invalid_request] / mark_invalid_request_from_decode_compute_flash,
        "generate_ready_error_channel_decision"_s <= "decode_compute_flash"_s + completion<EventGenerateRun> [guard_decode_compute_backend_unavailable] / mark_backend_error_from_decode_compute_flash,
        "decode_compute_flash_preselected_argmax"_s <= "decode_compute_flash"_s + completion<EventGenerateRun> [guard_decode_preselected_direct_ready],
        "decode_compute_flash_decision"_s <= "decode_compute_flash"_s + completion<EventGenerateRun> [guard_decode_materialized_streamed_scalar_packed_q8_0_ready] / request_decode_compute_flash_packed_q8_0_streamed,
        "decode_compute_flash_decision"_s <= "decode_compute_flash"_s + completion<EventGenerateRun> [guard_decode_materialized_streamed_scalar_q8_k_ready] / request_decode_compute_flash_q8_k_streamed,
        "decode_compute_flash_decision"_s <= "decode_compute_flash"_s + completion<EventGenerateRun> [guard_decode_materialized_streamed_scalar_native_quantized_q8_k_ready] / request_decode_compute_flash_native_quantized_q8_k_logits_streamed,
        "decode_compute_flash_decision"_s <= "decode_compute_flash"_s + completion<EventGenerateRun> [guard_decode_materialized_streamed_scalar_native_quantized_ready] / request_decode_compute_flash_native_quantized_streamed,
        "decode_compute_flash_decision"_s <= "decode_compute_flash"_s + completion<EventGenerateRun> [guard_decode_materialized_streamed_scalar_kernel_ready] / request_decode_compute_flash_kernel_streamed,
        "decode_compute_flash_decision"_s <= "decode_compute_flash"_s + completion<EventGenerateRun> [guard_decode_materialized_parallel_scalar_packed_q8_0_ready] / request_decode_compute_flash_parallel_packed_q8_0,
        "decode_compute_flash_decision"_s <= "decode_compute_flash"_s + completion<EventGenerateRun> [guard_decode_materialized_scalar_packed_q8_0_ready] / request_decode_compute_flash_packed_q8_0,
        "decode_compute_flash_decision"_s <= "decode_compute_flash"_s + completion<EventGenerateRun> [guard_decode_materialized_parallel_scalar_q8_k_ready] / request_decode_compute_flash_parallel_q8_k,
        "decode_compute_flash_decision"_s <= "decode_compute_flash"_s + completion<EventGenerateRun> [guard_decode_materialized_scalar_q8_k_ready] / request_decode_compute_flash_q8_k,
        "decode_compute_flash_decision"_s <= "decode_compute_flash"_s + completion<EventGenerateRun> [guard_decode_materialized_parallel_scalar_native_quantized_q8_k_ready] / request_decode_compute_flash_parallel_native_quantized_q8_k_logits,
        "decode_compute_flash_decision"_s <= "decode_compute_flash"_s + completion<EventGenerateRun> [guard_decode_materialized_scalar_native_quantized_q8_k_ready] / request_decode_compute_flash_native_quantized_q8_k_logits,
        "decode_compute_flash_decision"_s <= "decode_compute_flash"_s + completion<EventGenerateRun> [guard_decode_materialized_parallel_scalar_native_quantized_kernel_ready] / request_decode_compute_flash_parallel_native_quantized,
        "decode_compute_flash_decision"_s <= "decode_compute_flash"_s + completion<EventGenerateRun> [guard_decode_materialized_scalar_native_quantized_kernel_ready] / request_decode_compute_flash_native_quantized,
        "decode_compute_flash_decision"_s <= "decode_compute_flash"_s + completion<EventGenerateRun> [guard_decode_materialized_parallel_scalar_kernel_ready] / request_decode_compute_flash_parallel_kernel,
        "decode_compute_flash_decision"_s <= "decode_compute_flash"_s + completion<EventGenerateRun> [guard_decode_materialized_scalar_kernel_ready] / request_decode_compute_flash_kernel,
        "generate_ready_error_channel_decision"_s <= "decode_compute_nonflash"_s + completion<EventGenerateRun> [guard_decode_compute_invalid_request] / mark_invalid_request_from_decode_compute_nonflash,
        "generate_ready_error_channel_decision"_s <= "decode_compute_nonflash"_s + completion<EventGenerateRun> [guard_decode_compute_backend_unavailable] / mark_backend_error_from_decode_compute_nonflash,
        "decode_compute_nonflash_preselected_argmax"_s <= "decode_compute_nonflash"_s + completion<EventGenerateRun> [guard_decode_preselected_direct_ready],
        "decode_compute_nonflash_decision"_s <= "decode_compute_nonflash"_s + completion<EventGenerateRun> [guard_decode_materialized_streamed_scalar_packed_q8_0_ready] / request_decode_compute_nonflash_packed_q8_0_streamed,
        "decode_compute_nonflash_decision"_s <= "decode_compute_nonflash"_s + completion<EventGenerateRun> [guard_decode_materialized_streamed_scalar_q8_k_ready] / request_decode_compute_nonflash_q8_k_streamed,
        "decode_compute_nonflash_decision"_s <= "decode_compute_nonflash"_s + completion<EventGenerateRun> [guard_decode_materialized_streamed_scalar_native_quantized_q8_k_ready] / request_decode_compute_nonflash_native_quantized_q8_k_logits_streamed,
        "decode_compute_nonflash_decision"_s <= "decode_compute_nonflash"_s + completion<EventGenerateRun> [guard_decode_materialized_streamed_scalar_native_quantized_ready] / request_decode_compute_nonflash_native_quantized_streamed,
        "decode_compute_nonflash_decision"_s <= "decode_compute_nonflash"_s + completion<EventGenerateRun> [guard_decode_materialized_streamed_scalar_kernel_ready] / request_decode_compute_nonflash_kernel_streamed,
        "decode_compute_nonflash_decision"_s <= "decode_compute_nonflash"_s + completion<EventGenerateRun> [guard_decode_materialized_scalar_packed_q8_0_ready] / request_decode_compute_nonflash_packed_q8_0,
        "decode_compute_nonflash_decision"_s <= "decode_compute_nonflash"_s + completion<EventGenerateRun> [guard_decode_materialized_scalar_q8_k_ready] / request_decode_compute_nonflash_q8_k,
        "decode_compute_nonflash_decision"_s <= "decode_compute_nonflash"_s + completion<EventGenerateRun> [guard_decode_materialized_scalar_native_quantized_q8_k_ready] / request_decode_compute_nonflash_native_quantized_q8_k_logits,
        "decode_compute_nonflash_decision"_s <= "decode_compute_nonflash"_s + completion<EventGenerateRun> [guard_decode_materialized_scalar_native_quantized_kernel_ready] / request_decode_compute_nonflash_native_quantized,
        "decode_compute_nonflash_decision"_s <= "decode_compute_nonflash"_s + completion<EventGenerateRun> [guard_decode_materialized_scalar_kernel_ready] / request_decode_compute_nonflash_kernel,
        "generate_ready_error_channel_decision"_s <= "decode_compute_flash_preselected_argmax"_s + completion<EventGenerateRun> [guard_decode_preselected_compute_invalid_request] / mark_invalid_request_from_decode_compute_flash_preselected_argmax,
        "generate_ready_error_channel_decision"_s <= "decode_compute_flash_preselected_argmax"_s + completion<EventGenerateRun> [guard_decode_compute_backend_unavailable] / mark_backend_error_from_decode_compute_flash_preselected_argmax,
        "decode_compute_flash_preselected_argmax_decision"_s <= "decode_compute_flash_preselected_argmax"_s + completion<EventGenerateRun> [guard_decode_preselected_argmax_q8_k_streamed_ready] / request_decode_compute_flash_preselected_argmax_q8_k_streamed,
        "decode_compute_flash_preselected_argmax_decision"_s <= "decode_compute_flash_preselected_argmax"_s + completion<EventGenerateRun> [guard_decode_preselected_argmax_native_quantized_q8_k_streamed_ready] / request_decode_compute_flash_preselected_argmax_native_quantized_q8_k_streamed,
        "decode_compute_flash_preselected_argmax_decision"_s <= "decode_compute_flash_preselected_argmax"_s + completion<EventGenerateRun> [guard_decode_preselected_argmax_native_quantized_kernel_streamed_ready] / request_decode_compute_flash_preselected_argmax_native_quantized_kernel_streamed,
        "decode_compute_flash_preselected_argmax_decision"_s <= "decode_compute_flash_preselected_argmax"_s + completion<EventGenerateRun> [guard_decode_preselected_argmax_kernel_streamed_ready] / request_decode_compute_flash_preselected_argmax_kernel_streamed,
        "decode_compute_flash_preselected_argmax_decision"_s <= "decode_compute_flash_preselected_argmax"_s + completion<EventGenerateRun> [guard_decode_preselected_parallel_argmax_q8_k_ready] / request_decode_compute_flash_parallel_preselected_argmax_q8_k,
        "decode_compute_flash_preselected_argmax_decision"_s <= "decode_compute_flash_preselected_argmax"_s + completion<EventGenerateRun> [guard_decode_preselected_argmax_q8_k_ready] / request_decode_compute_flash_preselected_argmax_q8_k,
        "decode_compute_flash_preselected_argmax_decision"_s <= "decode_compute_flash_preselected_argmax"_s + completion<EventGenerateRun> [guard_decode_preselected_parallel_argmax_native_quantized_q8_k_ready] / request_decode_compute_flash_parallel_preselected_argmax_native_quantized_q8_k,
        "decode_compute_flash_preselected_argmax_decision"_s <= "decode_compute_flash_preselected_argmax"_s + completion<EventGenerateRun> [guard_decode_preselected_argmax_native_quantized_q8_k_ready] / request_decode_compute_flash_preselected_argmax_native_quantized_q8_k,
        "decode_compute_flash_preselected_argmax_decision"_s <= "decode_compute_flash_preselected_argmax"_s + completion<EventGenerateRun> [guard_decode_preselected_parallel_argmax_native_quantized_kernel_ready] / request_decode_compute_flash_parallel_preselected_argmax_native_quantized_kernel,
        "decode_compute_flash_preselected_argmax_decision"_s <= "decode_compute_flash_preselected_argmax"_s + completion<EventGenerateRun> [guard_decode_preselected_argmax_native_quantized_kernel_ready] / request_decode_compute_flash_preselected_argmax_native_quantized_kernel,
        "decode_compute_flash_preselected_argmax_decision"_s <= "decode_compute_flash_preselected_argmax"_s + completion<EventGenerateRun> [guard_decode_preselected_parallel_argmax_kernel_ready] / request_decode_compute_flash_parallel_preselected_argmax_kernel,
        "decode_compute_flash_preselected_argmax_decision"_s <= "decode_compute_flash_preselected_argmax"_s + completion<EventGenerateRun> [guard_decode_preselected_argmax_kernel_ready] / request_decode_compute_flash_preselected_argmax_kernel,
        "generate_ready_error_channel_decision"_s <= "decode_compute_nonflash_preselected_argmax"_s + completion<EventGenerateRun> [guard_decode_preselected_compute_invalid_request] / mark_invalid_request_from_decode_compute_nonflash_preselected_argmax,
        "generate_ready_error_channel_decision"_s <= "decode_compute_nonflash_preselected_argmax"_s + completion<EventGenerateRun> [guard_decode_compute_backend_unavailable] / mark_backend_error_from_decode_compute_nonflash_preselected_argmax,
        "decode_compute_nonflash_preselected_argmax_decision"_s <= "decode_compute_nonflash_preselected_argmax"_s + completion<EventGenerateRun> [guard_decode_preselected_argmax_q8_k_streamed_ready] / request_decode_compute_nonflash_preselected_argmax_q8_k_streamed,
        "decode_compute_nonflash_preselected_argmax_decision"_s <= "decode_compute_nonflash_preselected_argmax"_s + completion<EventGenerateRun> [guard_decode_preselected_argmax_native_quantized_q8_k_streamed_ready] / request_decode_compute_nonflash_preselected_argmax_native_quantized_q8_k_streamed,
        "decode_compute_nonflash_preselected_argmax_decision"_s <= "decode_compute_nonflash_preselected_argmax"_s + completion<EventGenerateRun> [guard_decode_preselected_argmax_native_quantized_kernel_streamed_ready] / request_decode_compute_nonflash_preselected_argmax_native_quantized_kernel_streamed,
        "decode_compute_nonflash_preselected_argmax_decision"_s <= "decode_compute_nonflash_preselected_argmax"_s + completion<EventGenerateRun> [guard_decode_preselected_argmax_kernel_streamed_ready] / request_decode_compute_nonflash_preselected_argmax_kernel_streamed,
        "decode_compute_nonflash_preselected_argmax_decision"_s <= "decode_compute_nonflash_preselected_argmax"_s + completion<EventGenerateRun> [guard_decode_preselected_argmax_q8_k_ready] / request_decode_compute_nonflash_preselected_argmax_q8_k,
        "decode_compute_nonflash_preselected_argmax_decision"_s <= "decode_compute_nonflash_preselected_argmax"_s + completion<EventGenerateRun> [guard_decode_preselected_argmax_native_quantized_q8_k_ready] / request_decode_compute_nonflash_preselected_argmax_native_quantized_q8_k,
        "decode_compute_nonflash_preselected_argmax_decision"_s <= "decode_compute_nonflash_preselected_argmax"_s + completion<EventGenerateRun> [guard_decode_preselected_argmax_native_quantized_kernel_ready] / request_decode_compute_nonflash_preselected_argmax_native_quantized_kernel,
        "decode_compute_nonflash_preselected_argmax_decision"_s <= "decode_compute_nonflash_preselected_argmax"_s + completion<EventGenerateRun> [guard_decode_preselected_argmax_kernel_ready] / request_decode_compute_nonflash_preselected_argmax_kernel,
        "decode_selection_mode_decision"_s <= "decode_compute_flash_decision"_s + completion<EventGenerateRun> [decode_compute_ok] / advance_kv_cache_from_decode_compute_flash_decision,
        "generate_ready_error_channel_decision"_s <= "decode_compute_flash_decision"_s + completion<EventGenerateRun> [decode_compute_invalid_request] / mark_invalid_request_from_decode_compute_flash_decision,
        "generate_ready_error_channel_decision"_s <= "decode_compute_flash_decision"_s + completion<EventGenerateRun> [decode_compute_backend_error] / mark_backend_error_from_decode_compute_flash_decision,
        "decode_selection_mode_decision"_s <= "decode_compute_nonflash_decision"_s + completion<EventGenerateRun> [decode_compute_ok] / advance_kv_cache_from_decode_compute_nonflash_decision,
        "generate_ready_error_channel_decision"_s <= "decode_compute_nonflash_decision"_s + completion<EventGenerateRun> [decode_compute_invalid_request] / mark_invalid_request_from_decode_compute_nonflash_decision,
        "generate_ready_error_channel_decision"_s <= "decode_compute_nonflash_decision"_s + completion<EventGenerateRun> [decode_compute_backend_error] / mark_backend_error_from_decode_compute_nonflash_decision,
        "decode_sample_preselected"_s <= "decode_compute_flash_preselected_argmax_decision"_s + completion<EventGenerateRun> [decode_compute_ok] / advance_kv_cache_from_decode_compute_flash_preselected_argmax_decision,
        "generate_ready_error_channel_decision"_s <= "decode_compute_flash_preselected_argmax_decision"_s + completion<EventGenerateRun> [decode_compute_invalid_request] / mark_invalid_request_from_decode_compute_flash_preselected_argmax_decision,
        "generate_ready_error_channel_decision"_s <= "decode_compute_flash_preselected_argmax_decision"_s + completion<EventGenerateRun> [decode_compute_backend_error] / mark_backend_error_from_decode_compute_flash_preselected_argmax_decision,
        "decode_sample_preselected"_s <= "decode_compute_nonflash_preselected_argmax_decision"_s + completion<EventGenerateRun> [decode_compute_ok] / advance_kv_cache_from_decode_compute_nonflash_preselected_argmax_decision,
        "generate_ready_error_channel_decision"_s <= "decode_compute_nonflash_preselected_argmax_decision"_s + completion<EventGenerateRun> [decode_compute_invalid_request] / mark_invalid_request_from_decode_compute_nonflash_preselected_argmax_decision,
        "generate_ready_error_channel_decision"_s <= "decode_compute_nonflash_preselected_argmax_decision"_s + completion<EventGenerateRun> [decode_compute_backend_error] / mark_backend_error_from_decode_compute_nonflash_preselected_argmax_decision,
        "flushing_decision"_s <= "flushing"_s + completion<EventGenerateRun> / request_flush,
        "generate_done_channel_decision"_s <= "flushing_decision"_s + completion<EventGenerateRun> [flush_ok] / commit_flush_output,
        "generate_ready_error_channel_decision"_s <= "flushing_decision"_s + completion<EventGenerateRun> [flush_invalid_request] / mark_invalid_request_from_flushing_decision,
        "generate_ready_error_channel_decision"_s <= "flushing_decision"_s + completion<EventGenerateRun> [flush_backend_error] / mark_backend_error_from_flushing_decision,
        "ready"_s <= "generate_done_channel_decision"_s + completion<EventGenerateRun> [generate_done_callback_with_error_out] / dispatch_generate_done_with_callback_and_error_out,
        "ready"_s <= "generate_done_channel_decision"_s + completion<EventGenerateRun> [generate_done_callback_without_error_out] / dispatch_generate_done_with_callback_only,
        "ready"_s <= "generate_done_channel_decision"_s + completion<EventGenerateRun> [generate_no_done_callback_with_error_out] / dispatch_generate_done_with_error_out_only,
        "ready"_s <= "generate_done_channel_decision"_s + completion<EventGenerateRun> [generate_no_done_callback_without_error_out] / dispatch_generate_done_without_channels,
        "ready"_s <= "generate_ready_error_channel_decision"_s + completion<EventGenerateRun> [generate_error_callback_with_error_out] / dispatch_generate_error_with_callback_and_error_out_from_generate_ready_error_channel_decision,
        "ready"_s <= "generate_ready_error_channel_decision"_s + completion<EventGenerateRun> [generate_error_callback_without_error_out] / dispatch_generate_error_with_callback_only_from_generate_ready_error_channel_decision,
        "ready"_s <= "generate_ready_error_channel_decision"_s + completion<EventGenerateRun> [generate_no_error_callback_with_error_out] / dispatch_generate_error_with_error_out_only_from_generate_ready_error_channel_decision,
        "ready"_s <= "generate_ready_error_channel_decision"_s + completion<EventGenerateRun> [generate_no_error_callback_without_error_out] / dispatch_generate_error_without_channels_from_generate_ready_error_channel_decision,
        "uninitialized"_s <= "generate_uninitialized_error_channel_decision"_s + completion<EventGenerateRun> [generate_error_callback_with_error_out] / dispatch_generate_error_with_callback_and_error_out_from_generate_uninitialized_error_channel_decision,
        "uninitialized"_s <= "generate_uninitialized_error_channel_decision"_s + completion<EventGenerateRun> [generate_error_callback_without_error_out] / dispatch_generate_error_with_callback_only_from_generate_uninitialized_error_channel_decision,
        "uninitialized"_s <= "generate_uninitialized_error_channel_decision"_s + completion<EventGenerateRun> [generate_no_error_callback_with_error_out] / dispatch_generate_error_with_error_out_only_from_generate_uninitialized_error_channel_decision,
        "uninitialized"_s <= "generate_uninitialized_error_channel_decision"_s + completion<EventGenerateRun> [generate_no_error_callback_without_error_out] / dispatch_generate_error_without_channels_from_generate_uninitialized_error_channel_decision,
        "uninitialized"_s <= "uninitialized"_s + event<EventCaptureDiagnostics> / capture_diagnostics_from_uninitialized,
        "ready"_s <= "ready"_s + event<EventCaptureDiagnostics> / capture_diagnostics_from_ready,
        "uninitialized"_s <= "uninitialized"_s + event<EventConfigureBenchmarkLane> [guard_benchmark_lane_single] / effect_disable_parallel_benchmark_lanes_from_uninitialized,
        "uninitialized"_s <= "uninitialized"_s + event<EventConfigureBenchmarkLane> [guard_benchmark_lane_multithreaded] / effect_enable_parallel_benchmark_lanes_from_uninitialized,
        "ready"_s <= "ready"_s + event<EventConfigureBenchmarkLane> [guard_benchmark_lane_single] / effect_disable_parallel_benchmark_lanes_from_ready,
        "ready"_s <= "ready"_s + event<EventConfigureBenchmarkLane> [guard_benchmark_lane_multithreaded] / effect_enable_parallel_benchmark_lanes_from_ready,
        "uninitialized"_s <= "uninitialized"_s + event<EventCaptureGraphLifecycle> [graph_lifecycle_runtime_tensor_unavailable] / capture_graph_lifecycle_without_runtime_tensor_from_uninitialized,
        "uninitialized"_s <= "uninitialized"_s + event<EventCaptureGraphLifecycle> [graph_lifecycle_runtime_tensor_available] / capture_graph_lifecycle_with_runtime_tensor_from_uninitialized,
        "ready"_s <= "ready"_s + event<EventCaptureGraphLifecycle> [graph_lifecycle_runtime_tensor_unavailable] / capture_graph_lifecycle_without_runtime_tensor_from_ready,
        "ready"_s <= "ready"_s + event<EventCaptureGraphLifecycle> [graph_lifecycle_runtime_tensor_available] / capture_graph_lifecycle_with_runtime_tensor_from_ready,
        "uninitialized"_s <= "uninitialized"_s + unexpected_event<_> / on_unexpected_from_uninitialized,
        "ready"_s <= "ready"_s + unexpected_event<_> / on_unexpected_from_ready,
        "uninitialized"_s <= "initializing"_s + unexpected_event<_> / on_unexpected_from_initializing,
        "uninitialized"_s <= "initializer_result_decision"_s + unexpected_event<_> / on_unexpected_from_initializer_result_decision,
        "ready"_s <= "initialize_done_channel_decision"_s + unexpected_event<_> / on_unexpected_from_initialize_done_channel_decision,
        "uninitialized"_s <= "initialize_error_channel_decision"_s + unexpected_event<_> / on_unexpected_from_initialize_error_channel_decision,
        "ready"_s <= "reset_sequence"_s + unexpected_event<_> / on_unexpected_from_reset_sequence,
        "ready"_s <= "reset_sequence_decision"_s + unexpected_event<_> / on_unexpected_from_reset_sequence_decision,
        "ready"_s <= "conditioning"_s + unexpected_event<_> / on_unexpected_from_conditioning,
        "ready"_s <= "conditioning_decision"_s + unexpected_event<_> / on_unexpected_from_conditioning_decision,
        "ready"_s <= "planning_chunk8"_s + unexpected_event<_> / on_unexpected_from_planning_chunk8,
        "ready"_s <= "planning_chunk4"_s + unexpected_event<_> / on_unexpected_from_planning_chunk4,
        "ready"_s <= "planning_scalar"_s + unexpected_event<_> / on_unexpected_from_planning_scalar,
        "ready"_s <= "planning_decision"_s + unexpected_event<_> / on_unexpected_from_planning_decision,
        "ready"_s <= "sequence_allocating"_s + unexpected_event<_> / on_unexpected_from_sequence_allocating,
        "ready"_s <= "sequence_allocating_decision"_s + unexpected_event<_> / on_unexpected_from_sequence_allocating_decision,
        "ready"_s <= "prefill_running"_s + unexpected_event<_> / on_unexpected_from_prefill_running,
        "ready"_s <= "prefill_result_decision"_s + unexpected_event<_> / on_unexpected_from_prefill_result_decision,
        "ready"_s <= "decode_selection_mode_decision"_s + unexpected_event<_> / on_unexpected_from_decode_selection_mode_decision,
        "ready"_s <= "decode_slots"_s + unexpected_event<_> / on_unexpected_from_decode_slots,
        "ready"_s <= "decode_slots_decision"_s + unexpected_event<_> / on_unexpected_from_decode_slots_decision,
        "ready"_s <= "snapshot_decode"_s + unexpected_event<_> / on_unexpected_from_snapshot_decode,
        "ready"_s <= "snapshot_decode_decision"_s + unexpected_event<_> / on_unexpected_from_snapshot_decode_decision,
        "ready"_s <= "decode_compute_runtime_decision"_s + unexpected_event<_> / on_unexpected_from_decode_compute_runtime_decision,
        "ready"_s <= "decode_compute_flash"_s + unexpected_event<_> / on_unexpected_from_decode_compute_flash,
        "ready"_s <= "decode_compute_flash_preselected_argmax"_s + unexpected_event<_> / on_unexpected_from_decode_compute_flash_preselected_argmax,
        "ready"_s <= "decode_compute_flash_preselected_argmax_decision"_s + unexpected_event<_> / on_unexpected_from_decode_compute_flash_preselected_argmax_decision,
        "ready"_s <= "decode_compute_flash_decision"_s + unexpected_event<_> / on_unexpected_from_decode_compute_flash_decision,
        "ready"_s <= "decode_compute_nonflash"_s + unexpected_event<_> / on_unexpected_from_decode_compute_nonflash,
        "ready"_s <= "decode_compute_nonflash_preselected_argmax"_s + unexpected_event<_> / on_unexpected_from_decode_compute_nonflash_preselected_argmax,
        "ready"_s <= "decode_compute_nonflash_preselected_argmax_decision"_s + unexpected_event<_> / on_unexpected_from_decode_compute_nonflash_preselected_argmax_decision,
        "ready"_s <= "decode_compute_nonflash_decision"_s + unexpected_event<_> / on_unexpected_from_decode_compute_nonflash_decision,
        "ready"_s <= "decode_sample"_s + unexpected_event<_> / on_unexpected_from_decode_sample,
        "ready"_s <= "decode_sample_decision"_s + unexpected_event<_> / on_unexpected_from_decode_sample_decision,
        "ready"_s <= "decode_preselected_argmax"_s + unexpected_event<_> / on_unexpected_from_decode_preselected_argmax,
        "ready"_s <= "decode_preselected_argmax_decision"_s + unexpected_event<_> / on_unexpected_from_decode_preselected_argmax_decision,
        "ready"_s <= "decode_sample_preselected"_s + unexpected_event<_> / on_unexpected_from_decode_sample_preselected,
        "ready"_s <= "decode_sample_preselected_decision"_s + unexpected_event<_> / on_unexpected_from_decode_sample_preselected_decision,
        "ready"_s <= "decode_render"_s + unexpected_event<_> / on_unexpected_from_decode_render,
        "ready"_s <= "decode_render_decision"_s + unexpected_event<_> / on_unexpected_from_decode_render_decision,
        "ready"_s <= "decode_loop_decision"_s + unexpected_event<_> / on_unexpected_from_decode_loop_decision,
        "ready"_s <= "flushing"_s + unexpected_event<_> / on_unexpected_from_flushing,
        "ready"_s <= "flushing_decision"_s + unexpected_event<_> / on_unexpected_from_flushing_decision,
        "ready"_s <= "generate_done_channel_decision"_s + unexpected_event<_> / on_unexpected_from_generate_done_channel_decision,
        "ready"_s <= "generate_ready_error_channel_decision"_s + unexpected_event<_> / on_unexpected_from_generate_ready_error_channel_decision,
        "uninitialized"_s <= "generate_uninitialized_error_channel_decision"_s + unexpected_event<_> / on_unexpected_from_generate_uninitialized_error_channel_decision,
    }
}

/// Context for `TextGenerator` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct TextGeneratorContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl TextGeneratorStateMachineContext for TextGeneratorContext {
    fn advance_kv_cache_from_decode_compute_flash_decision(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::advance_kv_cache
        todo!(
            "TODO: port action `advance_kv_cache` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn advance_kv_cache_from_decode_compute_flash_preselected_argmax_decision(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::advance_kv_cache
        todo!(
            "TODO: port action `advance_kv_cache` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn advance_kv_cache_from_decode_compute_nonflash_decision(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::advance_kv_cache
        todo!(
            "TODO: port action `advance_kv_cache` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn advance_kv_cache_from_decode_compute_nonflash_preselected_argmax_decision(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::advance_kv_cache
        todo!(
            "TODO: port action `advance_kv_cache` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn allocate_sequence_backend_error(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::allocate_sequence_backend_error
        todo!(
            "TODO: port guard `allocate_sequence_backend_error` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn allocate_sequence_invalid_request(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::allocate_sequence_invalid_request
        todo!(
            "TODO: port guard `allocate_sequence_invalid_request` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn allocate_sequence_ok(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::allocate_sequence_ok
        todo!(
            "TODO: port guard `allocate_sequence_ok` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn begin_generate_from_ready(&mut self, _event: &EventGenerateRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::begin_generate
        todo!(
            "TODO: port action `begin_generate` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn capture_diagnostics_from_ready(
        &mut self,
        _event: &EventCaptureDiagnostics,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::capture_diagnostics
        todo!(
            "TODO: port action `capture_diagnostics` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn capture_diagnostics_from_uninitialized(
        &mut self,
        _event: &EventCaptureDiagnostics,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::capture_diagnostics
        todo!(
            "TODO: port action `capture_diagnostics` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn capture_graph_lifecycle_with_runtime_tensor_from_ready(
        &mut self,
        _event: &EventCaptureGraphLifecycle,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::capture_graph_lifecycle_with_runtime_tensor
        todo!(
            "TODO: port action `capture_graph_lifecycle_with_runtime_tensor` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn capture_graph_lifecycle_with_runtime_tensor_from_uninitialized(
        &mut self,
        _event: &EventCaptureGraphLifecycle,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::capture_graph_lifecycle_with_runtime_tensor
        todo!(
            "TODO: port action `capture_graph_lifecycle_with_runtime_tensor` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn capture_graph_lifecycle_without_runtime_tensor_from_ready(
        &mut self,
        _event: &EventCaptureGraphLifecycle,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::capture_graph_lifecycle_without_runtime_tensor
        todo!(
            "TODO: port action `capture_graph_lifecycle_without_runtime_tensor` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn capture_graph_lifecycle_without_runtime_tensor_from_uninitialized(
        &mut self,
        _event: &EventCaptureGraphLifecycle,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::capture_graph_lifecycle_without_runtime_tensor
        todo!(
            "TODO: port action `capture_graph_lifecycle_without_runtime_tensor` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn commit_flush_output(&mut self, _event: &EventGenerateRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::commit_flush_output
        todo!(
            "TODO: port action `commit_flush_output` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn commit_render_output(&mut self, _event: &EventGenerateRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::commit_render_output
        todo!(
            "TODO: port action `commit_render_output` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn conditioning_backend_error(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::conditioning_backend_error
        todo!(
            "TODO: port guard `conditioning_backend_error` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn conditioning_invalid_request(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::conditioning_invalid_request
        todo!(
            "TODO: port guard `conditioning_invalid_request` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn conditioning_ok_with_chunk4_prefill(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::conditioning_ok_with_chunk4_prefill
        todo!(
            "TODO: port guard `conditioning_ok_with_chunk4_prefill` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn conditioning_ok_with_chunk8_prefill(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::conditioning_ok_with_chunk8_prefill
        todo!(
            "TODO: port guard `conditioning_ok_with_chunk8_prefill` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn conditioning_ok_with_scalar_prefill(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::conditioning_ok_with_scalar_prefill
        todo!(
            "TODO: port guard `conditioning_ok_with_scalar_prefill` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn decode_argmax_invalid_request(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::decode_argmax_invalid_request
        todo!(
            "TODO: port guard `decode_argmax_invalid_request` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn decode_argmax_ready(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::decode_argmax_ready
        todo!(
            "TODO: port guard `decode_argmax_ready` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn decode_complete(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::decode_complete
        todo!("TODO: port guard `decode_complete` from emel.cpp/src/emel/text/generator/guards.hpp")
    }
    fn decode_compute_backend_error(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::decode_compute_backend_error
        todo!(
            "TODO: port guard `decode_compute_backend_error` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn decode_compute_invalid_request(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::decode_compute_invalid_request
        todo!(
            "TODO: port guard `decode_compute_invalid_request` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn decode_compute_ok(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::decode_compute_ok
        todo!(
            "TODO: port guard `decode_compute_ok` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn decode_flash_runtime_supported(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::decode_flash_runtime_supported
        todo!(
            "TODO: port guard `decode_flash_runtime_supported` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn decode_nonflash_runtime_required(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::decode_nonflash_runtime_required
        todo!(
            "TODO: port guard `decode_nonflash_runtime_required` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn decode_render_backend_error(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::decode_render_backend_error
        todo!(
            "TODO: port guard `decode_render_backend_error` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn decode_render_invalid_request(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::decode_render_invalid_request
        todo!(
            "TODO: port guard `decode_render_invalid_request` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn decode_render_ok(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::decode_render_ok
        todo!(
            "TODO: port guard `decode_render_ok` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn decode_sample_backend_error(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::decode_sample_backend_error
        todo!(
            "TODO: port guard `decode_sample_backend_error` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn decode_sample_invalid_request(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::decode_sample_invalid_request
        todo!(
            "TODO: port guard `decode_sample_invalid_request` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn decode_sample_ok(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::decode_sample_ok
        todo!(
            "TODO: port guard `decode_sample_ok` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn decode_should_continue(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::decode_should_continue
        todo!(
            "TODO: port guard `decode_should_continue` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn decode_slots_backend_error(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::decode_slots_backend_error
        todo!(
            "TODO: port guard `decode_slots_backend_error` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn decode_slots_invalid_request(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::decode_slots_invalid_request
        todo!(
            "TODO: port guard `decode_slots_invalid_request` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn decode_slots_ok(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::decode_slots_ok
        todo!("TODO: port guard `decode_slots_ok` from emel.cpp/src/emel/text/generator/guards.hpp")
    }
    fn decode_snapshot_backend_error(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::decode_snapshot_backend_error
        todo!(
            "TODO: port guard `decode_snapshot_backend_error` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn decode_snapshot_invalid_request(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::decode_snapshot_invalid_request
        todo!(
            "TODO: port guard `decode_snapshot_invalid_request` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn decode_snapshot_ok(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::decode_snapshot_ok
        todo!(
            "TODO: port guard `decode_snapshot_ok` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn decode_uses_materialized_logits(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::decode_uses_materialized_logits
        todo!(
            "TODO: port guard `decode_uses_materialized_logits` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn decode_uses_preselected_argmax(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::decode_uses_preselected_argmax
        todo!(
            "TODO: port guard `decode_uses_preselected_argmax` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn dispatch_generate_done_with_callback_and_error_out(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::dispatch_generate_done_with_callback_and_error_out
        todo!(
            "TODO: port action `dispatch_generate_done_with_callback_and_error_out` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn dispatch_generate_done_with_callback_only(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::dispatch_generate_done_with_callback_only
        todo!(
            "TODO: port action `dispatch_generate_done_with_callback_only` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn dispatch_generate_done_with_error_out_only(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::dispatch_generate_done_with_error_out_only
        todo!(
            "TODO: port action `dispatch_generate_done_with_error_out_only` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn dispatch_generate_done_without_channels(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::dispatch_generate_done_without_channels
        todo!(
            "TODO: port action `dispatch_generate_done_without_channels` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn dispatch_generate_error_with_callback_and_error_out_from_generate_ready_error_channel_decision(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::dispatch_generate_error_with_callback_and_error_out
        todo!(
            "TODO: port action `dispatch_generate_error_with_callback_and_error_out` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn dispatch_generate_error_with_callback_and_error_out_from_generate_uninitialized_error_channel_decision(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::dispatch_generate_error_with_callback_and_error_out
        todo!(
            "TODO: port action `dispatch_generate_error_with_callback_and_error_out` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn dispatch_generate_error_with_callback_only_from_generate_ready_error_channel_decision(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::dispatch_generate_error_with_callback_only
        todo!(
            "TODO: port action `dispatch_generate_error_with_callback_only` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn dispatch_generate_error_with_callback_only_from_generate_uninitialized_error_channel_decision(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::dispatch_generate_error_with_callback_only
        todo!(
            "TODO: port action `dispatch_generate_error_with_callback_only` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn dispatch_generate_error_with_error_out_only_from_generate_ready_error_channel_decision(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::dispatch_generate_error_with_error_out_only
        todo!(
            "TODO: port action `dispatch_generate_error_with_error_out_only` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn dispatch_generate_error_with_error_out_only_from_generate_uninitialized_error_channel_decision(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::dispatch_generate_error_with_error_out_only
        todo!(
            "TODO: port action `dispatch_generate_error_with_error_out_only` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn dispatch_generate_error_without_channels_from_generate_ready_error_channel_decision(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::dispatch_generate_error_without_channels
        todo!(
            "TODO: port action `dispatch_generate_error_without_channels` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn dispatch_generate_error_without_channels_from_generate_uninitialized_error_channel_decision(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::dispatch_generate_error_without_channels
        todo!(
            "TODO: port action `dispatch_generate_error_without_channels` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn dispatch_initialize_done_with_callback_and_error_out(
        &mut self,
        _event: &EventInitializeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::dispatch_initialize_done_with_callback_and_error_out
        todo!(
            "TODO: port action `dispatch_initialize_done_with_callback_and_error_out` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn dispatch_initialize_done_with_callback_only(
        &mut self,
        _event: &EventInitializeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::dispatch_initialize_done_with_callback_only
        todo!(
            "TODO: port action `dispatch_initialize_done_with_callback_only` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn dispatch_initialize_done_with_error_out_only(
        &mut self,
        _event: &EventInitializeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::dispatch_initialize_done_with_error_out_only
        todo!(
            "TODO: port action `dispatch_initialize_done_with_error_out_only` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn dispatch_initialize_done_without_channels(
        &mut self,
        _event: &EventInitializeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::dispatch_initialize_done_without_channels
        todo!(
            "TODO: port action `dispatch_initialize_done_without_channels` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn dispatch_initialize_error_with_callback_and_error_out(
        &mut self,
        _event: &EventInitializeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::dispatch_initialize_error_with_callback_and_error_out
        todo!(
            "TODO: port action `dispatch_initialize_error_with_callback_and_error_out` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn dispatch_initialize_error_with_callback_only(
        &mut self,
        _event: &EventInitializeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::dispatch_initialize_error_with_callback_only
        todo!(
            "TODO: port action `dispatch_initialize_error_with_callback_only` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn dispatch_initialize_error_with_error_out_only(
        &mut self,
        _event: &EventInitializeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::dispatch_initialize_error_with_error_out_only
        todo!(
            "TODO: port action `dispatch_initialize_error_with_error_out_only` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn dispatch_initialize_error_without_channels(
        &mut self,
        _event: &EventInitializeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::dispatch_initialize_error_without_channels
        todo!(
            "TODO: port action `dispatch_initialize_error_without_channels` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn effect_disable_parallel_benchmark_lanes_from_ready(
        &mut self,
        _event: &EventConfigureBenchmarkLane,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::effect_disable_parallel_benchmark_lanes
        todo!(
            "TODO: port action `effect_disable_parallel_benchmark_lanes` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn effect_disable_parallel_benchmark_lanes_from_uninitialized(
        &mut self,
        _event: &EventConfigureBenchmarkLane,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::effect_disable_parallel_benchmark_lanes
        todo!(
            "TODO: port action `effect_disable_parallel_benchmark_lanes` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn effect_enable_parallel_benchmark_lanes_from_ready(
        &mut self,
        _event: &EventConfigureBenchmarkLane,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::effect_enable_parallel_benchmark_lanes
        todo!(
            "TODO: port action `effect_enable_parallel_benchmark_lanes` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn effect_enable_parallel_benchmark_lanes_from_uninitialized(
        &mut self,
        _event: &EventConfigureBenchmarkLane,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::effect_enable_parallel_benchmark_lanes
        todo!(
            "TODO: port action `effect_enable_parallel_benchmark_lanes` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn flush_backend_error(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::flush_backend_error
        todo!(
            "TODO: port guard `flush_backend_error` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn flush_invalid_request(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::flush_invalid_request
        todo!(
            "TODO: port guard `flush_invalid_request` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn flush_ok(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::flush_ok
        todo!("TODO: port guard `flush_ok` from emel.cpp/src/emel/text/generator/guards.hpp")
    }
    fn generate_done_callback_with_error_out(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::generate_done_callback_with_error_out
        todo!(
            "TODO: port guard `generate_done_callback_with_error_out` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn generate_done_callback_without_error_out(
        &self,
        _event: &EventGenerateRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::generate_done_callback_without_error_out
        todo!(
            "TODO: port guard `generate_done_callback_without_error_out` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn generate_error_callback_with_error_out(
        &self,
        _event: &EventGenerateRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::generate_error_callback_with_error_out
        todo!(
            "TODO: port guard `generate_error_callback_with_error_out` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn generate_error_callback_without_error_out(
        &self,
        _event: &EventGenerateRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::generate_error_callback_without_error_out
        todo!(
            "TODO: port guard `generate_error_callback_without_error_out` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn generate_no_done_callback_with_error_out(
        &self,
        _event: &EventGenerateRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::generate_no_done_callback_with_error_out
        todo!(
            "TODO: port guard `generate_no_done_callback_with_error_out` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn generate_no_done_callback_without_error_out(
        &self,
        _event: &EventGenerateRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::generate_no_done_callback_without_error_out
        todo!(
            "TODO: port guard `generate_no_done_callback_without_error_out` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn generate_no_error_callback_with_error_out(
        &self,
        _event: &EventGenerateRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::generate_no_error_callback_with_error_out
        todo!(
            "TODO: port guard `generate_no_error_callback_with_error_out` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn generate_no_error_callback_without_error_out(
        &self,
        _event: &EventGenerateRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::generate_no_error_callback_without_error_out
        todo!(
            "TODO: port guard `generate_no_error_callback_without_error_out` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn graph_lifecycle_runtime_tensor_available(
        &self,
        _event: &EventCaptureGraphLifecycle,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::graph_lifecycle_runtime_tensor_available
        todo!(
            "TODO: port guard `graph_lifecycle_runtime_tensor_available` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn graph_lifecycle_runtime_tensor_unavailable(
        &self,
        _event: &EventCaptureGraphLifecycle,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::graph_lifecycle_runtime_tensor_unavailable
        todo!(
            "TODO: port guard `graph_lifecycle_runtime_tensor_unavailable` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn guard_benchmark_lane_multithreaded(
        &self,
        _event: &EventConfigureBenchmarkLane,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::guard_benchmark_lane_multithreaded
        todo!(
            "TODO: port guard `guard_benchmark_lane_multithreaded` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn guard_benchmark_lane_single(
        &self,
        _event: &EventConfigureBenchmarkLane,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::guard_benchmark_lane_single
        todo!(
            "TODO: port guard `guard_benchmark_lane_single` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn guard_decode_compute_backend_unavailable(
        &self,
        _event: &EventGenerateRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::guard_decode_compute_backend_unavailable
        todo!(
            "TODO: port guard `guard_decode_compute_backend_unavailable` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn guard_decode_compute_invalid_request(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::guard_decode_compute_invalid_request
        todo!(
            "TODO: port guard `guard_decode_compute_invalid_request` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn guard_decode_materialized_parallel_scalar_kernel_ready(
        &self,
        _event: &EventGenerateRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::guard_decode_materialized_parallel_scalar_kernel_ready
        todo!(
            "TODO: port guard `guard_decode_materialized_parallel_scalar_kernel_ready` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn guard_decode_materialized_parallel_scalar_native_quantized_kernel_ready(
        &self,
        _event: &EventGenerateRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::guard_decode_materialized_parallel_scalar_native_quantized_kernel_ready
        todo!(
            "TODO: port guard `guard_decode_materialized_parallel_scalar_native_quantized_kernel_ready` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn guard_decode_materialized_parallel_scalar_native_quantized_q8_k_ready(
        &self,
        _event: &EventGenerateRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::guard_decode_materialized_parallel_scalar_native_quantized_q8_k_ready
        todo!(
            "TODO: port guard `guard_decode_materialized_parallel_scalar_native_quantized_q8_k_ready` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn guard_decode_materialized_parallel_scalar_packed_q8_0_ready(
        &self,
        _event: &EventGenerateRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::guard_decode_materialized_parallel_scalar_packed_q8_0_ready
        todo!(
            "TODO: port guard `guard_decode_materialized_parallel_scalar_packed_q8_0_ready` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn guard_decode_materialized_parallel_scalar_q8_k_ready(
        &self,
        _event: &EventGenerateRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::guard_decode_materialized_parallel_scalar_q8_k_ready
        todo!(
            "TODO: port guard `guard_decode_materialized_parallel_scalar_q8_k_ready` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn guard_decode_materialized_scalar_kernel_ready(
        &self,
        _event: &EventGenerateRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::guard_decode_materialized_scalar_kernel_ready
        todo!(
            "TODO: port guard `guard_decode_materialized_scalar_kernel_ready` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn guard_decode_materialized_scalar_native_quantized_kernel_ready(
        &self,
        _event: &EventGenerateRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::guard_decode_materialized_scalar_native_quantized_kernel_ready
        todo!(
            "TODO: port guard `guard_decode_materialized_scalar_native_quantized_kernel_ready` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn guard_decode_materialized_scalar_native_quantized_q8_k_ready(
        &self,
        _event: &EventGenerateRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::guard_decode_materialized_scalar_native_quantized_q8_k_ready
        todo!(
            "TODO: port guard `guard_decode_materialized_scalar_native_quantized_q8_k_ready` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn guard_decode_materialized_scalar_packed_q8_0_ready(
        &self,
        _event: &EventGenerateRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::guard_decode_materialized_scalar_packed_q8_0_ready
        todo!(
            "TODO: port guard `guard_decode_materialized_scalar_packed_q8_0_ready` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn guard_decode_materialized_scalar_q8_k_ready(
        &self,
        _event: &EventGenerateRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::guard_decode_materialized_scalar_q8_k_ready
        todo!(
            "TODO: port guard `guard_decode_materialized_scalar_q8_k_ready` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn guard_decode_materialized_streamed_scalar_kernel_ready(
        &self,
        _event: &EventGenerateRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::guard_decode_materialized_streamed_scalar_kernel_ready
        todo!(
            "TODO: port guard `guard_decode_materialized_streamed_scalar_kernel_ready` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn guard_decode_materialized_streamed_scalar_native_quantized_q8_k_ready(
        &self,
        _event: &EventGenerateRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::guard_decode_materialized_streamed_scalar_native_quantized_q8_k_ready
        todo!(
            "TODO: port guard `guard_decode_materialized_streamed_scalar_native_quantized_q8_k_ready` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn guard_decode_materialized_streamed_scalar_native_quantized_ready(
        &self,
        _event: &EventGenerateRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::guard_decode_materialized_streamed_scalar_native_quantized_ready
        todo!(
            "TODO: port guard `guard_decode_materialized_streamed_scalar_native_quantized_ready` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn guard_decode_materialized_streamed_scalar_packed_q8_0_ready(
        &self,
        _event: &EventGenerateRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::guard_decode_materialized_streamed_scalar_packed_q8_0_ready
        todo!(
            "TODO: port guard `guard_decode_materialized_streamed_scalar_packed_q8_0_ready` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn guard_decode_materialized_streamed_scalar_q8_k_ready(
        &self,
        _event: &EventGenerateRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::guard_decode_materialized_streamed_scalar_q8_k_ready
        todo!(
            "TODO: port guard `guard_decode_materialized_streamed_scalar_q8_k_ready` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn guard_decode_preselected_argmax_kernel_ready(
        &self,
        _event: &EventGenerateRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::guard_decode_preselected_argmax_kernel_ready
        todo!(
            "TODO: port guard `guard_decode_preselected_argmax_kernel_ready` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn guard_decode_preselected_argmax_kernel_streamed_ready(
        &self,
        _event: &EventGenerateRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::guard_decode_preselected_argmax_kernel_streamed_ready
        todo!(
            "TODO: port guard `guard_decode_preselected_argmax_kernel_streamed_ready` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn guard_decode_preselected_argmax_native_quantized_kernel_ready(
        &self,
        _event: &EventGenerateRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::guard_decode_preselected_argmax_native_quantized_kernel_ready
        todo!(
            "TODO: port guard `guard_decode_preselected_argmax_native_quantized_kernel_ready` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn guard_decode_preselected_argmax_native_quantized_kernel_streamed_ready(
        &self,
        _event: &EventGenerateRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::guard_decode_preselected_argmax_native_quantized_kernel_streamed_ready
        todo!(
            "TODO: port guard `guard_decode_preselected_argmax_native_quantized_kernel_streamed_ready` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn guard_decode_preselected_argmax_native_quantized_q8_k_ready(
        &self,
        _event: &EventGenerateRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::guard_decode_preselected_argmax_native_quantized_q8_k_ready
        todo!(
            "TODO: port guard `guard_decode_preselected_argmax_native_quantized_q8_k_ready` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn guard_decode_preselected_argmax_native_quantized_q8_k_streamed_ready(
        &self,
        _event: &EventGenerateRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::guard_decode_preselected_argmax_native_quantized_q8_k_streamed_ready
        todo!(
            "TODO: port guard `guard_decode_preselected_argmax_native_quantized_q8_k_streamed_ready` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn guard_decode_preselected_argmax_q8_k_ready(
        &self,
        _event: &EventGenerateRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::guard_decode_preselected_argmax_q8_k_ready
        todo!(
            "TODO: port guard `guard_decode_preselected_argmax_q8_k_ready` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn guard_decode_preselected_argmax_q8_k_streamed_ready(
        &self,
        _event: &EventGenerateRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::guard_decode_preselected_argmax_q8_k_streamed_ready
        todo!(
            "TODO: port guard `guard_decode_preselected_argmax_q8_k_streamed_ready` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn guard_decode_preselected_compute_invalid_request(
        &self,
        _event: &EventGenerateRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::guard_decode_preselected_compute_invalid_request
        todo!(
            "TODO: port guard `guard_decode_preselected_compute_invalid_request` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn guard_decode_preselected_direct_ready(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::guard_decode_preselected_direct_ready
        todo!(
            "TODO: port guard `guard_decode_preselected_direct_ready` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn guard_decode_preselected_parallel_argmax_kernel_ready(
        &self,
        _event: &EventGenerateRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::guard_decode_preselected_parallel_argmax_kernel_ready
        todo!(
            "TODO: port guard `guard_decode_preselected_parallel_argmax_kernel_ready` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn guard_decode_preselected_parallel_argmax_native_quantized_kernel_ready(
        &self,
        _event: &EventGenerateRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::guard_decode_preselected_parallel_argmax_native_quantized_kernel_ready
        todo!(
            "TODO: port guard `guard_decode_preselected_parallel_argmax_native_quantized_kernel_ready` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn guard_decode_preselected_parallel_argmax_native_quantized_q8_k_ready(
        &self,
        _event: &EventGenerateRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::guard_decode_preselected_parallel_argmax_native_quantized_q8_k_ready
        todo!(
            "TODO: port guard `guard_decode_preselected_parallel_argmax_native_quantized_q8_k_ready` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn guard_decode_preselected_parallel_argmax_q8_k_ready(
        &self,
        _event: &EventGenerateRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::guard_decode_preselected_parallel_argmax_q8_k_ready
        todo!(
            "TODO: port guard `guard_decode_preselected_parallel_argmax_q8_k_ready` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn initialize_done_callback_with_error_out(
        &self,
        _event: &EventInitializeRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::initialize_done_callback_with_error_out
        todo!(
            "TODO: port guard `initialize_done_callback_with_error_out` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn initialize_done_callback_without_error_out(
        &self,
        _event: &EventInitializeRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::initialize_done_callback_without_error_out
        todo!(
            "TODO: port guard `initialize_done_callback_without_error_out` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn initialize_error_callback_with_error_out(
        &self,
        _event: &EventInitializeRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::initialize_error_callback_with_error_out
        todo!(
            "TODO: port guard `initialize_error_callback_with_error_out` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn initialize_error_callback_without_error_out(
        &self,
        _event: &EventInitializeRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::initialize_error_callback_without_error_out
        todo!(
            "TODO: port guard `initialize_error_callback_without_error_out` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn initialize_no_done_callback_with_error_out(
        &self,
        _event: &EventInitializeRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::initialize_no_done_callback_with_error_out
        todo!(
            "TODO: port guard `initialize_no_done_callback_with_error_out` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn initialize_no_done_callback_without_error_out(
        &self,
        _event: &EventInitializeRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::initialize_no_done_callback_without_error_out
        todo!(
            "TODO: port guard `initialize_no_done_callback_without_error_out` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn initialize_no_error_callback_with_error_out(
        &self,
        _event: &EventInitializeRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::initialize_no_error_callback_with_error_out
        todo!(
            "TODO: port guard `initialize_no_error_callback_with_error_out` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn initialize_no_error_callback_without_error_out(
        &self,
        _event: &EventInitializeRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::initialize_no_error_callback_without_error_out
        todo!(
            "TODO: port guard `initialize_no_error_callback_without_error_out` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn initialize_result_backend(&self, _event: &EventInitializeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::initialize_result_backend
        todo!(
            "TODO: port guard `initialize_result_backend` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn initialize_result_invalid_request(&self, _event: &EventInitializeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::initialize_result_invalid_request
        todo!(
            "TODO: port guard `initialize_result_invalid_request` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn initialize_result_none(&self, _event: &EventInitializeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::initialize_result_none
        todo!(
            "TODO: port guard `initialize_result_none` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn invalid_generate(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::invalid_generate
        todo!(
            "TODO: port guard `invalid_generate` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn invalid_initialize(&self, _event: &EventInitializeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::invalid_initialize
        todo!(
            "TODO: port guard `invalid_initialize` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn mark_backend_error_from_conditioning_decision(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::mark_backend_error
        todo!(
            "TODO: port action `mark_backend_error` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn mark_backend_error_from_decode_compute_flash(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::mark_backend_error
        todo!(
            "TODO: port action `mark_backend_error` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn mark_backend_error_from_decode_compute_flash_decision(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::mark_backend_error
        todo!(
            "TODO: port action `mark_backend_error` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn mark_backend_error_from_decode_compute_flash_preselected_argmax(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::mark_backend_error
        todo!(
            "TODO: port action `mark_backend_error` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn mark_backend_error_from_decode_compute_flash_preselected_argmax_decision(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::mark_backend_error
        todo!(
            "TODO: port action `mark_backend_error` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn mark_backend_error_from_decode_compute_nonflash(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::mark_backend_error
        todo!(
            "TODO: port action `mark_backend_error` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn mark_backend_error_from_decode_compute_nonflash_decision(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::mark_backend_error
        todo!(
            "TODO: port action `mark_backend_error` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn mark_backend_error_from_decode_compute_nonflash_preselected_argmax(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::mark_backend_error
        todo!(
            "TODO: port action `mark_backend_error` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn mark_backend_error_from_decode_compute_nonflash_preselected_argmax_decision(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::mark_backend_error
        todo!(
            "TODO: port action `mark_backend_error` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn mark_backend_error_from_decode_preselected_argmax_decision(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::mark_backend_error
        todo!(
            "TODO: port action `mark_backend_error` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn mark_backend_error_from_decode_render_decision(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::mark_backend_error
        todo!(
            "TODO: port action `mark_backend_error` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn mark_backend_error_from_decode_sample_decision(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::mark_backend_error
        todo!(
            "TODO: port action `mark_backend_error` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn mark_backend_error_from_decode_sample_preselected_decision(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::mark_backend_error
        todo!(
            "TODO: port action `mark_backend_error` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn mark_backend_error_from_decode_slots_decision(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::mark_backend_error
        todo!(
            "TODO: port action `mark_backend_error` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn mark_backend_error_from_flushing_decision(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::mark_backend_error
        todo!(
            "TODO: port action `mark_backend_error` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn mark_backend_error_from_planning_decision(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::mark_backend_error
        todo!(
            "TODO: port action `mark_backend_error` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn mark_backend_error_from_prefill_running(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::mark_backend_error
        todo!(
            "TODO: port action `mark_backend_error` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn mark_backend_error_from_reset_sequence_decision(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::mark_backend_error
        todo!(
            "TODO: port action `mark_backend_error` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn mark_backend_error_from_sequence_allocating_decision(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::mark_backend_error
        todo!(
            "TODO: port action `mark_backend_error` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn mark_backend_error_from_snapshot_decode_decision(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::mark_backend_error
        todo!(
            "TODO: port action `mark_backend_error` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn mark_invalid_request_from_conditioning_decision(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn mark_invalid_request_from_decode_compute_flash(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn mark_invalid_request_from_decode_compute_flash_decision(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn mark_invalid_request_from_decode_compute_flash_preselected_argmax(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn mark_invalid_request_from_decode_compute_flash_preselected_argmax_decision(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn mark_invalid_request_from_decode_compute_nonflash(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn mark_invalid_request_from_decode_compute_nonflash_decision(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn mark_invalid_request_from_decode_compute_nonflash_preselected_argmax(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn mark_invalid_request_from_decode_compute_nonflash_preselected_argmax_decision(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn mark_invalid_request_from_decode_preselected_argmax(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn mark_invalid_request_from_decode_preselected_argmax_decision(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn mark_invalid_request_from_decode_render_decision(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn mark_invalid_request_from_decode_sample_decision(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn mark_invalid_request_from_decode_sample_preselected_decision(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn mark_invalid_request_from_decode_slots_decision(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn mark_invalid_request_from_flushing_decision(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn mark_invalid_request_from_planning_decision(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn mark_invalid_request_from_reset_sequence_decision(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn mark_invalid_request_from_sequence_allocating_decision(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn mark_invalid_request_from_snapshot_decode_decision(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::mark_invalid_request
        todo!(
            "TODO: port action `mark_invalid_request` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn mark_sequence_clear(&mut self, _event: &EventGenerateRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::mark_sequence_clear
        todo!(
            "TODO: port action `mark_sequence_clear` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn mark_sequence_live(&mut self, _event: &EventGenerateRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::mark_sequence_live
        todo!(
            "TODO: port action `mark_sequence_live` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn on_unexpected_from_conditioning(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_conditioning_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_decode_compute_flash(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_decode_compute_flash_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_decode_compute_flash_preselected_argmax(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_decode_compute_flash_preselected_argmax_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_decode_compute_nonflash(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_decode_compute_nonflash_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_decode_compute_nonflash_preselected_argmax(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_decode_compute_nonflash_preselected_argmax_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_decode_compute_runtime_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_decode_loop_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_decode_preselected_argmax(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_decode_preselected_argmax_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_decode_render(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_decode_render_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_decode_sample(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_decode_sample_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_decode_sample_preselected(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_decode_sample_preselected_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_decode_selection_mode_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_decode_slots(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_decode_slots_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_flushing(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_flushing_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_generate_done_channel_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_generate_ready_error_channel_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_generate_uninitialized_error_channel_decision(
        &mut self,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_initialize_done_channel_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_initialize_error_channel_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_initializer_result_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_initializing(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_planning_chunk4(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_planning_chunk8(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_planning_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_planning_scalar(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_prefill_result_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_prefill_running(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_ready(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_reset_sequence(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_reset_sequence_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_sequence_allocating(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_sequence_allocating_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_snapshot_decode(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_snapshot_decode_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn on_unexpected_from_uninitialized(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn planning_backend_error(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::planning_backend_error
        todo!(
            "TODO: port guard `planning_backend_error` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn planning_invalid_request(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::planning_invalid_request
        todo!(
            "TODO: port guard `planning_invalid_request` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn planning_ok(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::planning_ok
        todo!("TODO: port guard `planning_ok` from emel.cpp/src/emel/text/generator/guards.hpp")
    }
    fn prefill_dispatch_available(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::prefill_dispatch_available
        todo!(
            "TODO: port guard `prefill_dispatch_available` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn prefill_dispatch_unavailable(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::prefill_dispatch_unavailable
        todo!(
            "TODO: port guard `prefill_dispatch_unavailable` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn prefill_result_backend_error(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::prefill_result_backend_error
        todo!(
            "TODO: port guard `prefill_result_backend_error` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn prefill_result_invalid_request(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::prefill_result_invalid_request
        todo!(
            "TODO: port guard `prefill_result_invalid_request` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn prefill_result_ok_with_materialized_logits_contract(
        &self,
        _event: &EventGenerateRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::prefill_result_ok_with_materialized_logits_contract
        todo!(
            "TODO: port guard `prefill_result_ok_with_materialized_logits_contract` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn prefill_result_ok_with_preselected_argmax_contract(
        &self,
        _event: &EventGenerateRun,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::prefill_result_ok_with_preselected_argmax_contract
        todo!(
            "TODO: port guard `prefill_result_ok_with_preselected_argmax_contract` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn reject_initialize_from_ready(&mut self, _event: &EventInitializeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::reject_initialize
        todo!(
            "TODO: port action `reject_initialize` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn reject_initialize_from_uninitialized(
        &mut self,
        _event: &EventInitializeRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::reject_initialize
        todo!(
            "TODO: port action `reject_initialize` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn reject_invalid_generate(&mut self, _event: &EventGenerateRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::reject_invalid_generate
        todo!(
            "TODO: port action `reject_invalid_generate` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn reject_uninitialized_generate(&mut self, _event: &EventGenerateRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::reject_uninitialized_generate
        todo!(
            "TODO: port action `reject_uninitialized_generate` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_allocate_sequence(&mut self, _event: &EventGenerateRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_allocate_sequence
        todo!(
            "TODO: port action `request_allocate_sequence` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_conditioning(&mut self, _event: &EventGenerateRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_conditioning
        todo!(
            "TODO: port action `request_conditioning` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_flash_kernel(&mut self, _event: &EventGenerateRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_flash_kernel
        todo!(
            "TODO: port action `request_decode_compute_flash_kernel` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_flash_kernel_streamed(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_flash_kernel_streamed
        todo!(
            "TODO: port action `request_decode_compute_flash_kernel_streamed` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_flash_native_quantized(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_flash_native_quantized
        todo!(
            "TODO: port action `request_decode_compute_flash_native_quantized` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_flash_native_quantized_q8_k_logits(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_flash_native_quantized_q8_k_logits
        todo!(
            "TODO: port action `request_decode_compute_flash_native_quantized_q8_k_logits` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_flash_native_quantized_q8_k_logits_streamed(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_flash_native_quantized_q8_k_logits_streamed
        todo!(
            "TODO: port action `request_decode_compute_flash_native_quantized_q8_k_logits_streamed` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_flash_native_quantized_streamed(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_flash_native_quantized_streamed
        todo!(
            "TODO: port action `request_decode_compute_flash_native_quantized_streamed` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_flash_packed_q8_0(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_flash_packed_q8_0
        todo!(
            "TODO: port action `request_decode_compute_flash_packed_q8_0` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_flash_packed_q8_0_streamed(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_flash_packed_q8_0_streamed
        todo!(
            "TODO: port action `request_decode_compute_flash_packed_q8_0_streamed` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_flash_parallel_kernel(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_flash_parallel_kernel
        todo!(
            "TODO: port action `request_decode_compute_flash_parallel_kernel` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_flash_parallel_native_quantized(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_flash_parallel_native_quantized
        todo!(
            "TODO: port action `request_decode_compute_flash_parallel_native_quantized` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_flash_parallel_native_quantized_q8_k_logits(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_flash_parallel_native_quantized_q8_k_logits
        todo!(
            "TODO: port action `request_decode_compute_flash_parallel_native_quantized_q8_k_logits` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_flash_parallel_packed_q8_0(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_flash_parallel_packed_q8_0
        todo!(
            "TODO: port action `request_decode_compute_flash_parallel_packed_q8_0` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_flash_parallel_preselected_argmax_kernel(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_flash_parallel_preselected_argmax_kernel
        todo!(
            "TODO: port action `request_decode_compute_flash_parallel_preselected_argmax_kernel` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_flash_parallel_preselected_argmax_native_quantized_kernel(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_flash_parallel_preselected_argmax_native_quantized_kernel
        todo!(
            "TODO: port action `request_decode_compute_flash_parallel_preselected_argmax_native_quantized_kernel` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_flash_parallel_preselected_argmax_native_quantized_q8_k(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_flash_parallel_preselected_argmax_native_quantized_q8_k
        todo!(
            "TODO: port action `request_decode_compute_flash_parallel_preselected_argmax_native_quantized_q8_k` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_flash_parallel_preselected_argmax_q8_k(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_flash_parallel_preselected_argmax_q8_k
        todo!(
            "TODO: port action `request_decode_compute_flash_parallel_preselected_argmax_q8_k` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_flash_parallel_q8_k(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_flash_parallel_q8_k
        todo!(
            "TODO: port action `request_decode_compute_flash_parallel_q8_k` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_flash_preselected_argmax_kernel(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_flash_preselected_argmax_kernel
        todo!(
            "TODO: port action `request_decode_compute_flash_preselected_argmax_kernel` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_flash_preselected_argmax_kernel_streamed(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_flash_preselected_argmax_kernel_streamed
        todo!(
            "TODO: port action `request_decode_compute_flash_preselected_argmax_kernel_streamed` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_flash_preselected_argmax_native_quantized_kernel(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_flash_preselected_argmax_native_quantized_kernel
        todo!(
            "TODO: port action `request_decode_compute_flash_preselected_argmax_native_quantized_kernel` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_flash_preselected_argmax_native_quantized_kernel_streamed(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_flash_preselected_argmax_native_quantized_kernel_streamed
        todo!(
            "TODO: port action `request_decode_compute_flash_preselected_argmax_native_quantized_kernel_streamed` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_flash_preselected_argmax_native_quantized_q8_k(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_flash_preselected_argmax_native_quantized_q8_k
        todo!(
            "TODO: port action `request_decode_compute_flash_preselected_argmax_native_quantized_q8_k` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_flash_preselected_argmax_native_quantized_q8_k_streamed(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_flash_preselected_argmax_native_quantized_q8_k_streamed
        todo!(
            "TODO: port action `request_decode_compute_flash_preselected_argmax_native_quantized_q8_k_streamed` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_flash_preselected_argmax_q8_k(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_flash_preselected_argmax_q8_k
        todo!(
            "TODO: port action `request_decode_compute_flash_preselected_argmax_q8_k` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_flash_preselected_argmax_q8_k_streamed(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_flash_preselected_argmax_q8_k_streamed
        todo!(
            "TODO: port action `request_decode_compute_flash_preselected_argmax_q8_k_streamed` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_flash_q8_k(&mut self, _event: &EventGenerateRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_flash_q8_k
        todo!(
            "TODO: port action `request_decode_compute_flash_q8_k` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_flash_q8_k_streamed(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_flash_q8_k_streamed
        todo!(
            "TODO: port action `request_decode_compute_flash_q8_k_streamed` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_nonflash_kernel(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_nonflash_kernel
        todo!(
            "TODO: port action `request_decode_compute_nonflash_kernel` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_nonflash_kernel_streamed(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_nonflash_kernel_streamed
        todo!(
            "TODO: port action `request_decode_compute_nonflash_kernel_streamed` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_nonflash_native_quantized(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_nonflash_native_quantized
        todo!(
            "TODO: port action `request_decode_compute_nonflash_native_quantized` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_nonflash_native_quantized_q8_k_logits(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_nonflash_native_quantized_q8_k_logits
        todo!(
            "TODO: port action `request_decode_compute_nonflash_native_quantized_q8_k_logits` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_nonflash_native_quantized_q8_k_logits_streamed(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_nonflash_native_quantized_q8_k_logits_streamed
        todo!(
            "TODO: port action `request_decode_compute_nonflash_native_quantized_q8_k_logits_streamed` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_nonflash_native_quantized_streamed(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_nonflash_native_quantized_streamed
        todo!(
            "TODO: port action `request_decode_compute_nonflash_native_quantized_streamed` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_nonflash_packed_q8_0(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_nonflash_packed_q8_0
        todo!(
            "TODO: port action `request_decode_compute_nonflash_packed_q8_0` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_nonflash_packed_q8_0_streamed(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_nonflash_packed_q8_0_streamed
        todo!(
            "TODO: port action `request_decode_compute_nonflash_packed_q8_0_streamed` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_nonflash_preselected_argmax_kernel(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_nonflash_preselected_argmax_kernel
        todo!(
            "TODO: port action `request_decode_compute_nonflash_preselected_argmax_kernel` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_nonflash_preselected_argmax_kernel_streamed(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_nonflash_preselected_argmax_kernel_streamed
        todo!(
            "TODO: port action `request_decode_compute_nonflash_preselected_argmax_kernel_streamed` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_nonflash_preselected_argmax_native_quantized_kernel(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_nonflash_preselected_argmax_native_quantized_kernel
        todo!(
            "TODO: port action `request_decode_compute_nonflash_preselected_argmax_native_quantized_kernel` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_nonflash_preselected_argmax_native_quantized_kernel_streamed(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_nonflash_preselected_argmax_native_quantized_kernel_streamed
        todo!(
            "TODO: port action `request_decode_compute_nonflash_preselected_argmax_native_quantized_kernel_streamed` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_nonflash_preselected_argmax_native_quantized_q8_k(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_nonflash_preselected_argmax_native_quantized_q8_k
        todo!(
            "TODO: port action `request_decode_compute_nonflash_preselected_argmax_native_quantized_q8_k` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_nonflash_preselected_argmax_native_quantized_q8_k_streamed(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_nonflash_preselected_argmax_native_quantized_q8_k_streamed
        todo!(
            "TODO: port action `request_decode_compute_nonflash_preselected_argmax_native_quantized_q8_k_streamed` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_nonflash_preselected_argmax_q8_k(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_nonflash_preselected_argmax_q8_k
        todo!(
            "TODO: port action `request_decode_compute_nonflash_preselected_argmax_q8_k` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_nonflash_preselected_argmax_q8_k_streamed(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_nonflash_preselected_argmax_q8_k_streamed
        todo!(
            "TODO: port action `request_decode_compute_nonflash_preselected_argmax_q8_k_streamed` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_nonflash_q8_k(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_nonflash_q8_k
        todo!(
            "TODO: port action `request_decode_compute_nonflash_q8_k` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_compute_nonflash_q8_k_streamed(
        &mut self,
        _event: &EventGenerateRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_compute_nonflash_q8_k_streamed
        todo!(
            "TODO: port action `request_decode_compute_nonflash_q8_k_streamed` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_render(&mut self, _event: &EventGenerateRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_render
        todo!(
            "TODO: port action `request_decode_render` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_sample(&mut self, _event: &EventGenerateRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_sample
        todo!(
            "TODO: port action `request_decode_sample` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_sample_preselected(&mut self, _event: &EventGenerateRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_sample_preselected
        todo!(
            "TODO: port action `request_decode_sample_preselected` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_select_argmax(&mut self, _event: &EventGenerateRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_select_argmax
        todo!(
            "TODO: port action `request_decode_select_argmax` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_decode_slots(&mut self, _event: &EventGenerateRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_decode_slots
        todo!(
            "TODO: port action `request_decode_slots` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_flush(&mut self, _event: &EventGenerateRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_flush
        todo!("TODO: port action `request_flush` from emel.cpp/src/emel/text/generator/actions.hpp")
    }
    fn request_initializer(&mut self, _event: &EventInitializeRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_initializer
        todo!(
            "TODO: port action `request_initializer` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_memory_snapshot(&mut self, _event: &EventGenerateRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_memory_snapshot
        todo!(
            "TODO: port action `request_memory_snapshot` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_planning_chunk4(&mut self, _event: &EventGenerateRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_planning_chunk4
        todo!(
            "TODO: port action `request_planning_chunk4` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_planning_chunk8(&mut self, _event: &EventGenerateRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_planning_chunk8
        todo!(
            "TODO: port action `request_planning_chunk8` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_planning_scalar(&mut self, _event: &EventGenerateRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_planning_scalar
        todo!(
            "TODO: port action `request_planning_scalar` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_prefill(&mut self, _event: &EventGenerateRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_prefill
        todo!(
            "TODO: port action `request_prefill` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn request_reset_sequence(&mut self, _event: &EventGenerateRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/actions.hpp::request_reset_sequence
        todo!(
            "TODO: port action `request_reset_sequence` from emel.cpp/src/emel/text/generator/actions.hpp"
        )
    }
    fn reset_sequence_backend_error(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::reset_sequence_backend_error
        todo!(
            "TODO: port guard `reset_sequence_backend_error` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn reset_sequence_invalid_request(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::reset_sequence_invalid_request
        todo!(
            "TODO: port guard `reset_sequence_invalid_request` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn reset_sequence_ok(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::reset_sequence_ok
        todo!(
            "TODO: port guard `reset_sequence_ok` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn valid_generate_with_reset(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::valid_generate_with_reset
        todo!(
            "TODO: port guard `valid_generate_with_reset` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn valid_generate_without_reset(&self, _event: &EventGenerateRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::valid_generate_without_reset
        todo!(
            "TODO: port guard `valid_generate_without_reset` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
    fn valid_initialize(&self, _event: &EventInitializeRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/guards.hpp::valid_initialize
        todo!(
            "TODO: port guard `valid_initialize` from emel.cpp/src/emel/text/generator/guards.hpp"
        )
    }
}
