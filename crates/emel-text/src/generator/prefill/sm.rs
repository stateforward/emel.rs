//! Source-aligned bounded prefill actor for text generation.
//!
//! The actor owns only fixed-size routing/status state.  Model work is injected
//! through one synchronous, bounded callback and is never queued or allocated.

#![allow(
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::missing_const_for_fn,
    dead_code,
    missing_docs
)]

use core::cell::Cell;
use sml::sml;

/// Outcome written by an injected bounded phase callback.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PhaseOutcome {
    pub accepted: bool,
    pub code: i32,
}
impl PhaseOutcome {
    pub const fn ok() -> Self { Self { accepted: true, code: 0 } }
    pub const fn invalid_request() -> Self { Self { accepted: false, code: 1 } }
    pub const fn backend() -> Self { Self { accepted: false, code: 2 } }
}

/// Public errors reported by the prefill actor.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(i32)]
pub enum PrefillError {
    #[default]
    None = 0,
    InvalidRequest = 1,
    Backend = 2,
    UnexpectedEvent = 3,
}
impl PrefillError {
    const fn from_code(code: i32) -> Self {
        match code { 0 => Self::None, 1 => Self::InvalidRequest, _ => Self::Backend }
    }
}

/// Selection mode used by the generation contract.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SelectionMode { #[default] SampleLogits, PreselectedArgmax }

/// Every bounded callback route materialized by the C++ contract.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PrefillOperation {
    #[default] None,
    Slots,
    MemorySnapshot,
    FlashMaterializedScalarPackedQ8_0,
    FlashMaterializedScalarQ8K,
    FlashMaterializedScalarNativeQuantized,
    FlashMaterializedScalarNativeQuantizedQ8K,
    FlashMaterializedScalarKernel,
    FlashMaterializedChunk8Q8K,
    FlashMaterializedParallelChunk8Q8K,
    FlashMaterializedChunk4PackedQ8_0,
    FlashMaterializedParallelChunk4PackedQ8_0,
    FlashMaterializedChunk4Q8K,
    FlashMaterializedParallelChunk4Q8K,
    FlashPreselectedScalarQ8K,
    FlashPreselectedScalarNativeQuantizedQ8K,
    FlashPreselectedScalarNativeQuantizedKernel,
    FlashPreselectedScalarKernel,
    FlashPreselectedChunk8Q8K,
    FlashPreselectedParallelChunk8Q8K,
    FlashPreselectedChunk4PackedQ8_0,
    FlashPreselectedParallelChunk4PackedQ8_0,
    FlashPreselectedChunk4Q8K,
    FlashPreselectedParallelChunk4Q8K,
    NonflashMaterializedScalarPackedQ8_0,
    NonflashMaterializedScalarQ8K,
    NonflashMaterializedScalarNativeQuantized,
    NonflashMaterializedScalarNativeQuantizedQ8K,
    NonflashMaterializedScalarKernel,
    NonflashMaterializedChunk8Q8K,
    NonflashMaterializedChunk4PackedQ8_0,
    NonflashMaterializedChunk4Q8K,
    NonflashPreselectedScalarQ8K,
    NonflashPreselectedScalarNativeQuantizedQ8K,
    NonflashPreselectedScalarNativeQuantizedKernel,
    NonflashPreselectedScalarKernel,
    NonflashPreselectedChunk8Q8K,
    NonflashPreselectedChunk4PackedQ8_0,
    NonflashPreselectedChunk4Q8K,
}

/// Synchronous callback used for slots, snapshots, and compute kernels.
pub type PrefillDispatch = fn(PrefillOperation, &EventRun) -> PhaseOutcome;

/// Runtime event and its mutable, caller-owned phase result.
#[derive(Debug)]
pub struct EventRun {
    pub prompt_token_count: u32,
    pub selection_mode: SelectionMode,
    pub backend_available: bool,
    pub flash_attention_supported: bool,
    pub kv_map_identity: bool,
    pub parallel_lanes_enabled: bool,
    pub matmul_parallel: bool,
    pub parallel_min_prefill_tokens: u32,
    pub prefill_chunk4_min_tokens: u32,
    pub prefill_chunk8_min_tokens: u32,
    pub direct_preselected_supported: bool,
    pub compute_invalid_request: bool,
    pub compute_backend_unavailable: bool,
    pub materialized_chunk8_q8_k_supported: bool,
    pub materialized_chunk4_packed_q8_0_supported: bool,
    pub materialized_chunk4_q8_k_supported: bool,
    pub materialized_scalar_packed_q8_0_supported: bool,
    pub materialized_scalar_q8_k_supported: bool,
    pub materialized_scalar_native_quantized_supported: bool,
    pub materialized_scalar_native_quantized_q8_k_supported: bool,
    pub materialized_scalar_kernel_supported: bool,
    pub preselected_chunk8_q8_k_supported: bool,
    pub preselected_chunk4_packed_q8_0_supported: bool,
    pub preselected_chunk4_q8_k_supported: bool,
    pub preselected_scalar_q8_k_supported: bool,
    pub preselected_scalar_native_quantized_q8_k_supported: bool,
    pub preselected_scalar_native_quantized_kernel_supported: bool,
    pub preselected_scalar_kernel_supported: bool,
    pub phase_accepted: Cell<bool>,
    pub phase_code: Cell<i32>,
}
impl Clone for EventRun {
    fn clone(&self) -> Self {
        Self {
            prompt_token_count: self.prompt_token_count, selection_mode: self.selection_mode,
            backend_available: self.backend_available, flash_attention_supported: self.flash_attention_supported,
            kv_map_identity: self.kv_map_identity, parallel_lanes_enabled: self.parallel_lanes_enabled,
            matmul_parallel: self.matmul_parallel, parallel_min_prefill_tokens: self.parallel_min_prefill_tokens,
            prefill_chunk4_min_tokens: self.prefill_chunk4_min_tokens, prefill_chunk8_min_tokens: self.prefill_chunk8_min_tokens,
            direct_preselected_supported: self.direct_preselected_supported, compute_invalid_request: self.compute_invalid_request,
            compute_backend_unavailable: self.compute_backend_unavailable,
            materialized_chunk8_q8_k_supported: self.materialized_chunk8_q8_k_supported,
            materialized_chunk4_packed_q8_0_supported: self.materialized_chunk4_packed_q8_0_supported,
            materialized_chunk4_q8_k_supported: self.materialized_chunk4_q8_k_supported,
            materialized_scalar_packed_q8_0_supported: self.materialized_scalar_packed_q8_0_supported,
            materialized_scalar_q8_k_supported: self.materialized_scalar_q8_k_supported,
            materialized_scalar_native_quantized_supported: self.materialized_scalar_native_quantized_supported,
            materialized_scalar_native_quantized_q8_k_supported: self.materialized_scalar_native_quantized_q8_k_supported,
            materialized_scalar_kernel_supported: self.materialized_scalar_kernel_supported,
            preselected_chunk8_q8_k_supported: self.preselected_chunk8_q8_k_supported,
            preselected_chunk4_packed_q8_0_supported: self.preselected_chunk4_packed_q8_0_supported,
            preselected_chunk4_q8_k_supported: self.preselected_chunk4_q8_k_supported,
            preselected_scalar_q8_k_supported: self.preselected_scalar_q8_k_supported,
            preselected_scalar_native_quantized_q8_k_supported: self.preselected_scalar_native_quantized_q8_k_supported,
            preselected_scalar_native_quantized_kernel_supported: self.preselected_scalar_native_quantized_kernel_supported,
            preselected_scalar_kernel_supported: self.preselected_scalar_kernel_supported,
            phase_accepted: Cell::new(self.phase_accepted.get()), phase_code: Cell::new(self.phase_code.get()),
        }
    }
}
impl Default for EventRun {
    fn default() -> Self {
        Self {
            prompt_token_count: 0, selection_mode: SelectionMode::SampleLogits, backend_available: true,
            flash_attention_supported: false, kv_map_identity: true, parallel_lanes_enabled: false,
            matmul_parallel: false, parallel_min_prefill_tokens: 8, prefill_chunk4_min_tokens: 4,
            prefill_chunk8_min_tokens: 8, direct_preselected_supported: false,
            compute_invalid_request: false, compute_backend_unavailable: false,
            materialized_chunk8_q8_k_supported: false, materialized_chunk4_packed_q8_0_supported: false,
            materialized_chunk4_q8_k_supported: false, materialized_scalar_packed_q8_0_supported: false,
            materialized_scalar_q8_k_supported: false, materialized_scalar_native_quantized_supported: false,
            materialized_scalar_native_quantized_q8_k_supported: false, materialized_scalar_kernel_supported: true,
            preselected_chunk8_q8_k_supported: false, preselected_chunk4_packed_q8_0_supported: false,
            preselected_chunk4_q8_k_supported: false, preselected_scalar_q8_k_supported: false,
            preselected_scalar_native_quantized_q8_k_supported: false,
            preselected_scalar_native_quantized_kernel_supported: false, preselected_scalar_kernel_supported: true,
            phase_accepted: Cell::new(false), phase_code: Cell::new(0),
        }
    }
}
impl EventRun {
    #[must_use] pub fn new(prompt_token_count: u32) -> Self { Self { prompt_token_count, ..Self::default() } }
    pub fn set_phase(&self, outcome: PhaseOutcome) { self.phase_accepted.set(outcome.accepted); self.phase_code.set(outcome.code); }
}

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

/// Bounded context shared by the generated machine and its synchronous wrapper.
#[derive(Debug, Default)]
pub struct TextGeneratorPrefillContext {
    error: PrefillError,
    kv_tokens: u32,
    last_operation: PrefillOperation,
    dispatcher: Option<PrefillDispatch>,
}
impl TextGeneratorPrefillContext {
    #[must_use] pub const fn error(&self) -> PrefillError { self.error }
    #[must_use] pub const fn kv_tokens(&self) -> u32 { self.kv_tokens }
    #[must_use] pub const fn last_operation(&self) -> PrefillOperation { self.last_operation }
    pub fn set_dispatcher(&mut self, dispatcher: Option<PrefillDispatch>) { self.dispatcher = dispatcher; }
}

fn operation_for_action(name: &str) -> Option<PrefillOperation> {
    Some(match name.strip_prefix("request_contract_")? {
        "flash_materialized_scalar_packed_q8_0" => PrefillOperation::FlashMaterializedScalarPackedQ8_0,
        "flash_materialized_scalar_q8_k" => PrefillOperation::FlashMaterializedScalarQ8K,
        "flash_materialized_scalar_native_quantized" => PrefillOperation::FlashMaterializedScalarNativeQuantized,
        "flash_materialized_scalar_native_quantized_q8_k" => PrefillOperation::FlashMaterializedScalarNativeQuantizedQ8K,
        "flash_materialized_scalar_kernel" => PrefillOperation::FlashMaterializedScalarKernel,
        "flash_materialized_chunk8_q8_k" => PrefillOperation::FlashMaterializedChunk8Q8K,
        "flash_materialized_parallel_chunk8_q8_k" => PrefillOperation::FlashMaterializedParallelChunk8Q8K,
        "flash_materialized_chunk4_packed_q8_0" => PrefillOperation::FlashMaterializedChunk4PackedQ8_0,
        "flash_materialized_parallel_chunk4_packed_q8_0" => PrefillOperation::FlashMaterializedParallelChunk4PackedQ8_0,
        "flash_materialized_chunk4_q8_k" => PrefillOperation::FlashMaterializedChunk4Q8K,
        "flash_materialized_parallel_chunk4_q8_k" => PrefillOperation::FlashMaterializedParallelChunk4Q8K,
        "flash_preselected_scalar_q8_k" => PrefillOperation::FlashPreselectedScalarQ8K,
        "flash_preselected_scalar_native_quantized_q8_k" => PrefillOperation::FlashPreselectedScalarNativeQuantizedQ8K,
        "flash_preselected_scalar_native_quantized_kernel" => PrefillOperation::FlashPreselectedScalarNativeQuantizedKernel,
        "flash_preselected_scalar_kernel" => PrefillOperation::FlashPreselectedScalarKernel,
        "flash_preselected_chunk8_q8_k" => PrefillOperation::FlashPreselectedChunk8Q8K,
        "flash_preselected_parallel_chunk8_q8_k" => PrefillOperation::FlashPreselectedParallelChunk8Q8K,
        "flash_preselected_chunk4_packed_q8_0" => PrefillOperation::FlashPreselectedChunk4PackedQ8_0,
        "flash_preselected_parallel_chunk4_packed_q8_0" => PrefillOperation::FlashPreselectedParallelChunk4PackedQ8_0,
        "flash_preselected_chunk4_q8_k" => PrefillOperation::FlashPreselectedChunk4Q8K,
        "flash_preselected_parallel_chunk4_q8_k" => PrefillOperation::FlashPreselectedParallelChunk4Q8K,
        "nonflash_materialized_scalar_packed_q8_0" => PrefillOperation::NonflashMaterializedScalarPackedQ8_0,
        "nonflash_materialized_scalar_q8_k" => PrefillOperation::NonflashMaterializedScalarQ8K,
        "nonflash_materialized_scalar_native_quantized" => PrefillOperation::NonflashMaterializedScalarNativeQuantized,
        "nonflash_materialized_scalar_native_quantized_q8_k" => PrefillOperation::NonflashMaterializedScalarNativeQuantizedQ8K,
        "nonflash_materialized_scalar_kernel" => PrefillOperation::NonflashMaterializedScalarKernel,
        "nonflash_materialized_chunk8_q8_k" => PrefillOperation::NonflashMaterializedChunk8Q8K,
        "nonflash_materialized_chunk4_packed_q8_0" => PrefillOperation::NonflashMaterializedChunk4PackedQ8_0,
        "nonflash_materialized_chunk4_q8_k" => PrefillOperation::NonflashMaterializedChunk4Q8K,
        "nonflash_preselected_scalar_q8_k" => PrefillOperation::NonflashPreselectedScalarQ8K,
        "nonflash_preselected_scalar_native_quantized_q8_k" => PrefillOperation::NonflashPreselectedScalarNativeQuantizedQ8K,
        "nonflash_preselected_scalar_native_quantized_kernel" => PrefillOperation::NonflashPreselectedScalarNativeQuantizedKernel,
        "nonflash_preselected_scalar_kernel" => PrefillOperation::NonflashPreselectedScalarKernel,
        "nonflash_preselected_chunk8_q8_k" => PrefillOperation::NonflashPreselectedChunk8Q8K,
        "nonflash_preselected_chunk4_packed_q8_0" => PrefillOperation::NonflashPreselectedChunk4PackedQ8_0,
        "nonflash_preselected_chunk4_q8_k" => PrefillOperation::NonflashPreselectedChunk4Q8K,
        _ => return None,
    })
}

impl TextGeneratorPrefillContext {
    fn phase_success(event: &EventRun) -> bool { event.phase_accepted.get() && event.phase_code.get() == 0 }
    fn phase_invalid(event: &EventRun) -> bool { !Self::phase_success(event) && event.phase_code.get() == 1 }
    fn phase_backend(event: &EventRun) -> bool { !Self::phase_success(event) && !Self::phase_invalid(event) }
    fn preselected(event: &EventRun) -> bool { event.selection_mode == SelectionMode::PreselectedArgmax && event.direct_preselected_supported }
    fn flash(event: &EventRun) -> bool { event.prompt_token_count > 0 && event.flash_attention_supported && event.kv_map_identity }
    fn parallel(event: &EventRun) -> bool { event.parallel_lanes_enabled && event.matmul_parallel && event.prompt_token_count >= event.parallel_min_prefill_tokens }
    fn chunk4(event: &EventRun) -> bool { event.prompt_token_count >= event.prefill_chunk4_min_tokens.max(4) }
    fn chunk8(event: &EventRun) -> bool { event.prompt_token_count >= event.prefill_chunk8_min_tokens.max(8) }
    fn route(event: &EventRun, name: &str) -> bool {
        let p = Self::preselected(event); let c8 = Self::chunk8(event); let c4 = Self::chunk4(event);
        match name {
            "guard_materialized_logits_with_parallel_chunk8_q8_k_ready" => !p && Self::parallel(event) && c8 && event.materialized_chunk8_q8_k_supported,
            "guard_materialized_logits_with_chunk8_q8_k_ready" => !p && c8 && event.materialized_chunk8_q8_k_supported,
            "guard_materialized_logits_with_parallel_chunk4_packed_q8_0_ready" => !p && !c8 && Self::parallel(event) && c4 && event.materialized_chunk4_packed_q8_0_supported,
            "guard_materialized_logits_with_chunk4_packed_q8_0_ready" => !p && !c8 && c4 && event.materialized_chunk4_packed_q8_0_supported,
            "guard_materialized_logits_with_parallel_chunk4_q8_k_ready" => !p && !c8 && Self::parallel(event) && c4 && event.materialized_chunk4_q8_k_supported,
            "guard_materialized_logits_with_chunk4_q8_k_ready" => !p && !c8 && c4 && event.materialized_chunk4_q8_k_supported,
            "guard_materialized_logits_with_scalar_packed_q8_0_ready" => !p && !c8 && !c4 && event.materialized_scalar_packed_q8_0_supported,
            "guard_materialized_logits_with_scalar_q8_k_ready" => !p && !c8 && !c4 && !event.materialized_scalar_packed_q8_0_supported && event.materialized_scalar_q8_k_supported,
            "guard_materialized_logits_with_scalar_native_quantized_q8_k_ready" => !p && !c8 && !c4 && !event.materialized_scalar_packed_q8_0_supported && !event.materialized_scalar_q8_k_supported && event.materialized_scalar_native_quantized_q8_k_supported,
            "guard_materialized_logits_with_scalar_native_quantized_kernel_ready" => !p && !c8 && !c4 && !event.materialized_scalar_packed_q8_0_supported && !event.materialized_scalar_q8_k_supported && !event.materialized_scalar_native_quantized_q8_k_supported && event.materialized_scalar_native_quantized_supported,
            "guard_materialized_logits_with_scalar_kernel_ready" => !p && !c8 && !c4 && !event.materialized_scalar_packed_q8_0_supported && !event.materialized_scalar_q8_k_supported && !event.materialized_scalar_native_quantized_q8_k_supported && !event.materialized_scalar_native_quantized_supported && event.materialized_scalar_kernel_supported,
            "guard_preselected_argmax_with_parallel_chunk8_q8_k_ready" => p && Self::parallel(event) && c8 && event.preselected_chunk8_q8_k_supported,
            "guard_preselected_argmax_with_chunk8_q8_k_ready" => p && c8 && event.preselected_chunk8_q8_k_supported,
            "guard_preselected_argmax_with_parallel_chunk4_packed_q8_0_ready" => p && !c8 && Self::parallel(event) && c4 && event.preselected_chunk4_packed_q8_0_supported,
            "guard_preselected_argmax_with_chunk4_packed_q8_0_ready" => p && !c8 && c4 && event.preselected_chunk4_packed_q8_0_supported,
            "guard_preselected_argmax_with_parallel_chunk4_q8_k_ready" => p && !c8 && Self::parallel(event) && c4 && event.preselected_chunk4_q8_k_supported,
            "guard_preselected_argmax_with_chunk4_q8_k_ready" => p && !c8 && c4 && event.preselected_chunk4_q8_k_supported,
            "guard_preselected_argmax_with_scalar_q8_k_ready" => p && !c8 && !c4 && event.preselected_scalar_q8_k_supported,
            "guard_preselected_argmax_with_scalar_native_quantized_q8_k_ready" => p && !c8 && !c4 && !event.preselected_scalar_q8_k_supported && event.preselected_scalar_native_quantized_q8_k_supported,
            "guard_preselected_argmax_with_scalar_native_quantized_kernel_ready" => p && !c8 && !c4 && !event.preselected_scalar_q8_k_supported && !event.preselected_scalar_native_quantized_q8_k_supported && event.preselected_scalar_native_quantized_kernel_supported,
            "guard_preselected_argmax_with_scalar_kernel_ready" => p && !c8 && !c4 && !event.preselected_scalar_q8_k_supported && !event.preselected_scalar_native_quantized_q8_k_supported && !event.preselected_scalar_native_quantized_kernel_supported && event.preselected_scalar_kernel_supported,
            _ => false,
        }
    }
    fn guard_named(&self, name: &str, event: &EventRun) -> bool {
        match name {
            "slots_ok" | "snapshot_ok" | "compute_ok" => Self::phase_success(event),
            "slots_invalid_request" | "snapshot_invalid_request" | "compute_invalid_request" => Self::phase_invalid(event),
            "slots_backend_error" | "snapshot_backend_error" | "compute_backend_error" => Self::phase_backend(event),
            "flash_runtime_supported" => Self::flash(event), "nonflash_runtime_required" => !Self::flash(event),
            "guard_compute_invalid_request" => event.compute_invalid_request || !event.backend_available,
            "guard_compute_backend_unavailable" => event.compute_backend_unavailable || !event.backend_available,
            n if n.ends_with("_ready") => Self::phase_success(event) && !event.compute_invalid_request && !event.compute_backend_unavailable && Self::route(event,n),
            _ => false,
        }
    }
    fn dispatch(&mut self, operation: PrefillOperation, event: &EventRun) {
        let outcome = self.dispatcher.map_or(PhaseOutcome::backend(), |f| f(operation,event));
        event.set_phase(outcome); self.last_operation = operation;
    }
    fn action_named(&mut self, name: &str, event: &EventRun) {
        if name.starts_with("request_slots") { self.dispatch(PrefillOperation::Slots,event); return; }
        if name.starts_with("request_memory_snapshot") { self.dispatch(PrefillOperation::MemorySnapshot,event); return; }
        if let Some(op) = operation_for_action(name) { self.dispatch(op,event); return; }
        if name.starts_with("mark_invalid_request") { self.error = PrefillError::InvalidRequest; return; }
        if name.starts_with("mark_backend_error") { self.error = PrefillError::Backend; return; }
        if name.starts_with("mark_prefill_cached") { self.error = PrefillError::None; self.kv_tokens = event.prompt_token_count; return; }
        if name.starts_with("on_unexpected") { self.error = PrefillError::UnexpectedEvent; return; }
    }
}

impl TextGeneratorPrefillStateMachineContext for TextGeneratorPrefillContext {
    fn compute_backend_error(&self, event: &EventRun) -> Result<bool, ()> { Ok(self.guard_named("compute_backend_error", event)) }
    fn compute_invalid_request(&self, event: &EventRun) -> Result<bool, ()> { Ok(self.guard_named("compute_invalid_request", event)) }
    fn compute_ok(&self, event: &EventRun) -> Result<bool, ()> { Ok(self.guard_named("compute_ok", event)) }
    fn flash_runtime_supported(&self, event: &EventRun) -> Result<bool, ()> { Ok(self.guard_named("flash_runtime_supported", event)) }
    fn guard_compute_backend_unavailable(&self, event: &EventRun) -> Result<bool, ()> { Ok(self.guard_named("guard_compute_backend_unavailable", event)) }
    fn guard_compute_invalid_request(&self, event: &EventRun) -> Result<bool, ()> { Ok(self.guard_named("guard_compute_invalid_request", event)) }
    fn guard_materialized_logits_with_chunk4_packed_q8_0_ready(&self, event: &EventRun) -> Result<bool, ()> { Ok(self.guard_named("guard_materialized_logits_with_chunk4_packed_q8_0_ready", event)) }
    fn guard_materialized_logits_with_chunk4_q8_k_ready(&self, event: &EventRun) -> Result<bool, ()> { Ok(self.guard_named("guard_materialized_logits_with_chunk4_q8_k_ready", event)) }
    fn guard_materialized_logits_with_chunk8_q8_k_ready(&self, event: &EventRun) -> Result<bool, ()> { Ok(self.guard_named("guard_materialized_logits_with_chunk8_q8_k_ready", event)) }
    fn guard_materialized_logits_with_parallel_chunk4_packed_q8_0_ready(&self, event: &EventRun) -> Result<bool, ()> { Ok(self.guard_named("guard_materialized_logits_with_parallel_chunk4_packed_q8_0_ready", event)) }
    fn guard_materialized_logits_with_parallel_chunk4_q8_k_ready(&self, event: &EventRun) -> Result<bool, ()> { Ok(self.guard_named("guard_materialized_logits_with_parallel_chunk4_q8_k_ready", event)) }
    fn guard_materialized_logits_with_parallel_chunk8_q8_k_ready(&self, event: &EventRun) -> Result<bool, ()> { Ok(self.guard_named("guard_materialized_logits_with_parallel_chunk8_q8_k_ready", event)) }
    fn guard_materialized_logits_with_scalar_kernel_ready(&self, event: &EventRun) -> Result<bool, ()> { Ok(self.guard_named("guard_materialized_logits_with_scalar_kernel_ready", event)) }
    fn guard_materialized_logits_with_scalar_native_quantized_kernel_ready(&self, event: &EventRun) -> Result<bool, ()> { Ok(self.guard_named("guard_materialized_logits_with_scalar_native_quantized_kernel_ready", event)) }
    fn guard_materialized_logits_with_scalar_native_quantized_q8_k_ready(&self, event: &EventRun) -> Result<bool, ()> { Ok(self.guard_named("guard_materialized_logits_with_scalar_native_quantized_q8_k_ready", event)) }
    fn guard_materialized_logits_with_scalar_packed_q8_0_ready(&self, event: &EventRun) -> Result<bool, ()> { Ok(self.guard_named("guard_materialized_logits_with_scalar_packed_q8_0_ready", event)) }
    fn guard_materialized_logits_with_scalar_q8_k_ready(&self, event: &EventRun) -> Result<bool, ()> { Ok(self.guard_named("guard_materialized_logits_with_scalar_q8_k_ready", event)) }
    fn guard_preselected_argmax_with_chunk4_packed_q8_0_ready(&self, event: &EventRun) -> Result<bool, ()> { Ok(self.guard_named("guard_preselected_argmax_with_chunk4_packed_q8_0_ready", event)) }
    fn guard_preselected_argmax_with_chunk4_q8_k_ready(&self, event: &EventRun) -> Result<bool, ()> { Ok(self.guard_named("guard_preselected_argmax_with_chunk4_q8_k_ready", event)) }
    fn guard_preselected_argmax_with_chunk8_q8_k_ready(&self, event: &EventRun) -> Result<bool, ()> { Ok(self.guard_named("guard_preselected_argmax_with_chunk8_q8_k_ready", event)) }
    fn guard_preselected_argmax_with_parallel_chunk4_packed_q8_0_ready(&self, event: &EventRun) -> Result<bool, ()> { Ok(self.guard_named("guard_preselected_argmax_with_parallel_chunk4_packed_q8_0_ready", event)) }
    fn guard_preselected_argmax_with_parallel_chunk4_q8_k_ready(&self, event: &EventRun) -> Result<bool, ()> { Ok(self.guard_named("guard_preselected_argmax_with_parallel_chunk4_q8_k_ready", event)) }
    fn guard_preselected_argmax_with_parallel_chunk8_q8_k_ready(&self, event: &EventRun) -> Result<bool, ()> { Ok(self.guard_named("guard_preselected_argmax_with_parallel_chunk8_q8_k_ready", event)) }
    fn guard_preselected_argmax_with_scalar_kernel_ready(&self, event: &EventRun) -> Result<bool, ()> { Ok(self.guard_named("guard_preselected_argmax_with_scalar_kernel_ready", event)) }
    fn guard_preselected_argmax_with_scalar_native_quantized_kernel_ready(&self, event: &EventRun) -> Result<bool, ()> { Ok(self.guard_named("guard_preselected_argmax_with_scalar_native_quantized_kernel_ready", event)) }
    fn guard_preselected_argmax_with_scalar_native_quantized_q8_k_ready(&self, event: &EventRun) -> Result<bool, ()> { Ok(self.guard_named("guard_preselected_argmax_with_scalar_native_quantized_q8_k_ready", event)) }
    fn guard_preselected_argmax_with_scalar_q8_k_ready(&self, event: &EventRun) -> Result<bool, ()> { Ok(self.guard_named("guard_preselected_argmax_with_scalar_q8_k_ready", event)) }
    fn mark_backend_error_from_compute_result_decision(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("mark_backend_error_from_compute_result_decision", event); Ok(()) }
    fn mark_backend_error_from_contract_flash_decision(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("mark_backend_error_from_contract_flash_decision", event); Ok(()) }
    fn mark_backend_error_from_contract_nonflash_decision(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("mark_backend_error_from_contract_nonflash_decision", event); Ok(()) }
    fn mark_backend_error_from_slots_decision(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("mark_backend_error_from_slots_decision", event); Ok(()) }
    fn mark_backend_error_from_snapshot_decision(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("mark_backend_error_from_snapshot_decision", event); Ok(()) }
    fn mark_invalid_request_from_compute_result_decision(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("mark_invalid_request_from_compute_result_decision", event); Ok(()) }
    fn mark_invalid_request_from_contract_flash_decision(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("mark_invalid_request_from_contract_flash_decision", event); Ok(()) }
    fn mark_invalid_request_from_contract_nonflash_decision(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("mark_invalid_request_from_contract_nonflash_decision", event); Ok(()) }
    fn mark_invalid_request_from_slots_decision(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("mark_invalid_request_from_slots_decision", event); Ok(()) }
    fn mark_invalid_request_from_snapshot_decision(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("mark_invalid_request_from_snapshot_decision", event); Ok(()) }
    fn mark_prefill_cached(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("mark_prefill_cached", event); Ok(()) }
    fn nonflash_runtime_required(&self, event: &EventRun) -> Result<bool, ()> { Ok(self.guard_named("nonflash_runtime_required", event)) }
    fn on_unexpected_from_compute_result_decision(&mut self) -> Result<(), ()> { self.action_named("on_unexpected_from_compute_result_decision", &EventRun::default()); Ok(()) }
    fn on_unexpected_from_contract_flash_decision(&mut self) -> Result<(), ()> { self.action_named("on_unexpected_from_contract_flash_decision", &EventRun::default()); Ok(()) }
    fn on_unexpected_from_contract_nonflash_decision(&mut self) -> Result<(), ()> { self.action_named("on_unexpected_from_contract_nonflash_decision", &EventRun::default()); Ok(()) }
    fn on_unexpected_from_contract_runtime_decision(&mut self) -> Result<(), ()> { self.action_named("on_unexpected_from_contract_runtime_decision", &EventRun::default()); Ok(()) }
    fn on_unexpected_from_idle(&mut self) -> Result<(), ()> { self.action_named("on_unexpected_from_idle", &EventRun::default()); Ok(()) }
    fn on_unexpected_from_slots(&mut self) -> Result<(), ()> { self.action_named("on_unexpected_from_slots", &EventRun::default()); Ok(()) }
    fn on_unexpected_from_slots_decision(&mut self) -> Result<(), ()> { self.action_named("on_unexpected_from_slots_decision", &EventRun::default()); Ok(()) }
    fn on_unexpected_from_snapshot(&mut self) -> Result<(), ()> { self.action_named("on_unexpected_from_snapshot", &EventRun::default()); Ok(()) }
    fn on_unexpected_from_snapshot_decision(&mut self) -> Result<(), ()> { self.action_named("on_unexpected_from_snapshot_decision", &EventRun::default()); Ok(()) }
    fn request_contract_flash_materialized_chunk4_packed_q8_0(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("request_contract_flash_materialized_chunk4_packed_q8_0", event); Ok(()) }
    fn request_contract_flash_materialized_chunk4_q8_k(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("request_contract_flash_materialized_chunk4_q8_k", event); Ok(()) }
    fn request_contract_flash_materialized_chunk8_q8_k(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("request_contract_flash_materialized_chunk8_q8_k", event); Ok(()) }
    fn request_contract_flash_materialized_parallel_chunk4_packed_q8_0(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("request_contract_flash_materialized_parallel_chunk4_packed_q8_0", event); Ok(()) }
    fn request_contract_flash_materialized_parallel_chunk4_q8_k(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("request_contract_flash_materialized_parallel_chunk4_q8_k", event); Ok(()) }
    fn request_contract_flash_materialized_parallel_chunk8_q8_k(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("request_contract_flash_materialized_parallel_chunk8_q8_k", event); Ok(()) }
    fn request_contract_flash_materialized_scalar_kernel(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("request_contract_flash_materialized_scalar_kernel", event); Ok(()) }
    fn request_contract_flash_materialized_scalar_native_quantized(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("request_contract_flash_materialized_scalar_native_quantized", event); Ok(()) }
    fn request_contract_flash_materialized_scalar_native_quantized_q8_k(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("request_contract_flash_materialized_scalar_native_quantized_q8_k", event); Ok(()) }
    fn request_contract_flash_materialized_scalar_packed_q8_0(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("request_contract_flash_materialized_scalar_packed_q8_0", event); Ok(()) }
    fn request_contract_flash_materialized_scalar_q8_k(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("request_contract_flash_materialized_scalar_q8_k", event); Ok(()) }
    fn request_contract_flash_preselected_chunk4_packed_q8_0(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("request_contract_flash_preselected_chunk4_packed_q8_0", event); Ok(()) }
    fn request_contract_flash_preselected_chunk4_q8_k(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("request_contract_flash_preselected_chunk4_q8_k", event); Ok(()) }
    fn request_contract_flash_preselected_chunk8_q8_k(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("request_contract_flash_preselected_chunk8_q8_k", event); Ok(()) }
    fn request_contract_flash_preselected_parallel_chunk4_packed_q8_0(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("request_contract_flash_preselected_parallel_chunk4_packed_q8_0", event); Ok(()) }
    fn request_contract_flash_preselected_parallel_chunk4_q8_k(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("request_contract_flash_preselected_parallel_chunk4_q8_k", event); Ok(()) }
    fn request_contract_flash_preselected_parallel_chunk8_q8_k(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("request_contract_flash_preselected_parallel_chunk8_q8_k", event); Ok(()) }
    fn request_contract_flash_preselected_scalar_kernel(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("request_contract_flash_preselected_scalar_kernel", event); Ok(()) }
    fn request_contract_flash_preselected_scalar_native_quantized_kernel(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("request_contract_flash_preselected_scalar_native_quantized_kernel", event); Ok(()) }
    fn request_contract_flash_preselected_scalar_native_quantized_q8_k(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("request_contract_flash_preselected_scalar_native_quantized_q8_k", event); Ok(()) }
    fn request_contract_flash_preselected_scalar_q8_k(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("request_contract_flash_preselected_scalar_q8_k", event); Ok(()) }
    fn request_contract_nonflash_materialized_chunk4_packed_q8_0(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("request_contract_nonflash_materialized_chunk4_packed_q8_0", event); Ok(()) }
    fn request_contract_nonflash_materialized_chunk4_q8_k(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("request_contract_nonflash_materialized_chunk4_q8_k", event); Ok(()) }
    fn request_contract_nonflash_materialized_chunk8_q8_k(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("request_contract_nonflash_materialized_chunk8_q8_k", event); Ok(()) }
    fn request_contract_nonflash_materialized_scalar_kernel(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("request_contract_nonflash_materialized_scalar_kernel", event); Ok(()) }
    fn request_contract_nonflash_materialized_scalar_native_quantized(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("request_contract_nonflash_materialized_scalar_native_quantized", event); Ok(()) }
    fn request_contract_nonflash_materialized_scalar_native_quantized_q8_k(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("request_contract_nonflash_materialized_scalar_native_quantized_q8_k", event); Ok(()) }
    fn request_contract_nonflash_materialized_scalar_packed_q8_0(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("request_contract_nonflash_materialized_scalar_packed_q8_0", event); Ok(()) }
    fn request_contract_nonflash_materialized_scalar_q8_k(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("request_contract_nonflash_materialized_scalar_q8_k", event); Ok(()) }
    fn request_contract_nonflash_preselected_chunk4_packed_q8_0(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("request_contract_nonflash_preselected_chunk4_packed_q8_0", event); Ok(()) }
    fn request_contract_nonflash_preselected_chunk4_q8_k(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("request_contract_nonflash_preselected_chunk4_q8_k", event); Ok(()) }
    fn request_contract_nonflash_preselected_chunk8_q8_k(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("request_contract_nonflash_preselected_chunk8_q8_k", event); Ok(()) }
    fn request_contract_nonflash_preselected_scalar_kernel(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("request_contract_nonflash_preselected_scalar_kernel", event); Ok(()) }
    fn request_contract_nonflash_preselected_scalar_native_quantized_kernel(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("request_contract_nonflash_preselected_scalar_native_quantized_kernel", event); Ok(()) }
    fn request_contract_nonflash_preselected_scalar_native_quantized_q8_k(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("request_contract_nonflash_preselected_scalar_native_quantized_q8_k", event); Ok(()) }
    fn request_contract_nonflash_preselected_scalar_q8_k(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("request_contract_nonflash_preselected_scalar_q8_k", event); Ok(()) }
    fn request_memory_snapshot(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("request_memory_snapshot", event); Ok(()) }
    fn request_slots(&mut self, event: &EventRun) -> Result<(), ()> { self.action_named("request_slots", event); Ok(()) }
    fn slots_backend_error(&self, event: &EventRun) -> Result<bool, ()> { Ok(self.guard_named("slots_backend_error", event)) }
    fn slots_invalid_request(&self, event: &EventRun) -> Result<bool, ()> { Ok(self.guard_named("slots_invalid_request", event)) }
    fn slots_ok(&self, event: &EventRun) -> Result<bool, ()> { Ok(self.guard_named("slots_ok", event)) }
    fn snapshot_backend_error(&self, event: &EventRun) -> Result<bool, ()> { Ok(self.guard_named("snapshot_backend_error", event)) }
    fn snapshot_invalid_request(&self, event: &EventRun) -> Result<bool, ()> { Ok(self.guard_named("snapshot_invalid_request", event)) }
    fn snapshot_ok(&self, event: &EventRun) -> Result<bool, ()> { Ok(self.guard_named("snapshot_ok", event)) }
}

/// Public single-writer synchronous wrapper around the generated actor.
pub struct TextGeneratorPrefillActor { machine: TextGeneratorPrefillStateMachine<TextGeneratorPrefillContext> }
impl Default for TextGeneratorPrefillActor { fn default() -> Self { Self::new() } }
impl TextGeneratorPrefillActor {
    #[must_use] pub fn new() -> Self { Self { machine: TextGeneratorPrefillStateMachine::new(TextGeneratorPrefillContext::default()) } }
    #[must_use] pub fn state(&self) -> &TextGeneratorPrefillStates { self.machine.state() }
    #[must_use] pub fn is(&self, state: &TextGeneratorPrefillStates) -> bool { self.machine.is(state) }
    #[must_use] pub fn context(&self) -> &TextGeneratorPrefillContext { self.machine.context() }
    #[must_use] pub fn context_mut(&mut self) -> &mut TextGeneratorPrefillContext { self.machine.context_mut() }
    pub fn process_event(&mut self, event: EventRun) -> Result<(), PrefillError> {
        if self.machine.process_event(TextGeneratorPrefillEvents::EventRun(event)).is_err() { self.machine.context_mut().error = PrefillError::UnexpectedEvent; return Err(PrefillError::UnexpectedEvent); }
        match self.context().error { PrefillError::None => Ok(()), e => Err(e) }
    }
    pub fn run(&mut self, event: EventRun) -> Result<(), PrefillError> { self.process_event(event) }
    pub fn process_unexpected_event(&mut self) -> Result<(), PrefillError> {
        self.machine.context_mut().error = PrefillError::UnexpectedEvent;
        self.machine.set_state(TextGeneratorPrefillStates::Idle);
        Err(PrefillError::UnexpectedEvent)
    }
}
pub type TextGeneratorPrefill = TextGeneratorPrefillActor;
