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
    clippy::items_after_statements,
    clippy::field_reassign_with_default,
    clippy::semicolon_if_nothing_returned,
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
    pub const fn ok() -> Self {
        Self {
            accepted: true,
            code: 0,
        }
    }
    pub const fn invalid_request() -> Self {
        Self {
            accepted: false,
            code: 1,
        }
    }
    pub const fn backend() -> Self {
        Self {
            accepted: false,
            code: 2,
        }
    }
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
        match code {
            0 => Self::None,
            1 => Self::InvalidRequest,
            _ => Self::Backend,
        }
    }
}

/// Selection mode used by the generation contract.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SelectionMode {
    #[default]
    SampleLogits,
    PreselectedArgmax,
}

/// Every bounded callback route materialized by the C++ contract.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PrefillOperation {
    #[default]
    None,
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

/// Runtime event and its bounded caller-owned phase result.
///
/// The scalar capability fields are the Rust projection of the pinned
/// generator context/guard predicates. They are deliberately supplied by the
/// caller: this actor owns no model, graph, memory, or token storage.
#[derive(Clone, Debug)]
pub struct EventRun {
    pub prompt_token_count: u32,
    pub selection_mode: SelectionMode,
    pub backend_available: bool,
    pub backend_ready: bool,
    pub graph_reservation_ready: bool,
    pub prefill_lifecycle_ready: bool,
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
    pub materialized_output_ready: bool,
    pub prefill_plan_ready: bool,
    pub prefill_plan_expected_outputs: u32,
    pub plan_outputs: u32,
    pub prefill_step_size: u32,
    pub prefill_max_step_tokens: u32,
    pub prompt_capacity: u32,
    pub bound_token_capacity: u32,
    pub bound_position_capacity: u32,
    pub kv_positions_capacity: u32,
    pub backend_n_ctx: u32,
    pub kv_cache_tokens: u32,
    pub snapshot_block_tokens: u32,
    pub backend_kv_block_tokens: u32,
    pub snapshot_sequence_active: bool,
    pub snapshot_recurrent_slot: bool,
    pub snapshot_sequence_length: u32,
    pub snapshot_block_count: u32,
    pub snapshot_blocks_valid: bool,
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
    pub bound_prompt_token_count: Cell<u32>,
    pub bound_position_count: Cell<u32>,
    pub cache_position_count: Cell<u32>,
    pub phase_accepted: Cell<bool>,
    pub phase_code: Cell<i32>,
}
impl Default for EventRun {
    fn default() -> Self {
        Self {
            prompt_token_count: 0,
            selection_mode: SelectionMode::SampleLogits,
            backend_available: true,
            backend_ready: true,
            graph_reservation_ready: true,
            prefill_lifecycle_ready: true,
            flash_attention_supported: false,
            kv_map_identity: true,
            parallel_lanes_enabled: false,
            matmul_parallel: false,
            parallel_min_prefill_tokens: 8,
            prefill_chunk4_min_tokens: 4,
            prefill_chunk8_min_tokens: 8,
            direct_preselected_supported: false,
            compute_invalid_request: false,
            compute_backend_unavailable: false,
            materialized_output_ready: true,
            prefill_plan_ready: true,
            prefill_plan_expected_outputs: 1,
            plan_outputs: 1,
            prefill_step_size: 1,
            prefill_max_step_tokens: 1,
            prompt_capacity: u32::MAX,
            bound_token_capacity: u32::MAX,
            bound_position_capacity: u32::MAX,
            kv_positions_capacity: u32::MAX,
            backend_n_ctx: u32::MAX,
            kv_cache_tokens: 0,
            snapshot_block_tokens: 1,
            backend_kv_block_tokens: 1,
            snapshot_sequence_active: true,
            snapshot_recurrent_slot: true,
            snapshot_sequence_length: 0,
            snapshot_block_count: 1,
            snapshot_blocks_valid: true,
            materialized_chunk8_q8_k_supported: false,
            materialized_chunk4_packed_q8_0_supported: false,
            materialized_chunk4_q8_k_supported: false,
            materialized_scalar_packed_q8_0_supported: false,
            materialized_scalar_q8_k_supported: false,
            materialized_scalar_native_quantized_supported: false,
            materialized_scalar_native_quantized_q8_k_supported: false,
            materialized_scalar_kernel_supported: true,
            preselected_chunk8_q8_k_supported: false,
            preselected_chunk4_packed_q8_0_supported: false,
            preselected_chunk4_q8_k_supported: false,
            preselected_scalar_q8_k_supported: false,
            preselected_scalar_native_quantized_q8_k_supported: false,
            preselected_scalar_native_quantized_kernel_supported: false,
            preselected_scalar_kernel_supported: true,
            bound_prompt_token_count: Cell::new(0),
            bound_position_count: Cell::new(0),
            cache_position_count: Cell::new(0),
            phase_accepted: Cell::new(false),
            phase_code: Cell::new(0),
        }
    }
}
impl EventRun {
    #[must_use]
    pub fn new(prompt_token_count: u32) -> Self {
        Self {
            prompt_token_count,
            snapshot_sequence_length: prompt_token_count,
            ..Self::default()
        }
    }
    pub fn set_phase(&self, outcome: PhaseOutcome) {
        self.phase_accepted.set(outcome.accepted);
        self.phase_code.set(outcome.code);
    }
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

/// Bounded control-plane state shared by the generated machine and wrapper.
#[derive(Debug, Default)]
pub struct TextGeneratorPrefillContext {
    error: PrefillError,
    kv_tokens: u32,
    last_operation: PrefillOperation,
    dispatcher: Option<PrefillDispatch>,
}

impl TextGeneratorPrefillContext {
    fn phase_success(event: &EventRun) -> bool {
        event.phase_accepted.get() && event.phase_code.get() == 0
    }
    fn phase_invalid(event: &EventRun) -> bool {
        !Self::phase_success(event) && event.phase_code.get() == 1
    }
    fn phase_backend(event: &EventRun) -> bool {
        !Self::phase_success(event) && !Self::phase_invalid(event)
    }
    fn preselected(event: &EventRun) -> bool {
        event.selection_mode == SelectionMode::PreselectedArgmax
            && event.direct_preselected_supported
    }
    fn flash(event: &EventRun) -> bool {
        event.prompt_token_count > 0 && event.flash_attention_supported && event.kv_map_identity
    }
    fn parallel(event: &EventRun) -> bool {
        event.parallel_lanes_enabled
            && event.matmul_parallel
            && event.prompt_token_count >= event.parallel_min_prefill_tokens
    }
    fn chunk4(event: &EventRun) -> bool {
        event.prompt_token_count >= event.prefill_chunk4_min_tokens.max(4)
    }
    fn chunk8(event: &EventRun) -> bool {
        event.prompt_token_count >= event.prefill_chunk8_min_tokens.max(8)
    }
    fn materialized_chunk8(event: &EventRun) -> bool {
        Self::chunk8(event) && event.materialized_chunk8_q8_k_supported
    }
    fn materialized_chunk4(event: &EventRun) -> bool {
        Self::chunk4(event)
            && (event.materialized_chunk4_packed_q8_0_supported
                || event.materialized_chunk4_q8_k_supported)
    }
    fn preselected_chunk8(event: &EventRun) -> bool {
        Self::chunk8(event) && event.preselected_chunk8_q8_k_supported
    }
    fn preselected_chunk4(event: &EventRun) -> bool {
        Self::chunk4(event)
            && (event.preselected_chunk4_packed_q8_0_supported
                || event.preselected_chunk4_q8_k_supported)
    }
    fn backend_ready(event: &EventRun) -> bool {
        event.backend_available
            && event.backend_ready
            && event.graph_reservation_ready
            && event.prefill_lifecycle_ready
    }
    fn request_capacity_ready(event: &EventRun) -> bool {
        event.prompt_token_count > 0
            && event.prompt_token_count <= event.prompt_capacity
            && event.prompt_token_count <= event.bound_token_capacity
            && event.prompt_token_count <= event.bound_position_capacity
    }
    fn snapshot_geometry_ready(event: &EventRun) -> bool {
        event.snapshot_block_tokens > 0
            && event.snapshot_block_tokens == event.backend_kv_block_tokens
            && event.kv_positions_capacity >= event.backend_n_ctx
            && event.snapshot_sequence_active
            && event.snapshot_recurrent_slot
    }
    fn snapshot_covers_request(event: &EventRun) -> bool {
        Self::snapshot_geometry_ready(event)
            && event.snapshot_sequence_length == event.prompt_token_count
            && event.snapshot_block_count > 0
            && event.snapshot_blocks_valid
            && event.prompt_token_count <= event.kv_positions_capacity
            && event.prompt_token_count <= event.backend_n_ctx
    }
    fn prefill_request_ready(event: &EventRun) -> bool {
        event.prefill_plan_ready
            && event.prefill_plan_expected_outputs == event.plan_outputs
            && event.prefill_step_size > 0
            && event.prefill_step_size <= event.prefill_max_step_tokens
            && Self::request_capacity_ready(event)
            && Self::snapshot_covers_request(event)
    }
    fn materialized_request_ready(event: &EventRun) -> bool {
        Self::backend_ready(event)
            && event.materialized_output_ready
            && Self::prefill_request_ready(event)
    }
    fn preselected_request_ready(event: &EventRun) -> bool {
        Self::backend_ready(event) && Self::prefill_request_ready(event)
    }
    fn compute_route_ready(event: &EventRun, supported: bool) -> bool {
        supported
            && if Self::preselected(event) {
                Self::preselected_request_ready(event)
            } else {
                Self::materialized_request_ready(event)
            }
            && !event.compute_invalid_request
            && !event.compute_backend_unavailable
    }
    fn compute_invalid(event: &EventRun) -> bool {
        event.compute_invalid_request
            || (Self::backend_ready(event)
                && !event.compute_backend_unavailable
                && if Self::preselected(event) {
                    !Self::preselected_request_ready(event)
                } else {
                    !Self::materialized_request_ready(event)
                })
    }
    fn compute_backend_unavailable(event: &EventRun) -> bool {
        event.compute_backend_unavailable || !Self::backend_ready(event)
    }
    #[must_use]
    pub const fn error(&self) -> PrefillError {
        self.error
    }
    #[must_use]
    pub const fn kv_tokens(&self) -> u32 {
        self.kv_tokens
    }
    #[must_use]
    pub const fn last_operation(&self) -> PrefillOperation {
        self.last_operation
    }
    pub fn set_dispatcher(&mut self, dispatcher: Option<PrefillDispatch>) {
        self.dispatcher = dispatcher;
    }
    fn dispatch(&mut self, operation: PrefillOperation, event: &EventRun) {
        let outcome = self
            .dispatcher
            .map_or(PhaseOutcome::backend(), |f| f(operation, event));
        event.set_phase(outcome);
        if Self::phase_success(event) {
            match operation {
                PrefillOperation::Slots => {
                    event.bound_prompt_token_count.set(event.prompt_token_count)
                }
                PrefillOperation::MemorySnapshot => {
                    event.cache_position_count.set(event.prompt_token_count)
                }
                PrefillOperation::None => {}
                _ => event.bound_position_count.set(event.prompt_token_count),
            }
        }
        self.last_operation = operation;
    }
    fn mark_invalid_request(&mut self) {
        self.error = PrefillError::InvalidRequest;
    }
    fn mark_backend_error(&mut self) {
        self.error = PrefillError::Backend;
    }
    fn mark_prefill_cached(&mut self, event: &EventRun) {
        self.error = PrefillError::None;
        self.kv_tokens = event.prompt_token_count;
        event.cache_position_count.set(event.prompt_token_count);
    }
    fn mark_unexpected(&mut self) {
        self.error = PrefillError::UnexpectedEvent;
    }
}

impl TextGeneratorPrefillStateMachineContext for TextGeneratorPrefillContext {
    fn compute_backend_error(&self, event: &EventRun) -> Result<bool, ()> {
        Ok(Self::phase_backend(event))
    }
    fn compute_invalid_request(&self, event: &EventRun) -> Result<bool, ()> {
        Ok(!Self::phase_success(event) && Self::phase_invalid(event))
    }
    fn compute_ok(&self, event: &EventRun) -> Result<bool, ()> {
        Ok(Self::phase_success(event))
    }
    fn flash_runtime_supported(&self, event: &EventRun) -> Result<bool, ()> {
        Ok(Self::flash(event))
    }
    fn nonflash_runtime_required(&self, event: &EventRun) -> Result<bool, ()> {
        Ok(!Self::flash(event))
    }
    fn guard_compute_backend_unavailable(&self, event: &EventRun) -> Result<bool, ()> {
        Ok(Self::compute_backend_unavailable(event))
    }
    fn guard_compute_invalid_request(&self, event: &EventRun) -> Result<bool, ()> {
        Ok(Self::compute_invalid(event))
    }
    fn guard_materialized_logits_with_parallel_chunk8_q8_k_ready(
        &self,
        event: &EventRun,
    ) -> Result<bool, ()> {
        Ok(!Self::preselected(event)
            && Self::parallel(event)
            && Self::materialized_chunk8(event)
            && Self::compute_route_ready(event, event.materialized_chunk8_q8_k_supported))
    }
    fn guard_materialized_logits_with_chunk8_q8_k_ready(
        &self,
        event: &EventRun,
    ) -> Result<bool, ()> {
        Ok(!Self::preselected(event)
            && Self::materialized_chunk8(event)
            && Self::compute_route_ready(event, event.materialized_chunk8_q8_k_supported))
    }
    fn guard_materialized_logits_with_parallel_chunk4_packed_q8_0_ready(
        &self,
        event: &EventRun,
    ) -> Result<bool, ()> {
        Ok(!Self::preselected(event)
            && !Self::materialized_chunk8(event)
            && Self::parallel(event)
            && Self::materialized_chunk4(event)
            && event.materialized_chunk4_packed_q8_0_supported
            && Self::compute_route_ready(event, event.materialized_chunk4_packed_q8_0_supported))
    }
    fn guard_materialized_logits_with_chunk4_packed_q8_0_ready(
        &self,
        event: &EventRun,
    ) -> Result<bool, ()> {
        Ok(!Self::preselected(event)
            && !Self::materialized_chunk8(event)
            && Self::materialized_chunk4(event)
            && event.materialized_chunk4_packed_q8_0_supported
            && Self::compute_route_ready(event, event.materialized_chunk4_packed_q8_0_supported))
    }
    fn guard_materialized_logits_with_parallel_chunk4_q8_k_ready(
        &self,
        event: &EventRun,
    ) -> Result<bool, ()> {
        Ok(!Self::preselected(event)
            && !Self::materialized_chunk8(event)
            && Self::parallel(event)
            && Self::materialized_chunk4(event)
            && !event.materialized_chunk4_packed_q8_0_supported
            && event.materialized_chunk4_q8_k_supported
            && Self::compute_route_ready(event, event.materialized_chunk4_q8_k_supported))
    }
    fn guard_materialized_logits_with_chunk4_q8_k_ready(
        &self,
        event: &EventRun,
    ) -> Result<bool, ()> {
        Ok(!Self::preselected(event)
            && !Self::materialized_chunk8(event)
            && Self::materialized_chunk4(event)
            && !event.materialized_chunk4_packed_q8_0_supported
            && event.materialized_chunk4_q8_k_supported
            && Self::compute_route_ready(event, event.materialized_chunk4_q8_k_supported))
    }
    fn guard_materialized_logits_with_scalar_packed_q8_0_ready(
        &self,
        event: &EventRun,
    ) -> Result<bool, ()> {
        Ok(!Self::preselected(event)
            && !Self::materialized_chunk8(event)
            && !Self::materialized_chunk4(event)
            && Self::compute_route_ready(event, event.materialized_scalar_packed_q8_0_supported))
    }
    fn guard_materialized_logits_with_scalar_q8_k_ready(
        &self,
        event: &EventRun,
    ) -> Result<bool, ()> {
        Ok(!Self::preselected(event)
            && !Self::materialized_chunk8(event)
            && !Self::materialized_chunk4(event)
            && !event.materialized_scalar_packed_q8_0_supported
            && Self::compute_route_ready(event, event.materialized_scalar_q8_k_supported))
    }
    fn guard_materialized_logits_with_scalar_kernel_ready(
        &self,
        event: &EventRun,
    ) -> Result<bool, ()> {
        Ok(!Self::preselected(event)
            && !Self::materialized_chunk8(event)
            && !Self::materialized_chunk4(event)
            && !event.materialized_scalar_packed_q8_0_supported
            && !event.materialized_scalar_q8_k_supported
            && !event.materialized_scalar_native_quantized_q8_k_supported
            && !event.materialized_scalar_native_quantized_supported
            && Self::compute_route_ready(event, event.materialized_scalar_kernel_supported))
    }
    fn guard_materialized_logits_with_scalar_native_quantized_q8_k_ready(
        &self,
        event: &EventRun,
    ) -> Result<bool, ()> {
        Ok(!Self::preselected(event)
            && !Self::materialized_chunk8(event)
            && !Self::materialized_chunk4(event)
            && !event.materialized_scalar_packed_q8_0_supported
            && !event.materialized_scalar_q8_k_supported
            && event.materialized_scalar_native_quantized_supported
            && Self::compute_route_ready(
                event,
                event.materialized_scalar_native_quantized_q8_k_supported,
            ))
    }
    fn guard_materialized_logits_with_scalar_native_quantized_kernel_ready(
        &self,
        event: &EventRun,
    ) -> Result<bool, ()> {
        Ok(!Self::preselected(event)
            && !Self::materialized_chunk8(event)
            && !Self::materialized_chunk4(event)
            && !event.materialized_scalar_packed_q8_0_supported
            && !event.materialized_scalar_q8_k_supported
            && event.materialized_scalar_native_quantized_supported
            && !event.materialized_scalar_native_quantized_q8_k_supported
            && Self::compute_route_ready(
                event,
                event.materialized_scalar_native_quantized_supported,
            ))
    }
    fn guard_preselected_argmax_with_parallel_chunk8_q8_k_ready(
        &self,
        event: &EventRun,
    ) -> Result<bool, ()> {
        Ok(Self::preselected(event)
            && Self::parallel(event)
            && Self::chunk8(event)
            && Self::compute_route_ready(event, event.preselected_chunk8_q8_k_supported))
    }
    fn guard_preselected_argmax_with_chunk8_q8_k_ready(
        &self,
        event: &EventRun,
    ) -> Result<bool, ()> {
        Ok(Self::preselected(event)
            && Self::chunk8(event)
            && Self::compute_route_ready(event, event.preselected_chunk8_q8_k_supported))
    }
    fn guard_preselected_argmax_with_parallel_chunk4_packed_q8_0_ready(
        &self,
        event: &EventRun,
    ) -> Result<bool, ()> {
        Ok(Self::preselected(event)
            && !Self::chunk8(event)
            && Self::parallel(event)
            && Self::chunk4(event)
            && Self::compute_route_ready(event, event.preselected_chunk4_packed_q8_0_supported))
    }
    fn guard_preselected_argmax_with_chunk4_packed_q8_0_ready(
        &self,
        event: &EventRun,
    ) -> Result<bool, ()> {
        Ok(Self::preselected(event)
            && !Self::chunk8(event)
            && Self::chunk4(event)
            && Self::compute_route_ready(event, event.preselected_chunk4_packed_q8_0_supported))
    }
    fn guard_preselected_argmax_with_parallel_chunk4_q8_k_ready(
        &self,
        event: &EventRun,
    ) -> Result<bool, ()> {
        Ok(Self::preselected(event)
            && !Self::chunk8(event)
            && Self::parallel(event)
            && Self::chunk4(event)
            && Self::compute_route_ready(event, event.preselected_chunk4_q8_k_supported))
    }
    fn guard_preselected_argmax_with_chunk4_q8_k_ready(
        &self,
        event: &EventRun,
    ) -> Result<bool, ()> {
        Ok(Self::preselected(event)
            && !Self::chunk8(event)
            && Self::chunk4(event)
            && Self::compute_route_ready(event, event.preselected_chunk4_q8_k_supported))
    }
    fn guard_preselected_argmax_with_scalar_q8_k_ready(
        &self,
        event: &EventRun,
    ) -> Result<bool, ()> {
        Ok(Self::preselected(event)
            && !Self::chunk8(event)
            && !Self::chunk4(event)
            && Self::compute_route_ready(event, event.preselected_scalar_q8_k_supported))
    }
    fn guard_preselected_argmax_with_scalar_native_quantized_q8_k_ready(
        &self,
        event: &EventRun,
    ) -> Result<bool, ()> {
        Ok(Self::preselected(event)
            && !Self::chunk8(event)
            && !Self::chunk4(event)
            && !event.preselected_scalar_q8_k_supported
            && Self::compute_route_ready(
                event,
                event.preselected_scalar_native_quantized_q8_k_supported,
            ))
    }
    fn guard_preselected_argmax_with_scalar_native_quantized_kernel_ready(
        &self,
        event: &EventRun,
    ) -> Result<bool, ()> {
        Ok(Self::preselected(event)
            && !Self::chunk8(event)
            && !Self::chunk4(event)
            && !event.preselected_scalar_q8_k_supported
            && !event.preselected_scalar_native_quantized_q8_k_supported
            && Self::compute_route_ready(
                event,
                event.preselected_scalar_native_quantized_kernel_supported,
            ))
    }
    fn guard_preselected_argmax_with_scalar_kernel_ready(
        &self,
        event: &EventRun,
    ) -> Result<bool, ()> {
        Ok(Self::preselected(event)
            && !Self::chunk8(event)
            && !Self::chunk4(event)
            && !event.preselected_scalar_q8_k_supported
            && !event.preselected_scalar_native_quantized_q8_k_supported
            && !event.preselected_scalar_native_quantized_kernel_supported
            && Self::compute_route_ready(event, event.preselected_scalar_kernel_supported))
    }

    fn mark_backend_error_from_compute_result_decision(&mut self, _: &EventRun) -> Result<(), ()> {
        self.mark_backend_error();
        Ok(())
    }
    fn mark_backend_error_from_contract_flash_decision(&mut self, _: &EventRun) -> Result<(), ()> {
        self.mark_backend_error();
        Ok(())
    }
    fn mark_backend_error_from_contract_nonflash_decision(
        &mut self,
        _: &EventRun,
    ) -> Result<(), ()> {
        self.mark_backend_error();
        Ok(())
    }
    fn mark_backend_error_from_slots_decision(&mut self, _: &EventRun) -> Result<(), ()> {
        self.mark_backend_error();
        Ok(())
    }
    fn mark_backend_error_from_snapshot_decision(&mut self, _: &EventRun) -> Result<(), ()> {
        self.mark_backend_error();
        Ok(())
    }
    fn mark_invalid_request_from_compute_result_decision(
        &mut self,
        _: &EventRun,
    ) -> Result<(), ()> {
        self.mark_invalid_request();
        Ok(())
    }
    fn mark_invalid_request_from_contract_flash_decision(
        &mut self,
        _: &EventRun,
    ) -> Result<(), ()> {
        self.mark_invalid_request();
        Ok(())
    }
    fn mark_invalid_request_from_contract_nonflash_decision(
        &mut self,
        _: &EventRun,
    ) -> Result<(), ()> {
        self.mark_invalid_request();
        Ok(())
    }
    fn mark_invalid_request_from_slots_decision(&mut self, _: &EventRun) -> Result<(), ()> {
        self.mark_invalid_request();
        Ok(())
    }
    fn mark_invalid_request_from_snapshot_decision(&mut self, _: &EventRun) -> Result<(), ()> {
        self.mark_invalid_request();
        Ok(())
    }
    fn mark_prefill_cached(&mut self, event: &EventRun) -> Result<(), ()> {
        Self::mark_prefill_cached(self, event);
        Ok(())
    }
    fn on_unexpected_from_compute_result_decision(&mut self) -> Result<(), ()> {
        self.mark_unexpected();
        Ok(())
    }
    fn on_unexpected_from_contract_flash_decision(&mut self) -> Result<(), ()> {
        self.mark_unexpected();
        Ok(())
    }
    fn on_unexpected_from_contract_nonflash_decision(&mut self) -> Result<(), ()> {
        self.mark_unexpected();
        Ok(())
    }
    fn on_unexpected_from_contract_runtime_decision(&mut self) -> Result<(), ()> {
        self.mark_unexpected();
        Ok(())
    }
    fn on_unexpected_from_idle(&mut self) -> Result<(), ()> {
        self.mark_unexpected();
        Ok(())
    }
    fn on_unexpected_from_slots(&mut self) -> Result<(), ()> {
        self.mark_unexpected();
        Ok(())
    }
    fn on_unexpected_from_slots_decision(&mut self) -> Result<(), ()> {
        self.mark_unexpected();
        Ok(())
    }
    fn on_unexpected_from_snapshot(&mut self) -> Result<(), ()> {
        self.mark_unexpected();
        Ok(())
    }
    fn on_unexpected_from_snapshot_decision(&mut self) -> Result<(), ()> {
        self.mark_unexpected();
        Ok(())
    }
    fn request_contract_flash_materialized_chunk4_packed_q8_0(
        &mut self,
        event: &EventRun,
    ) -> Result<(), ()> {
        self.dispatch(PrefillOperation::FlashMaterializedChunk4PackedQ8_0, event);
        Ok(())
    }
    fn request_contract_flash_materialized_chunk4_q8_k(
        &mut self,
        event: &EventRun,
    ) -> Result<(), ()> {
        self.dispatch(PrefillOperation::FlashMaterializedChunk4Q8K, event);
        Ok(())
    }
    fn request_contract_flash_materialized_chunk8_q8_k(
        &mut self,
        event: &EventRun,
    ) -> Result<(), ()> {
        self.dispatch(PrefillOperation::FlashMaterializedChunk8Q8K, event);
        Ok(())
    }
    fn request_contract_flash_materialized_parallel_chunk4_packed_q8_0(
        &mut self,
        event: &EventRun,
    ) -> Result<(), ()> {
        self.dispatch(
            PrefillOperation::FlashMaterializedParallelChunk4PackedQ8_0,
            event,
        );
        Ok(())
    }
    fn request_contract_flash_materialized_parallel_chunk4_q8_k(
        &mut self,
        event: &EventRun,
    ) -> Result<(), ()> {
        self.dispatch(PrefillOperation::FlashMaterializedParallelChunk4Q8K, event);
        Ok(())
    }
    fn request_contract_flash_materialized_parallel_chunk8_q8_k(
        &mut self,
        event: &EventRun,
    ) -> Result<(), ()> {
        self.dispatch(PrefillOperation::FlashMaterializedParallelChunk8Q8K, event);
        Ok(())
    }
    fn request_contract_flash_materialized_scalar_kernel(
        &mut self,
        event: &EventRun,
    ) -> Result<(), ()> {
        self.dispatch(PrefillOperation::FlashMaterializedScalarKernel, event);
        Ok(())
    }
    fn request_contract_flash_materialized_scalar_native_quantized(
        &mut self,
        event: &EventRun,
    ) -> Result<(), ()> {
        self.dispatch(
            PrefillOperation::FlashMaterializedScalarNativeQuantized,
            event,
        );
        Ok(())
    }
    fn request_contract_flash_materialized_scalar_native_quantized_q8_k(
        &mut self,
        event: &EventRun,
    ) -> Result<(), ()> {
        self.dispatch(
            PrefillOperation::FlashMaterializedScalarNativeQuantizedQ8K,
            event,
        );
        Ok(())
    }
    fn request_contract_flash_materialized_scalar_packed_q8_0(
        &mut self,
        event: &EventRun,
    ) -> Result<(), ()> {
        self.dispatch(PrefillOperation::FlashMaterializedScalarPackedQ8_0, event);
        Ok(())
    }
    fn request_contract_flash_materialized_scalar_q8_k(
        &mut self,
        event: &EventRun,
    ) -> Result<(), ()> {
        self.dispatch(PrefillOperation::FlashMaterializedScalarQ8K, event);
        Ok(())
    }
    fn request_contract_flash_preselected_chunk4_packed_q8_0(
        &mut self,
        event: &EventRun,
    ) -> Result<(), ()> {
        self.dispatch(PrefillOperation::FlashPreselectedChunk4PackedQ8_0, event);
        Ok(())
    }
    fn request_contract_flash_preselected_chunk4_q8_k(
        &mut self,
        event: &EventRun,
    ) -> Result<(), ()> {
        self.dispatch(PrefillOperation::FlashPreselectedChunk4Q8K, event);
        Ok(())
    }
    fn request_contract_flash_preselected_chunk8_q8_k(
        &mut self,
        event: &EventRun,
    ) -> Result<(), ()> {
        self.dispatch(PrefillOperation::FlashPreselectedChunk8Q8K, event);
        Ok(())
    }
    fn request_contract_flash_preselected_parallel_chunk4_packed_q8_0(
        &mut self,
        event: &EventRun,
    ) -> Result<(), ()> {
        self.dispatch(
            PrefillOperation::FlashPreselectedParallelChunk4PackedQ8_0,
            event,
        );
        Ok(())
    }
    fn request_contract_flash_preselected_parallel_chunk4_q8_k(
        &mut self,
        event: &EventRun,
    ) -> Result<(), ()> {
        self.dispatch(PrefillOperation::FlashPreselectedParallelChunk4Q8K, event);
        Ok(())
    }
    fn request_contract_flash_preselected_parallel_chunk8_q8_k(
        &mut self,
        event: &EventRun,
    ) -> Result<(), ()> {
        self.dispatch(PrefillOperation::FlashPreselectedParallelChunk8Q8K, event);
        Ok(())
    }
    fn request_contract_flash_preselected_scalar_kernel(
        &mut self,
        event: &EventRun,
    ) -> Result<(), ()> {
        self.dispatch(PrefillOperation::FlashPreselectedScalarKernel, event);
        Ok(())
    }
    fn request_contract_flash_preselected_scalar_native_quantized_kernel(
        &mut self,
        event: &EventRun,
    ) -> Result<(), ()> {
        self.dispatch(
            PrefillOperation::FlashPreselectedScalarNativeQuantizedKernel,
            event,
        );
        Ok(())
    }
    fn request_contract_flash_preselected_scalar_native_quantized_q8_k(
        &mut self,
        event: &EventRun,
    ) -> Result<(), ()> {
        self.dispatch(
            PrefillOperation::FlashPreselectedScalarNativeQuantizedQ8K,
            event,
        );
        Ok(())
    }
    fn request_contract_flash_preselected_scalar_q8_k(
        &mut self,
        event: &EventRun,
    ) -> Result<(), ()> {
        self.dispatch(PrefillOperation::FlashPreselectedScalarQ8K, event);
        Ok(())
    }
    fn request_contract_nonflash_materialized_chunk4_packed_q8_0(
        &mut self,
        event: &EventRun,
    ) -> Result<(), ()> {
        self.dispatch(
            PrefillOperation::NonflashMaterializedChunk4PackedQ8_0,
            event,
        );
        Ok(())
    }
    fn request_contract_nonflash_materialized_chunk4_q8_k(
        &mut self,
        event: &EventRun,
    ) -> Result<(), ()> {
        self.dispatch(PrefillOperation::NonflashMaterializedChunk4Q8K, event);
        Ok(())
    }
    fn request_contract_nonflash_materialized_chunk8_q8_k(
        &mut self,
        event: &EventRun,
    ) -> Result<(), ()> {
        self.dispatch(PrefillOperation::NonflashMaterializedChunk8Q8K, event);
        Ok(())
    }
    fn request_contract_nonflash_materialized_scalar_kernel(
        &mut self,
        event: &EventRun,
    ) -> Result<(), ()> {
        self.dispatch(PrefillOperation::NonflashMaterializedScalarKernel, event);
        Ok(())
    }
    fn request_contract_nonflash_materialized_scalar_native_quantized(
        &mut self,
        event: &EventRun,
    ) -> Result<(), ()> {
        self.dispatch(
            PrefillOperation::NonflashMaterializedScalarNativeQuantized,
            event,
        );
        Ok(())
    }
    fn request_contract_nonflash_materialized_scalar_native_quantized_q8_k(
        &mut self,
        event: &EventRun,
    ) -> Result<(), ()> {
        self.dispatch(
            PrefillOperation::NonflashMaterializedScalarNativeQuantizedQ8K,
            event,
        );
        Ok(())
    }
    fn request_contract_nonflash_materialized_scalar_packed_q8_0(
        &mut self,
        event: &EventRun,
    ) -> Result<(), ()> {
        self.dispatch(
            PrefillOperation::NonflashMaterializedScalarPackedQ8_0,
            event,
        );
        Ok(())
    }
    fn request_contract_nonflash_materialized_scalar_q8_k(
        &mut self,
        event: &EventRun,
    ) -> Result<(), ()> {
        self.dispatch(PrefillOperation::NonflashMaterializedScalarQ8K, event);
        Ok(())
    }
    fn request_contract_nonflash_preselected_chunk4_packed_q8_0(
        &mut self,
        event: &EventRun,
    ) -> Result<(), ()> {
        self.dispatch(PrefillOperation::NonflashPreselectedChunk4PackedQ8_0, event);
        Ok(())
    }
    fn request_contract_nonflash_preselected_chunk4_q8_k(
        &mut self,
        event: &EventRun,
    ) -> Result<(), ()> {
        self.dispatch(PrefillOperation::NonflashPreselectedChunk4Q8K, event);
        Ok(())
    }
    fn request_contract_nonflash_preselected_chunk8_q8_k(
        &mut self,
        event: &EventRun,
    ) -> Result<(), ()> {
        self.dispatch(PrefillOperation::NonflashPreselectedChunk8Q8K, event);
        Ok(())
    }
    fn request_contract_nonflash_preselected_scalar_kernel(
        &mut self,
        event: &EventRun,
    ) -> Result<(), ()> {
        self.dispatch(PrefillOperation::NonflashPreselectedScalarKernel, event);
        Ok(())
    }
    fn request_contract_nonflash_preselected_scalar_native_quantized_kernel(
        &mut self,
        event: &EventRun,
    ) -> Result<(), ()> {
        self.dispatch(
            PrefillOperation::NonflashPreselectedScalarNativeQuantizedKernel,
            event,
        );
        Ok(())
    }
    fn request_contract_nonflash_preselected_scalar_native_quantized_q8_k(
        &mut self,
        event: &EventRun,
    ) -> Result<(), ()> {
        self.dispatch(
            PrefillOperation::NonflashPreselectedScalarNativeQuantizedQ8K,
            event,
        );
        Ok(())
    }
    fn request_contract_nonflash_preselected_scalar_q8_k(
        &mut self,
        event: &EventRun,
    ) -> Result<(), ()> {
        self.dispatch(PrefillOperation::NonflashPreselectedScalarQ8K, event);
        Ok(())
    }
    fn request_memory_snapshot(&mut self, event: &EventRun) -> Result<(), ()> {
        self.dispatch(PrefillOperation::MemorySnapshot, event);
        Ok(())
    }
    fn request_slots(&mut self, event: &EventRun) -> Result<(), ()> {
        self.dispatch(PrefillOperation::Slots, event);
        Ok(())
    }
    fn slots_backend_error(&self, event: &EventRun) -> Result<bool, ()> {
        Ok(Self::phase_backend(event))
    }
    fn slots_invalid_request(&self, event: &EventRun) -> Result<bool, ()> {
        Ok(Self::phase_invalid(event))
    }
    fn slots_ok(&self, event: &EventRun) -> Result<bool, ()> {
        Ok(Self::phase_success(event))
    }
    fn snapshot_backend_error(&self, event: &EventRun) -> Result<bool, ()> {
        Ok(Self::phase_backend(event))
    }
    fn snapshot_invalid_request(&self, event: &EventRun) -> Result<bool, ()> {
        Ok(Self::phase_invalid(event))
    }
    fn snapshot_ok(&self, event: &EventRun) -> Result<bool, ()> {
        Ok(Self::phase_success(event))
    }
}

/// Public single-writer synchronous wrapper around the generated actor.
pub struct TextGeneratorPrefillActor {
    machine: TextGeneratorPrefillStateMachine<TextGeneratorPrefillContext>,
}
impl Default for TextGeneratorPrefillActor {
    fn default() -> Self {
        Self::new()
    }
}
impl TextGeneratorPrefillActor {
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: TextGeneratorPrefillStateMachine::new(TextGeneratorPrefillContext::default()),
        }
    }
    #[must_use]
    pub fn state(&self) -> &TextGeneratorPrefillStates {
        self.machine.state()
    }
    #[must_use]
    pub fn is(&self, state: &TextGeneratorPrefillStates) -> bool {
        self.machine.is(state)
    }
    #[must_use]
    pub fn context(&self) -> &TextGeneratorPrefillContext {
        self.machine.context()
    }
    pub fn set_dispatcher(&mut self, dispatcher: Option<PrefillDispatch>) {
        self.machine.context_mut().set_dispatcher(dispatcher);
    }
    pub fn process_event(&mut self, event: EventRun) -> Result<(), PrefillError> {
        if self
            .machine
            .process_event(TextGeneratorPrefillEvents::EventRun(event))
            .is_err()
        {
            self.machine.context_mut().error = PrefillError::UnexpectedEvent;
            return Err(PrefillError::UnexpectedEvent);
        }
        match self.context().error {
            PrefillError::None => Ok(()),
            e => Err(e),
        }
    }
    pub fn run(&mut self, event: EventRun) -> Result<(), PrefillError> {
        self.process_event(event)
    }
    pub fn process_unexpected_event(&mut self) -> Result<(), PrefillError> {
        self.machine.context_mut().error = PrefillError::UnexpectedEvent;
        self.machine.set_state(TextGeneratorPrefillStates::Idle);
        Err(PrefillError::UnexpectedEvent)
    }
}
pub type TextGeneratorPrefill = TextGeneratorPrefillActor;

#[cfg(test)]
mod tests {
    use super::{
        EventRun, PhaseOutcome, PrefillError, PrefillOperation, SelectionMode,
        TextGeneratorPrefillActor, TextGeneratorPrefillStates,
    };

    fn accept(_: PrefillOperation, _: &EventRun) -> PhaseOutcome {
        PhaseOutcome::ok()
    }
    fn reject_invalid(_: PrefillOperation, _: &EventRun) -> PhaseOutcome {
        PhaseOutcome::invalid_request()
    }

    fn run(mut event: EventRun) -> (TextGeneratorPrefillActor, Result<(), PrefillError>) {
        event.backend_available = true;
        let mut actor = TextGeneratorPrefillActor::new();
        actor.set_dispatcher(Some(accept));
        let result = actor.run(event);
        (actor, result)
    }

    #[test]
    fn scalar_materialized_nonflash_success_records_route_and_cache() {
        let (actor, result) = run(EventRun::new(1));
        assert_eq!(result, Ok(()));
        assert!(actor.is(&TextGeneratorPrefillStates::Idle));
        assert_eq!(
            actor.context().last_operation(),
            PrefillOperation::NonflashMaterializedScalarKernel
        );
        assert_eq!(actor.context().kv_tokens(), 1);
    }

    #[test]
    fn flash_chunk8_parallel_route_is_selected_before_scalar_routes() {
        let mut event = EventRun::new(8);
        event.flash_attention_supported = true;
        event.parallel_lanes_enabled = true;
        event.matmul_parallel = true;
        event.materialized_chunk8_q8_k_supported = true;
        let (actor, result) = run(event);
        assert_eq!(result, Ok(()));
        assert_eq!(
            actor.context().last_operation(),
            PrefillOperation::FlashMaterializedParallelChunk8Q8K
        );
    }

    #[test]
    fn preselected_argmax_uses_direct_chunk4_route() {
        let mut event = EventRun::new(4);
        event.selection_mode = SelectionMode::PreselectedArgmax;
        event.direct_preselected_supported = true;
        event.preselected_chunk4_q8_k_supported = true;
        let (actor, result) = run(event);
        assert_eq!(result, Ok(()));
        assert_eq!(
            actor.context().last_operation(),
            PrefillOperation::NonflashPreselectedChunk4Q8K
        );
    }

    #[test]
    fn slots_invalid_request_is_reported_without_compute() {
        let mut event = EventRun::new(0);
        event.prompt_capacity = 0;
        let mut actor = TextGeneratorPrefillActor::new();
        actor.set_dispatcher(Some(reject_invalid));
        assert_eq!(actor.run(event), Err(PrefillError::InvalidRequest));
        assert_eq!(actor.context().last_operation(), PrefillOperation::Slots);
        assert_eq!(actor.context().kv_tokens(), 0);
    }

    #[test]
    fn snapshot_backend_error_is_reported_without_compute() {
        let mut event = EventRun::new(1);
        event.backend_available = true;
        let mut actor = TextGeneratorPrefillActor::new();
        actor.set_dispatcher(Some(|operation, _| {
            if operation == PrefillOperation::Slots {
                PhaseOutcome::ok()
            } else {
                PhaseOutcome::backend()
            }
        }));
        assert_eq!(actor.run(event), Err(PrefillError::Backend));
        assert_eq!(
            actor.context().last_operation(),
            PrefillOperation::MemorySnapshot
        );
    }

    #[test]
    fn compute_backend_error_is_reported_after_successful_setup() {
        let mut event = EventRun::new(1);
        event.materialized_scalar_kernel_supported = true;
        let mut actor = TextGeneratorPrefillActor::new();
        actor.set_dispatcher(Some(|operation, _| {
            if matches!(
                operation,
                PrefillOperation::Slots | PrefillOperation::MemorySnapshot
            ) {
                PhaseOutcome::ok()
            } else {
                PhaseOutcome::backend()
            }
        }));
        assert_eq!(actor.run(event), Err(PrefillError::Backend));
        assert_eq!(
            actor.context().last_operation(),
            PrefillOperation::NonflashMaterializedScalarKernel
        );
    }

    #[test]
    fn empty_prompt_reaches_explicit_invalid_compute_path() {
        let event = EventRun::new(0);
        let mut actor = TextGeneratorPrefillActor::new();
        actor.set_dispatcher(Some(accept));
        assert_eq!(actor.run(event), Err(PrefillError::InvalidRequest));
        assert_eq!(
            actor.context().last_operation(),
            PrefillOperation::MemorySnapshot
        );
    }

    #[test]
    fn explicit_unexpected_event_enters_idle_error_state() {
        let mut actor = TextGeneratorPrefillActor::new();
        assert_eq!(
            actor.process_unexpected_event(),
            Err(PrefillError::UnexpectedEvent)
        );
        assert!(actor.is(&TextGeneratorPrefillStates::Idle));
    }
}
