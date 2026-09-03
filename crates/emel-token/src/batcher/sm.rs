//! Full token batch preparation contract.

#![allow(
    clippy::all,
    clippy::derive_partial_eq_without_eq,
    clippy::missing_const_for_fn,
    clippy::manual_is_variant_and,
    clippy::match_same_arms,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    private_interfaces
)]

use core::cell::{Cell, RefCell};

use sml::sml;

pub const MAX_TOKENS: usize = 4096;
pub const MAX_SEQ: usize = 256;
pub const SEQ_WORDS: usize = MAX_SEQ.div_ceil(64);

/// Synchronous success outcome delivered to a batch callback.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BatchDone {
    /// Number of tokens in the completed request.
    pub token_count: usize,
    /// Number of selected output tokens.
    pub outputs_total: usize,
}

/// Synchronous failure outcome delivered to a batch callback.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BatchFailure {
    /// Failure classification.
    pub error: BatchError,
    /// Number of tokens in the rejected request.
    pub token_count: usize,
}

/// Statically dispatched, non-retained success callback.
pub type BatchDoneCallback = fn(BatchDone);
/// Statically dispatched, non-retained failure callback.
pub type BatchErrorCallback = fn(BatchFailure);

/// Failure reported by a position-seed provider.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PositionSeedError {
    /// The provider could not access its backing state.
    Backend,
    /// The provider rejected the sequence identifier.
    InvalidRequest,
}

/// Request-specific, statically dispatched position-seed context.
#[derive(Clone, Copy, Debug)]
pub struct PositionSeedContext<'a> {
    /// Caller-owned seed positions indexed by sequence identifier.
    pub seeds: &'a [i32],
}

/// Safe callback/context shape for position-seed resolution.
pub type PositionSeedFn = fn(&PositionSeedContext<'_>, i32) -> Result<i32, PositionSeedError>;

/// A borrowed position-seed resolver and its request-specific context.
#[derive(Clone, Copy, Debug)]
pub struct PositionSeedResolver<'a> {
    /// Typed request context.
    pub context: PositionSeedContext<'a>,
    /// Statically dispatched resolver.
    pub resolve: PositionSeedFn,
}

/// Caller-owned output buffers populated by the batcher.
pub struct BatchOutputs<'a> {
    pub seq_primary_ids: &'a mut [i32],
    pub seq_masks: &'a mut [u64],
    pub positions: &'a mut [i32],
    pub output_mask: &'a mut [i8],
}

impl core::fmt::Debug for BatchOutputs<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BatchOutputs")
            .field("seq_primary_ids_len", &self.seq_primary_ids.len())
            .field("seq_masks_len", &self.seq_masks.len())
            .field("positions_len", &self.positions.len())
            .field("output_mask_len", &self.output_mask.len())
            .finish()
    }
}

/// Complete safe Rust equivalent of `token::batcher::event::batch`.
pub struct BatchRequest<'a> {
    pub token_ids: &'a [i32],
    pub vocab_size: i32,
    pub seq_masks: Option<&'a [u64]>,
    pub seq_mask_words: usize,
    pub seq_primary_ids: Option<&'a [i32]>,
    pub positions: Option<&'a [i32]>,
    pub output_mask_input: Option<&'a [i8]>,
    pub output_all: bool,
    pub enforce_single_output_per_seq: bool,
    pub resolve_position_seed: Option<PositionSeedResolver<'a>>,
    pub seq_mask_words_out: Option<&'a mut usize>,
    pub positions_count_out: Option<&'a mut usize>,
    pub outputs_total_out: Option<&'a mut usize>,
    /// Optional synchronous completion callback.
    pub on_done: Option<BatchDoneCallback>,
    /// Optional synchronous error callback.
    pub on_error: Option<BatchErrorCallback>,
    pub outputs: BatchOutputs<'a>,
}

impl core::fmt::Debug for BatchRequest<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BatchRequest")
            .field("token_ids_len", &self.token_ids.len())
            .field("vocab_size", &self.vocab_size)
            .field("seq_masks", &self.seq_masks.map(<[u64]>::len))
            .field("seq_mask_words", &self.seq_mask_words)
            .field("seq_primary_ids", &self.seq_primary_ids.map(<[i32]>::len))
            .field("positions", &self.positions.map(<[i32]>::len))
            .field(
                "output_mask_input",
                &self.output_mask_input.map(<[i8]>::len),
            )
            .field("output_all", &self.output_all)
            .field(
                "enforce_single_output_per_seq",
                &self.enforce_single_output_per_seq,
            )
            .field("outputs", &self.outputs)
            .finish()
    }
}

/// Successful batch result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BatchResult {
    pub token_count: usize,
    pub seq_mask_words: usize,
    pub positions_count: usize,
    pub outputs_total: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchError {
    InvalidRequest,
    Backend,
    Internal,
    UnexpectedEvent,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DispatchError {
    None,
    InvalidRequest,
    Backend,
    Internal,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProbeOutcome {
    Pending,
    Ok,
    InvalidRequest,
    Backend,
    Internal,
}

#[derive(Debug)]
struct Scratch {
    next_pos: [i32; MAX_SEQ],
    seed_pos: [i32; MAX_SEQ],
    seen: [bool; MAX_SEQ],
    seq_output_count: [usize; MAX_SEQ],
    seq_last_pos: [i32; MAX_SEQ],
    seq_min_pos: [i32; MAX_SEQ],
    seq_max_pos: [i32; MAX_SEQ],
    seq_pos_count: [usize; MAX_SEQ],
    seq_seen: [bool; MAX_SEQ],
    active: [usize; MAX_SEQ],
    active_count: usize,
    current_masks: [u64; MAX_SEQ * SEQ_WORDS],
}

impl Scratch {
    fn reset(&mut self) {
        self.next_pos.fill(0);
        self.seed_pos.fill(0);
        self.seen.fill(false);
        self.seq_output_count.fill(0);
        self.seq_last_pos.fill(-1);
        self.seq_min_pos.fill(i32::MAX);
        self.seq_max_pos.fill(i32::MIN);
        self.seq_pos_count.fill(0);
        self.seq_seen.fill(false);
        self.active.fill(0);
        self.active_count = 0;
        self.current_masks.fill(0);
    }
}

impl Default for Scratch {
    fn default() -> Self {
        Self {
            next_pos: [0; MAX_SEQ],
            seed_pos: [0; MAX_SEQ],
            seen: [false; MAX_SEQ],
            seq_output_count: [0; MAX_SEQ],
            seq_last_pos: [-1; MAX_SEQ],
            seq_min_pos: [i32::MAX; MAX_SEQ],
            seq_max_pos: [i32::MIN; MAX_SEQ],
            seq_pos_count: [0; MAX_SEQ],
            seq_seen: [false; MAX_SEQ],
            active: [0; MAX_SEQ],
            active_count: 0,
            current_masks: [0; MAX_SEQ * SEQ_WORDS],
        }
    }
}

#[derive(Debug)]
struct DispatchContext {
    error: Cell<DispatchError>,
    probe_outcome: Cell<ProbeOutcome>,
    mask_words: Cell<usize>,
    positions_count: Cell<usize>,
    outputs_total: Cell<usize>,
    scratch: RefCell<Scratch>,
}

impl DispatchContext {
    fn new() -> Self {
        Self {
            error: Cell::new(DispatchError::None),
            probe_outcome: Cell::new(ProbeOutcome::Pending),
            mask_words: Cell::new(1),
            positions_count: Cell::new(0),
            outputs_total: Cell::new(0),
            scratch: RefCell::new(Scratch::default()),
        }
    }
}

#[derive(Clone, Copy)]
struct RequestView<'a> {
    token_ids: &'a [i32],
    vocab_size: i32,
    seq_masks: Option<&'a [u64]>,
    seq_mask_words: usize,
    seq_primary_ids: Option<&'a [i32]>,
    positions: Option<&'a [i32]>,
    output_mask_input: Option<&'a [i8]>,
    output_all: bool,
    enforce_single_output_per_seq: bool,
    resolve_position_seed: Option<PositionSeedResolver<'a>>,
    seq_mask_words_out: Option<&'a Cell<usize>>,
    positions_count_out: Option<&'a Cell<usize>>,
    outputs_total_out: Option<&'a Cell<usize>>,
    on_done: Option<BatchDoneCallback>,
    on_error: Option<BatchErrorCallback>,
}

#[derive(Clone, Copy)]
struct OutputCells<'a> {
    seq_primary_ids: &'a [Cell<i32>],
    seq_masks: &'a [Cell<u64>],
    positions: &'a [Cell<i32>],
    output_mask: &'a [Cell<i8>],
}

#[derive(Clone, Copy)]
struct BatchRuntime<'a> {
    request: RequestView<'a>,
    outputs: OutputCells<'a>,
    context: &'a DispatchContext,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Context {
    unexpected: bool,
}

sml! {
    TokenBatcher {
        "request_decision"_s <= *"ready"_s + Batch(&'dispatch BatchRuntime<'dispatch>) / begin_batch,
        "request_validation_probe"_s <= "request_decision"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>),
        "outputs_decision"_s <= "request_validation_probe"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>),
        "errored"_s <= "outputs_decision"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [request_outputs_missing] / invalid,
        "counts_decision"_s <= "outputs_decision"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [request_outputs_present],
        "errored"_s <= "counts_decision"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [token_counts_invalid] / invalid,
        "capacity_decision"_s <= "counts_decision"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [capacities_valid],
        "errored"_s <= "counts_decision"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [capacities_invalid] / invalid,
        "vocab_decision"_s <= "capacity_decision"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [vocab_valid],
        "errored"_s <= "capacity_decision"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [vocab_invalid] / invalid,
        "seq_payload_decision"_s <= "vocab_decision"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [seq_payload_valid],
        "errored"_s <= "vocab_decision"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [seq_payload_invalid] / invalid,

        "seq_masks"_s <= "seq_payload_decision"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [seq_masks_mode] / normalize_masks,
        "seq_primary"_s <= "seq_payload_decision"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [seq_primary_mode] / normalize_primary,
        "seq_default"_s <= "seq_payload_decision"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [seq_default_mode] / normalize_default,
        "errored"_s <= "seq_payload_decision"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) / internal,
        "seq_mask_words_decision"_s <= "seq_masks"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_ok],
        "errored"_s <= "seq_masks"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_invalid] / invalid,
        "errored"_s <= "seq_masks"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_backend] / backend,
        "errored"_s <= "seq_masks"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_internal] / internal,
        "errored"_s <= "seq_masks"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_unknown] / internal,
        "seq_mask_words_decision"_s <= "seq_primary"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_ok],
        "errored"_s <= "seq_primary"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_invalid] / invalid,
        "errored"_s <= "seq_primary"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_backend] / backend,
        "errored"_s <= "seq_primary"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_internal] / internal,
        "errored"_s <= "seq_primary"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_unknown] / internal,
        "seq_mask_words_decision"_s <= "seq_default"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_ok],
        "errored"_s <= "seq_default"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_invalid] / invalid,
        "errored"_s <= "seq_default"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_backend] / backend,
        "errored"_s <= "seq_default"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_internal] / internal,
        "errored"_s <= "seq_default"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_unknown] / internal,
        "positions_mask_publish"_s <= "seq_mask_words_decision"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [mask_words_output_present] / publish_mask_words,
        "positions_decision"_s <= "seq_mask_words_decision"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [mask_words_output_absent],
        "positions_decision"_s <= "positions_mask_publish"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_ok],
        "errored"_s <= "positions_mask_publish"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_invalid] / invalid,
        "errored"_s <= "positions_mask_publish"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_backend] / backend,
        "errored"_s <= "positions_mask_publish"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_internal] / internal,
        "errored"_s <= "positions_mask_publish"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_unknown] / internal,
        "positions_stride_three"_s <= "positions_decision"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [stride_three] / copy_positions_three,
        "positions_stride_one"_s <= "positions_decision"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [stride_one] / copy_positions_one,
        "positions_seed_probe"_s <= "positions_decision"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [seeded_mode] / probe_seeded,
        "positions_unseed_probe"_s <= "positions_decision"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [unseeded_mode] / probe_unseeded,
        "errored"_s <= "positions_decision"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) / internal,
        "positions_seeded"_s <= "positions_seed_probe"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [seeded_probe_ok] / generate_seeded,
        "errored"_s <= "positions_seed_probe"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [seeded_probe_backend] / backend,
        "errored"_s <= "positions_seed_probe"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [seeded_probe_invalid] / invalid,
        "errored"_s <= "positions_seed_probe"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [seeded_probe_internal] / internal,
        "errored"_s <= "positions_seed_probe"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [seeded_probe_pending] / internal,
        "positions_unseeded"_s <= "positions_unseed_probe"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [unseeded_probe_ok] / generate_unseeded,
        "errored"_s <= "positions_unseed_probe"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [unseeded_probe_backend] / backend,
        "errored"_s <= "positions_unseed_probe"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [unseeded_probe_invalid] / invalid,
        "errored"_s <= "positions_unseed_probe"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [unseeded_probe_internal] / internal,
        "errored"_s <= "positions_unseed_probe"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [unseeded_probe_pending] / internal,
        "positions_count_decision"_s <= "positions_stride_three"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_ok],
        "errored"_s <= "positions_stride_three"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_invalid] / invalid,
        "errored"_s <= "positions_stride_three"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_backend] / backend,
        "errored"_s <= "positions_stride_three"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_internal] / internal,
        "errored"_s <= "positions_stride_three"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_unknown] / internal,
        "positions_count_decision"_s <= "positions_stride_one"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_ok],
        "errored"_s <= "positions_stride_one"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_invalid] / invalid,
        "errored"_s <= "positions_stride_one"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_backend] / backend,
        "errored"_s <= "positions_stride_one"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_internal] / internal,
        "errored"_s <= "positions_stride_one"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_unknown] / internal,
        "positions_count_decision"_s <= "positions_seeded"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_ok],
        "errored"_s <= "positions_seeded"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_invalid] / invalid,
        "errored"_s <= "positions_seeded"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_backend] / backend,
        "errored"_s <= "positions_seeded"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_internal] / internal,
        "errored"_s <= "positions_seeded"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_unknown] / internal,
        "positions_count_decision"_s <= "positions_unseeded"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_ok],
        "errored"_s <= "positions_unseeded"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_invalid] / invalid,
        "errored"_s <= "positions_unseeded"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_backend] / backend,
        "errored"_s <= "positions_unseeded"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_internal] / internal,
        "errored"_s <= "positions_unseeded"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_unknown] / internal,
        "positions_count_publish"_s <= "positions_count_decision"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [positions_count_output_present] / publish_positions_count,
        "output_decision"_s <= "positions_count_decision"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [positions_count_output_absent],
        "output_decision"_s <= "positions_count_publish"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_ok],
        "errored"_s <= "positions_count_publish"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_invalid] / invalid,
        "errored"_s <= "positions_count_publish"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_backend] / backend,
        "errored"_s <= "positions_count_publish"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_internal] / internal,
        "errored"_s <= "positions_count_publish"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_unknown] / internal,

        "output_all"_s <= "output_decision"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [output_all_mode] / set_output_all,
        "output_copy"_s <= "output_decision"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [output_copy_mode] / copy_output,
        "output_last"_s <= "output_decision"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [output_last_mode] / set_output_last,
        "errored"_s <= "output_decision"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) / internal,
        "count_outputs"_s <= "output_all"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_ok] / count_outputs_from_output_all,
        "count_outputs"_s <= "output_copy"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_ok] / count_outputs_from_output_copy,
        "count_outputs"_s <= "output_last"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_ok] / count_outputs_from_output_last,
        "errored"_s <= "output_all"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_invalid] / invalid,
        "errored"_s <= "output_all"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_backend] / backend,
        "errored"_s <= "output_all"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_internal] / internal,
        "errored"_s <= "output_all"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_unknown] / internal,
        "errored"_s <= "output_copy"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_invalid] / invalid,
        "errored"_s <= "output_copy"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_backend] / backend,
        "errored"_s <= "output_copy"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_internal] / internal,
        "errored"_s <= "output_copy"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_unknown] / internal,
        "errored"_s <= "output_last"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_invalid] / invalid,
        "errored"_s <= "output_last"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_backend] / backend,
        "errored"_s <= "output_last"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_internal] / internal,
        "errored"_s <= "output_last"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_unknown] / internal,
        "outputs_total_publish_decision"_s <= "count_outputs"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_ok],
        "errored"_s <= "count_outputs"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_invalid] / invalid,
        "errored"_s <= "count_outputs"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_backend] / backend,
        "errored"_s <= "count_outputs"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_internal] / internal,
        "errored"_s <= "count_outputs"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [phase_unknown] / internal,
        "single_output_decision"_s <= "outputs_total_publish_decision"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [outputs_total_output_present] / publish_outputs_total,
        "single_output_probe"_s <= "single_output_decision"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [single_output_required] / probe_single_output,
        "single_output_decision"_s <= "outputs_total_publish_decision"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [outputs_total_output_absent],
        "continuity_decision"_s <= "single_output_decision"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [single_output_skipped],
        "continuity_decision"_s <= "single_output_probe"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [single_output_probe_ok],
        "errored"_s <= "single_output_probe"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [single_output_probe_invalid] / invalid,
        "errored"_s <= "single_output_probe"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [single_output_probe_backend] / backend,
        "errored"_s <= "single_output_probe"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [single_output_probe_internal] / internal,
        "errored"_s <= "single_output_probe"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) / internal,
        "done"_s <= "continuity_decision"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [continuity_skipped],
        "continuity_probe"_s <= "continuity_decision"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [continuity_required] / probe_continuity,
        "done"_s <= "continuity_probe"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [continuity_probe_ok],
        "errored"_s <= "continuity_probe"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [continuity_probe_invalid] / invalid,
        "errored"_s <= "continuity_probe"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [continuity_probe_backend] / backend,
        "errored"_s <= "continuity_probe"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [continuity_probe_internal] / internal,
        "errored"_s <= "continuity_probe"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) / internal,
        "ready"_s <= "done"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [done_callback_present] / publish_done_callback,
        "ready"_s <= "done"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [done_callback_absent],
        "error_callback_decision"_s <= "errored"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>),
        "ready"_s <= "error_callback_decision"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [error_callback_present] / publish_error_callback,
        "ready"_s <= "error_callback_decision"_s + completion<Batch>(&'dispatch BatchRuntime<'dispatch>) [error_callback_absent],
        "ready"_s <= "ready"_s + unexpected<_> / unexpected,
        "ready"_s <= "request_decision"_s + unexpected<_> / unexpected,
        "ready"_s <= "request_validation_probe"_s + unexpected<_> / unexpected,
        "ready"_s <= "outputs_decision"_s + unexpected<_> / unexpected,
        "ready"_s <= "counts_decision"_s + unexpected<_> / unexpected,
        "ready"_s <= "capacity_decision"_s + unexpected<_> / unexpected,
        "ready"_s <= "vocab_decision"_s + unexpected<_> / unexpected,
        "ready"_s <= "seq_payload_decision"_s + unexpected<_> / unexpected,
        "ready"_s <= "seq_masks"_s + unexpected<_> / unexpected,
        "ready"_s <= "seq_primary"_s + unexpected<_> / unexpected,
        "ready"_s <= "seq_default"_s + unexpected<_> / unexpected,
        "ready"_s <= "seq_mask_words_decision"_s + unexpected<_> / unexpected,
        "ready"_s <= "positions_decision"_s + unexpected<_> / unexpected,
        "ready"_s <= "positions_stride_three"_s + unexpected<_> / unexpected,
        "ready"_s <= "positions_stride_one"_s + unexpected<_> / unexpected,
        "ready"_s <= "positions_seed_probe"_s + unexpected<_> / unexpected,
        "ready"_s <= "positions_unseed_probe"_s + unexpected<_> / unexpected,
        "ready"_s <= "positions_seeded"_s + unexpected<_> / unexpected,
        "ready"_s <= "positions_unseeded"_s + unexpected<_> / unexpected,
        "ready"_s <= "positions_count_decision"_s + unexpected<_> / unexpected,
        "ready"_s <= "positions_count_publish"_s + unexpected<_> / unexpected,
        "ready"_s <= "output_decision"_s + unexpected<_> / unexpected,
        "ready"_s <= "output_all"_s + unexpected<_> / unexpected,
        "ready"_s <= "output_copy"_s + unexpected<_> / unexpected,
        "ready"_s <= "output_last"_s + unexpected<_> / unexpected,
        "ready"_s <= "count_outputs"_s + unexpected<_> / unexpected,
        "ready"_s <= "single_output_decision"_s + unexpected<_> / unexpected,
        "ready"_s <= "single_output_probe"_s + unexpected<_> / unexpected,
        "ready"_s <= "positions_mask_publish"_s + unexpected<_> / unexpected,
        "ready"_s <= "outputs_total_publish_decision"_s + unexpected<_> / unexpected,
        "ready"_s <= "continuity_decision"_s + unexpected<_> / unexpected,
        "ready"_s <= "continuity_probe"_s + unexpected<_> / unexpected,
        "ready"_s <= "error_callback_decision"_s + unexpected<_> / unexpected,
        "ready"_s <= "done"_s + unexpected<_> / unexpected,
        "ready"_s <= "errored"_s + unexpected<_> / unexpected,
    }
}
impl TokenBatcherStateMachine<Context> {
    pub(super) fn dispatch(
        &mut self,
        request: BatchRequest<'_>,
    ) -> Result<BatchResult, BatchError> {
        let BatchRequest {
            token_ids,
            vocab_size,
            seq_masks,
            seq_mask_words,
            seq_primary_ids,
            positions,
            output_mask_input,
            output_all,
            enforce_single_output_per_seq,
            resolve_position_seed,
            seq_mask_words_out,
            positions_count_out,
            outputs_total_out,
            on_done,
            on_error,
            outputs,
        } = request;
        let primary_out = Cell::from_mut(outputs.seq_primary_ids).as_slice_of_cells();
        let masks_out = Cell::from_mut(outputs.seq_masks).as_slice_of_cells();
        let positions_out = Cell::from_mut(outputs.positions).as_slice_of_cells();
        let output_out = Cell::from_mut(outputs.output_mask).as_slice_of_cells();
        let mask_words_cell = seq_mask_words_out.map(Cell::from_mut);
        let positions_count_cell = positions_count_out.map(Cell::from_mut);
        let total_cell = outputs_total_out.map(Cell::from_mut);
        let context = DispatchContext::new();
        let runtime = BatchRuntime {
            request: RequestView {
                token_ids,
                vocab_size,
                seq_masks,
                seq_mask_words,
                seq_primary_ids,
                positions,
                output_mask_input,
                output_all,
                enforce_single_output_per_seq,
                resolve_position_seed,
                seq_mask_words_out: mask_words_cell.as_ref().map(|c| &**c),
                positions_count_out: positions_count_cell.as_ref().map(|c| &**c),
                outputs_total_out: total_cell.as_ref().map(|c| &**c),
                on_done,
                on_error,
            },
            outputs: OutputCells {
                seq_primary_ids: primary_out,
                seq_masks: masks_out,
                positions: positions_out,
                output_mask: output_out,
            },
            context: &context,
        };
        if self
            .process_event(TokenBatcherEvents::Batch(&runtime))
            .is_err()
        {
            return Err(BatchError::Internal);
        }
        match context.error.get() {
            DispatchError::None => Ok(BatchResult {
                token_count: token_ids.len(),
                seq_mask_words: context.mask_words.get(),
                positions_count: context.positions_count.get(),
                outputs_total: context.outputs_total.get(),
            }),
            DispatchError::InvalidRequest => Err(BatchError::InvalidRequest),
            DispatchError::Backend => Err(BatchError::Backend),
            DispatchError::Internal => Err(BatchError::Internal),
        }
    }

    pub(super) fn dispatch_unexpected(&mut self) -> Result<(), BatchError> {
        self.context_mut().unexpected = false;
        if self.is_ready() {
            self.context_mut().unexpected = true;
        }
        Err(BatchError::UnexpectedEvent)
    }

    pub(super) fn is_ready(&self) -> bool {
        self.is(&TokenBatcherStates::Ready)
    }
}
impl TokenBatcherStateMachineContext for Context {
    fn begin_batch(&mut self, event: &BatchRuntime<'_>) -> Result<(), ()> {
        self.unexpected = false;
        event.context.error.set(DispatchError::None);
        event.context.probe_outcome.set(ProbeOutcome::Pending);
        event
            .context
            .mask_words
            .set(effective_mask_words(&event.request));
        event.context.positions_count.set(0);
        event.context.outputs_total.set(0);
        event.context.scratch.borrow_mut().reset();
        Ok(())
    }
    fn request_outputs_present(&self, _event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(true)
    }
    fn request_outputs_missing(&self, _event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(false)
    }
    fn token_counts_invalid(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(event.request.token_ids.is_empty() || event.request.token_ids.len() > MAX_TOKENS)
    }
    fn capacities_valid(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        let n = event.request.token_ids.len();
        let words = effective_mask_words(&event.request);
        let mask_n = n.checked_mul(words).unwrap_or(usize::MAX);
        let pos_n = positions_capacity(&event.request);
        Ok(position_stride(&event.request) >= 0
            && event.outputs.seq_primary_ids.len() >= n
            && event.outputs.seq_masks.len() >= mask_n
            && event.outputs.positions.len() >= pos_n
            && event.outputs.output_mask.len() >= n)
    }
    fn capacities_invalid(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(!self.capacities_valid(event)?)
    }
    fn vocab_valid(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(vocab_valid(&event.request))
    }
    fn vocab_invalid(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(!vocab_valid(&event.request))
    }
    fn seq_payload_valid(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(seq_payload_valid(&event.request))
    }
    fn seq_payload_invalid(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(!seq_payload_valid(&event.request))
    }
    fn seq_masks_mode(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(has_masks(&event.request))
    }
    fn seq_primary_mode(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(!has_masks(&event.request) && has_primary(&event.request))
    }
    fn seq_default_mode(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(!has_masks(&event.request) && !has_primary(&event.request))
    }
    fn normalize_default(&mut self, event: &BatchRuntime<'_>) -> Result<(), ()> {
        normalize_default(event);
        Ok(())
    }
    fn normalize_masks(&mut self, event: &BatchRuntime<'_>) -> Result<(), ()> {
        normalize_masks(event);
        Ok(())
    }
    fn normalize_primary(&mut self, event: &BatchRuntime<'_>) -> Result<(), ()> {
        normalize_primary(event);
        Ok(())
    }
    fn seeded_probe_ok(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(event.context.probe_outcome.get() == ProbeOutcome::Ok)
    }
    fn seeded_probe_backend(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(event.context.probe_outcome.get() == ProbeOutcome::Backend)
    }
    fn seeded_probe_invalid(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(event.context.probe_outcome.get() == ProbeOutcome::InvalidRequest)
    }
    fn seeded_probe_internal(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(event.context.probe_outcome.get() == ProbeOutcome::Internal)
    }
    fn seeded_probe_pending(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(event.context.probe_outcome.get() == ProbeOutcome::Pending)
    }
    fn unseeded_probe_ok(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(event.context.probe_outcome.get() == ProbeOutcome::Ok)
    }
    fn unseeded_probe_backend(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(event.context.probe_outcome.get() == ProbeOutcome::Backend)
    }
    fn unseeded_probe_invalid(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(event.context.probe_outcome.get() == ProbeOutcome::InvalidRequest)
    }
    fn unseeded_probe_internal(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(event.context.probe_outcome.get() == ProbeOutcome::Internal)
    }
    fn unseeded_probe_pending(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(event.context.probe_outcome.get() == ProbeOutcome::Pending)
    }
    fn phase_ok(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(event.context.error.get() == DispatchError::None)
    }
    fn phase_invalid(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(event.context.error.get() == DispatchError::InvalidRequest)
    }
    fn phase_backend(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(event.context.error.get() == DispatchError::Backend)
    }
    fn phase_internal(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(event.context.error.get() == DispatchError::Internal)
    }
    fn phase_unknown(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(!matches!(
            event.context.error.get(),
            DispatchError::None
                | DispatchError::InvalidRequest
                | DispatchError::Backend
                | DispatchError::Internal
        ))
    }
    fn stride_three(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(position_stride(&event.request) == 3)
    }
    fn stride_one(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(position_stride(&event.request) == 1)
    }
    fn seeded_mode(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(position_stride(&event.request) == 0 && event.request.resolve_position_seed.is_some())
    }
    fn unseeded_mode(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(position_stride(&event.request) == 0 && event.request.resolve_position_seed.is_none())
    }
    fn copy_positions_three(&mut self, event: &BatchRuntime<'_>) -> Result<(), ()> {
        copy_positions(event, 3);
        Ok(())
    }
    fn copy_positions_one(&mut self, event: &BatchRuntime<'_>) -> Result<(), ()> {
        copy_positions(event, 1);
        Ok(())
    }
    fn probe_seeded(&mut self, event: &BatchRuntime<'_>) -> Result<(), ()> {
        probe_seeded(event);
        Ok(())
    }
    fn probe_unseeded(&mut self, event: &BatchRuntime<'_>) -> Result<(), ()> {
        probe_unseeded(event);
        Ok(())
    }
    fn generate_seeded(&mut self, event: &BatchRuntime<'_>) -> Result<(), ()> {
        generate_seeded(event);
        Ok(())
    }
    fn generate_unseeded(&mut self, event: &BatchRuntime<'_>) -> Result<(), ()> {
        generate_unseeded(event);
        Ok(())
    }
    fn mask_words_output_present(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(event.request.seq_mask_words_out.is_some())
    }
    fn mask_words_output_absent(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(event.request.seq_mask_words_out.is_none())
    }
    fn positions_count_output_present(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(event.request.positions_count_out.is_some())
    }
    fn positions_count_output_absent(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(event.request.positions_count_out.is_none())
    }
    fn publish_mask_words(&mut self, event: &BatchRuntime<'_>) -> Result<(), ()> {
        event
            .request
            .seq_mask_words_out
            .expect("guard selected mask output")
            .set(event.context.mask_words.get());
        Ok(())
    }
    fn publish_positions_count(&mut self, event: &BatchRuntime<'_>) -> Result<(), ()> {
        event
            .request
            .positions_count_out
            .expect("guard selected position output")
            .set(event.context.positions_count.get());
        Ok(())
    }
    fn output_all_mode(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(event.request.output_all)
    }
    fn output_copy_mode(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.request.output_all && output_mask_present(&event.request))
    }
    fn output_last_mode(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.request.output_all && !output_mask_present(&event.request))
    }
    fn set_output_all(&mut self, event: &BatchRuntime<'_>) -> Result<(), ()> {
        let count = event.request.token_ids.len();
        for output in event.outputs.output_mask.iter().take(count) {
            output.set(1);
        }
        Ok(())
    }
    fn copy_output(&mut self, event: &BatchRuntime<'_>) -> Result<(), ()> {
        let input = event.request.output_mask_input.ok_or(())?;
        let count = event.request.token_ids.len();
        for (output, input) in event
            .outputs
            .output_mask
            .iter()
            .take(count)
            .zip(input.iter().take(count))
        {
            output.set(*input);
        }
        Ok(())
    }
    fn set_output_last(&mut self, event: &BatchRuntime<'_>) -> Result<(), ()> {
        let count = event.request.token_ids.len();
        for output in event.outputs.output_mask.iter().take(count) {
            output.set(0);
        }
        if let Some(output) = event.outputs.output_mask.get(count.saturating_sub(1)) {
            output.set(1);
        }
        Ok(())
    }
    fn count_outputs_from_output_all(&mut self, event: &BatchRuntime<'_>) -> Result<(), ()> {
        let count = event.request.token_ids.len();
        let total = event
            .outputs
            .output_mask
            .iter()
            .take(count)
            .filter(|output| output.get() != 0)
            .count();
        event.context.outputs_total.set(total);
        Ok(())
    }
    fn count_outputs_from_output_copy(&mut self, event: &BatchRuntime<'_>) -> Result<(), ()> {
        self.count_outputs_from_output_all(event)
    }
    fn count_outputs_from_output_last(&mut self, event: &BatchRuntime<'_>) -> Result<(), ()> {
        self.count_outputs_from_output_all(event)
    }
    fn single_output_probe_ok(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(event.context.probe_outcome.get() == ProbeOutcome::Ok)
    }
    fn single_output_probe_invalid(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(event.context.probe_outcome.get() == ProbeOutcome::InvalidRequest)
    }
    fn single_output_probe_backend(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(event.context.probe_outcome.get() == ProbeOutcome::Backend)
    }
    fn single_output_probe_internal(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(event.context.probe_outcome.get() == ProbeOutcome::Internal)
    }
    fn continuity_probe_ok(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(event.context.probe_outcome.get() == ProbeOutcome::Ok)
    }
    fn continuity_probe_invalid(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(event.context.probe_outcome.get() == ProbeOutcome::InvalidRequest)
    }
    fn continuity_probe_backend(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(event.context.probe_outcome.get() == ProbeOutcome::Backend)
    }
    fn continuity_probe_internal(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(event.context.probe_outcome.get() == ProbeOutcome::Internal)
    }
    fn single_output_required(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(event.request.enforce_single_output_per_seq)
    }
    fn single_output_skipped(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.request.enforce_single_output_per_seq)
    }
    fn probe_single_output(&mut self, event: &BatchRuntime<'_>) -> Result<(), ()> {
        event.context.probe_outcome.set(match single_output_ok(event) {
            true => ProbeOutcome::Ok,
            false => ProbeOutcome::InvalidRequest,
        });
        Ok(())
    }
    fn continuity_required(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(position_stride(&event.request) <= 1)
    }
    fn continuity_skipped(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(position_stride(&event.request) > 1)
    }
    fn probe_continuity(&mut self, event: &BatchRuntime<'_>) -> Result<(), ()> {
        event.context.probe_outcome.set(match continuity_ok(event) {
            true => ProbeOutcome::Ok,
            false => ProbeOutcome::InvalidRequest,
        });
        Ok(())
    }
    fn outputs_total_output_present(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(event.request.outputs_total_out.is_some())
    }
    fn outputs_total_output_absent(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(event.request.outputs_total_out.is_none())
    }
    fn publish_outputs_total(&mut self, event: &BatchRuntime<'_>) -> Result<(), ()> {
        event
            .request
            .outputs_total_out
            .expect("guard selected total output")
            .set(event.context.outputs_total.get());
        Ok(())
    }
    fn done_callback_present(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(event.request.on_done.is_some())
    }
    fn done_callback_absent(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(event.request.on_done.is_none())
    }
    fn error_callback_present(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(event.request.on_error.is_some())
    }
    fn error_callback_absent(&self, event: &BatchRuntime<'_>) -> Result<bool, ()> {
        Ok(event.request.on_error.is_none())
    }
    fn publish_done_callback(&mut self, event: &BatchRuntime<'_>) -> Result<(), ()> {
        (event.request.on_done.expect("guard selected done callback"))(BatchDone {
            token_count: event.request.token_ids.len(),
            outputs_total: event.context.outputs_total.get(),
        });
        Ok(())
    }
    fn publish_error_callback(&mut self, event: &BatchRuntime<'_>) -> Result<(), ()> {
        (event
            .request
            .on_error
            .expect("guard selected error callback"))(BatchFailure {
            error: match event.context.error.get() {
                DispatchError::InvalidRequest => BatchError::InvalidRequest,
                DispatchError::Backend => BatchError::Backend,
                DispatchError::Internal | DispatchError::None => BatchError::Internal,
            },
            token_count: event.request.token_ids.len(),
        });
        Ok(())
    }
    fn invalid(&mut self, event: &BatchRuntime<'_>) -> Result<(), ()> {
        event.context.error.set(DispatchError::InvalidRequest);
        Ok(())
    }
    fn backend(&mut self, event: &BatchRuntime<'_>) -> Result<(), ()> {
        event.context.error.set(DispatchError::Backend);
        Ok(())
    }
    fn internal(&mut self, event: &BatchRuntime<'_>) -> Result<(), ()> {
        event.context.error.set(DispatchError::Internal);
        Ok(())
    }
    fn unexpected(&mut self) -> Result<(), ()> {
        self.unexpected = true;
        Ok(())
    }
}

fn has_masks(r: &RequestView<'_>) -> bool {
    r.seq_masks.is_some_and(|values| {
        r.seq_mask_words > 0
            && r.seq_mask_words <= SEQ_WORDS
            && r.token_ids
                .len()
                .checked_mul(r.seq_mask_words)
                .is_some_and(|required| values.len() >= required)
    })
}
fn has_primary(r: &RequestView<'_>) -> bool {
    r.seq_primary_ids
        .is_some_and(|v| v.len() >= r.token_ids.len())
}
fn output_mask_present(r: &RequestView<'_>) -> bool {
    r.output_mask_input
        .is_some_and(|v| v.len() >= r.token_ids.len())
}
fn effective_mask_words(r: &RequestView<'_>) -> usize {
    if has_masks(r) { r.seq_mask_words } else { 1 }
}
const fn position_stride(r: &RequestView<'_>) -> i32 {
    match r.positions {
        None => 0,
        Some(v) if v.len() >= r.token_ids.len().saturating_mul(3) => 3,
        Some(v) if v.len() >= r.token_ids.len() => 1,
        Some(_) => -1,
    }
}
const fn positions_capacity(r: &RequestView<'_>) -> usize {
    match position_stride(r) {
        3 => r.token_ids.len().saturating_mul(3),
        _ => r.token_ids.len(),
    }
}
const fn normalized_positions_count(r: &RequestView<'_>) -> usize {
    if position_stride(r) == 3 {
        r.token_ids.len().saturating_mul(3)
    } else {
        r.token_ids.len()
    }
}
fn vocab_valid(r: &RequestView<'_>) -> bool {
    r.vocab_size >= 0
        && (r.vocab_size == 0 || r.token_ids.iter().all(|t| *t >= 0 && *t < r.vocab_size))
}
fn mask_has(mask: &[u64], id: i32) -> bool {
    usize::try_from(id)
        .ok()
        .is_some_and(|id| id / 64 < mask.len() && mask[id / 64] & (1u64 << (id % 64)) != 0)
}
fn explicit_masks_malformed(r: &RequestView<'_>) -> bool {
    r.seq_masks.is_some_and(|values| {
        r.seq_mask_words == 0
            || r.seq_mask_words > SEQ_WORDS
            || r.token_ids
                .len()
                .checked_mul(r.seq_mask_words)
                .is_none_or(|required| values.len() < required)
    })
}
fn seq_payload_valid(r: &RequestView<'_>) -> bool {
    if explicit_masks_malformed(r) || position_stride(r) < 0 {
        return false;
    }
    let masks = has_masks(r);
    let primary = has_primary(r);
    let words = effective_mask_words(r);
    let mask_len = r
        .token_ids
        .len()
        .checked_mul(r.seq_mask_words)
        .unwrap_or(usize::MAX);
    let masks_ok = !masks
        || (r.seq_masks.is_some_and(|values| values.len() >= mask_len)
            && r.seq_masks
                .unwrap()
                .chunks_exact(r.seq_mask_words)
                .take(r.token_ids.len())
                .all(|row| row.iter().any(|v| *v != 0)));
    let ids_ok = !primary
        || r.seq_primary_ids
            .unwrap()
            .iter()
            .take(r.token_ids.len())
            .all(|id| *id >= 0 && usize::try_from(*id).is_ok_and(|id| id < words * 64));
    let relation_ok = !(masks && primary)
        || r.seq_masks
            .unwrap()
            .chunks_exact(r.seq_mask_words)
            .zip(r.seq_primary_ids.unwrap())
            .take(r.token_ids.len())
            .all(|(row, id)| mask_has(row, *id));
    masks_ok && ids_ok && relation_ok
}
fn normalize_masks(e: &BatchRuntime<'_>) {
    let input = e.request.seq_masks.unwrap();
    let words = e.request.seq_mask_words;
    for i in 0..e.request.token_ids.len() {
        let in_row = &input[i * words..(i + 1) * words];
        for (o, v) in e.outputs.seq_masks[i * words..(i + 1) * words]
            .iter()
            .zip(in_row)
        {
            o.set(*v);
        }
        let id = in_row
            .iter()
            .enumerate()
            .find_map(|(w, v)| (*v != 0).then_some(w * 64 + v.trailing_zeros() as usize))
            .unwrap_or(0);
        e.outputs.seq_primary_ids[i].set(i32::try_from(id).unwrap_or(0));
    }
}
fn normalize_primary(e: &BatchRuntime<'_>) {
    for (i, id) in e
        .request
        .seq_primary_ids
        .unwrap()
        .iter()
        .copied()
        .take(e.request.token_ids.len())
        .enumerate()
    {
        e.outputs.seq_primary_ids[i].set(id);
        e.outputs.seq_masks[i].set(1u64 << u32::try_from(id).unwrap());
    }
}
fn normalize_default(e: &BatchRuntime<'_>) {
    for i in 0..e.request.token_ids.len() {
        e.outputs.seq_primary_ids[i].set(0);
        e.outputs.seq_masks[i].set(1);
    }
}
fn copy_positions(e: &BatchRuntime<'_>, stride: usize) {
    let n = e.request.token_ids.len() * stride;
    for (o, v) in e
        .outputs
        .positions
        .iter()
        .take(n)
        .zip(e.request.positions.unwrap().iter().take(n))
    {
        o.set(*v);
    }
    e.context.positions_count.set(n);
}
fn probe_seeded(e: &BatchRuntime<'_>) {
    let mut s = e.context.scratch.borrow_mut();
    let resolver = e.request.resolve_position_seed.unwrap();
    let mut backend_error = false;
    let mut invalid_error = false;
    for id in 0..MAX_SEQ {
        match (resolver.resolve)(&resolver.context, id as i32) {
            Ok(v) => {
                invalid_error |= v < 0;
                s.next_pos[id] = v;
                s.seed_pos[id] = v;
            }
            Err(PositionSeedError::Backend) => {
                backend_error = true;
            }
            Err(PositionSeedError::InvalidRequest) => {
                invalid_error = true;
            }
        }
    }
    let mut valid = !backend_error && !invalid_error;
    let words = effective_mask_words(&e.request);
    for i in 0..e.request.token_ids.len() {
        let primary = e.outputs.seq_primary_ids[i].get() as usize;
        let pos = s.next_pos[primary];
        valid = valid && pos != i32::MAX;
        let row = &e.outputs.seq_masks[i * words..(i + 1) * words];
        let compatible = valid
            && row.iter().enumerate().all(|(w, bits)| {
                let mut b = bits.get();
                let mut ok = true;
                while b != 0 {
                    let bit = b.trailing_zeros() as usize;
                    ok &= s.next_pos[w * 64 + bit] == pos;
                    b &= b - 1;
                }
                ok
            });
        valid = valid && compatible;
        let advance = i64::from(valid as i32);
        let next = i64::from(pos) + 1;
        for (w, bits) in row.iter().enumerate() {
            let mut b = bits.get();
            while b != 0 {
                let bit = b.trailing_zeros() as usize;
                let id = w * 64 + bit;
                let current = i64::from(s.next_pos[id]);
                s.next_pos[id] = (advance * next + (1 - advance) * current) as i32;
                b &= b - 1;
            }
        }
    }
    e.context.probe_outcome.set(if backend_error {
        ProbeOutcome::Backend
    } else if !valid {
        ProbeOutcome::InvalidRequest
    } else {
        ProbeOutcome::Ok
    });
}
fn probe_unseeded(e: &BatchRuntime<'_>) {
    let mut s = e.context.scratch.borrow_mut();
    let words = effective_mask_words(&e.request);
    let mut valid = true;
    for i in 0..e.request.token_ids.len() {
        let primary = e.outputs.seq_primary_ids[i].get() as usize;
        let pos = s.next_pos[primary];
        valid = valid && pos != i32::MAX;
        let row = &e.outputs.seq_masks[i * words..(i + 1) * words];
        let aligned = valid
            && row.iter().enumerate().all(|(w, bits)| {
                let mut b = bits.get();
                let mut ok = true;
                while b != 0 {
                    let bit = b.trailing_zeros() as usize;
                    let id = w * 64 + bit;
                    let current = [pos, s.next_pos[id]][usize::from(s.seen[id])];
                    ok &= current == pos;
                    b &= b - 1;
                }
                ok
            });
        valid = valid && aligned;
        let advance = i64::from(valid as i32);
        let next = i64::from(pos) + 1;
        for (w, bits) in row.iter().enumerate() {
            let mut b = bits.get();
            while b != 0 {
                let bit = b.trailing_zeros() as usize;
                let id = w * 64 + bit;
                s.seen[id] |= advance != 0;
                let current = i64::from(s.next_pos[id]);
                s.next_pos[id] = (advance * next + (1 - advance) * current) as i32;
                b &= b - 1;
            }
        }
    }
    e.context.probe_outcome.set(if valid {
        ProbeOutcome::Ok
    } else {
        ProbeOutcome::InvalidRequest
    });
}
fn generate_seeded(e: &BatchRuntime<'_>) {
    let mut s = e.context.scratch.borrow_mut();
    let words = effective_mask_words(&e.request);
    s.next_pos = s.seed_pos;
    for i in 0..e.request.token_ids.len() {
        let pos = s.next_pos[e.outputs.seq_primary_ids[i].get() as usize];
        e.outputs.positions[i].set(pos);
        let row = &e.outputs.seq_masks[i * words..(i + 1) * words];
        for (w, bits) in row.iter().enumerate() {
            let mut b = bits.get();
            while b != 0 {
                let bit = b.trailing_zeros() as usize;
                let Some(next) = pos.checked_add(1) else {
                    e.context.error.set(DispatchError::InvalidRequest);
                    return;
                };
                s.next_pos[w * 64 + bit] = next;
                b &= b - 1;
            }
        }
    }
    e.context.positions_count.set(e.request.token_ids.len());
}
fn generate_unseeded(e: &BatchRuntime<'_>) {
    let mut s = e.context.scratch.borrow_mut();
    s.next_pos.fill(0);
    s.seen.fill(false);
    let words = effective_mask_words(&e.request);
    for i in 0..e.request.token_ids.len() {
        let pos = s.next_pos[e.outputs.seq_primary_ids[i].get() as usize];
        e.outputs.positions[i].set(pos);
        let row = &e.outputs.seq_masks[i * words..(i + 1) * words];
        for (w, bits) in row.iter().enumerate() {
            let mut b = bits.get();
            while b != 0 {
                let bit = b.trailing_zeros() as usize;
                let id = w * 64 + bit;
                s.seen[id] = true;
                let Some(next) = pos.checked_add(1) else {
                    e.context.error.set(DispatchError::InvalidRequest);
                    return;
                };
                s.next_pos[id] = next;
                b &= b - 1;
            }
        }
    }
    e.context.positions_count.set(e.request.token_ids.len());
}
fn single_output_ok(e: &BatchRuntime<'_>) -> bool {
    let mut s = e.context.scratch.borrow_mut();
    s.seq_output_count.fill(0);
    let words = effective_mask_words(&e.request);
    for i in 0..e.request.token_ids.len() {
        if e.outputs.output_mask[i].get() == 0 {
            continue;
        }
        for (w, bits) in e.outputs.seq_masks[i * words..(i + 1) * words]
            .iter()
            .enumerate()
        {
            let mut b = bits.get();
            while b != 0 {
                let id = w * 64 + b.trailing_zeros() as usize;
                s.seq_output_count[id] += 1;
                if s.seq_output_count[id] > 1 {
                    return false;
                }
                b &= b - 1;
            }
        }
    }
    true
}
fn continuity_ok(e: &BatchRuntime<'_>) -> bool {
    let mut s = e.context.scratch.borrow_mut();
    s.seq_last_pos.fill(-1);
    s.seq_min_pos.fill(i32::MAX);
    s.seq_max_pos.fill(i32::MIN);
    s.seq_pos_count.fill(0);
    s.seq_seen.fill(false);
    s.active.fill(0);
    s.active_count = 0;
    s.current_masks.fill(0);

    let words = effective_mask_words(&e.request);
    let mut ok = true;
    for i in 0..e.request.token_ids.len() {
        let pos = e.outputs.positions[i].get();
        let row = &e.outputs.seq_masks[i * words..(i + 1) * words];
        for (w, bits) in row.iter().enumerate() {
            let mut bits = bits.get();
            while bits != 0 {
                let id = w * 64 + bits.trailing_zeros() as usize;
                let last = s.seq_last_pos[id];
                let first_seen = !s.seq_seen[id];
                let monotonic = (last < 0) || pos >= last;
                let pos_changed = pos != last;
                s.seq_pos_count[id] += usize::from(pos_changed);
                s.seq_last_pos[id] = pos;
                s.seq_min_pos[id] = s.seq_min_pos[id].min(pos);
                s.seq_max_pos[id] = s.seq_max_pos[id].max(pos);
                let active_count = s.active_count;
                let has_active_slot = active_count < MAX_SEQ;
                if first_seen && has_active_slot {
                    s.active[active_count] = id;
                    s.active_count = active_count + 1;
                }
                s.seq_seen[id] = true;

                let mask_start = id * SEQ_WORDS;
                if first_seen {
                    for (current, value) in s.current_masks[mask_start..mask_start + words]
                        .iter_mut()
                        .zip(row.iter())
                    {
                        *current = value.get();
                    }
                } else {
                    for (current, value) in s.current_masks[mask_start..mask_start + words]
                        .iter_mut()
                        .zip(row.iter())
                    {
                        *current &= value.get();
                    }
                }
                let intersection_non_empty = s.current_masks[mask_start..mask_start + words]
                    .iter()
                    .any(|value| *value != 0);
                ok = ok && (!first_seen || has_active_slot) && monotonic && intersection_non_empty;
                bits &= bits - 1;
            }
        }
    }

    for index in 0..s.active_count {
        let id = s.active[index];
        let span = i64::from(s.seq_max_pos[id]) - i64::from(s.seq_min_pos[id]) + 1;
        ok = ok && span <= s.seq_pos_count[id] as i64;
    }
    ok
}
