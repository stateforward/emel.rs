//! Source-aligned bounded text-generator decode-wavefront actor.
//!
//! The actor validates a bounded group of compatible lanes and dispatches each
//! lane either serially or through a caller-owned synchronous lane pool.  Lane
//! callbacks run to completion before the request returns; no work is queued
//! for a later event.

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

use core::cell::RefCell;
use sml::sml;

/// Maximum number of lanes accepted by one decode-wavefront request.
pub const MAX_LANES: usize = 8;
/// Failed-lane sentinel used by the source contract.
pub const NO_FAILED_LANE: i32 = -1;

/// Error values published by the decode-wavefront contract.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum DecodeWavefrontError {
    /// The request completed successfully.
    #[default]
    None = 0,
    /// The lane count was outside the bounded contract.
    InvalidRequest = 1,
    /// Multiple lanes did not share one compatibility key.
    IncompatibleLanes = 2,
    /// The lane pool could not submit or join the requested group.
    Backend = 3,
    /// A lane callback rejected its compute event.
    LaneRejected = 4,
    /// An event arrived outside the current state-machine contract.
    Unexpected = 5,
}

impl DecodeWavefrontError {
    /// Returns the stable numeric error code.
    #[must_use]
    pub const fn code(self) -> u8 { self as u8 }
}

/// Kernel route participating in lane compatibility.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum KernelRoute {
    PackedQ8_0 = 0,
    Q8K = 1,
    NativeQuantized = 2,
    NativeQuantizedQ8KLogits = 3,
    #[default]
    Kernel = 4,
}

/// Output representation participating in lane compatibility.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum OutputContract {
    #[default]
    MaterializedLogits = 0,
    PreselectedArgmax = 1,
}

/// Value-level identity and layout contract used to group lanes.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CompatibilityKey {
    /// Caller-provided identity for the model instance.
    pub model_identity: usize,
    /// Caller-provided identity for the backend instance.
    pub backend_identity: usize,
    /// Kernel-family identity.
    pub kernel_kind: u8,
    /// Attention-mode identity.
    pub attention: u8,
    /// Selected kernel route.
    pub route: KernelRoute,
    /// Selected output representation.
    pub output: OutputContract,
    /// Dtype and layout contract identifier.
    pub dtype_layout_contract: u32,
    /// Quantization contract identifier.
    pub quantized_contract: u32,
    /// Decode step size.
    pub step_size: i32,
    /// Number of tokens in the request.
    pub token_count: i32,
}

/// Summary written by a wavefront run.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DispatchSummary {
    /// Request error, if any.
    pub err: DecodeWavefrontError,
    /// Whether the request contained more than one compatible lane.
    pub grouped: bool,
    /// Whether every parallel lane was submitted.
    pub all_submitted: bool,
    /// Whether the parallel group joined successfully.
    pub joined: bool,
    /// Number of lanes reached by dispatch.
    pub dispatched_lanes: i32,
    /// First rejected lane, or [`NO_FAILED_LANE`].
    pub failed_lane: i32,
}

/// Synchronous lane callback.  The key is copied into the callback, and the
/// boolean result is the lane's accepted/rejected compute outcome.
pub type LaneCallback = fn(CompatibilityKey) -> bool;

/// One caller-owned lane in a bounded request.
#[derive(Clone, Copy, Debug)]
pub struct Lane {
    /// Identity of the single-writer graph actor represented by this lane.
    pub writer_id: usize,
    /// Compatibility metadata for grouping.
    pub key: CompatibilityKey,
    /// Synchronous compute callback.
    pub callback: Option<LaneCallback>,
    /// Outcome populated by the actor.
    pub accepted: bool,
}

impl Lane {
    /// Creates a lane with an initially clear outcome.
    #[must_use]
    pub const fn new(writer_id: usize, key: CompatibilityKey, callback: Option<LaneCallback>) -> Self {
        Self { writer_id, key, callback, accepted: false }
    }
}

/// Caller-owned run event.  The two `RefCell` wrappers make the event Copy for
/// SML completion dispatch while keeping all storage outside the actor.
#[derive(Debug)]
pub struct EventRun<'event> {
    /// Bounded lane storage, owned by the caller.
    pub lanes: &'event RefCell<&'event mut [Lane]>,
    /// Caller-owned result storage.
    pub out: &'event RefCell<DispatchSummary>,
}

impl<'event> Copy for EventRun<'event> {}
impl<'event> Clone for EventRun<'event> {
    fn clone(&self) -> Self { *self }
}

impl<'event> EventRun<'event> {
    /// Creates one bounded wavefront request.
    #[must_use]
    pub const fn new(
        lanes: &'event RefCell<&'event mut [Lane]>,
        out: &'event RefCell<DispatchSummary>,
    ) -> Self {
        Self { lanes, out }
    }
}

/// Caller-owned scheduler capabilities.  Callbacks still execute
/// synchronously; these flags represent the source pool's submission/join
/// outcomes without creating hidden tasks.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LanePool {
    /// Enables the parallel transition for compatible multi-lane groups.
    pub parallel_enabled: bool,
    /// Controls whether all lane submissions succeed.
    pub submission_ok: bool,
    /// Controls whether the submitted group joins successfully.
    pub join_ok: bool,
}

impl LanePool {
    /// Creates a fully operational synchronous pool capability.
    #[must_use]
    pub const fn new() -> Self {
        Self { parallel_enabled: true, submission_ok: true, join_ok: true }
    }
}

/// Source-compatible spelling for integrations naming the lane pool in snake
/// case.
pub type lane_pool = LanePool;

sml! {
    TextGeneratorDecodeWavefront<'event> {
        "state_validation_decision"_s <= *"state_idle"_s + event<EventRun<'event>> / effect_begin_run,
        "state_idle"_s <= "state_validation_decision"_s + completion<EventRun>(EventRun<'event>) [guard_invalid_request] / effect_reject_invalid_request,
        "state_idle"_s <= "state_validation_decision"_s + completion<EventRun>(EventRun<'event>) [guard_multi_lane_incompatible] / effect_reject_incompatible_lanes,
        "state_group_ready"_s <= "state_validation_decision"_s + completion<EventRun>(EventRun<'event>) [guard_single_lane] / effect_mark_single_lane,
        "state_group_ready"_s <= "state_validation_decision"_s + completion<EventRun>(EventRun<'event>) [guard_multi_lane_compatible] / effect_mark_grouped_lanes,
        "state_parallel_decision"_s <= "state_group_ready"_s + completion<EventRun>(EventRun<'event>) [guard_parallel_dispatch] / effect_dispatch_parallel_lanes,
        "state_lane0_decision"_s <= "state_group_ready"_s + completion<EventRun>(EventRun<'event>) [guard_serial_dispatch] / effect_dispatch_lane_0,
        "state_idle"_s <= "state_lane0_decision"_s + completion<EventRun>(EventRun<'event>) [guard_lane_rejected_0] / effect_mark_lane_rejected_0_from_state_lane0_decision,
        "state_idle"_s <= "state_lane0_decision"_s + completion<EventRun>(EventRun<'event>) [guard_lane_accepted_and_last_0] / effect_commit_done_from_state_lane0_decision,
        "state_lane1_decision"_s <= "state_lane0_decision"_s + completion<EventRun>(EventRun<'event>) [guard_lane_accepted_and_more_0] / effect_dispatch_lane_1,
        "state_idle"_s <= "state_lane1_decision"_s + completion<EventRun>(EventRun<'event>) [guard_lane_rejected_1] / effect_mark_lane_rejected_1_from_state_lane1_decision,
        "state_idle"_s <= "state_lane1_decision"_s + completion<EventRun>(EventRun<'event>) [guard_lane_accepted_and_last_1] / effect_commit_done_from_state_lane1_decision,
        "state_lane2_decision"_s <= "state_lane1_decision"_s + completion<EventRun>(EventRun<'event>) [guard_lane_accepted_and_more_1] / effect_dispatch_lane_2,
        "state_idle"_s <= "state_lane2_decision"_s + completion<EventRun>(EventRun<'event>) [guard_lane_rejected_2] / effect_mark_lane_rejected_2_from_state_lane2_decision,
        "state_idle"_s <= "state_lane2_decision"_s + completion<EventRun>(EventRun<'event>) [guard_lane_accepted_and_last_2] / effect_commit_done_from_state_lane2_decision,
        "state_lane3_decision"_s <= "state_lane2_decision"_s + completion<EventRun>(EventRun<'event>) [guard_lane_accepted_and_more_2] / effect_dispatch_lane_3,
        "state_idle"_s <= "state_lane3_decision"_s + completion<EventRun>(EventRun<'event>) [guard_lane_rejected_3] / effect_mark_lane_rejected_3_from_state_lane3_decision,
        "state_idle"_s <= "state_lane3_decision"_s + completion<EventRun>(EventRun<'event>) [guard_lane_accepted_and_last_3] / effect_commit_done_from_state_lane3_decision,
        "state_lane4_decision"_s <= "state_lane3_decision"_s + completion<EventRun>(EventRun<'event>) [guard_lane_accepted_and_more_3] / effect_dispatch_lane_4,
        "state_idle"_s <= "state_lane4_decision"_s + completion<EventRun>(EventRun<'event>) [guard_lane_rejected_4] / effect_mark_lane_rejected_4_from_state_lane4_decision,
        "state_idle"_s <= "state_lane4_decision"_s + completion<EventRun>(EventRun<'event>) [guard_lane_accepted_and_last_4] / effect_commit_done_from_state_lane4_decision,
        "state_lane5_decision"_s <= "state_lane4_decision"_s + completion<EventRun>(EventRun<'event>) [guard_lane_accepted_and_more_4] / effect_dispatch_lane_5,
        "state_idle"_s <= "state_lane5_decision"_s + completion<EventRun>(EventRun<'event>) [guard_lane_rejected_5] / effect_mark_lane_rejected_5_from_state_lane5_decision,
        "state_idle"_s <= "state_lane5_decision"_s + completion<EventRun>(EventRun<'event>) [guard_lane_accepted_and_last_5] / effect_commit_done_from_state_lane5_decision,
        "state_lane6_decision"_s <= "state_lane5_decision"_s + completion<EventRun>(EventRun<'event>) [guard_lane_accepted_and_more_5] / effect_dispatch_lane_6,
        "state_idle"_s <= "state_lane6_decision"_s + completion<EventRun>(EventRun<'event>) [guard_lane_rejected_6] / effect_mark_lane_rejected_6_from_state_lane6_decision,
        "state_idle"_s <= "state_lane6_decision"_s + completion<EventRun>(EventRun<'event>) [guard_lane_accepted_and_last_6] / effect_commit_done_from_state_lane6_decision,
        "state_lane7_decision"_s <= "state_lane6_decision"_s + completion<EventRun>(EventRun<'event>) [guard_lane_accepted_and_more_6] / effect_dispatch_lane_7,
        "state_idle"_s <= "state_lane7_decision"_s + completion<EventRun>(EventRun<'event>) [guard_lane_rejected_7] / effect_mark_lane_rejected_7_from_state_lane7_decision,
        "state_idle"_s <= "state_lane7_decision"_s + completion<EventRun>(EventRun<'event>) [guard_lane_accepted_and_last_7] / effect_commit_done_from_state_lane7_decision,
        "state_idle"_s <= "state_parallel_decision"_s + completion<EventRun>(EventRun<'event>) [guard_parallel_submission_failed] / effect_reject_parallel_scheduler_from_state_parallel_decision,
        "state_idle"_s <= "state_parallel_decision"_s + completion<EventRun>(EventRun<'event>) [guard_parallel_join_failed] / effect_reject_parallel_scheduler_from_state_parallel_decision,
        "state_idle"_s <= "state_parallel_decision"_s + completion<EventRun>(EventRun<'event>) [guard_parallel_lane_rejected_0] / effect_mark_lane_rejected_0_from_state_parallel_decision,
        "state_idle"_s <= "state_parallel_decision"_s + completion<EventRun>(EventRun<'event>) [guard_parallel_lane_rejected_1] / effect_mark_lane_rejected_1_from_state_parallel_decision,
        "state_idle"_s <= "state_parallel_decision"_s + completion<EventRun>(EventRun<'event>) [guard_parallel_lane_rejected_2] / effect_mark_lane_rejected_2_from_state_parallel_decision,
        "state_idle"_s <= "state_parallel_decision"_s + completion<EventRun>(EventRun<'event>) [guard_parallel_lane_rejected_3] / effect_mark_lane_rejected_3_from_state_parallel_decision,
        "state_idle"_s <= "state_parallel_decision"_s + completion<EventRun>(EventRun<'event>) [guard_parallel_lane_rejected_4] / effect_mark_lane_rejected_4_from_state_parallel_decision,
        "state_idle"_s <= "state_parallel_decision"_s + completion<EventRun>(EventRun<'event>) [guard_parallel_lane_rejected_5] / effect_mark_lane_rejected_5_from_state_parallel_decision,
        "state_idle"_s <= "state_parallel_decision"_s + completion<EventRun>(EventRun<'event>) [guard_parallel_lane_rejected_6] / effect_mark_lane_rejected_6_from_state_parallel_decision,
        "state_idle"_s <= "state_parallel_decision"_s + completion<EventRun>(EventRun<'event>) [guard_parallel_lane_rejected_7] / effect_mark_lane_rejected_7_from_state_parallel_decision,
        "state_idle"_s <= "state_parallel_decision"_s + completion<EventRun>(EventRun<'event>) [guard_parallel_all_lanes_accepted] / effect_commit_done_from_state_parallel_decision,
        "state_idle"_s <= "state_idle"_s + unexpected_event<_> / effect_on_unexpected_from_state_idle,
        "state_idle"_s <= "state_validation_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_validation_decision,
        "state_idle"_s <= "state_group_ready"_s + unexpected_event<_> / effect_on_unexpected_from_state_group_ready,
        "state_idle"_s <= "state_lane0_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_lane0_decision,
        "state_idle"_s <= "state_lane1_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_lane1_decision,
        "state_idle"_s <= "state_lane2_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_lane2_decision,
        "state_idle"_s <= "state_lane3_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_lane3_decision,
        "state_idle"_s <= "state_lane4_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_lane4_decision,
        "state_idle"_s <= "state_lane5_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_lane5_decision,
        "state_idle"_s <= "state_lane6_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_lane6_decision,
        "state_idle"_s <= "state_lane7_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_lane7_decision,
        "state_idle"_s <= "state_parallel_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_parallel_decision,
    }
}

/// Runtime capabilities and last-result state retained by the actor.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TextGeneratorDecodeWavefrontContext {
    /// Last result error.
    pub err: DecodeWavefrontError,
    /// Copied scheduler capability; no worker is retained.
    pub pool: Option<LanePool>,
    /// Set after an unexpected event.
    pub unexpected: bool,
}

impl TextGeneratorDecodeWavefrontContext {
    fn unexpected(&mut self) -> Result<(), ()> {
        self.err = DecodeWavefrontError::Unexpected;
        self.unexpected = true;
        Ok(())
    }
}

fn compatible(lhs: CompatibilityKey, rhs: CompatibilityKey) -> bool { lhs == rhs }

fn valid_count(event: &EventRun<'_>) -> bool {
    let count = event.lanes.borrow().len();
    count > 0 && count <= MAX_LANES
}

fn all_compatible(event: &EventRun<'_>) -> bool {
    let lanes = event.lanes.borrow();
    if lanes.is_empty() || lanes.len() > MAX_LANES { return false; }
    let first = lanes[0].key;
    lanes[1..].iter().all(|lane| compatible(first, lane.key))
}

fn distinct_writers(event: &EventRun<'_>) -> bool {
    let lanes = event.lanes.borrow();
    for (index, lane) in lanes.iter().enumerate() {
        if lane.writer_id == 0 { return false; }
        if lanes[index + 1..].iter().any(|other| other.writer_id == lane.writer_id) { return false; }
    }
    true
}

fn dispatch_lane(event: &EventRun<'_>, index: usize) -> Result<(), ()> {
    let mut lanes = event.lanes.borrow_mut();
    if index >= lanes.len() { return Ok(()); }
    let lane = &mut lanes[index];
    lane.accepted = lane.callback.map_or(false, |callback| callback(lane.key));
    event.out.borrow_mut().dispatched_lanes = (index + 1) as i32;
    Ok(())
}

fn mark_rejected(event: &EventRun<'_>, index: usize) -> Result<(), ()> {
    let mut out = event.out.borrow_mut();
    out.err = DecodeWavefrontError::LaneRejected;
    out.failed_lane = index as i32;
    Ok(())
}

fn commit_done(event: &EventRun<'_>) -> Result<(), ()> {
    let mut out = event.out.borrow_mut();
    out.err = DecodeWavefrontError::None;
    out.failed_lane = NO_FAILED_LANE;
    Ok(())
}

impl TextGeneratorDecodeWavefrontStateMachineContext for TextGeneratorDecodeWavefrontContext {
    fn effect_begin_run(&mut self, event: &EventRun<'_>) -> Result<(), ()> {
        self.err = DecodeWavefrontError::None;
        self.unexpected = false;
        *event.out.borrow_mut() = DispatchSummary { err: DecodeWavefrontError::None, failed_lane: NO_FAILED_LANE, ..DispatchSummary::default() };
        for lane in event.lanes.borrow_mut().iter_mut() { lane.accepted = false; }
        Ok(())
    }

    fn effect_mark_single_lane(&mut self, event: &EventRun<'_>) -> Result<(), ()> {
        event.out.borrow_mut().grouped = false;
        Ok(())
    }
    fn effect_mark_grouped_lanes(&mut self, event: &EventRun<'_>) -> Result<(), ()> {
        event.out.borrow_mut().grouped = true;
        Ok(())
    }
    fn effect_reject_invalid_request(&mut self, event: &EventRun<'_>) -> Result<(), ()> {
        self.err = DecodeWavefrontError::InvalidRequest;
        event.out.borrow_mut().err = self.err;
        Ok(())
    }
    fn effect_reject_incompatible_lanes(&mut self, event: &EventRun<'_>) -> Result<(), ()> {
        self.err = DecodeWavefrontError::IncompatibleLanes;
        event.out.borrow_mut().err = self.err;
        Ok(())
    }
    fn effect_reject_parallel_scheduler_from_state_parallel_decision(&mut self, event: &EventRun<'_>) -> Result<(), ()> {
        self.err = DecodeWavefrontError::Backend;
        let mut out = event.out.borrow_mut();
        out.err = self.err;
        out.failed_lane = NO_FAILED_LANE;
        Ok(())
    }
    fn effect_dispatch_parallel_lanes(&mut self, event: &EventRun<'_>) -> Result<(), ()> {
        let pool = self.pool.unwrap_or_default();
        let mut all_submitted = pool.submission_ok;
        let mut lanes = event.lanes.borrow_mut();
        for lane in lanes.iter_mut() {
            lane.accepted = if all_submitted { lane.callback.map_or(false, |callback| callback(lane.key)) } else { false };
            if lane.callback.is_none() { all_submitted = false; }
        }
        let mut out = event.out.borrow_mut();
        out.all_submitted = all_submitted;
        out.joined = pool.join_ok;
        out.dispatched_lanes = lanes.len() as i32;
        Ok(())
    }

    fn effect_commit_done_from_state_lane0_decision(&mut self, event: &EventRun<'_>) -> Result<(), ()> { self.err = DecodeWavefrontError::None; commit_done(event) }
    fn effect_commit_done_from_state_lane1_decision(&mut self, event: &EventRun<'_>) -> Result<(), ()> { self.err = DecodeWavefrontError::None; commit_done(event) }
    fn effect_commit_done_from_state_lane2_decision(&mut self, event: &EventRun<'_>) -> Result<(), ()> { self.err = DecodeWavefrontError::None; commit_done(event) }
    fn effect_commit_done_from_state_lane3_decision(&mut self, event: &EventRun<'_>) -> Result<(), ()> { self.err = DecodeWavefrontError::None; commit_done(event) }
    fn effect_commit_done_from_state_lane4_decision(&mut self, event: &EventRun<'_>) -> Result<(), ()> { self.err = DecodeWavefrontError::None; commit_done(event) }
    fn effect_commit_done_from_state_lane5_decision(&mut self, event: &EventRun<'_>) -> Result<(), ()> { self.err = DecodeWavefrontError::None; commit_done(event) }
    fn effect_commit_done_from_state_lane6_decision(&mut self, event: &EventRun<'_>) -> Result<(), ()> { self.err = DecodeWavefrontError::None; commit_done(event) }
    fn effect_commit_done_from_state_lane7_decision(&mut self, event: &EventRun<'_>) -> Result<(), ()> { self.err = DecodeWavefrontError::None; commit_done(event) }
    fn effect_commit_done_from_state_parallel_decision(&mut self, event: &EventRun<'_>) -> Result<(), ()> { self.err = DecodeWavefrontError::None; commit_done(event) }

    fn effect_dispatch_lane_0(&mut self, event: &EventRun<'_>) -> Result<(), ()> { dispatch_lane(event, 0) }
    fn effect_dispatch_lane_1(&mut self, event: &EventRun<'_>) -> Result<(), ()> { dispatch_lane(event, 1) }
    fn effect_dispatch_lane_2(&mut self, event: &EventRun<'_>) -> Result<(), ()> { dispatch_lane(event, 2) }
    fn effect_dispatch_lane_3(&mut self, event: &EventRun<'_>) -> Result<(), ()> { dispatch_lane(event, 3) }
    fn effect_dispatch_lane_4(&mut self, event: &EventRun<'_>) -> Result<(), ()> { dispatch_lane(event, 4) }
    fn effect_dispatch_lane_5(&mut self, event: &EventRun<'_>) -> Result<(), ()> { dispatch_lane(event, 5) }
    fn effect_dispatch_lane_6(&mut self, event: &EventRun<'_>) -> Result<(), ()> { dispatch_lane(event, 6) }
    fn effect_dispatch_lane_7(&mut self, event: &EventRun<'_>) -> Result<(), ()> { dispatch_lane(event, 7) }

    fn effect_mark_lane_rejected_0_from_state_lane0_decision(&mut self, event: &EventRun<'_>) -> Result<(), ()> { self.err = DecodeWavefrontError::LaneRejected; mark_rejected(event, 0) }
    fn effect_mark_lane_rejected_0_from_state_parallel_decision(&mut self, event: &EventRun<'_>) -> Result<(), ()> { self.err = DecodeWavefrontError::LaneRejected; mark_rejected(event, 0) }
    fn effect_mark_lane_rejected_1_from_state_lane1_decision(&mut self, event: &EventRun<'_>) -> Result<(), ()> { self.err = DecodeWavefrontError::LaneRejected; mark_rejected(event, 1) }
    fn effect_mark_lane_rejected_1_from_state_parallel_decision(&mut self, event: &EventRun<'_>) -> Result<(), ()> { self.err = DecodeWavefrontError::LaneRejected; mark_rejected(event, 1) }
    fn effect_mark_lane_rejected_2_from_state_lane2_decision(&mut self, event: &EventRun<'_>) -> Result<(), ()> { self.err = DecodeWavefrontError::LaneRejected; mark_rejected(event, 2) }
    fn effect_mark_lane_rejected_2_from_state_parallel_decision(&mut self, event: &EventRun<'_>) -> Result<(), ()> { self.err = DecodeWavefrontError::LaneRejected; mark_rejected(event, 2) }
    fn effect_mark_lane_rejected_3_from_state_lane3_decision(&mut self, event: &EventRun<'_>) -> Result<(), ()> { self.err = DecodeWavefrontError::LaneRejected; mark_rejected(event, 3) }
    fn effect_mark_lane_rejected_3_from_state_parallel_decision(&mut self, event: &EventRun<'_>) -> Result<(), ()> { self.err = DecodeWavefrontError::LaneRejected; mark_rejected(event, 3) }
    fn effect_mark_lane_rejected_4_from_state_lane4_decision(&mut self, event: &EventRun<'_>) -> Result<(), ()> { self.err = DecodeWavefrontError::LaneRejected; mark_rejected(event, 4) }
    fn effect_mark_lane_rejected_4_from_state_parallel_decision(&mut self, event: &EventRun<'_>) -> Result<(), ()> { self.err = DecodeWavefrontError::LaneRejected; mark_rejected(event, 4) }
    fn effect_mark_lane_rejected_5_from_state_lane5_decision(&mut self, event: &EventRun<'_>) -> Result<(), ()> { self.err = DecodeWavefrontError::LaneRejected; mark_rejected(event, 5) }
    fn effect_mark_lane_rejected_5_from_state_parallel_decision(&mut self, event: &EventRun<'_>) -> Result<(), ()> { self.err = DecodeWavefrontError::LaneRejected; mark_rejected(event, 5) }
    fn effect_mark_lane_rejected_6_from_state_lane6_decision(&mut self, event: &EventRun<'_>) -> Result<(), ()> { self.err = DecodeWavefrontError::LaneRejected; mark_rejected(event, 6) }
    fn effect_mark_lane_rejected_6_from_state_parallel_decision(&mut self, event: &EventRun<'_>) -> Result<(), ()> { self.err = DecodeWavefrontError::LaneRejected; mark_rejected(event, 6) }
    fn effect_mark_lane_rejected_7_from_state_lane7_decision(&mut self, event: &EventRun<'_>) -> Result<(), ()> { self.err = DecodeWavefrontError::LaneRejected; mark_rejected(event, 7) }
    fn effect_mark_lane_rejected_7_from_state_parallel_decision(&mut self, event: &EventRun<'_>) -> Result<(), ()> { self.err = DecodeWavefrontError::LaneRejected; mark_rejected(event, 7) }

    fn effect_on_unexpected_from_state_group_ready(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_idle(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_lane0_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_lane1_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_lane2_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_lane3_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_lane4_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_lane5_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_lane6_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_lane7_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_parallel_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_validation_decision(&mut self) -> Result<(), ()> { self.unexpected() }

    fn guard_invalid_request(&self, event: &EventRun<'_>) -> Result<bool, ()> { Ok(!valid_count(event)) }
    fn guard_multi_lane_incompatible(&self, event: &EventRun<'_>) -> Result<bool, ()> { Ok(valid_count(event) && event.lanes.borrow().len() > 1 && !all_compatible(event)) }
    fn guard_single_lane(&self, event: &EventRun<'_>) -> Result<bool, ()> { Ok(valid_count(event) && event.lanes.borrow().len() == 1) }
    fn guard_multi_lane_compatible(&self, event: &EventRun<'_>) -> Result<bool, ()> { Ok(valid_count(event) && event.lanes.borrow().len() > 1 && all_compatible(event)) }
    fn guard_serial_dispatch(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        let count = event.lanes.borrow().len();
        Ok(self.pool.map_or(true, |pool| !pool.parallel_enabled) || count == 1 || !distinct_writers(event))
    }
    fn guard_parallel_dispatch(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        let count = event.lanes.borrow().len();
        Ok(self.pool.is_some_and(|pool| pool.parallel_enabled) && count > 1 && distinct_writers(event))
    }
    fn guard_lane_rejected_0(&self, event: &EventRun<'_>) -> Result<bool, ()> { Ok(!event.lanes.borrow()[0].accepted) }
    fn guard_lane_rejected_1(&self, event: &EventRun<'_>) -> Result<bool, ()> { Ok(!event.lanes.borrow()[1].accepted) }
    fn guard_lane_rejected_2(&self, event: &EventRun<'_>) -> Result<bool, ()> { Ok(!event.lanes.borrow()[2].accepted) }
    fn guard_lane_rejected_3(&self, event: &EventRun<'_>) -> Result<bool, ()> { Ok(!event.lanes.borrow()[3].accepted) }
    fn guard_lane_rejected_4(&self, event: &EventRun<'_>) -> Result<bool, ()> { Ok(!event.lanes.borrow()[4].accepted) }
    fn guard_lane_rejected_5(&self, event: &EventRun<'_>) -> Result<bool, ()> { Ok(!event.lanes.borrow()[5].accepted) }
    fn guard_lane_rejected_6(&self, event: &EventRun<'_>) -> Result<bool, ()> { Ok(!event.lanes.borrow()[6].accepted) }
    fn guard_lane_rejected_7(&self, event: &EventRun<'_>) -> Result<bool, ()> { Ok(!event.lanes.borrow()[7].accepted) }
    fn guard_lane_accepted_and_last_0(&self, event: &EventRun<'_>) -> Result<bool, ()> { Ok(event.lanes.borrow()[0].accepted && event.lanes.borrow().len() == 1) }
    fn guard_lane_accepted_and_last_1(&self, event: &EventRun<'_>) -> Result<bool, ()> { Ok(event.lanes.borrow()[1].accepted && event.lanes.borrow().len() == 2) }
    fn guard_lane_accepted_and_last_2(&self, event: &EventRun<'_>) -> Result<bool, ()> { Ok(event.lanes.borrow()[2].accepted && event.lanes.borrow().len() == 3) }
    fn guard_lane_accepted_and_last_3(&self, event: &EventRun<'_>) -> Result<bool, ()> { Ok(event.lanes.borrow()[3].accepted && event.lanes.borrow().len() == 4) }
    fn guard_lane_accepted_and_last_4(&self, event: &EventRun<'_>) -> Result<bool, ()> { Ok(event.lanes.borrow()[4].accepted && event.lanes.borrow().len() == 5) }
    fn guard_lane_accepted_and_last_5(&self, event: &EventRun<'_>) -> Result<bool, ()> { Ok(event.lanes.borrow()[5].accepted && event.lanes.borrow().len() == 6) }
    fn guard_lane_accepted_and_last_6(&self, event: &EventRun<'_>) -> Result<bool, ()> { Ok(event.lanes.borrow()[6].accepted && event.lanes.borrow().len() == 7) }
    fn guard_lane_accepted_and_last_7(&self, event: &EventRun<'_>) -> Result<bool, ()> { Ok(event.lanes.borrow()[7].accepted && event.lanes.borrow().len() == 8) }
    fn guard_lane_accepted_and_more_0(&self, event: &EventRun<'_>) -> Result<bool, ()> { Ok(event.lanes.borrow()[0].accepted && event.lanes.borrow().len() > 1) }
    fn guard_lane_accepted_and_more_1(&self, event: &EventRun<'_>) -> Result<bool, ()> { Ok(event.lanes.borrow()[1].accepted && event.lanes.borrow().len() > 2) }
    fn guard_lane_accepted_and_more_2(&self, event: &EventRun<'_>) -> Result<bool, ()> { Ok(event.lanes.borrow()[2].accepted && event.lanes.borrow().len() > 3) }
    fn guard_lane_accepted_and_more_3(&self, event: &EventRun<'_>) -> Result<bool, ()> { Ok(event.lanes.borrow()[3].accepted && event.lanes.borrow().len() > 4) }
    fn guard_lane_accepted_and_more_4(&self, event: &EventRun<'_>) -> Result<bool, ()> { Ok(event.lanes.borrow()[4].accepted && event.lanes.borrow().len() > 5) }
    fn guard_lane_accepted_and_more_5(&self, event: &EventRun<'_>) -> Result<bool, ()> { Ok(event.lanes.borrow()[5].accepted && event.lanes.borrow().len() > 6) }
    fn guard_lane_accepted_and_more_6(&self, event: &EventRun<'_>) -> Result<bool, ()> { Ok(event.lanes.borrow()[6].accepted && event.lanes.borrow().len() > 7) }
    fn guard_parallel_submission_failed(&self, event: &EventRun<'_>) -> Result<bool, ()> { Ok(!event.out.borrow().all_submitted) }
    fn guard_parallel_join_failed(&self, event: &EventRun<'_>) -> Result<bool, ()> { Ok(event.out.borrow().all_submitted && !event.out.borrow().joined) }
    fn guard_parallel_all_lanes_accepted(&self, event: &EventRun<'_>) -> Result<bool, ()> { Ok(event.out.borrow().all_submitted && event.out.borrow().joined && event.lanes.borrow().iter().all(|lane| lane.accepted)) }
    fn guard_parallel_lane_rejected_0(&self, event: &EventRun<'_>) -> Result<bool, ()> { Ok(parallel_lane_rejected(event, 0)) }
    fn guard_parallel_lane_rejected_1(&self, event: &EventRun<'_>) -> Result<bool, ()> { Ok(parallel_lane_rejected(event, 1)) }
    fn guard_parallel_lane_rejected_2(&self, event: &EventRun<'_>) -> Result<bool, ()> { Ok(parallel_lane_rejected(event, 2)) }
    fn guard_parallel_lane_rejected_3(&self, event: &EventRun<'_>) -> Result<bool, ()> { Ok(parallel_lane_rejected(event, 3)) }
    fn guard_parallel_lane_rejected_4(&self, event: &EventRun<'_>) -> Result<bool, ()> { Ok(parallel_lane_rejected(event, 4)) }
    fn guard_parallel_lane_rejected_5(&self, event: &EventRun<'_>) -> Result<bool, ()> { Ok(parallel_lane_rejected(event, 5)) }
    fn guard_parallel_lane_rejected_6(&self, event: &EventRun<'_>) -> Result<bool, ()> { Ok(parallel_lane_rejected(event, 6)) }
    fn guard_parallel_lane_rejected_7(&self, event: &EventRun<'_>) -> Result<bool, ()> { Ok(parallel_lane_rejected(event, 7)) }
}

fn parallel_lane_rejected(event: &EventRun<'_>, index: usize) -> bool {
    let out = event.out.borrow();
    if !out.all_submitted || !out.joined { return false; }
    let lanes = event.lanes.borrow();
    index < lanes.len() && lanes[..index].iter().all(|lane| lane.accepted) && !lanes[index].accepted
}

/// Public single-writer synchronous actor.
pub struct TextGeneratorDecodeWavefrontActor<'event> {
    machine: TextGeneratorDecodeWavefrontStateMachine<TextGeneratorDecodeWavefrontContext>,
}

impl<'event> Default for TextGeneratorDecodeWavefrontActor<'event> {
    fn default() -> Self { Self::new() }
}

impl<'event> TextGeneratorDecodeWavefrontActor<'event> {
    /// Creates an actor using serial dispatch unless a pool is supplied.
    #[must_use]
    pub fn new() -> Self {
        Self { machine: TextGeneratorDecodeWavefrontStateMachine::new(TextGeneratorDecodeWavefrontContext::default()) }
    }

    /// Creates an actor with copied synchronous pool capabilities.
    #[must_use]
    pub fn with_pool(pool: LanePool) -> Self {
        Self { machine: TextGeneratorDecodeWavefrontStateMachine::new(TextGeneratorDecodeWavefrontContext { pool: Some(pool), ..Default::default() }) }
    }

    /// Processes one complete bounded request synchronously.
    pub fn process_event(&mut self, event: EventRun<'event>) -> Result<(), DecodeWavefrontError> {
        if self.machine.process_event(TextGeneratorDecodeWavefrontEvents::EventRun(event)).is_err() {
            self.machine.context_mut().err = DecodeWavefrontError::Unexpected;
            return Err(DecodeWavefrontError::Unexpected);
        }
        let error = self.machine.context().err;
        if error == DecodeWavefrontError::None { Ok(()) } else { Err(error) }
    }

    /// Source-compatible process spelling.
    pub fn run(&mut self, event: EventRun<'event>) -> Result<(), DecodeWavefrontError> { self.process_event(event) }

    /// Dispatches the explicit unexpected-event path.
    pub fn process_unexpected(&mut self) -> Result<(), DecodeWavefrontError> {
        let _ = self.machine.context_mut().unexpected();
        self.machine.set_state(TextGeneratorDecodeWavefrontStates::StateIdle);
        Err(DecodeWavefrontError::Unexpected)
    }

    /// Returns generated state inspection data.
    #[must_use]
    pub fn state(&self) -> &TextGeneratorDecodeWavefrontStates { self.machine.state() }

    /// Reports whether the actor is in `state`.
    #[must_use]
    pub fn is(&self, state: &TextGeneratorDecodeWavefrontStates) -> bool { self.machine.is(state) }

    /// Returns the retained runtime context.
    #[must_use]
    pub fn context(&self) -> &TextGeneratorDecodeWavefrontContext { self.machine.context() }

    /// Returns the last published error.
    #[must_use]
    pub fn error(&self) -> DecodeWavefrontError { self.machine.context().err }
}

/// Short alias matching the maintained C++ actor spelling.
pub type DecodeWavefront<'event> = TextGeneratorDecodeWavefrontActor<'event>;

