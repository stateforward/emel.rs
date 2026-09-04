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
    clippy::cast_possible_truncation,
    clippy::elidable_lifetime_names,
    clippy::clone_on_copy,
    clippy::unnecessary_map_or,
    clippy::needless_lifetimes,
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
    Backend = 8,
    /// A lane callback rejected its compute event.
    LaneRejected = 4,
    /// An event arrived outside the current state-machine contract.
    Unexpected = 16,
    /// An accepted lane did not provide the selected-token handoff.
    MissingSelectedToken = 32,
    /// The selected-token handoff was explicitly present but invalid.
    InvalidSelectedToken = 64,
}

impl DecodeWavefrontError {
    /// Returns the stable numeric error code.
    #[must_use]
    pub const fn code(self) -> u8 {
        self as u8
    }
}

/// Caller-owned selected-token handoff written by the actual lane callback.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SelectedToken {
    /// Token selected by the callback.
    pub token: i32,
    /// Whether `token` contains a real selected result.
    pub valid: bool,
}

impl SelectedToken {
    /// Creates a valid caller-owned handoff.
    #[must_use]
    pub const fn new(token: i32) -> Self {
        Self { token, valid: true }
    }
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
    /// Selected token copied from the accepted callback's lane handoff.
    pub selected_token: i32,
    /// Whether `selected_token` is a real callback-produced handoff.
    pub selected_token_valid: bool,
}

/// Synchronous lane callback. The callback writes its selected token before
/// returning `true`; the boolean remains the lane's acceptance outcome.
pub type LaneCallback = fn(CompatibilityKey, &mut SelectedToken) -> bool;

/// One caller-owned lane in a bounded request.
#[derive(Clone, Copy, Debug)]
pub struct Lane {
    /// Identity of the single-writer graph actor represented by this lane.
    pub writer_id: usize,
    /// Identity of the caller-owned outcome slot.
    pub outcome_id: usize,
    /// Compatibility metadata for grouping.
    pub key: CompatibilityKey,
    /// Synchronous compute callback.
    pub callback: Option<LaneCallback>,
    /// Outcome populated by the actor.
    pub accepted: bool,
    /// Callback-owned selected-token result for this lane.
    pub selected_token: SelectedToken,
}

impl Lane {
    /// Creates a lane with initially clear outcomes.
    #[must_use]
    pub const fn new(
        writer_id: usize,
        key: CompatibilityKey,
        callback: Option<LaneCallback>,
    ) -> Self {
        Self {
            writer_id,
            outcome_id: writer_id,
            key,
            callback,
            accepted: false,
            selected_token: SelectedToken {
                token: 0,
                valid: false,
            },
        }
    }
}

/// Caller-owned run event. The wrappers make the event Copy for SML completion
/// dispatch while keeping all storage outside the actor. The selected-token
/// slot is the event-level handoff owned by the caller; lane-local slots remain
/// only for synchronous pool safety.
#[derive(Clone, Copy, Debug)]
pub struct EventRun<'event> {
    /// Bounded lane storage, owned by the caller.
    pub lanes: &'event RefCell<&'event mut [Lane]>,
    /// Caller-owned result storage.
    pub out: &'event RefCell<DispatchSummary>,
    /// Caller-owned selected-token handoff for this complete event.
    pub selected_token: &'event RefCell<SelectedToken>,
    /// Lane designated by the caller as the selection owner.
    pub selection_owner: usize,
}

impl<'event> EventRun<'event> {
    /// Creates one bounded wavefront request with an explicit selection owner.
    #[must_use]
    pub const fn new(
        lanes: &'event RefCell<&'event mut [Lane]>,
        out: &'event RefCell<DispatchSummary>,
        selected_token: &'event RefCell<SelectedToken>,
        selection_owner: usize,
    ) -> Self {
        Self {
            lanes,
            out,
            selected_token,
            selection_owner,
        }
    }
}

fn commit_done(
    context: &mut TextGeneratorDecodeWavefrontContext,
    event: &EventRun<'_>,
) -> Result<(), ()> {
    let lanes = event.lanes.borrow();
    let Some(owner) = lanes.get(event.selection_owner) else {
        context.err = DecodeWavefrontError::InvalidSelectedToken;
        let mut out = event.out.borrow_mut();
        out.err = context.err;
        out.selected_token_valid = false;
        return Ok(());
    };
    if !owner.accepted {
        context.err = DecodeWavefrontError::LaneRejected;
        let mut out = event.out.borrow_mut();
        out.err = context.err;
        out.selected_token_valid = false;
        return Ok(());
    }
    if !owner.selected_token.valid {
        context.err = DecodeWavefrontError::MissingSelectedToken;
        let mut out = event.out.borrow_mut();
        out.err = context.err;
        out.selected_token_valid = false;
        return Ok(());
    }
    if owner.selected_token.token < 0 {
        context.err = DecodeWavefrontError::InvalidSelectedToken;
        let mut out = event.out.borrow_mut();
        out.err = context.err;
        out.selected_token_valid = false;
        return Ok(());
    }
    let selected = owner.selected_token;
    *event.selected_token.borrow_mut() = selected;
    context.err = DecodeWavefrontError::None;
    let mut out = event.out.borrow_mut();
    out.err = DecodeWavefrontError::None;
    out.failed_lane = NO_FAILED_LANE;
    out.selected_token = selected.token;
    out.selected_token_valid = true;
    Ok(())
}

/// Synchronous submission callback supplied by a caller-owned lane pool.
///
/// A successful submission must arrange for `accepted` and `selected_token`
/// to be written before the enclosing join returns.
pub type LaneSubmit = fn(
    context: *mut (),
    lane_index: usize,
    key: CompatibilityKey,
    callback: Option<LaneCallback>,
    accepted: &mut bool,
    selected_token: &mut SelectedToken,
) -> bool;

/// Synchronous join callback supplied by a caller-owned lane pool.
pub type LaneJoin = fn(context: *mut ()) -> bool;

/// Caller-owned scheduler capabilities. Callbacks still execute synchronously
#[allow(unpredictable_function_pointer_comparisons)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct LanePool {
    /// Opaque caller-owned pool state.
    pub context: *mut (),
    /// Submits one bounded lane without retaining borrowed inputs.
    pub submit: Option<LaneSubmit>,
    /// Joins all lanes submitted for the current dispatch.
    pub join: Option<LaneJoin>,
}

impl LanePool {
    /// Creates an inline synchronous pool capability.
    #[must_use]
    pub const fn new() -> Self {
        Self::inline()
    }

    /// Creates an inline synchronous pool capability.
    #[must_use]
    pub const fn inline() -> Self {
        Self {
            context: core::ptr::null_mut(),
            submit: Some(inline_lane_submit),
            join: Some(inline_lane_join),
        }
    }

    /// Creates a pool backed by caller-owned synchronous callbacks.
    #[must_use]
    pub const fn with_callbacks(context: *mut (), submit: LaneSubmit, join: LaneJoin) -> Self {
        Self {
            context,
            submit: Some(submit),
            join: Some(join),
        }
    }

    fn submit_lane(
        self,
        lane_index: usize,
        key: CompatibilityKey,
        callback: Option<LaneCallback>,
        accepted: &mut bool,
        selected_token: &mut SelectedToken,
    ) -> bool {
        self.submit.is_some_and(|submit| {
            submit(
                self.context,
                lane_index,
                key,
                callback,
                accepted,
                selected_token,
            )
        })
    }

    fn join(self) -> bool {
        self.join.is_some_and(|join| join(self.context))
    }
}

fn inline_lane_submit(
    _context: *mut (),
    _lane_index: usize,
    key: CompatibilityKey,
    callback: Option<LaneCallback>,
    accepted: &mut bool,
    selected_token: &mut SelectedToken,
) -> bool {
    *selected_token = SelectedToken::default();
    *accepted = callback.is_some_and(|callback| callback(key, selected_token));
    true
}

fn inline_lane_join(_context: *mut ()) -> bool {
    true
}

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
        // The pinned action::effect_on_unexpected publishes the backend bit;
        // retain the Rust-only flag so callers can still inspect the path.
        self.err = DecodeWavefrontError::Backend;
        self.unexpected = true;
        Ok(())
    }
}

fn compatible(lhs: CompatibilityKey, rhs: CompatibilityKey) -> bool {
    lhs == rhs
}

fn valid_count(event: &EventRun<'_>) -> bool {
    let count = event.lanes.borrow().len();
    count > 0 && count <= MAX_LANES
}

fn all_compatible(event: &EventRun<'_>) -> bool {
    let lanes = event.lanes.borrow();
    if lanes.is_empty() || lanes.len() > MAX_LANES {
        return false;
    }
    let first = lanes[0].key;
    lanes[1..].iter().all(|lane| compatible(first, lane.key))
}

fn distinct_writers(event: &EventRun<'_>) -> bool {
    let lanes = event.lanes.borrow();
    for (index, lane) in lanes.iter().enumerate() {
        if lane.writer_id == 0
            || lanes[index + 1..]
                .iter()
                .any(|other| other.writer_id == lane.writer_id)
        {
            return false;
        }
    }
    true
}

fn distinct_outcomes(event: &EventRun<'_>) -> bool {
    let lanes = event.lanes.borrow();
    for (index, lane) in lanes.iter().enumerate() {
        if lanes[index + 1..]
            .iter()
            .any(|other| other.outcome_id == lane.outcome_id)
        {
            return false;
        }
    }
    true
}

fn dispatch_lane(event: &EventRun<'_>, index: usize) -> Result<(), ()> {
    let mut lanes = event.lanes.borrow_mut();
    if index >= lanes.len() {
        return Err(());
    }
    let lane = &mut lanes[index];
    lane.selected_token = SelectedToken::default();
    lane.accepted = lane
        .callback
        .is_some_and(|callback| callback(lane.key, &mut lane.selected_token));
    event.out.borrow_mut().dispatched_lanes =
        i32::try_from(index + 1).expect("bounded lane index fits in i32");
    Ok(())
}

fn mark_rejected(event: &EventRun<'_>, index: usize) -> Result<(), ()> {
    let mut out = event.out.borrow_mut();
    out.err = DecodeWavefrontError::LaneRejected;
    out.failed_lane = i32::try_from(index).expect("bounded lane index fits in i32");
    Ok(())
}

impl TextGeneratorDecodeWavefrontStateMachineContext for TextGeneratorDecodeWavefrontContext {
    fn effect_begin_run(&mut self, event: &EventRun<'_>) -> Result<(), ()> {
        self.err = DecodeWavefrontError::None;
        self.unexpected = false;
        *event.selected_token.borrow_mut() = SelectedToken::default();
        *event.out.borrow_mut() = DispatchSummary {
            err: DecodeWavefrontError::None,
            failed_lane: NO_FAILED_LANE,
            ..DispatchSummary::default()
        };
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
    fn effect_dispatch_parallel_lanes(&mut self, event: &EventRun<'_>) -> Result<(), ()> {
        let pool = self.pool.unwrap_or_default();
        let lane_count;
        let mut all_submitted = true;
        {
            let mut lanes = event.lanes.borrow_mut();
            lane_count = lanes.len();
            for (lane_index, lane) in lanes.iter_mut().enumerate() {
                lane.accepted = false;
                lane.selected_token = SelectedToken::default();
                let submitted = pool.submit_lane(
                    lane_index,
                    lane.key,
                    lane.callback,
                    &mut lane.accepted,
                    &mut lane.selected_token,
                );
                all_submitted &= submitted;
            }
        }
        let mut out = event.out.borrow_mut();
        out.all_submitted = all_submitted;
        out.joined = pool.join();
        out.dispatched_lanes = i32::try_from(lane_count).expect("bounded lane count fits in i32");
        Ok(())
    }
    fn effect_commit_done_from_state_lane0_decision(
        &mut self,
        event: &EventRun<'_>,
    ) -> Result<(), ()> {
        commit_done(self, event)
    }
    fn effect_commit_done_from_state_lane1_decision(
        &mut self,
        event: &EventRun<'_>,
    ) -> Result<(), ()> {
        commit_done(self, event)
    }
    fn effect_commit_done_from_state_lane2_decision(
        &mut self,
        event: &EventRun<'_>,
    ) -> Result<(), ()> {
        commit_done(self, event)
    }
    fn effect_commit_done_from_state_lane3_decision(
        &mut self,
        event: &EventRun<'_>,
    ) -> Result<(), ()> {
        commit_done(self, event)
    }
    fn effect_commit_done_from_state_lane4_decision(
        &mut self,
        event: &EventRun<'_>,
    ) -> Result<(), ()> {
        commit_done(self, event)
    }
    fn effect_commit_done_from_state_lane5_decision(
        &mut self,
        event: &EventRun<'_>,
    ) -> Result<(), ()> {
        commit_done(self, event)
    }
    fn effect_commit_done_from_state_lane6_decision(
        &mut self,
        event: &EventRun<'_>,
    ) -> Result<(), ()> {
        commit_done(self, event)
    }
    fn effect_commit_done_from_state_lane7_decision(
        &mut self,
        event: &EventRun<'_>,
    ) -> Result<(), ()> {
        commit_done(self, event)
    }
    fn effect_commit_done_from_state_parallel_decision(
        &mut self,
        event: &EventRun<'_>,
    ) -> Result<(), ()> {
        commit_done(self, event)
    }

    fn effect_dispatch_lane_0(&mut self, event: &EventRun<'_>) -> Result<(), ()> {
        dispatch_lane(event, 0)
    }
    fn effect_dispatch_lane_1(&mut self, event: &EventRun<'_>) -> Result<(), ()> {
        dispatch_lane(event, 1)
    }
    fn effect_dispatch_lane_2(&mut self, event: &EventRun<'_>) -> Result<(), ()> {
        dispatch_lane(event, 2)
    }
    fn effect_dispatch_lane_3(&mut self, event: &EventRun<'_>) -> Result<(), ()> {
        dispatch_lane(event, 3)
    }
    fn effect_dispatch_lane_4(&mut self, event: &EventRun<'_>) -> Result<(), ()> {
        dispatch_lane(event, 4)
    }
    fn effect_dispatch_lane_5(&mut self, event: &EventRun<'_>) -> Result<(), ()> {
        dispatch_lane(event, 5)
    }
    fn effect_dispatch_lane_6(&mut self, event: &EventRun<'_>) -> Result<(), ()> {
        dispatch_lane(event, 6)
    }
    fn effect_dispatch_lane_7(&mut self, event: &EventRun<'_>) -> Result<(), ()> {
        dispatch_lane(event, 7)
    }

    fn effect_mark_lane_rejected_0_from_state_lane0_decision(
        &mut self,
        event: &EventRun<'_>,
    ) -> Result<(), ()> {
        self.err = DecodeWavefrontError::LaneRejected;
        mark_rejected(event, 0)
    }
    fn effect_mark_lane_rejected_0_from_state_parallel_decision(
        &mut self,
        event: &EventRun<'_>,
    ) -> Result<(), ()> {
        self.err = DecodeWavefrontError::LaneRejected;
        mark_rejected(event, 0)
    }
    fn effect_mark_lane_rejected_1_from_state_lane1_decision(
        &mut self,
        event: &EventRun<'_>,
    ) -> Result<(), ()> {
        self.err = DecodeWavefrontError::LaneRejected;
        mark_rejected(event, 1)
    }
    fn effect_mark_lane_rejected_1_from_state_parallel_decision(
        &mut self,
        event: &EventRun<'_>,
    ) -> Result<(), ()> {
        self.err = DecodeWavefrontError::LaneRejected;
        mark_rejected(event, 1)
    }
    fn effect_mark_lane_rejected_2_from_state_lane2_decision(
        &mut self,
        event: &EventRun<'_>,
    ) -> Result<(), ()> {
        self.err = DecodeWavefrontError::LaneRejected;
        mark_rejected(event, 2)
    }
    fn effect_mark_lane_rejected_2_from_state_parallel_decision(
        &mut self,
        event: &EventRun<'_>,
    ) -> Result<(), ()> {
        self.err = DecodeWavefrontError::LaneRejected;
        mark_rejected(event, 2)
    }
    fn effect_mark_lane_rejected_3_from_state_lane3_decision(
        &mut self,
        event: &EventRun<'_>,
    ) -> Result<(), ()> {
        self.err = DecodeWavefrontError::LaneRejected;
        mark_rejected(event, 3)
    }
    fn effect_mark_lane_rejected_3_from_state_parallel_decision(
        &mut self,
        event: &EventRun<'_>,
    ) -> Result<(), ()> {
        self.err = DecodeWavefrontError::LaneRejected;
        mark_rejected(event, 3)
    }
    fn effect_mark_lane_rejected_4_from_state_lane4_decision(
        &mut self,
        event: &EventRun<'_>,
    ) -> Result<(), ()> {
        self.err = DecodeWavefrontError::LaneRejected;
        mark_rejected(event, 4)
    }
    fn effect_mark_lane_rejected_4_from_state_parallel_decision(
        &mut self,
        event: &EventRun<'_>,
    ) -> Result<(), ()> {
        self.err = DecodeWavefrontError::LaneRejected;
        mark_rejected(event, 4)
    }
    fn effect_mark_lane_rejected_5_from_state_lane5_decision(
        &mut self,
        event: &EventRun<'_>,
    ) -> Result<(), ()> {
        self.err = DecodeWavefrontError::LaneRejected;
        mark_rejected(event, 5)
    }
    fn effect_mark_lane_rejected_5_from_state_parallel_decision(
        &mut self,
        event: &EventRun<'_>,
    ) -> Result<(), ()> {
        self.err = DecodeWavefrontError::LaneRejected;
        mark_rejected(event, 5)
    }
    fn effect_mark_lane_rejected_6_from_state_lane6_decision(
        &mut self,
        event: &EventRun<'_>,
    ) -> Result<(), ()> {
        self.err = DecodeWavefrontError::LaneRejected;
        mark_rejected(event, 6)
    }
    fn effect_mark_lane_rejected_6_from_state_parallel_decision(
        &mut self,
        event: &EventRun<'_>,
    ) -> Result<(), ()> {
        self.err = DecodeWavefrontError::LaneRejected;
        mark_rejected(event, 6)
    }
    fn effect_mark_lane_rejected_7_from_state_lane7_decision(
        &mut self,
        event: &EventRun<'_>,
    ) -> Result<(), ()> {
        self.err = DecodeWavefrontError::LaneRejected;
        mark_rejected(event, 7)
    }
    fn effect_mark_lane_rejected_7_from_state_parallel_decision(
        &mut self,
        event: &EventRun<'_>,
    ) -> Result<(), ()> {
        self.err = DecodeWavefrontError::LaneRejected;
        mark_rejected(event, 7)
    }
    fn effect_reject_parallel_scheduler_from_state_parallel_decision(
        &mut self,
        event: &EventRun<'_>,
    ) -> Result<(), ()> {
        self.err = DecodeWavefrontError::Backend;
        event.out.borrow_mut().err = self.err;
        Ok(())
    }

    fn effect_on_unexpected_from_state_group_ready(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_idle(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_lane0_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_lane1_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_lane2_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_lane3_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_lane4_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_lane5_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_lane6_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_lane7_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_parallel_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn effect_on_unexpected_from_state_validation_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }

    fn guard_invalid_request(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        Ok(!valid_count(event))
    }
    fn guard_multi_lane_incompatible(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        Ok(valid_count(event) && event.lanes.borrow().len() > 1 && !all_compatible(event))
    }
    fn guard_multi_lane_compatible(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        Ok(valid_count(event) && event.lanes.borrow().len() > 1 && all_compatible(event))
    }
    fn guard_single_lane(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        Ok(valid_count(event) && event.lanes.borrow().len() == 1)
    }
    fn guard_serial_dispatch(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        let count = event.lanes.borrow().len();
        Ok(self.pool.is_none()
            || count == 1
            || !distinct_writers(event)
            || !distinct_outcomes(event))
    }
    fn guard_parallel_dispatch(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        let count = event.lanes.borrow().len();
        Ok(self.pool.is_some() && count > 1 && distinct_writers(event) && distinct_outcomes(event))
    }
    fn guard_lane_rejected_0(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        Ok(!event.lanes.borrow()[0].accepted)
    }
    fn guard_lane_rejected_1(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        Ok(!event.lanes.borrow()[1].accepted)
    }
    fn guard_lane_rejected_2(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        Ok(!event.lanes.borrow()[2].accepted)
    }
    fn guard_lane_rejected_3(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        Ok(!event.lanes.borrow()[3].accepted)
    }
    fn guard_lane_rejected_4(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        Ok(!event.lanes.borrow()[4].accepted)
    }
    fn guard_lane_rejected_5(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        Ok(!event.lanes.borrow()[5].accepted)
    }
    fn guard_lane_rejected_6(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        Ok(!event.lanes.borrow()[6].accepted)
    }
    fn guard_lane_rejected_7(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        Ok(!event.lanes.borrow()[7].accepted)
    }
    fn guard_lane_accepted_and_last_0(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        Ok(event.lanes.borrow()[0].accepted && event.lanes.borrow().len() == 1)
    }
    fn guard_lane_accepted_and_last_1(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        Ok(event.lanes.borrow()[1].accepted && event.lanes.borrow().len() == 2)
    }
    fn guard_lane_accepted_and_last_2(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        Ok(event.lanes.borrow()[2].accepted && event.lanes.borrow().len() == 3)
    }
    fn guard_lane_accepted_and_last_3(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        Ok(event.lanes.borrow()[3].accepted && event.lanes.borrow().len() == 4)
    }
    fn guard_lane_accepted_and_last_4(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        Ok(event.lanes.borrow()[4].accepted && event.lanes.borrow().len() == 5)
    }
    fn guard_lane_accepted_and_last_5(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        Ok(event.lanes.borrow()[5].accepted && event.lanes.borrow().len() == 6)
    }
    fn guard_lane_accepted_and_last_6(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        Ok(event.lanes.borrow()[6].accepted && event.lanes.borrow().len() == 7)
    }
    fn guard_lane_accepted_and_last_7(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        Ok(event.lanes.borrow()[7].accepted && event.lanes.borrow().len() == 8)
    }
    fn guard_lane_accepted_and_more_0(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        Ok(event.lanes.borrow()[0].accepted && event.lanes.borrow().len() > 1)
    }
    fn guard_lane_accepted_and_more_1(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        Ok(event.lanes.borrow()[1].accepted && event.lanes.borrow().len() > 2)
    }
    fn guard_lane_accepted_and_more_2(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        Ok(event.lanes.borrow()[2].accepted && event.lanes.borrow().len() > 3)
    }
    fn guard_lane_accepted_and_more_3(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        Ok(event.lanes.borrow()[3].accepted && event.lanes.borrow().len() > 4)
    }
    fn guard_lane_accepted_and_more_4(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        Ok(event.lanes.borrow()[4].accepted && event.lanes.borrow().len() > 5)
    }
    fn guard_lane_accepted_and_more_5(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        Ok(event.lanes.borrow()[5].accepted && event.lanes.borrow().len() > 6)
    }
    fn guard_lane_accepted_and_more_6(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        Ok(event.lanes.borrow()[6].accepted && event.lanes.borrow().len() > 7)
    }
    fn guard_parallel_submission_failed(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        Ok(!event.out.borrow().all_submitted)
    }
    fn guard_parallel_join_failed(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        Ok(event.out.borrow().all_submitted && !event.out.borrow().joined)
    }
    fn guard_parallel_all_lanes_accepted(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        Ok(event.out.borrow().all_submitted
            && event.out.borrow().joined
            && event.lanes.borrow().iter().all(|lane| lane.accepted))
    }
    fn guard_parallel_lane_rejected_0(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        Ok(parallel_lane_rejected(event, 0))
    }
    fn guard_parallel_lane_rejected_1(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        Ok(parallel_lane_rejected(event, 1))
    }
    fn guard_parallel_lane_rejected_2(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        Ok(parallel_lane_rejected(event, 2))
    }
    fn guard_parallel_lane_rejected_3(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        Ok(parallel_lane_rejected(event, 3))
    }
    fn guard_parallel_lane_rejected_4(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        Ok(parallel_lane_rejected(event, 4))
    }
    fn guard_parallel_lane_rejected_5(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        Ok(parallel_lane_rejected(event, 5))
    }
    fn guard_parallel_lane_rejected_6(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        Ok(parallel_lane_rejected(event, 6))
    }
    fn guard_parallel_lane_rejected_7(&self, event: &EventRun<'_>) -> Result<bool, ()> {
        Ok(parallel_lane_rejected(event, 7))
    }
}

fn parallel_lane_rejected(event: &EventRun<'_>, index: usize) -> bool {
    let out = event.out.borrow();
    if !out.all_submitted || !out.joined {
        return false;
    }
    let lanes = event.lanes.borrow();
    index < lanes.len() && lanes[..index].iter().all(|lane| lane.accepted) && !lanes[index].accepted
}

/// Public single-writer synchronous actor.
pub struct TextGeneratorDecodeWavefrontActor {
    machine: TextGeneratorDecodeWavefrontStateMachine<TextGeneratorDecodeWavefrontContext>,
}

impl Default for TextGeneratorDecodeWavefrontActor {
    fn default() -> Self {
        Self::new()
    }
}

impl TextGeneratorDecodeWavefrontActor {
    /// Creates an actor using serial dispatch unless a pool is supplied.
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: TextGeneratorDecodeWavefrontStateMachine::new(
                TextGeneratorDecodeWavefrontContext::default(),
            ),
        }
    }

    /// Creates an actor with copied synchronous pool capabilities.
    #[must_use]
    pub fn with_pool(pool: LanePool) -> Self {
        Self {
            machine: TextGeneratorDecodeWavefrontStateMachine::new(
                TextGeneratorDecodeWavefrontContext {
                    pool: Some(pool),
                    ..Default::default()
                },
            ),
        }
    }

    /// Processes one complete bounded request synchronously.
    pub fn process_event(&mut self, event: EventRun<'_>) -> Result<(), DecodeWavefrontError> {
        if self
            .machine
            .process_event(TextGeneratorDecodeWavefrontEvents::EventRun(event))
            .is_err()
        {
            self.machine.context_mut().err = DecodeWavefrontError::Unexpected;
            return Err(DecodeWavefrontError::Unexpected);
        }
        let error = self.machine.context().err;
        if error == DecodeWavefrontError::None {
            Ok(())
        } else {
            Err(error)
        }
    }

    /// Source-compatible process spelling.
    pub fn run(&mut self, event: EventRun<'_>) -> Result<(), DecodeWavefrontError> {
        self.process_event(event)
    }

    /// Dispatches the explicit unexpected-event path.
    pub fn process_unexpected(&mut self) -> Result<(), DecodeWavefrontError> {
        let _ = self.machine.context_mut().unexpected();
        self.machine.context_mut().err = DecodeWavefrontError::Unexpected;
        self.machine
            .set_state(TextGeneratorDecodeWavefrontStates::StateIdle);
        Err(DecodeWavefrontError::Unexpected)
    }
    /// Returns the generated machine state.
    #[must_use]
    pub fn state(&self) -> &TextGeneratorDecodeWavefrontStates {
        self.machine.state()
    }

    /// Source-compatible unexpected-event spelling.
    pub fn process_unexpected_event(&mut self) -> Result<(), DecodeWavefrontError> {
        self.process_unexpected()
    }

    #[must_use]
    pub fn is(&self, state: &TextGeneratorDecodeWavefrontStates) -> bool {
        self.machine.is(state)
    }
    /// Returns the retained runtime context.
    #[must_use]
    pub fn context(&self) -> &TextGeneratorDecodeWavefrontContext {
        self.machine.context()
    }

    /// Returns the last published error.
    #[must_use]
    pub fn error(&self) -> DecodeWavefrontError {
        self.machine.context().err
    }
}

/// Short alias matching the maintained C++ actor spelling.
pub type DecodeWavefront = TextGeneratorDecodeWavefrontActor;
