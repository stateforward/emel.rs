#![allow(clippy::field_reassign_with_default)]

use core::cell::RefCell;

use super::sm::{
    CompatibilityKey, DecodeWavefrontError, DispatchSummary, EventRun, Lane, LanePool, MAX_LANES,
    NO_FAILED_LANE, SelectedToken, TextGeneratorDecodeWavefrontActor,
};

fn accept(_: CompatibilityKey, selected: &mut SelectedToken) -> bool {
    *selected = SelectedToken::new(42);
    true
}
fn reject(_: CompatibilityKey, _selected: &mut SelectedToken) -> bool {
    false
}
fn submit_fail(
    _context: *mut (),
    _lane_index: usize,
    _key: CompatibilityKey,
    _callback: Option<fn(CompatibilityKey, &mut SelectedToken) -> bool>,
    _accepted: &mut bool,
    _selected_token: &mut SelectedToken,
) -> bool {
    false
}

fn submit_inline(
    _context: *mut (),
    _lane_index: usize,
    key: CompatibilityKey,
    callback: Option<fn(CompatibilityKey, &mut SelectedToken) -> bool>,
    accepted: &mut bool,
    selected_token: &mut SelectedToken,
) -> bool {
    *selected_token = SelectedToken::default();
    *accepted = callback.is_some_and(|callback| callback(key, selected_token));
    true
}

fn join_fail(_context: *mut ()) -> bool {
    false
}
static SUBMISSIONS: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);
static CALLBACKS: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);
static JOINS: core::sync::atomic::AtomicUsize = core::sync::atomic::AtomicUsize::new(0);

fn submit_observing(
    _context: *mut (),
    _lane_index: usize,
    key: CompatibilityKey,
    callback: Option<fn(CompatibilityKey, &mut SelectedToken) -> bool>,
    accepted: &mut bool,
    selected_token: &mut SelectedToken,
) -> bool {
    SUBMISSIONS.fetch_add(1, core::sync::atomic::Ordering::SeqCst);
    *selected_token = SelectedToken::default();
    *accepted = callback.is_some_and(|callback| callback(key, selected_token));
    true
}

fn join_observing(_context: *mut ()) -> bool {
    JOINS.fetch_add(1, core::sync::atomic::Ordering::SeqCst);
    true
}

fn accept_observing(_: CompatibilityKey, selected: &mut SelectedToken) -> bool {
    CALLBACKS.fetch_add(1, core::sync::atomic::Ordering::SeqCst);
    *selected = SelectedToken::new(42);
    true
}

fn missing_selected(_: CompatibilityKey, _selected: &mut SelectedToken) -> bool {
    true
}

fn invalid_selected(_: CompatibilityKey, selected: &mut SelectedToken) -> bool {
    *selected = SelectedToken {
        token: -1,
        valid: true,
    };
    true
}

#[test]
fn propagates_selected_token_synchronously() {
    let key = CompatibilityKey::default();
    let mut lanes = [Lane::new(1, key, Some(accept))];
    let summary = RefCell::new(DispatchSummary::default());
    let mut actor = TextGeneratorDecodeWavefrontActor::new();

    assert_eq!(run(&mut actor, &mut lanes, &summary), Ok(()));
    let result = summary.borrow();
    assert_eq!(result.selected_token, 42);
    assert!(result.selected_token_valid);
    assert_eq!(lanes[0].selected_token, SelectedToken::new(42));
}

#[test]
fn rejects_missing_or_invalid_selected_token_handoff() {
    let key = CompatibilityKey::default();
    let mut lanes = [Lane::new(1, key, Some(missing_selected))];
    let summary = RefCell::new(DispatchSummary::default());
    let mut actor = TextGeneratorDecodeWavefrontActor::new();
    assert_eq!(
        run(&mut actor, &mut lanes, &summary),
        Err(DecodeWavefrontError::MissingSelectedToken)
    );
    assert!(!summary.borrow().selected_token_valid);

    let mut lanes = [Lane::new(1, key, Some(invalid_selected))];
    let summary = RefCell::new(DispatchSummary::default());
    assert_eq!(
        run(&mut actor, &mut lanes, &summary),
        Err(DecodeWavefrontError::InvalidSelectedToken)
    );
    assert!(!summary.borrow().selected_token_valid);
}

fn run<'a>(
    actor: &mut TextGeneratorDecodeWavefrontActor,
    lanes: &'a mut [Lane],
    summary: &'a RefCell<DispatchSummary>,
) -> Result<(), DecodeWavefrontError> {
    let lane_storage = RefCell::new(lanes);
    let selected_token = RefCell::new(SelectedToken::default());
    actor.process_event(EventRun::new(&lane_storage, summary, &selected_token, 0))
}

fn run_with_owner<'a>(
    actor: &mut TextGeneratorDecodeWavefrontActor,
    lanes: &'a mut [Lane],
    summary: &'a RefCell<DispatchSummary>,
    selection_owner: usize,
) -> (Result<(), DecodeWavefrontError>, SelectedToken) {
    let lane_storage = RefCell::new(lanes);
    let selected_token = RefCell::new(SelectedToken::default());
    let result = actor.process_event(EventRun::new(
        &lane_storage,
        summary,
        &selected_token,
        selection_owner,
    ));
    (result, *selected_token.borrow())
}

fn accept_7(_: CompatibilityKey, selected: &mut SelectedToken) -> bool {
    *selected = SelectedToken::new(7);
    true
}

#[test]
fn accepts_distinct_lane_tokens_from_explicit_selection_owner() {
    let key = CompatibilityKey::default();
    let mut lanes = [
        Lane::new(1, key, Some(accept)),
        Lane::new(2, key, Some(accept_7)),
    ];
    let summary = RefCell::new(DispatchSummary::default());
    let mut actor = TextGeneratorDecodeWavefrontActor::with_pool(LanePool::new());
    let (result, selected) = run_with_owner(&mut actor, &mut lanes, &summary, 1);
    assert_eq!(result, Ok(()));
    assert_eq!(selected, SelectedToken::new(7));
    assert_eq!(summary.borrow().selected_token, 7);
}

#[test]
fn rejects_invalid_selection_owner_handoffs_without_publication() {
    let key = CompatibilityKey::default();
    let mut lanes = [Lane::new(1, key, Some(accept))];
    let summary = RefCell::new(DispatchSummary::default());
    let mut actor = TextGeneratorDecodeWavefrontActor::new();
    let (result, selected) = run_with_owner(&mut actor, &mut lanes, &summary, 1);
    assert_eq!(result, Err(DecodeWavefrontError::InvalidSelectedToken));
    assert_eq!(selected, SelectedToken::default());
    assert!(!summary.borrow().selected_token_valid);

    let mut lanes = [Lane::new(1, key, Some(invalid_selected))];
    let summary = RefCell::new(DispatchSummary::default());
    let (result, selected) = run_with_owner(&mut actor, &mut lanes, &summary, 0);
    assert_eq!(result, Err(DecodeWavefrontError::InvalidSelectedToken));
    assert_eq!(selected, SelectedToken::default());
    assert!(!summary.borrow().selected_token_valid);
}

#[test]
fn rejects_empty_and_over_bound_requests_without_dispatch() {
    let mut actor = TextGeneratorDecodeWavefrontActor::new();
    let summary = RefCell::new(DispatchSummary::default());
    let mut empty: [Lane; 0] = [];
    assert_eq!(
        run(&mut actor, &mut empty, &summary),
        Err(DecodeWavefrontError::InvalidRequest)
    );
    assert_eq!(summary.borrow().dispatched_lanes, 0);

    let key = CompatibilityKey::default();
    let mut lanes = [Lane::new(1, key, Some(accept)); MAX_LANES + 1];
    assert_eq!(
        run(&mut actor, &mut lanes, &summary),
        Err(DecodeWavefrontError::InvalidRequest)
    );
    assert_eq!(summary.borrow().dispatched_lanes, 0);
    assert!(lanes.iter().all(|lane| !lane.accepted));
}

#[test]
fn rejects_incompatible_lanes_before_callbacks() {
    let mut actor = TextGeneratorDecodeWavefrontActor::new();
    let summary = RefCell::new(DispatchSummary::default());
    let first = CompatibilityKey {
        model_identity: 1,
        ..CompatibilityKey::default()
    };
    let second = CompatibilityKey {
        model_identity: 2,
        ..first
    };
    let mut lanes = [
        Lane::new(1, first, Some(accept)),
        Lane::new(2, second, Some(accept)),
    ];

    assert_eq!(
        run(&mut actor, &mut lanes, &summary),
        Err(DecodeWavefrontError::IncompatibleLanes)
    );
    let result = summary.borrow();
    assert!(!result.grouped);
    assert_eq!(result.dispatched_lanes, 0);
    assert_eq!(result.failed_lane, NO_FAILED_LANE);
    assert!(lanes.iter().all(|lane| !lane.accepted));
}

#[test]
fn dispatches_compatible_lanes_successfully() {
    let key = CompatibilityKey::default();
    let mut lanes = [
        Lane::new(1, key, Some(accept)),
        Lane::new(2, key, Some(accept)),
    ];
    let summary = RefCell::new(DispatchSummary::default());
    let mut actor = TextGeneratorDecodeWavefrontActor::with_pool(LanePool::new());

    assert_eq!(run(&mut actor, &mut lanes, &summary), Ok(()));
    let result = summary.borrow();
    assert!(result.grouped);
    assert!(result.all_submitted);
    assert!(result.joined);
    assert_eq!(result.dispatched_lanes, 2);
    assert_eq!(result.failed_lane, NO_FAILED_LANE);
    assert!(lanes.iter().all(|lane| lane.accepted));
}

#[test]
fn reports_lane_failure_and_stops_serial_dispatch() {
    let key = CompatibilityKey::default();
    let mut lanes = [
        Lane::new(1, key, Some(accept)),
        Lane::new(2, key, Some(reject)),
        Lane::new(3, key, Some(accept)),
    ];
    let summary = RefCell::new(DispatchSummary::default());
    let mut actor = TextGeneratorDecodeWavefrontActor::new();

    assert_eq!(
        run(&mut actor, &mut lanes, &summary),
        Err(DecodeWavefrontError::LaneRejected)
    );
    let result = summary.borrow();
    assert!(result.grouped);
    assert_eq!(result.dispatched_lanes, 2);
    assert_eq!(result.failed_lane, 1);
    assert!(lanes[0].accepted);
    assert!(!lanes[1].accepted);
    assert!(!lanes[2].accepted);
}

#[test]
fn treats_missing_lane_callback_as_rejection() {
    let key = CompatibilityKey::default();
    let mut lanes = [Lane::new(1, key, None)];
    let summary = RefCell::new(DispatchSummary::default());
    let mut actor = TextGeneratorDecodeWavefrontActor::new();

    assert_eq!(
        run(&mut actor, &mut lanes, &summary),
        Err(DecodeWavefrontError::LaneRejected)
    );
    let result = summary.borrow();
    assert_eq!(result.dispatched_lanes, 1);
    assert_eq!(result.failed_lane, 0);
    assert!(!lanes[0].accepted);
}

#[test]
fn routes_explicitly_aliased_outcomes_serially_with_pool() {
    let key = CompatibilityKey::default();
    let mut lanes = [
        Lane::new(1, key, Some(accept)),
        Lane::new(2, key, Some(reject)),
    ];
    lanes[1].outcome_id = lanes[0].outcome_id;
    let summary = RefCell::new(DispatchSummary::default());
    let mut actor = TextGeneratorDecodeWavefrontActor::with_pool(LanePool::new());

    assert_eq!(
        run(&mut actor, &mut lanes, &summary),
        Err(DecodeWavefrontError::LaneRejected)
    );
    let result = summary.borrow();
    assert_eq!(result.dispatched_lanes, 2);
    assert!(!result.all_submitted);
    assert!(!result.joined);
}
#[test]
fn routes_duplicate_writer_lanes_serially_as_slot_safety_fallback() {
    let key = CompatibilityKey::default();
    let mut lanes = [
        Lane::new(7, key, Some(accept)),
        Lane::new(7, key, Some(reject)),
    ];
    let summary = RefCell::new(DispatchSummary::default());
    let mut actor = TextGeneratorDecodeWavefrontActor::with_pool(LanePool::new());

    assert_eq!(
        run(&mut actor, &mut lanes, &summary),
        Err(DecodeWavefrontError::LaneRejected)
    );
    let result = summary.borrow();
    assert!(result.grouped);
    assert_eq!(result.dispatched_lanes, 2);
    assert_eq!(result.failed_lane, 1);
}

#[test]
fn reports_pool_submission_and_join_failures() {
    let key = CompatibilityKey::default();
    let mut lanes = [
        Lane::new(1, key, Some(accept)),
        Lane::new(2, key, Some(accept)),
    ];
    let summary = RefCell::new(DispatchSummary::default());
    let mut actor = TextGeneratorDecodeWavefrontActor::with_pool(LanePool::with_callbacks(
        core::ptr::null_mut(),
        submit_fail,
        join_fail,
    ));

    assert_eq!(
        run(&mut actor, &mut lanes, &summary),
        Err(DecodeWavefrontError::Backend)
    );
    assert!(!summary.borrow().all_submitted);

    let mut lanes = [
        Lane::new(1, key, Some(accept)),
        Lane::new(2, key, Some(accept)),
    ];
    let summary = RefCell::new(DispatchSummary::default());
    let mut actor = TextGeneratorDecodeWavefrontActor::with_pool(LanePool::with_callbacks(
        core::ptr::null_mut(),
        submit_inline,
        join_fail,
    ));
    assert_eq!(
        run(&mut actor, &mut lanes, &summary),
        Err(DecodeWavefrontError::Backend)
    );
    assert!(summary.borrow().all_submitted);
    assert!(!summary.borrow().joined);
}

#[test]
fn pool_dispatch_finishes_all_callbacks_before_return() {
    SUBMISSIONS.store(0, core::sync::atomic::Ordering::SeqCst);
    CALLBACKS.store(0, core::sync::atomic::Ordering::SeqCst);
    JOINS.store(0, core::sync::atomic::Ordering::SeqCst);
    let key = CompatibilityKey::default();
    let mut lanes = [
        Lane::new(1, key, Some(accept_observing)),
        Lane::new(2, key, Some(accept_observing)),
    ];
    let summary = RefCell::new(DispatchSummary::default());
    let pool = LanePool::with_callbacks(core::ptr::null_mut(), submit_observing, join_observing);
    let mut actor = TextGeneratorDecodeWavefrontActor::with_pool(pool);

    assert_eq!(run(&mut actor, &mut lanes, &summary), Ok(()));
    assert_eq!(SUBMISSIONS.load(core::sync::atomic::Ordering::SeqCst), 2);
    assert_eq!(CALLBACKS.load(core::sync::atomic::Ordering::SeqCst), 2);
    assert_eq!(JOINS.load(core::sync::atomic::Ordering::SeqCst), 1);
    assert!(lanes.iter().all(|lane| lane.accepted));
}

#[test]
fn marks_unexpected_events_explicitly_and_recovers_to_idle() {
    let mut actor = TextGeneratorDecodeWavefrontActor::new();
    assert_eq!(
        actor.process_unexpected(),
        Err(DecodeWavefrontError::Unexpected)
    );
    assert_eq!(actor.error(), DecodeWavefrontError::Unexpected);
    assert!(actor.context().unexpected);
}
