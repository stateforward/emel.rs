//! Static architecture regressions for the maintained read machines.

use emel_io as _;
use sml as _;

#[test]
fn batch_failure_selection_stays_out_of_actions() {
    let source = include_str!("../src/read/sm.rs");
    assert!(
        !source.contains("first_failure_index("),
        "batch failure selection must be completed by guards before publication actions",
    );
    let classification_action = source
        .split_once("fn effect_classify_batch")
        .expect("batch classification action must exist")
        .1
        .split_once("fn guard_batch_requests_valid")
        .expect("batch request guard must follow classification")
        .0;
    assert!(
        classification_action.contains("self.batch_classifier.process_event(BatchSpanRuntime"),
        "the parent action must dispatch through the owned child actor wrapper",
    );
    assert!(
        !classification_action.contains("BatchSpanClassifierEvents"),
        "the parent action must not bypass the child actor wrapper",
    );
}

#[test]
fn refactored_machine_states_keep_the_state_prefix_and_phase_sections() {
    let source = include_str!("../src/read/sm.rs");
    for line in source.lines().filter(|line| line.contains("_s <=")) {
        assert!(
            line.trim_start().starts_with("\"state_"),
            "transition destination must use the state_ prefix: {line}",
        );
    }
    for phase in [
        "// Single request validation.",
        "// Single source validation and copy.",
        "// Single callback publication.",
        "// Batch count and child classification.",
        "// Batch phase-priority outcome selection.",
        "// Batch callback publication.",
        "// Explicit unexpected-event recovery.",
    ] {
        assert!(
            source.contains(phase),
            "missing visible phase section: {phase}"
        );
    }
}

#[test]
fn parity_snapshot_covers_batch_failure_and_count_boundaries() {
    let snapshot = include_str!("../../../snapshots/parity/io-read/manifest.txt");
    for case_name in [
        "batch_mixed_short_then_file_read",
        "batch_mixed_seek_then_open",
        "batch_mixed_resource_then_invalid",
        "batch_same_phase_two_file_reads",
        "batch_count_exact_cap",
        "batch_count_over_cap",
    ] {
        assert!(
            snapshot.contains(case_name),
            "missing parity case: {case_name}"
        );
    }
}

#[test]
fn over_cap_proof_uses_spans_that_are_individually_valid() {
    let lifecycle = include_str!("../src/read/mod.rs");
    let rust_parity = include_str!("../examples/read_parity.rs");
    let cpp_parity = include_str!("../../../tools/emel-io-read-reference/main.cpp");

    for (owner, source) in [
        ("ordinary Rust lifecycle", lifecycle),
        ("public Rust parity", rust_parity),
        ("archived C++ parity", cpp_parity),
    ] {
        assert!(
            source.contains("batch-over-cap.bin"),
            "{owner} must make every over-cap span independently valid",
        );
    }
}

#[test]
fn production_machine_has_no_test_only_unexpected_event_hook() {
    let machine = include_str!("../src/read/sm.rs");
    let actor = include_str!("../src/read/actor.rs");

    for forbidden in [
        "UnexpectedRuntime",
        "Unexpected(UnexpectedRuntime)",
        "effect_on_explicit_unexpected",
    ] {
        assert!(
            !machine.contains(forbidden),
            "production machine contains synthetic test hook: {forbidden}",
        );
    }
    assert!(
        !actor.contains("IoReadEvents::Unexpected"),
        "actor must not expose a synthetic unexpected-event dispatch hook",
    );
    assert_eq!(
        machine
            .matches("unexpected_event<_> / effect_on_unexpected")
            .count(),
        27,
        "every reachable production state must recover through the generic unexpected-event handler",
    );
}
