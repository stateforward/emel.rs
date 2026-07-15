//! Loader ownership and no-hidden-control checks.

use emel_io as _;
#[cfg(unix)]
use libc as _;
#[cfg(unix)]
use rustix as _;
use sml as _;
#[cfg(windows)]
use windows_sys as _;

const LIB_SOURCE: &str = include_str!("../src/lib.rs");
const MOD_SOURCE: &str = include_str!("../src/loader/mod.rs");
const ACTOR_SOURCE: &str = include_str!("../src/loader/actor.rs");
const EVENT_SOURCE: &str = include_str!("../src/loader/event.rs");
const SM_SOURCE: &str = include_str!("../src/loader/sm.rs");

#[test]
fn public_surface_is_loader_actor_and_typed_event_module() {
    assert!(LIB_SOURCE.contains("pub mod loader;"));
    assert!(MOD_SOURCE.contains("pub use actor::Loader;"));
    assert!(MOD_SOURCE.contains("pub mod event;"));
    assert!(!MOD_SOURCE.contains("pub mod sm"));
    assert!(!MOD_SOURCE.contains("pub use sm"));
}

#[test]
fn dependencies_are_static_actor_contracts_without_internal_access() {
    assert!(EVENT_SOURCE.contains("pub trait ReadActor"));
    assert!(EVENT_SOURCE.contains("pub trait StagedReadActor"));
    assert!(ACTOR_SOURCE.contains("Loader<R = NoActor, S = NoActor>"));
    assert!(!ACTOR_SOURCE.contains("dyn "));
    assert!(!EVENT_SOURCE.contains("dyn "));
    for forbidden in ["context(", "state()", "states()", "visit_current_state"] {
        assert!(!EVENT_SOURCE.contains(forbidden));
        assert!(!ACTOR_SOURCE.contains(forbidden));
    }
}

#[test]
fn strategy_and_child_outcome_choices_are_explicit_machine_guards() {
    for guard in [
        "guard_single_read_available",
        "guard_single_read_unavailable",
        "guard_single_staged_available",
        "guard_single_staged_source_valid",
        "guard_single_staged_chunk_smaller",
        "guard_single_read_succeeded",
        "guard_single_read_failed",
        "guard_batch_read_available",
        "guard_batch_staged_available",
        "guard_batch_staged_sources_valid",
        "guard_batch_staged_succeeded",
        "guard_batch_staged_failed",
    ] {
        assert!(SM_SOURCE.contains(guard), "missing {guard}");
    }
    assert!(!SM_SOURCE.contains("unsafe"));
    assert!(!SM_SOURCE.contains("process("));
    assert!(!SM_SOURCE.contains("defer"));
}

#[test]
fn child_dispatch_actions_each_send_exactly_one_public_event() {
    for (action, call) in [
        ("effect_dispatch_single_read", ".process_read("),
        ("effect_dispatch_batch_read", ".process_read_batch("),
        (
            "effect_dispatch_batch_staged",
            ".process_staged_read_batch(",
        ),
    ] {
        let source = function_source(SM_SOURCE, action);
        assert_eq!(source.matches(call).count(), 1, "{action}");
        assert!(!source.contains("for "));
        assert!(!source.contains("while "));
    }
}

fn function_source<'a>(source: &'a str, name: &str) -> &'a str {
    let signature = format!("fn {name}");
    let start = source.find(&signature).expect("function exists");
    let body = source[start..].find('{').expect("function body") + start;
    let mut depth = 0_u32;
    for (offset, byte) in source[body..].bytes().enumerate() {
        depth += u32::from(byte == b'{');
        depth -= u32::from(byte == b'}');
        if depth == 0 {
            return &source[start..=body + offset];
        }
    }
    panic!("function body is closed")
}
