//! Static ownership and orchestration regressions for staged-read.

use emel_io as _;
#[cfg(unix)]
use libc as _;
#[cfg(unix)]
use rustix as _;
use sml as _;
#[cfg(windows)]
use windows_sys as _;

#[test]
fn public_surface_is_actor_and_typed_events_only() {
    let module = include_str!("../src/staged_read/mod.rs");
    let actor = include_str!("../src/staged_read/actor.rs");
    let events = include_str!("../src/staged_read/event.rs");
    assert!(module.contains("pub use actor::Stager"));
    assert!(module.contains("pub mod event"));
    assert!(!module.contains("pub mod sm"));
    assert!(!actor.contains("unsafe"));
    assert!(!actor.contains("if "));
    assert!(!actor.contains("match "));
    assert!(!actor.contains('?'));
    assert!(actor.contains("cfg!(any(unix, windows))"));
    assert!(!events.contains("pub fn bytes"));
}

#[test]
fn runtime_choices_stay_in_guards_and_copy_loops_are_data_plane_only() {
    let machine = include_str!("../src/staged_read/sm.rs");
    assert!(machine.contains("guard_single_chunk_aligned"));
    assert!(machine.contains("guard_single_chunk_remainder"));
    assert!(machine.contains("effect_copy_single_aligned"));
    assert!(machine.contains("effect_copy_single_remainder"));
    assert!(machine.contains("effect_compute_batch_assessment"));
    assert!(machine.contains("guard_batch_assessment_valid"));
    assert!(machine.contains("guard_batch_assessment_invalid"));
    assert!(!machine.contains("first_invalid_index"));
    assert!(machine.contains("while progressed < covered"));
    assert!(!machine.contains("todo!"));
    assert!(!machine.contains("process_event("));
    assert!(!machine.contains("defer"));
}

#[test]
fn callbacks_are_non_panicking_storage_slots() {
    let events = include_str!("../src/staged_read/event.rs");
    assert!(events.contains("slot.set(Some(value))"));
    assert!(!events.contains("FnMut"));
    assert!(!events.contains("catch_unwind"));
}
