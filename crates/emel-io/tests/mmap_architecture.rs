//! Static architecture regressions for the maintained mmap actor.

use emel_io as _;
#[cfg(unix)]
use libc as _;
#[cfg(unix)]
use rustix as _;
use sml as _;
#[cfg(windows)]
use windows_sys as _;

#[test]
fn unsafe_boundary_stays_exactly_scoped_and_single_view_conversion() {
    let root = include_str!("../src/lib.rs");
    let module = include_str!("../src/mmap/mod.rs");
    let actor = include_str!("../src/mmap/actor.rs");
    let events = include_str!("../src/mmap/event.rs");
    let machine = include_str!("../src/mmap/sm.rs");
    let platform = include_str!("../src/mmap/platform/mod.rs");
    let unix = include_str!("../src/mmap/platform/unix.rs");
    let windows = include_str!("../src/mmap/platform/windows.rs");

    assert!(root.contains("#![deny(unsafe_code)]"));
    assert_eq!(
        [root, module, actor, events, machine]
            .iter()
            .map(|source| source.matches("unsafe {").count())
            .sum::<usize>(),
        0,
        "no unsafe operation may escape mmap/platform",
    );
    assert_eq!(
        [platform, unix, windows]
            .iter()
            .map(|source| source.matches("from_raw_parts").count())
            .sum::<usize>(),
        1,
        "the approved immutable slice conversion must have one source owner",
    );
    assert_eq!(platform.matches("#![allow(unsafe_code)]").count(), 1);
    assert!(platform.contains("abort_on_release_failure"));
    assert!(!module.contains("allow(unsafe_code)"));
    assert_eq!(events.matches("pub unsafe fn open").count(), 1);
    assert_eq!(events.matches("allow(\n        unsafe_code,").count(), 1);
    assert!(events.contains("Arc<File>"));
    assert!(events.contains("# Safety"));

    assert!(unix.contains("rustix::param::page_size()"));
    assert!(!unix.contains("libc::sysconf"));
    assert!(!unix.contains("libc::close"));
    assert!(unix.contains("_file: MmapSource"));
    assert!(unix.contains("abort_on_release_failure(release(self))"));
    assert!(!unix.contains("let _ = release(self)"));

    assert_eq!(windows.matches("GetSystemInfo(").count(), 1);
    assert_eq!(windows.matches("CloseHandle(").count(), 1);
    assert!(windows.contains("CloseHandle(mapping)"));
    assert!(windows.contains("mapping: Option<windows_sys::Win32::Foundation::HANDLE>"));
    assert!(windows.contains("region.mapping = None"));
    assert!(windows.contains("abort_on_release_failure(release(self))"));
    assert!(!windows.contains("let _ = release(self)"));
    assert!(windows.contains("_file: MmapSource"));
}

#[test]
fn mapper_wrapper_contains_no_runtime_routing_or_pending_flags() {
    let actor = include_str!("../src/mmap/actor.rs");
    let machine = include_str!("../src/mmap/sm.rs");
    let core_wrapper = machine
        .split_once("impl<P: Platform> MapperCore<P>")
        .expect("MapperCore wrapper implementation must exist")
        .1
        .split_once("impl<P: Platform> IoMmapStateMachineContext")
        .expect("SML context implementation must follow MapperCore")
        .0;
    for forbidden in [
        "if ",
        "match ",
        "Option<",
        ".is_some()",
        ".is_none()",
        "?",
        ".and_then(",
        ".or_else(",
        ".map_err(",
    ] {
        assert!(
            !actor.contains(forbidden) && !core_wrapper.contains(forbidden),
            "Mapper or MapperCore wrapper contains runtime routing: {forbidden}",
        );
    }

    let context = machine
        .split_once("pub(super) struct Context")
        .expect("mapper Context must exist")
        .1
        .split_once("impl<P: Platform> Context")
        .expect("Context constructor must follow its fields")
        .0;
    for forbidden in ["pending", "phase", "waiting", "attempt", "failure"] {
        assert!(
            !context.contains(forbidden),
            "persistent Context contains dispatch-local status: {forbidden}",
        );
    }
}

#[test]
fn dispatch_has_no_escaped_phase_tokens_and_advice_is_statically_typed() {
    let events = include_str!("../src/mmap/event.rs");
    let machine = include_str!("../src/mmap/sm.rs");
    let actor = include_str!("../src/mmap/actor.rs");
    let platform = include_str!("../src/mmap/platform/mod.rs");
    let unix = include_str!("../src/mmap/platform/unix.rs");
    let windows = include_str!("../src/mmap/platform/windows.rs");

    for escaped in [
        "MapToken",
        "ReleaseToken",
        "AdviceToken",
        "AdvicePhase",
        "CancelMap",
        "CancelRelease",
        "CancelAdvice",
    ] {
        assert!(
            !events.contains(escaped) && !machine.contains(escaped),
            "escaped RTC phase remains public or dispatchable: {escaped}",
        );
    }
    for typed_advice in ["AdviseSequential", "AdviseWillNeed", "AdviseDontNeed"] {
        assert!(events.contains(typed_advice));
        assert!(machine.contains(typed_advice.trim_start_matches("Advise")));
    }
    for source in [actor, machine, platform, unix, windows] {
        assert!(!source.contains("match advice"));
        assert!(!source.contains("enum Advice"));
    }
    let unexpected = machine
        .split_once("fn effect_unexpected")
        .expect("explicit unexpected-event action must exist")
        .1
        .split_once("}\n")
        .expect("unexpected-event action must be bounded")
        .0;
    assert!(unexpected.contains("std::process::abort()"));
    for rtc_method in [
        "mapper.map(&self)",
        "mapper.release(self)",
        "mapper.advise_sequential(self.0)",
        "mapper.advise_will_need(self.0)",
        "mapper.advise_dont_need(self.0)",
    ] {
        assert!(events.contains(rtc_method));
    }
}

#[test]
fn map_setup_and_native_outcomes_stay_in_one_explicit_dispatch() {
    let events = include_str!("../src/mmap/event.rs");
    let machine = include_str!("../src/mmap/sm.rs");
    let platform = include_str!("../src/mmap/platform/mod.rs");
    let unix = include_str!("../src/mmap/platform/unix.rs");
    let windows = include_str!("../src/mmap/platform/windows.rs");

    assert!(machine.contains("effect_reserve_and_prepare_mapping"));
    assert!(machine.contains("state_setup_decision"));
    assert!(machine.contains("end <= event.request.file.len()"));
    assert!(machine.contains("self.platform.offset_supported(event.request.offset)"));
    assert!(!machine.contains("state_map_waiting"));
    assert!(!machine.contains("FinishMap"));
    for source in [platform, unix, windows] {
        assert!(!source.contains("SetupError::UnsupportedResource"));
        assert!(!source.contains("end > file_len"));
    }
    assert!(events.contains("mapper.map(&self)"));
    assert!(!events.contains("PreparedMapping"));

    assert!(machine.contains("IoMmapAccess"));
    assert!(machine.contains("[guard_access_owned] / effect_invoke_access"));
    assert!(machine.contains("mapping_view_len_supported(event.request.len)"));
    assert!(events.contains("pub trait MappingOperation"));
    assert!(!events.contains("catch_unwind"));
    assert!(!events.contains("dyn for<'view>"));
    assert!(!events.contains("Box<dyn"));
    assert!(!events.contains("AccessToken"));
}

#[test]
fn unix_null_success_is_owned_classified_and_released_before_failure() {
    let machine = include_str!("../src/mmap/sm.rs");
    let unix = include_str!("../src/mmap/platform/unix.rs");

    assert!(unix.contains("base: Some(raw.cast())"));
    assert!(unix.contains("including\n        // a POSIX-successful address-zero mapping"));
    assert!(machine.contains("state_setup_view_decision"));
    assert!(machine.contains("state_setup_cleanup_decision"));
    assert!(machine.contains("guard_setup_view_representable"));
    assert!(machine.contains("guard_setup_view_unrepresentable"));
    assert!(machine.contains("effect_release_unrepresentable_mapping"));
    assert!(machine.contains("effect_abort_unrepresentable_cleanup"));
}

#[test]
fn mandatory_test_path_executes_mmap_capability_compile_fail_docs() {
    let cargo_config = include_str!("../../../.cargo/config.toml");
    let ci = include_str!("../../../.github/workflows/ci.yml");

    let alias = cargo_config
        .lines()
        .find(|line| line.starts_with("test-all = "))
        .expect("test-all alias must remain configured");
    assert!(alias.contains("--all-targets"));
    assert!(ci.contains("cargo test-all"));
    assert!(ci.contains("cargo test --locked --workspace --doc --all-features"));
}
