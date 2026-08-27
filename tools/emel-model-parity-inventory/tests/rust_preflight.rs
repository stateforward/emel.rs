//! Git-object integration coverage for the bounded Rust-only preflight.

use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::Duration;
#[cfg(unix)]
use std::time::Instant;

use emel_model_parity_inventory::rust_preflight;
#[cfg(windows)]
use winapi_util as _;
#[cfg(windows)]
use windows_sys as _;
#[cfg(unix)]
use {cap_std as _, cap_tempfile as _, uuid as _};
use {
    proc_macro2 as _, quote as _, serde as _, serde_json as _, sha2 as _, shell_words as _,
    syn as _,
};

fn git(repo: &Path, arguments: &[&str]) -> String {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(arguments)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "git {} failed: {}",
        arguments.join(" "),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap().trim().to_owned()
}

fn committed_fixture(source: &str) -> (tempfile::TempDir, String) {
    let temp = tempfile::tempdir().unwrap();
    let crate_dir = temp.path().join("crates/emel-model");
    fs::create_dir_all(crate_dir.join("src")).unwrap();
    fs::write(
        crate_dir.join("Cargo.toml"),
        "[package]\nname = \"fixture-model\"\nversion = \"0.0.0\"\nedition = \"2024\"\n",
    )
    .unwrap();
    fs::write(crate_dir.join("src/lib.rs"), source).unwrap();
    git(temp.path(), &["init", "--quiet"]);
    git(
        temp.path(),
        &["config", "user.email", "inventory@example.invalid"],
    );
    git(temp.path(), &["config", "user.name", "Inventory Test"]);
    git(temp.path(), &["add", "crates/emel-model"]);
    git(temp.path(), &["commit", "--quiet", "-m", "fixture"]);
    let commit = git(temp.path(), &["rev-parse", "HEAD"]);
    (temp, commit)
}

#[test]
fn committed_preflight_is_deterministic_and_ignores_dirty_worktree() {
    let source = r"
pub struct Model {
    pub value: u32,
}

#[test]
fn works() {
    assert_eq!(1, 1);
}

sml! {
    table = {
        state_done <= state_idle + Start,
    }
}
";
    let (temp, commit) = committed_fixture(source);
    let first = rust_preflight(temp.path(), &commit, Duration::from_secs(10)).unwrap();
    assert_eq!(first.declaration_count, 2);
    assert_eq!(first.test_count, 1);
    assert_eq!(first.assertion_count, 1);
    assert_eq!(first.sml_transition_count, 1);

    fs::write(
        temp.path().join("crates/emel-model/src/lib.rs"),
        "this dirty worktree is intentionally invalid Rust",
    )
    .unwrap();
    fs::write(
        temp.path().join("crates/emel-model/src/untracked.rs"),
        "also invalid",
    )
    .unwrap();

    let second = rust_preflight(temp.path(), &commit, Duration::from_secs(10)).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.render(), second.render());
}

#[test]
fn committed_syn_parse_failure_is_rejected() {
    let (temp, commit) = committed_fixture("pub fn broken( {");
    let error = rust_preflight(temp.path(), &commit, Duration::from_secs(10)).unwrap_err();
    assert!(error.contains("syn rejected"), "{error}");
}

#[test]
#[cfg(unix)]
fn external_watchdog_bounds_pre_main_equivalent_and_preserves_streams() {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/run-with-deadline.sh");
    let fixture = tempfile::tempdir().unwrap();
    let descendant_file = fixture.path().join("descendant-pid");
    let started = Instant::now();
    let output = Command::new("bash")
        .arg(script)
        .args([
            "--seconds",
            "1",
            "--grace-seconds",
            "1",
            "--",
            "bash",
            "-c",
            "trap '' TERM; (trap '' TERM; while :; do :; done) & child=$!; printf '%s' \"$child\" >\"$1\"; printf stdout-proof; printf stderr-proof >&2; wait \"$child\"",
            "watchdog-fixture",
            descendant_file.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(124));
    assert!(started.elapsed() < Duration::from_secs(5));
    assert_eq!(String::from_utf8(output.stdout).unwrap(), "stdout-proof");
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("stderr-proof"), "{stderr}");
    assert!(stderr.contains("timeout after 1 second"), "{stderr}");
    assert!(stderr.contains("sent SIGTERM then SIGKILL"), "{stderr}");
    let descendant = fs::read_to_string(descendant_file).unwrap();
    let survived = Command::new("bash")
        .args([
            "-c",
            "kill -0 \"$1\" 2>/dev/null",
            "watchdog-check",
            &descendant,
        ])
        .status()
        .unwrap()
        .success();
    assert!(!survived, "watchdog descendant {descendant} survived");
}

#[test]
#[cfg(unix)]
fn external_watchdog_preserves_ordinary_exit_and_streams() {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/run-with-deadline.sh");
    let output = Command::new("bash")
        .arg(script)
        .args([
            "--seconds",
            "3",
            "--grace-seconds",
            "1",
            "--",
            "bash",
            "-c",
            "printf ordinary-stdout; printf ordinary-stderr >&2; exit 7",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(7));
    assert_eq!(String::from_utf8(output.stdout).unwrap(), "ordinary-stdout");
    assert_eq!(String::from_utf8(output.stderr).unwrap(), "ordinary-stderr");
}

#[test]
#[cfg(unix)]
fn external_watchdog_cleans_nohup_descendant_after_successful_leader_exit() {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/run-with-deadline.sh");
    let fixture = tempfile::tempdir().unwrap();
    let descendant_file = fixture.path().join("success-descendant-pid");
    let output = Command::new("bash")
        .arg(script)
        .args([
            "--seconds",
            "5",
            "--grace-seconds",
            "1",
            "--",
            "bash",
            "-c",
            "nohup bash -c 'trap \"\" TERM; while :; do sleep 1; done' >/dev/null 2>&1 & child=$!; printf '%s' \"$child\" >\"$1\"; printf leader-success; exit 0",
            "watchdog-fixture",
            descendant_file.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(125));
    assert_eq!(String::from_utf8(output.stdout).unwrap(), "leader-success");
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("target exited while descendants remained"),
        "{stderr}"
    );
    assert!(stderr.contains("sent SIGTERM then SIGKILL"), "{stderr}");
    let descendant = fs::read_to_string(descendant_file).unwrap();
    let survived = Command::new("bash")
        .args([
            "-c",
            "kill -0 \"$1\" 2>/dev/null",
            "watchdog-check",
            &descendant,
        ])
        .status()
        .unwrap()
        .success();
    assert!(!survived, "watchdog descendant {descendant} survived");
}

#[test]
#[cfg(unix)]
fn external_watchdog_cleans_nohup_descendant_after_crashed_leader() {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/run-with-deadline.sh");
    let fixture = tempfile::tempdir().unwrap();
    let descendant_file = fixture.path().join("crash-descendant-pid");
    let output = Command::new("bash")
        .arg(script)
        .args([
            "--seconds",
            "5",
            "--grace-seconds",
            "1",
            "--",
            "bash",
            "-c",
            "nohup bash -c 'trap \"\" TERM; while :; do sleep 1; done' >/dev/null 2>&1 & child=$!; printf '%s' \"$child\" >\"$1\"; printf leader-crash; kill -ABRT \"$$\"",
            "watchdog-fixture",
            descendant_file.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(125));
    assert_eq!(String::from_utf8(output.stdout).unwrap(), "leader-crash");
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stderr.contains("target exited while descendants remained"),
        "{stderr}"
    );
    assert!(stderr.contains("sent SIGTERM then SIGKILL"), "{stderr}");
    let descendant = fs::read_to_string(descendant_file).unwrap();
    let survived = Command::new("bash")
        .args([
            "-c",
            "kill -0 \"$1\" 2>/dev/null",
            "watchdog-check",
            &descendant,
        ])
        .status()
        .unwrap()
        .success();
    assert!(!survived, "watchdog descendant {descendant} survived");
}

#[test]
#[cfg(unix)]
fn external_watchdog_does_not_depend_on_external_process_inspection() {
    use std::os::unix::fs::PermissionsExt as _;

    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/run-with-deadline.sh");
    let fixture = tempfile::tempdir().unwrap();
    let fake_ps = fixture.path().join("ps");
    fs::write(&fake_ps, "#!/bin/sh\nexit 91\n").unwrap();
    let mut permissions = fs::metadata(&fake_ps).unwrap().permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&fake_ps, permissions).unwrap();
    let path = format!(
        "{}:{}",
        fixture.path().display(),
        std::env::var("PATH").unwrap()
    );
    let output = Command::new("bash")
        .arg(script)
        .args([
            "--seconds",
            "3",
            "--grace-seconds",
            "1",
            "--",
            "bash",
            "-c",
            "exit 0",
        ])
        .env("PATH", path)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(0));
    assert!(output.stdout.is_empty());
    assert!(output.stderr.is_empty());
}

#[test]
#[cfg(unix)]
fn external_watchdog_rejects_non_bash_shell_before_target_launch() {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/run-with-deadline.sh");
    let output = Command::new("dash")
        .arg(script)
        .args([
            "--seconds",
            "1",
            "--grace-seconds",
            "1",
            "--",
            "sh",
            "-c",
            "exit 0",
        ])
        .env("BASH_VERSION", "spoofed")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(125));
    assert!(output.stdout.is_empty());
    assert_eq!(
        String::from_utf8(output.stderr).unwrap(),
        "emel-model-parity-inventory-watchdog: Bash is required\n"
    );
}
