//! Checked-in mmap parity snapshot coverage through the public Rust actor.

#[path = "../examples/mmap_parity.rs"]
mod parity;

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(unix)]
use libc as _;
#[cfg(unix)]
use rustix as _;
use sml as _;
#[cfg(windows)]
use windows_sys as _;

const SNAPSHOT: &str = include_str!("../../../snapshots/parity/io-mmap/manifest.txt");
static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

#[test]
fn pinned_cpp_mmap_cases_match_the_public_rust_actor() {
    let path = fixture_path();
    parity::write_fixture(&path).expect("write mmap parity fixture");
    let actual = parity::render_manifest(&path);
    let _ = std::fs::remove_file(&path);
    assert_eq!(actual, SNAPSHOT);
}

fn fixture_path() -> PathBuf {
    let sequence = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "emel-io-mmap-parity-test-{}-{sequence}.bin",
        std::process::id()
    ))
}
