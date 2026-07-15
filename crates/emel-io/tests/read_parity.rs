//! Source-backed golden parity cases for the pinned emel.cpp read actor.

#[path = "../examples/read_parity.rs"]
mod parity;

#[cfg(unix)]
use libc as _;
#[cfg(unix)]
use rustix as _;
use sml as _;
#[cfg(windows)]
use windows_sys as _;

const SNAPSHOT: &str = include_str!("../../../snapshots/parity/io-read/manifest.txt");

#[test]
fn pinned_cpp_read_cases_match_the_rust_actor() {
    assert_eq!(parity::render_manifest(), SNAPSHOT);
}
