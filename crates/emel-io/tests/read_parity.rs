//! Source-backed golden parity cases for the pinned emel.cpp read actor.

#[path = "../examples/read_parity.rs"]
mod parity;

use sml as _;

const SNAPSHOT: &str = include_str!("../../../snapshots/parity/io-read/manifest.txt");

#[test]
fn pinned_cpp_read_cases_match_the_rust_actor() {
    assert_eq!(parity::render_manifest(), SNAPSHOT);
}
