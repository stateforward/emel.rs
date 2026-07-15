//! Pinned emel.cpp staged-read parity through the public Rust actor.

#[path = "../examples/staged_read_parity.rs"]
mod staged_read_parity;

const SNAPSHOT: &str = include_str!("../../../snapshots/parity/io-staged-read/manifest.txt");

#[test]
fn pinned_cpp_staged_read_cases_match_the_public_rust_actor() {
    assert_eq!(staged_read_parity::render_manifest(), SNAPSHOT);
}
