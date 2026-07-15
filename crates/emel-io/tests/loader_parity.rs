//! Pinned emel.cpp loader parity through the public Rust actor.

#[path = "../examples/loader_parity.rs"]
mod loader_parity;

#[test]
fn pinned_cpp_loader_cases_match_the_public_rust_actor() {
    assert_eq!(
        loader_parity::render_manifest(),
        include_str!("../../../snapshots/parity/io-loader/manifest.txt")
    );
}
