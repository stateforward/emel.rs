#[path = "../fuzz_targets/omniembed_support.rs"]
mod omniembed_support;

#[test]
fn canonical_structured_fixture_reaches_ready_and_publishes_name() {
    omniembed_support::exercise(&[]);
}
