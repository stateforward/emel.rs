//! External tokenizer profile actor contract checks.

use allocation_counter as _;
use emel_token::profile::{Resolver, event::Resolve};
use sml as _;

#[test]
fn unknown_names_resolve_successfully_for_late_model_validation() {
    let mut actor = Resolver::new();
    let resolved = actor
        .process_event(Resolve::new("future-tokenizer", "future-pre"))
        .expect("unknown identities remain a successful classification");

    assert!(resolved.model().is_unknown());
    assert_eq!(resolved.defaults().bos_id(), -1);
}
