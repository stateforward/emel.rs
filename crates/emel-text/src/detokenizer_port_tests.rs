//! Detokenizer topology completeness.

#[test]
fn detokenizer_keeps_typed_unexpected_events() {
    let src = include_str!("detokenizer/sm.rs");
    let n = src.matches("unexpected_event").count();
    assert!(n >= 60, "expected ~65 typed unexpected handlers, got {n}");
    assert!(src.contains("unexpected_event<EventBind>") || src.contains("EventBind"));
}
