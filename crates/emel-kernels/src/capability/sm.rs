//! Explicit contract-scope and serialized-type selection table.

#![allow(
    clippy::derive_partial_eq_without_eq,
    reason = "SML generates partial equality for its internal event union"
)]

use sml::sml;

use super::{QueryRuntime, UnexpectedRuntime};

sml! {
    CapabilityResolver<'event, 'output>
    where
        'output: 'event,
    {
        "ready"_s <= *"ready"_s + event<&'event mut QueryRuntime<'output>> [guard_vector_f32] / effect_publish_dense_f32,
        "ready"_s <= "ready"_s + event<&'event mut QueryRuntime<'output>> [guard_vector_native] / effect_publish_dense_f32,
        "ready"_s <= "ready"_s + event<&'event mut QueryRuntime<'output>> [guard_vector_explicit_no_claim] / effect_publish_explicit_no_claim,

        "ready"_s <= "ready"_s + event<&'event mut QueryRuntime<'output>> [guard_matrix_native] / effect_publish_native,
        "ready"_s <= "ready"_s + event<&'event mut QueryRuntime<'output>> [guard_matrix_f32] / effect_publish_disallowed,
        "ready"_s <= "ready"_s + event<&'event mut QueryRuntime<'output>> [guard_matrix_explicit_no_claim] / effect_publish_explicit_no_claim,

        "other"_s <= "other"_s + event<UnexpectedRuntime> [guard_never],
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_unexpected,
    }
}
