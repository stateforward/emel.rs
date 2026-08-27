//! Destination-first quantized-path classification table.

#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(private_interfaces)]

use sml::sml;

use super::Runtime;

sml! {
    QuantizedPathMachine<'dispatch> {
        "ready"_s <= *"ready"_s + Query(Runtime<'dispatch>) [guard_vector_f32] / effect_publish_dense_f32,
        "ready"_s <= "ready"_s + Query(Runtime<'dispatch>) [guard_vector_native] / effect_publish_dense_f32,
        "ready"_s <= "ready"_s + Query(Runtime<'dispatch>) [guard_vector_explicit_no_claim] / effect_publish_explicit_no_claim,

        "ready"_s <= "ready"_s + Query(Runtime<'dispatch>) [guard_matrix_native] / effect_publish_native,
        "ready"_s <= "ready"_s + Query(Runtime<'dispatch>) [guard_matrix_f32] / effect_publish_disallowed,
        "ready"_s <= "ready"_s + Query(Runtime<'dispatch>) [guard_matrix_explicit_no_claim] / effect_publish_explicit_no_claim,

        "ready"_s <= "ready"_s + unexpected_event<_> / effect_unexpected,
    }
}
