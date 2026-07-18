//! Canonical semantic tensor query transition table.

#![allow(
    clippy::derive_partial_eq_without_eq,
    clippy::needless_lifetimes,
    reason = "SML-generated generic callback signatures mirror event lifetimes"
)]

use sml::sml;

use super::{Operation, TensorRequest};

sml! {
    TensorQuery<'query, 'data, O>
    where
        'data: 'query,
        O: Operation + 'data,
    {
        X <= *"ready"_s + event<&'query mut TensorRequest<'data, O>> [guard_not_parsed] / effect_not_parsed,
        X <= "ready"_s + event<&'query mut TensorRequest<'data, O>> [guard_missing] / effect_missing,
        X <= "ready"_s + event<&'query mut TensorRequest<'data, O>> [guard_malformed] / effect_malformed,
        X <= "ready"_s + event<&'query mut TensorRequest<'data, O>> [guard_ready] / effect_apply,
        X <= "ready"_s + unexpected_event<_> / effect_unexpected,
    }
}
