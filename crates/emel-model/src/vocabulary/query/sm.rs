//! Canonical generic vocabulary query transition table.

#![allow(
    clippy::derive_partial_eq_without_eq,
    clippy::needless_lifetimes,
    reason = "SML-generated generic callback signatures mirror event lifetimes"
)]

use super::{Operation, QueryRequest};
use sml::sml;

sml! {
    VocabularyQuery<'query, 'data, O>
    where
        'data: 'query,
        O: Operation + 'data,
    {
        X <= *"state_ready"_s + event<&'query mut QueryRequest<'data, O>> [guard_not_loaded] / effect_not_loaded,
        X <= "state_ready"_s + event<&'query mut QueryRequest<'data, O>> [guard_loaded_valid] / effect_apply,
        X <= "state_ready"_s + event<&'query mut QueryRequest<'data, O>> [guard_loaded_out_of_range] / effect_not_found,
        X <= "state_ready"_s + unexpected_event<_> / effect_unexpected,
    }
}
