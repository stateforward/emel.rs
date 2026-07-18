//! Canonical indexed metadata descriptor transition table.

#![allow(
    clippy::derive_partial_eq_without_eq,
    clippy::needless_lifetimes,
    reason = "SML-generated generic callback signatures mirror event lifetimes"
)]

use sml::sml;

use super::{MetadataRequest, Operation};

sml! {
    MetadataDescriptor<'query, 'data, O>
    where
        'data: 'query,
        O: Operation + 'data,
    {
        X <= *"ready"_s + event<&'query mut MetadataRequest<'data, O>> [guard_not_parsed] / effect_not_parsed,
        X <= "ready"_s + event<&'query mut MetadataRequest<'data, O>> [guard_missing] / effect_missing,
        X <= "ready"_s + event<&'query mut MetadataRequest<'data, O>> [guard_malformed] / effect_malformed,
        X <= "ready"_s + event<&'query mut MetadataRequest<'data, O>> [guard_uint8] / effect_uint8,
        X <= "ready"_s + event<&'query mut MetadataRequest<'data, O>> [guard_int8] / effect_int8,
        X <= "ready"_s + event<&'query mut MetadataRequest<'data, O>> [guard_uint16] / effect_uint16,
        X <= "ready"_s + event<&'query mut MetadataRequest<'data, O>> [guard_int16] / effect_int16,
        X <= "ready"_s + event<&'query mut MetadataRequest<'data, O>> [guard_uint32] / effect_uint32,
        X <= "ready"_s + event<&'query mut MetadataRequest<'data, O>> [guard_int32] / effect_int32,
        X <= "ready"_s + event<&'query mut MetadataRequest<'data, O>> [guard_float32] / effect_float32,
        X <= "ready"_s + event<&'query mut MetadataRequest<'data, O>> [guard_bool] / effect_bool,
        X <= "ready"_s + event<&'query mut MetadataRequest<'data, O>> [guard_string] / effect_string,
        X <= "ready"_s + event<&'query mut MetadataRequest<'data, O>> [guard_uint64] / effect_uint64,
        X <= "ready"_s + event<&'query mut MetadataRequest<'data, O>> [guard_int64] / effect_int64,
        X <= "ready"_s + event<&'query mut MetadataRequest<'data, O>> [guard_float64] / effect_float64,

        X <= "ready"_s + event<&'query mut MetadataRequest<'data, O>> [guard_array_uint8] / effect_array_uint8,
        X <= "ready"_s + event<&'query mut MetadataRequest<'data, O>> [guard_array_int8] / effect_array_int8,
        X <= "ready"_s + event<&'query mut MetadataRequest<'data, O>> [guard_array_uint16] / effect_array_uint16,
        X <= "ready"_s + event<&'query mut MetadataRequest<'data, O>> [guard_array_int16] / effect_array_int16,
        X <= "ready"_s + event<&'query mut MetadataRequest<'data, O>> [guard_array_uint32] / effect_array_uint32,
        X <= "ready"_s + event<&'query mut MetadataRequest<'data, O>> [guard_array_int32] / effect_array_int32,
        X <= "ready"_s + event<&'query mut MetadataRequest<'data, O>> [guard_array_float32] / effect_array_float32,
        X <= "ready"_s + event<&'query mut MetadataRequest<'data, O>> [guard_array_bool] / effect_array_bool,
        X <= "ready"_s + event<&'query mut MetadataRequest<'data, O>> [guard_array_string] / effect_array_string,
        X <= "ready"_s + event<&'query mut MetadataRequest<'data, O>> [guard_array_uint64] / effect_array_uint64,
        X <= "ready"_s + event<&'query mut MetadataRequest<'data, O>> [guard_array_int64] / effect_array_int64,
        X <= "ready"_s + event<&'query mut MetadataRequest<'data, O>> [guard_array_float64] / effect_array_float64,

        X <= "ready"_s + unexpected_event<_> / effect_unexpected,
    }
}
