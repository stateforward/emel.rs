//! Catalog ownership and query orchestration.

#![allow(clippy::derive_partial_eq_without_eq)]

use sml::sml;

use super::name_query::{NameOperation, NameRequest};
use super::{
    BindRuntime, DescribeModelRuntime, DescribeTensorRuntime, FindRuntime, ReleaseRuntime,
    ResetRuntime, ScanPrefixRuntime, SealRuntime, UnexpectedRuntime, ValidateTensorShapeRuntime,
};

sml! {
    CatalogMachine<'dispatch> {
        "state_bound"_s <= *"state_empty"_s + Bind(BindRuntime<'dispatch>) [guard_bind_valid] / effect_bind,
        "state_empty"_s <= "state_empty"_s + Bind(BindRuntime<'dispatch>) [guard_bind_invalid] / effect_bind_invalid,
        "state_bound"_s <= "state_bound"_s + Bind(BindRuntime<'dispatch>) / effect_bind_busy,
        "state_sealed"_s <= "state_sealed"_s + Bind(BindRuntime<'dispatch>) / effect_bind_busy,

        "state_sealed"_s <= "state_bound"_s + Seal(SealRuntime<'dispatch>) [guard_seal_valid] / effect_seal,
        "state_bound"_s <= "state_bound"_s + Seal(SealRuntime<'dispatch>) [guard_seal_capacity] / effect_seal_capacity,
        "state_bound"_s <= "state_bound"_s + Seal(SealRuntime<'dispatch>) [guard_seal_model_invalid] / effect_seal_model_invalid,
        "state_empty"_s <= "state_empty"_s + Seal(SealRuntime<'dispatch>) / effect_seal_storage_unavailable,
        "state_sealed"_s <= "state_sealed"_s + Seal(SealRuntime<'dispatch>) / effect_seal_busy,

        "state_sealed"_s <= "state_sealed"_s + DescribeModel(DescribeModelRuntime<'dispatch>) [guard_model_wrong] / effect_model_wrong,
        "state_sealed"_s <= "state_sealed"_s + DescribeModel(DescribeModelRuntime<'dispatch>) [guard_model_stale] / effect_model_stale,
        "state_sealed"_s <= "state_sealed"_s + DescribeModel(DescribeModelRuntime<'dispatch>) [guard_model_valid] / effect_describe_model,
        "state_empty"_s <= "state_empty"_s + DescribeModel(DescribeModelRuntime<'dispatch>) / effect_model_storage_unavailable,
        "state_bound"_s <= "state_bound"_s + DescribeModel(DescribeModelRuntime<'dispatch>) / effect_model_storage_unavailable,

        "state_sealed"_s <= "state_sealed"_s + Find(FindRuntime<'dispatch>) [guard_find_wrong] / effect_find_wrong,
        "state_sealed"_s <= "state_sealed"_s + Find(FindRuntime<'dispatch>) [guard_find_stale] / effect_find_stale,
        "state_sealed"_s <= "state_sealed"_s + Find(FindRuntime<'dispatch>) [guard_find_present_bound] / effect_find_present_bound,
        "state_sealed"_s <= "state_sealed"_s + Find(FindRuntime<'dispatch>) [guard_find_present_unbound] / effect_find_present_unbound,
        "state_sealed"_s <= "state_sealed"_s + Find(FindRuntime<'dispatch>) [guard_find_absent] / effect_find_absent,
        "state_empty"_s <= "state_empty"_s + Find(FindRuntime<'dispatch>) / effect_find_storage_unavailable,
        "state_bound"_s <= "state_bound"_s + Find(FindRuntime<'dispatch>) / effect_find_storage_unavailable,

        "state_sealed"_s <= "state_sealed"_s + DescribeTensor(DescribeTensorRuntime<'dispatch>) [guard_tensor_wrong] / effect_tensor_wrong,
        "state_sealed"_s <= "state_sealed"_s + DescribeTensor(DescribeTensorRuntime<'dispatch>) [guard_tensor_stale] / effect_tensor_stale,
        "state_sealed"_s <= "state_sealed"_s + DescribeTensor(DescribeTensorRuntime<'dispatch>) [guard_tensor_valid_bound] / effect_describe_tensor_bound,
        "state_sealed"_s <= "state_sealed"_s + DescribeTensor(DescribeTensorRuntime<'dispatch>) [guard_tensor_valid_unbound] / effect_describe_tensor_unbound,
        "state_empty"_s <= "state_empty"_s + DescribeTensor(DescribeTensorRuntime<'dispatch>) / effect_tensor_storage_unavailable,
        "state_bound"_s <= "state_bound"_s + DescribeTensor(DescribeTensorRuntime<'dispatch>) / effect_tensor_storage_unavailable,

        "state_sealed"_s <= "state_sealed"_s + ScanPrefix(ScanPrefixRuntime<'dispatch>) [guard_scan_wrong] / effect_scan_wrong,
        "state_sealed"_s <= "state_sealed"_s + ScanPrefix(ScanPrefixRuntime<'dispatch>) [guard_scan_stale] / effect_scan_stale,
        "state_sealed"_s <= "state_sealed"_s + ScanPrefix(ScanPrefixRuntime<'dispatch>) [guard_scan_valid] / effect_scan_prefix,
        "state_empty"_s <= "state_empty"_s + ScanPrefix(ScanPrefixRuntime<'dispatch>) / effect_scan_storage_unavailable,
        "state_bound"_s <= "state_bound"_s + ScanPrefix(ScanPrefixRuntime<'dispatch>) / effect_scan_storage_unavailable,

        "state_sealed"_s <= "state_sealed"_s + ValidateTensorShape(ValidateTensorShapeRuntime<'dispatch>) [guard_shape_wrong] / effect_shape_wrong,
        "state_sealed"_s <= "state_sealed"_s + ValidateTensorShape(ValidateTensorShapeRuntime<'dispatch>) [guard_shape_stale] / effect_shape_stale,
        "state_sealed"_s <= "state_sealed"_s + ValidateTensorShape(ValidateTensorShapeRuntime<'dispatch>) [guard_shape_valid] / effect_validate_tensor_shape,
        "state_empty"_s <= "state_empty"_s + ValidateTensorShape(ValidateTensorShapeRuntime<'dispatch>) / effect_shape_storage_unavailable,
        "state_bound"_s <= "state_bound"_s + ValidateTensorShape(ValidateTensorShapeRuntime<'dispatch>) / effect_shape_storage_unavailable,

        "state_bound"_s <= "state_sealed"_s + Reset(ResetRuntime<'dispatch>) / effect_reset,
        "state_empty"_s <= "state_empty"_s + Reset(ResetRuntime<'dispatch>) / effect_reset_storage_unavailable,
        "state_bound"_s <= "state_bound"_s + Reset(ResetRuntime<'dispatch>) / effect_reset_invalid_request,

        "state_empty"_s <= "state_bound"_s + Release(ReleaseRuntime<'dispatch>) / effect_release,
        "state_empty"_s <= "state_empty"_s + Release(ReleaseRuntime<'dispatch>) / effect_release_storage_unavailable,
        "state_sealed"_s <= "state_sealed"_s + Release(ReleaseRuntime<'dispatch>) / effect_release_busy,

        "other"_s <= "other"_s + event<UnexpectedRuntime> [guard_never],
        "state_empty"_s <= "state_empty"_s + unexpected_event<_> / effect_unexpected,
        "state_bound"_s <= "state_bound"_s + unexpected_event<_> / effect_unexpected,
        "state_sealed"_s <= "state_sealed"_s + unexpected_event<_> / effect_unexpected,
    }
}

sml! {
    CatalogNameQuery<'query, 'data, O>
    where
        'data: 'query,
        O: NameOperation + 'data,
    {
        X <= *"state_ready"_s + event<&'query mut NameRequest<'data, O>> [guard_storage_unavailable] / effect_storage_unavailable,
        X <= "state_ready"_s + event<&'query mut NameRequest<'data, O>> [guard_wrong] / effect_wrong,
        X <= "state_ready"_s + event<&'query mut NameRequest<'data, O>> [guard_stale] / effect_stale,
        X <= "state_ready"_s + event<&'query mut NameRequest<'data, O>> [guard_valid] / effect_apply,
        X <= "state_ready"_s + unexpected_event<_> / effect_unexpected,
    }
}
