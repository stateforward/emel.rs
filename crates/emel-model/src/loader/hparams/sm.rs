//! Explicit architecture-neutral hyperparameter query transitions.

#![allow(clippy::derive_partial_eq_without_eq)]

use sml::sml;

use super::Runtime;

sml! {
    HparamAccess<'dispatch> {
        // Operation selection.
        "state_i32_scalar_decision"_s <= *"state_ready"_s + Request(Runtime<'dispatch>) [guard_optional_i32] / effect_read_scalar,
        "state_i32_or_array_scalar_decision"_s <= "state_ready"_s + Request(Runtime<'dispatch>) [guard_optional_i32_or_array] / effect_read_scalar,
        "state_nonzero_scan_decision"_s <= "state_ready"_s + Request(Runtime<'dispatch>) [guard_required_nonzero] / effect_scan_nonzero,
        "state_f32_decision"_s <= "state_ready"_s + Request(Runtime<'dispatch>) [guard_optional_f32] / effect_read_float,
        "state_flags_bool_length_decision"_s <= "state_ready"_s + Request(Runtime<'dispatch>) [guard_required_flags] / effect_read_bool_flag_length,

        // Optional i32: scalar integer only.
        "state_done"_s <= "state_i32_scalar_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_unsigned_value_in_range] / effect_assign_i32,
        "state_done"_s <= "state_i32_scalar_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_unsigned_missing] / effect_preserve,
        "state_error"_s <= "state_i32_scalar_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_unsigned_value_out_of_range] / effect_range,
        "state_error"_s <= "state_i32_scalar_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_unsigned_wrong_kind] / effect_wrong_kind,
        "state_error"_s <= "state_i32_scalar_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_unsigned_malformed] / effect_malformed,
        "state_error"_s <= "state_i32_scalar_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_unsigned_query_range] / effect_range,
        "state_error"_s <= "state_i32_scalar_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_unsigned_query] / effect_query,
        "state_error"_s <= "state_i32_scalar_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_unsigned_internal] / effect_internal,

        // Optional i32 or first integer-array element.
        "state_done"_s <= "state_i32_or_array_scalar_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_unsigned_value_in_range] / effect_assign_i32,
        "state_done"_s <= "state_i32_or_array_scalar_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_unsigned_missing] / effect_preserve,
        "state_i32_or_array_first_decision"_s <= "state_i32_or_array_scalar_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_unsigned_wrong_kind] / effect_read_first,
        "state_error"_s <= "state_i32_or_array_scalar_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_unsigned_value_out_of_range] / effect_range,
        "state_error"_s <= "state_i32_or_array_scalar_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_unsigned_malformed] / effect_malformed,
        "state_error"_s <= "state_i32_or_array_scalar_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_unsigned_query_range] / effect_range,
        "state_error"_s <= "state_i32_or_array_scalar_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_unsigned_query] / effect_query,
        "state_error"_s <= "state_i32_or_array_scalar_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_unsigned_internal] / effect_internal,
        "state_done"_s <= "state_i32_or_array_first_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_unsigned_value_in_range] / effect_assign_i32,
        "state_error"_s <= "state_i32_or_array_first_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_unsigned_value_out_of_range] / effect_range,
        "state_error"_s <= "state_i32_or_array_first_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_unsigned_count] / effect_count,
        "state_error"_s <= "state_i32_or_array_first_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_unsigned_wrong_kind] / effect_wrong_kind,
        "state_error"_s <= "state_i32_or_array_first_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_unsigned_malformed] / effect_malformed,
        "state_error"_s <= "state_i32_or_array_first_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_unsigned_query_range] / effect_range,
        "state_error"_s <= "state_i32_or_array_first_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_unsigned_query] / effect_query,
        "state_error"_s <= "state_i32_or_array_first_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_unsigned_internal] / effect_internal,

        // Required first nonzero integer array value.
        "state_done"_s <= "state_nonzero_scan_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_scan_value_in_range] / effect_assign_scanned_i32,
        "state_error"_s <= "state_nonzero_scan_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_scan_value_out_of_range] / effect_range,
        "state_error"_s <= "state_nonzero_scan_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_scan_missing] / effect_scan_missing,
        "state_error"_s <= "state_nonzero_scan_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_scan_count] / effect_scan_count,
        "state_error"_s <= "state_nonzero_scan_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_scan_wrong_kind] / effect_scan_wrong_kind,
        "state_error"_s <= "state_nonzero_scan_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_scan_malformed] / effect_malformed,
        "state_error"_s <= "state_nonzero_scan_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_scan_range] / effect_range,
        "state_error"_s <= "state_nonzero_scan_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_scan_query] / effect_query,
        "state_error"_s <= "state_nonzero_scan_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_scan_internal] / effect_internal,

        // Optional f32 accepts f32/f64 through the concrete typed query event.
        "state_done"_s <= "state_f32_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_float_value] / effect_assign_f32,
        "state_done"_s <= "state_f32_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_float_missing] / effect_preserve,
        "state_error"_s <= "state_f32_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_float_wrong_kind] / effect_wrong_kind,
        "state_error"_s <= "state_f32_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_float_malformed] / effect_malformed,
        "state_error"_s <= "state_f32_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_float_range] / effect_range,
        "state_error"_s <= "state_f32_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_float_query] / effect_query,
        "state_error"_s <= "state_f32_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_float_internal] / effect_internal,

        // Required flag array: validate length and kind before one selected bulk copy.
        "state_flags_bool_copy_decision"_s <= "state_flags_bool_length_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_bool_length_fits] / effect_visit_bool_flags,
        "state_flags_integer_metrics_decision"_s <= "state_flags_bool_length_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_bool_length_wrong_kind] / effect_read_integer_flag_metrics,
        "state_error"_s <= "state_flags_bool_length_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_bool_length_missing] / effect_scan_missing,
        "state_error"_s <= "state_flags_bool_length_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_bool_length_exceeds] / effect_array_capacity,
        "state_error"_s <= "state_flags_bool_length_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_bool_length_malformed] / effect_malformed,
        "state_error"_s <= "state_flags_bool_length_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_bool_length_range] / effect_range,
        "state_error"_s <= "state_flags_bool_length_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_bool_length_query] / effect_query,
        "state_error"_s <= "state_flags_bool_length_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_bool_length_internal] / effect_internal,
        "state_done"_s <= "state_flags_bool_copy_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_array_visit_done] / effect_publish_array_count,
        "state_error"_s <= "state_flags_bool_copy_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_array_visit_count_range] / effect_range,
        "state_error"_s <= "state_flags_bool_copy_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_array_visit_missing] / effect_scan_missing,
        "state_error"_s <= "state_flags_bool_copy_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_array_visit_wrong_kind] / effect_wrong_kind,
        "state_error"_s <= "state_flags_bool_copy_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_array_visit_count] / effect_count,
        "state_error"_s <= "state_flags_bool_copy_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_array_visit_malformed] / effect_malformed,
        "state_error"_s <= "state_flags_bool_copy_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_array_visit_range] / effect_range,
        "state_error"_s <= "state_flags_bool_copy_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_array_visit_query] / effect_query,
        "state_error"_s <= "state_flags_bool_copy_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_array_visit_internal] / effect_internal,
        "state_flags_integer_copy_decision"_s <= "state_flags_integer_metrics_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_integer_metrics_fit] / effect_visit_integer_flags,
        "state_error"_s <= "state_flags_integer_metrics_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_integer_metrics_exceed] / effect_array_capacity,
        "state_error"_s <= "state_flags_integer_metrics_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_integer_metrics_missing] / effect_scan_missing,
        "state_error"_s <= "state_flags_integer_metrics_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_integer_metrics_wrong_kind] / effect_scan_wrong_kind,
        "state_error"_s <= "state_flags_integer_metrics_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_integer_metrics_malformed] / effect_malformed,
        "state_error"_s <= "state_flags_integer_metrics_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_integer_metrics_range] / effect_range,
        "state_error"_s <= "state_flags_integer_metrics_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_integer_metrics_query] / effect_query,
        "state_error"_s <= "state_flags_integer_metrics_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_integer_metrics_internal] / effect_internal,
        "state_done"_s <= "state_flags_integer_copy_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_array_visit_done] / effect_publish_array_count,
        "state_error"_s <= "state_flags_integer_copy_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_array_visit_count_range] / effect_range,
        "state_error"_s <= "state_flags_integer_copy_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_array_visit_missing] / effect_scan_missing,
        "state_error"_s <= "state_flags_integer_copy_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_array_visit_wrong_kind] / effect_wrong_kind,
        "state_error"_s <= "state_flags_integer_copy_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_array_visit_count] / effect_count,
        "state_error"_s <= "state_flags_integer_copy_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_array_visit_malformed] / effect_malformed,
        "state_error"_s <= "state_flags_integer_copy_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_array_visit_range] / effect_range,
        "state_error"_s <= "state_flags_integer_copy_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_array_visit_query] / effect_query,
        "state_error"_s <= "state_flags_integer_copy_decision"_s + completion<Request>(Runtime<'dispatch>) [guard_array_visit_internal] / effect_internal,

        "state_ready"_s <= "state_done"_s + completion<Request>(Runtime<'dispatch>),
        "state_ready"_s <= "state_error"_s + completion<Request>(Runtime<'dispatch>),

        "state_ready"_s <= "state_ready"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_i32_scalar_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_i32_or_array_scalar_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_i32_or_array_first_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_nonzero_scan_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_f32_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_flags_bool_length_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_flags_bool_copy_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_flags_integer_metrics_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_flags_integer_copy_decision"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_done"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_error"_s + unexpected_event<_> / effect_unexpected,
    }
}
