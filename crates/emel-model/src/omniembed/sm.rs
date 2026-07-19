//! Canonical `OmniEmbed` execution-contract transition tables.

#![allow(clippy::derive_partial_eq_without_eq)]

use super::actor::{
    BeginRuntime, FinishRuntime, ObserveRuntime, ReleaseRuntime, ResetRuntime, UnexpectedRuntime,
    VisitRuntime,
};
use super::hparams::HparamLoadRuntime;
use super::query::{NameOperation, NameRequest};
use sml::sml;

sml! {
    OmniEmbedMachine<'dispatch> {
        "state_scanning"_s <= *"state_empty"_s + Begin(BeginRuntime<'dispatch>) [guard_begin_valid] / effect_begin,
        "state_rejected"_s <= "state_empty"_s + Begin(BeginRuntime<'dispatch>) [guard_begin_invalid] / effect_model_invalid,
        "state_rejected"_s <= "state_empty"_s + Begin(BeginRuntime<'dispatch>) [guard_begin_storage_unavailable] / effect_storage_unavailable_begin,
        "state_scanning"_s <= "state_scanning"_s + Begin(BeginRuntime<'dispatch>) / effect_busy_begin,
        "state_ready"_s <= "state_ready"_s + Begin(BeginRuntime<'dispatch>) / effect_busy_begin,

        "state_scan_error"_s <= "state_scanning"_s + Observe(ObserveRuntime<'dispatch>) [guard_observation_index_invalid] / effect_invalid_request,
        "state_scanning"_s <= "state_scanning"_s + Observe(ObserveRuntime<'dispatch>) [guard_text_encoder_first_fits] / effect_text_encoder_first,
        "state_scan_error"_s <= "state_scanning"_s + Observe(ObserveRuntime<'dispatch>) [guard_text_encoder_first_capacity] / effect_capacity,
        "state_scanning"_s <= "state_scanning"_s + Observe(ObserveRuntime<'dispatch>) [guard_text_encoder_additional] / effect_text_encoder_additional,
        "state_scanning"_s <= "state_scanning"_s + Observe(ObserveRuntime<'dispatch>) [guard_text_projection_first_fits] / effect_text_projection_first,
        "state_scan_error"_s <= "state_scanning"_s + Observe(ObserveRuntime<'dispatch>) [guard_text_projection_first_capacity] / effect_capacity,
        "state_scanning"_s <= "state_scanning"_s + Observe(ObserveRuntime<'dispatch>) [guard_text_projection_additional] / effect_text_projection_additional,
        "state_scanning"_s <= "state_scanning"_s + Observe(ObserveRuntime<'dispatch>) [guard_image_encoder_first_fits] / effect_image_encoder_first,
        "state_scan_error"_s <= "state_scanning"_s + Observe(ObserveRuntime<'dispatch>) [guard_image_encoder_first_capacity] / effect_capacity,
        "state_scanning"_s <= "state_scanning"_s + Observe(ObserveRuntime<'dispatch>) [guard_image_encoder_additional] / effect_image_encoder_additional,
        "state_scanning"_s <= "state_scanning"_s + Observe(ObserveRuntime<'dispatch>) [guard_image_projection_first_fits] / effect_image_projection_first,
        "state_scan_error"_s <= "state_scanning"_s + Observe(ObserveRuntime<'dispatch>) [guard_image_projection_first_capacity] / effect_capacity,
        "state_scanning"_s <= "state_scanning"_s + Observe(ObserveRuntime<'dispatch>) [guard_image_projection_additional] / effect_image_projection_additional,
        "state_scanning"_s <= "state_scanning"_s + Observe(ObserveRuntime<'dispatch>) [guard_audio_encoder_first_fits] / effect_audio_encoder_first,
        "state_scan_error"_s <= "state_scanning"_s + Observe(ObserveRuntime<'dispatch>) [guard_audio_encoder_first_capacity] / effect_capacity,
        "state_scanning"_s <= "state_scanning"_s + Observe(ObserveRuntime<'dispatch>) [guard_audio_encoder_additional] / effect_audio_encoder_additional,
        "state_scanning"_s <= "state_scanning"_s + Observe(ObserveRuntime<'dispatch>) [guard_audio_projection_first_fits] / effect_audio_projection_first,
        "state_scan_error"_s <= "state_scanning"_s + Observe(ObserveRuntime<'dispatch>) [guard_audio_projection_first_capacity] / effect_capacity,
        "state_scanning"_s <= "state_scanning"_s + Observe(ObserveRuntime<'dispatch>) [guard_audio_projection_additional] / effect_audio_projection_additional,
        "state_scanning"_s <= "state_scanning"_s + Observe(ObserveRuntime<'dispatch>) [guard_observation_ignored] / effect_observation_ignored,
        "state_rejected"_s <= "state_rejected"_s + Observe(ObserveRuntime<'dispatch>) / effect_preserve_observe,
        "state_scan_error"_s <= "state_scan_error"_s + Observe(ObserveRuntime<'dispatch>) / effect_preserve_observe,
        "state_empty"_s <= "state_empty"_s + Observe(ObserveRuntime<'dispatch>) [guard_storage_available_observe] / effect_invalid_request,
        "state_empty"_s <= "state_empty"_s + Observe(ObserveRuntime<'dispatch>) [guard_storage_unavailable_observe] / effect_storage_unavailable_observe,
        "state_ready"_s <= "state_ready"_s + Observe(ObserveRuntime<'dispatch>) / effect_busy_observe,

        "state_ready"_s <= "state_scanning"_s + Finish(FinishRuntime<'dispatch>) [guard_families_complete] / effect_finish,
        "state_empty"_s <= "state_scanning"_s + Finish(FinishRuntime<'dispatch>) [guard_families_incomplete] / effect_model_invalid_finish,
        "state_empty"_s <= "state_rejected"_s + Finish(FinishRuntime<'dispatch>) / effect_preserve_finish,
        "state_empty"_s <= "state_scan_error"_s + Finish(FinishRuntime<'dispatch>) / effect_preserve_finish,
        "state_empty"_s <= "state_empty"_s + Finish(FinishRuntime<'dispatch>) [guard_storage_available_finish] / effect_invalid_finish,
        "state_empty"_s <= "state_empty"_s + Finish(FinishRuntime<'dispatch>) [guard_storage_unavailable_finish] / effect_storage_unavailable_finish,
        "state_ready"_s <= "state_ready"_s + Finish(FinishRuntime<'dispatch>) / effect_busy_finish,

        "state_ready"_s <= "state_ready"_s + Visit(VisitRuntime<'dispatch>) / effect_visit,
        "state_empty"_s <= "state_empty"_s + Visit(VisitRuntime<'dispatch>) [guard_storage_available_visit] / effect_invalid_visit,
        "state_empty"_s <= "state_empty"_s + Visit(VisitRuntime<'dispatch>) [guard_storage_unavailable_visit] / effect_storage_unavailable,
        "state_scanning"_s <= "state_scanning"_s + Visit(VisitRuntime<'dispatch>) / effect_busy_visit,

        "state_empty"_s <= "state_scanning"_s + Reset(ResetRuntime<'dispatch>) / effect_reset_storage,
        "state_empty"_s <= "state_ready"_s + Reset(ResetRuntime<'dispatch>) / effect_reset_storage,
        "state_empty"_s <= "state_rejected"_s + Reset(ResetRuntime<'dispatch>) / effect_reset_storage,
        "state_empty"_s <= "state_scan_error"_s + Reset(ResetRuntime<'dispatch>) / effect_reset_storage,
        "state_empty"_s <= "state_empty"_s + Reset(ResetRuntime<'dispatch>) [guard_storage_available_reset] / effect_reset_storage,
        "state_empty"_s <= "state_empty"_s + Reset(ResetRuntime<'dispatch>) [guard_storage_unavailable_reset] / effect_reset_without_storage,

        "state_empty"_s <= "state_empty"_s + Release(ReleaseRuntime<'dispatch>) [guard_storage_available_release] / effect_release_storage,
        "state_empty"_s <= "state_empty"_s + Release(ReleaseRuntime<'dispatch>) [guard_storage_unavailable_release] / effect_storage_unavailable_release,
        "state_scanning"_s <= "state_scanning"_s + Release(ReleaseRuntime<'dispatch>) / effect_busy_release,
        "state_ready"_s <= "state_ready"_s + Release(ReleaseRuntime<'dispatch>) / effect_busy_release,
        "state_rejected"_s <= "state_rejected"_s + Release(ReleaseRuntime<'dispatch>) / effect_busy_release,
        "state_scan_error"_s <= "state_scan_error"_s + Release(ReleaseRuntime<'dispatch>) / effect_busy_release,

        "other"_s <= "other"_s + event<UnexpectedRuntime> [guard_never],
        "state_empty"_s <= "state_empty"_s + unexpected_event<_> / effect_unexpected,
        "state_scanning"_s <= "state_scanning"_s + unexpected_event<_> / effect_unexpected,
        "state_ready"_s <= "state_ready"_s + unexpected_event<_> / effect_unexpected,
        "state_rejected"_s <= "state_rejected"_s + unexpected_event<_> / effect_unexpected,
        "state_scan_error"_s <= "state_scan_error"_s + unexpected_event<_> / effect_unexpected,
    }
}

sml! {
    OmniEmbedHparams<'dispatch> {
        "state_architecture_decision"_s <= *"state_idle"_s + Load(HparamLoadRuntime<'dispatch>) / effect_query_architecture,
        "state_embedding_decision"_s <= "state_architecture_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_architecture_valid] / effect_query_embedding,
        X <= "state_architecture_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_architecture_invalid] / effect_contract,
        X <= "state_architecture_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_string_missing] / effect_missing,
        X <= "state_architecture_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_query_wrong_kind] / effect_wrong_kind,
        X <= "state_architecture_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_query_range] / effect_range,
        X <= "state_architecture_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_query_internal] / effect_internal,
        X <= "state_architecture_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_query_other] / effect_query,

        "state_image_name_decision"_s <= "state_embedding_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_unsigned_value] / effect_assign_embedding_query_image_name,
        "state_image_name_decision"_s <= "state_embedding_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_unsigned_missing] / effect_query_image_name,
        X <= "state_embedding_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_unsigned_range] / effect_range,
        X <= "state_embedding_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_query_wrong_kind] / effect_wrong_kind,
        X <= "state_embedding_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_query_range] / effect_range,
        X <= "state_embedding_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_query_internal] / effect_internal,
        X <= "state_embedding_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_query_other] / effect_query,

        "state_image_length_decision"_s <= "state_image_name_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_string_canonical_fits] / effect_set_image_canonical_query_length,
        "state_image_length_decision"_s <= "state_image_name_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_string_other_fits] / effect_set_image_other_query_length,
        "state_image_length_decision"_s <= "state_image_name_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_string_empty] / effect_query_image_length,
        "state_image_length_decision"_s <= "state_image_name_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_string_missing] / effect_query_image_length,
        X <= "state_image_name_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_string_capacity] / effect_capacity,
        X <= "state_image_name_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_query_wrong_kind] / effect_wrong_kind,
        X <= "state_image_name_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_query_range] / effect_range,
        X <= "state_image_name_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_query_internal] / effect_internal,
        X <= "state_image_name_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_query_other] / effect_query,

        "state_audio_name_decision"_s <= "state_image_length_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_unsigned_value] / effect_assign_image_query_audio_name,
        "state_audio_name_decision"_s <= "state_image_length_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_unsigned_missing] / effect_query_audio_name,
        X <= "state_image_length_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_unsigned_range] / effect_range,
        X <= "state_image_length_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_query_wrong_kind] / effect_wrong_kind,
        X <= "state_image_length_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_query_range] / effect_range,
        X <= "state_image_length_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_query_internal] / effect_internal,
        X <= "state_image_length_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_query_other] / effect_query,

        "state_audio_length_decision"_s <= "state_audio_name_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_string_canonical_fits] / effect_set_audio_canonical_query_length,
        "state_audio_length_decision"_s <= "state_audio_name_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_string_other_fits] / effect_set_audio_other_query_length,
        "state_audio_length_decision"_s <= "state_audio_name_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_string_empty] / effect_query_audio_length,
        "state_audio_length_decision"_s <= "state_audio_name_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_string_missing] / effect_query_audio_length,
        X <= "state_audio_name_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_string_capacity] / effect_capacity,
        X <= "state_audio_name_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_query_wrong_kind] / effect_wrong_kind,
        X <= "state_audio_name_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_query_range] / effect_range,
        X <= "state_audio_name_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_query_internal] / effect_internal,
        X <= "state_audio_name_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_query_other] / effect_query,

        "state_array_count_decision"_s <= "state_audio_length_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_unsigned_value] / effect_assign_audio_query_array,
        "state_array_count_decision"_s <= "state_audio_length_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_unsigned_missing] / effect_query_array,
        X <= "state_audio_length_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_unsigned_range] / effect_range,
        X <= "state_audio_length_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_query_wrong_kind] / effect_wrong_kind,
        X <= "state_audio_length_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_query_range] / effect_range,
        X <= "state_audio_length_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_query_internal] / effect_internal,
        X <= "state_audio_length_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_query_other] / effect_query,

        "state_array_values_decision"_s <= "state_array_count_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_array_count_valid] / effect_visit_array,
        X <= "state_array_count_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_array_missing] / effect_success,
        X <= "state_array_count_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_array_capacity] / effect_range,
        X <= "state_array_count_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_query_wrong_kind] / effect_wrong_kind,
        X <= "state_array_count_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_query_range] / effect_range,
        X <= "state_array_count_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_query_internal] / effect_internal,
        X <= "state_array_count_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_query_other] / effect_query,

        X <= "state_array_values_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_array_values_valid] / effect_assign_array_success,
        X <= "state_array_values_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_array_values_range] / effect_range,
        X <= "state_array_values_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_query_wrong_kind] / effect_wrong_kind,
        X <= "state_array_values_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_query_range] / effect_range,
        X <= "state_array_values_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_query_internal] / effect_internal,
        X <= "state_array_values_decision"_s + completion<Load>(HparamLoadRuntime<'dispatch>) [guard_query_other] / effect_query,

        "other"_s <= "other"_s + event<UnexpectedRuntime> [guard_never],
        X <= "state_idle"_s + unexpected_event<_> / effect_unexpected,
        X <= "state_architecture_decision"_s + unexpected_event<_> / effect_unexpected,
        X <= "state_embedding_decision"_s + unexpected_event<_> / effect_unexpected,
        X <= "state_image_name_decision"_s + unexpected_event<_> / effect_unexpected,
        X <= "state_image_length_decision"_s + unexpected_event<_> / effect_unexpected,
        X <= "state_audio_name_decision"_s + unexpected_event<_> / effect_unexpected,
        X <= "state_audio_length_decision"_s + unexpected_event<_> / effect_unexpected,
        X <= "state_array_count_decision"_s + unexpected_event<_> / effect_unexpected,
        X <= "state_array_values_decision"_s + unexpected_event<_> / effect_unexpected,
    }
}

sml! {
    OmniEmbedNameQuery<'query, 'data, O>
    where
        'data: 'query,
        O: NameOperation + 'data,
    {
        X <= *"state_ready"_s + event<&'query mut NameRequest<'data, O>> [guard_storage_unavailable] / effect_storage_unavailable,
        X <= "state_ready"_s + event<&'query mut NameRequest<'data, O>> [guard_invalid] / effect_invalid,
        X <= "state_ready"_s + event<&'query mut NameRequest<'data, O>> [guard_busy] / effect_busy,
        X <= "state_ready"_s + event<&'query mut NameRequest<'data, O>> [guard_text_encoder] / effect_apply_text_encoder,
        X <= "state_ready"_s + event<&'query mut NameRequest<'data, O>> [guard_text_projection] / effect_apply_text_projection,
        X <= "state_ready"_s + event<&'query mut NameRequest<'data, O>> [guard_image_encoder] / effect_apply_image_encoder,
        X <= "state_ready"_s + event<&'query mut NameRequest<'data, O>> [guard_image_projection] / effect_apply_image_projection,
        X <= "state_ready"_s + event<&'query mut NameRequest<'data, O>> [guard_audio_encoder] / effect_apply_audio_encoder,
        X <= "state_ready"_s + event<&'query mut NameRequest<'data, O>> [guard_audio_projection] / effect_apply_audio_projection,
        X <= "state_ready"_s + unexpected_event<_> / effect_unexpected,
    }
}
