//! Explicit common generation protocol and outcome-routing table.

#![allow(clippy::derive_partial_eq_without_eq)]

use sml::sml;

use super::{
    AttentionRuntime, AuditRuntime, BeginRuntime, BlockVisitRuntime, GlobalRuntime, PlanRuntime,
    RejectRuntime, ResetRuntime, ShortconvRuntime, StageRuntime, StorageBindRuntime,
    StorageReleaseRuntime, TopologyRuntime, UnexpectedRuntime, ValidateRuntime, VisitRuntime,
};

sml! {
    GenerationBuilder<'dispatch> {
        // Storage lifecycle.
        "state_bound"_s <= *"state_empty"_s + StorageBind(StorageBindRuntime<'dispatch>) / effect_storage_bind,
        "state_bound"_s <= "state_bound"_s + StorageBind(StorageBindRuntime<'dispatch>) / effect_storage_bind_busy,
        "state_building"_s <= "state_building"_s + StorageBind(StorageBindRuntime<'dispatch>) / effect_storage_bind_busy,

        // Begin validates the public catalog identity through the child actor.
        "state_begin_decision"_s <= "state_bound"_s + Begin(BeginRuntime<'dispatch>) [guard_begin_request_valid] / effect_describe_model,
        "state_bound"_s <= "state_bound"_s + Begin(BeginRuntime<'dispatch>) [guard_begin_request_invalid] / effect_invalid_begin,
        "state_building"_s <= "state_begin_decision"_s + completion<Begin>(BeginRuntime<'dispatch>) [guard_begin_catalog_valid] / effect_begin,
        "state_bound"_s <= "state_begin_decision"_s + completion<Begin>(BeginRuntime<'dispatch>) [guard_begin_wrong_identity] / effect_begin_wrong_identity,
        "state_bound"_s <= "state_begin_decision"_s + completion<Begin>(BeginRuntime<'dispatch>) [guard_begin_stale_identity] / effect_begin_stale_identity,
        "state_bound"_s <= "state_begin_decision"_s + completion<Begin>(BeginRuntime<'dispatch>) [guard_begin_storage_unavailable] / effect_begin_storage_unavailable,
        "state_bound"_s <= "state_begin_decision"_s + completion<Begin>(BeginRuntime<'dispatch>) [guard_begin_model_invalid] / effect_begin_model_invalid,
        "state_bound"_s <= "state_begin_decision"_s + completion<Begin>(BeginRuntime<'dispatch>) [guard_begin_dependency_error] / effect_begin_dependency_error,
        "state_building"_s <= "state_building"_s + Begin(BeginRuntime<'dispatch>) / effect_busy_begin,
        "state_empty"_s <= "state_empty"_s + Begin(BeginRuntime<'dispatch>) / effect_storage_unavailable_begin,

        // Global tensor binding, including explicit tied-output fallback.
        "state_global_decision"_s <= "state_building"_s + Global(GlobalRuntime<'dispatch>) [guard_global_request_direct] / effect_query_globals,
        "state_global_tied_decision"_s <= "state_building"_s + Global(GlobalRuntime<'dispatch>) [guard_global_request_tied] / effect_query_globals,
        "state_building"_s <= "state_building"_s + Global(GlobalRuntime<'dispatch>) [guard_global_request_invalid] / effect_invalid_global,
        "state_building"_s <= "state_global_decision"_s + completion<Global>(GlobalRuntime<'dispatch>) [guard_globals_direct] / effect_bind_globals_direct,
        "state_building"_s <= "state_global_decision"_s + completion<Global>(GlobalRuntime<'dispatch>) [guard_globals_missing] / effect_model_invalid_global,
        "state_building"_s <= "state_global_decision"_s + completion<Global>(GlobalRuntime<'dispatch>) [guard_globals_dependency_error] / effect_global_dependency_error,
        "state_building"_s <= "state_global_tied_decision"_s + completion<Global>(GlobalRuntime<'dispatch>) [guard_globals_direct] / effect_bind_globals_direct,
        "state_building"_s <= "state_global_tied_decision"_s + completion<Global>(GlobalRuntime<'dispatch>) [guard_globals_tied] / effect_bind_globals_tied,
        "state_building"_s <= "state_global_tied_decision"_s + completion<Global>(GlobalRuntime<'dispatch>) [guard_globals_missing] / effect_model_invalid_global,
        "state_building"_s <= "state_global_tied_decision"_s + completion<Global>(GlobalRuntime<'dispatch>) [guard_globals_dependency_error] / effect_global_dependency_error,

        // Exactly one attention block per dispatch; configuration is selected in guards.
        "state_attention_plain"_s <= "state_building"_s + Attention(AttentionRuntime<'dispatch>) [guard_attention_plain] / effect_query_attention_plain,
        "state_attention_qk"_s <= "state_building"_s + Attention(AttentionRuntime<'dispatch>) [guard_attention_qk] / effect_query_attention_qk,
        "state_attention_shared"_s <= "state_building"_s + Attention(AttentionRuntime<'dispatch>) [guard_attention_shared] / effect_query_attention_shared,
        "state_attention_qk_shared"_s <= "state_building"_s + Attention(AttentionRuntime<'dispatch>) [guard_attention_qk_shared] / effect_query_attention_qk_shared,
        "state_attention_shared_required_value"_s <= "state_building"_s + Attention(AttentionRuntime<'dispatch>) [guard_attention_shared_required_value] / effect_query_attention_shared_required_value,
        "state_attention_qk_shared_required_value"_s <= "state_building"_s + Attention(AttentionRuntime<'dispatch>) [guard_attention_qk_shared_required_value] / effect_query_attention_qk_shared_required_value,
        "state_building"_s <= "state_building"_s + Attention(AttentionRuntime<'dispatch>) [guard_attention_invalid] / effect_invalid_attention,
        "state_building"_s <= "state_attention_plain"_s + completion<Attention>(AttentionRuntime<'dispatch>) [guard_block_queries_valid] / effect_bind_attention_plain,
        "state_building"_s <= "state_attention_qk"_s + completion<Attention>(AttentionRuntime<'dispatch>) [guard_block_queries_valid] / effect_bind_attention_qk,
        "state_building"_s <= "state_attention_shared"_s + completion<Attention>(AttentionRuntime<'dispatch>) [guard_block_queries_valid] / effect_bind_attention_shared,
        "state_building"_s <= "state_attention_qk_shared"_s + completion<Attention>(AttentionRuntime<'dispatch>) [guard_block_queries_valid] / effect_bind_attention_qk_shared,
        "state_building"_s <= "state_attention_shared_required_value"_s + completion<Attention>(AttentionRuntime<'dispatch>) [guard_block_queries_valid] / effect_bind_attention_shared,
        "state_building"_s <= "state_attention_qk_shared_required_value"_s + completion<Attention>(AttentionRuntime<'dispatch>) [guard_block_queries_valid] / effect_bind_attention_qk_shared,
        "state_building"_s <= "state_attention_plain"_s + completion<Attention>(AttentionRuntime<'dispatch>) [guard_block_queries_missing] / effect_model_invalid_attention,
        "state_building"_s <= "state_attention_qk"_s + completion<Attention>(AttentionRuntime<'dispatch>) [guard_block_queries_missing] / effect_model_invalid_attention,
        "state_building"_s <= "state_attention_shared"_s + completion<Attention>(AttentionRuntime<'dispatch>) [guard_block_queries_missing] / effect_model_invalid_attention,
        "state_building"_s <= "state_attention_qk_shared"_s + completion<Attention>(AttentionRuntime<'dispatch>) [guard_block_queries_missing] / effect_model_invalid_attention,
        "state_building"_s <= "state_attention_shared_required_value"_s + completion<Attention>(AttentionRuntime<'dispatch>) [guard_block_queries_missing] / effect_model_invalid_attention,
        "state_building"_s <= "state_attention_qk_shared_required_value"_s + completion<Attention>(AttentionRuntime<'dispatch>) [guard_block_queries_missing] / effect_model_invalid_attention,
        "state_building"_s <= "state_attention_plain"_s + completion<Attention>(AttentionRuntime<'dispatch>) [guard_block_queries_dependency_error] / effect_attention_dependency_error,
        "state_building"_s <= "state_attention_qk"_s + completion<Attention>(AttentionRuntime<'dispatch>) [guard_block_queries_dependency_error] / effect_attention_dependency_error,
        "state_building"_s <= "state_attention_shared"_s + completion<Attention>(AttentionRuntime<'dispatch>) [guard_block_queries_dependency_error] / effect_attention_dependency_error,
        "state_building"_s <= "state_attention_qk_shared"_s + completion<Attention>(AttentionRuntime<'dispatch>) [guard_block_queries_dependency_error] / effect_attention_dependency_error,
        "state_building"_s <= "state_attention_shared_required_value"_s + completion<Attention>(AttentionRuntime<'dispatch>) [guard_block_queries_dependency_error] / effect_attention_dependency_error,
        "state_building"_s <= "state_attention_qk_shared_required_value"_s + completion<Attention>(AttentionRuntime<'dispatch>) [guard_block_queries_dependency_error] / effect_attention_dependency_error,

        // Exactly one shortconv block per dispatch.
        "state_shortconv_decision"_s <= "state_building"_s + Shortconv(ShortconvRuntime<'dispatch>) [guard_shortconv_valid] / effect_query_shortconv,
        "state_building"_s <= "state_building"_s + Shortconv(ShortconvRuntime<'dispatch>) [guard_shortconv_invalid] / effect_invalid_shortconv,
        "state_building"_s <= "state_shortconv_decision"_s + completion<Shortconv>(ShortconvRuntime<'dispatch>) [guard_shortconv_queries_valid] / effect_bind_shortconv,
        "state_building"_s <= "state_shortconv_decision"_s + completion<Shortconv>(ShortconvRuntime<'dispatch>) [guard_shortconv_queries_missing] / effect_model_invalid_shortconv,
        "state_building"_s <= "state_shortconv_decision"_s + completion<Shortconv>(ShortconvRuntime<'dispatch>) [guard_shortconv_queries_dependency_error] / effect_shortconv_dependency_error,

        // Source-selected opposite-family tensor rejection.
        "state_reject_shortconv"_s <= "state_building"_s + Reject(RejectRuntime<'dispatch>) [guard_reject_shortconv_valid] / effect_query_reject_shortconv,
        "state_reject_attention"_s <= "state_building"_s + Reject(RejectRuntime<'dispatch>) [guard_reject_attention_valid] / effect_query_reject_attention,
        "state_building"_s <= "state_building"_s + Reject(RejectRuntime<'dispatch>) [guard_reject_invalid] / effect_invalid_reject,
        "state_building"_s <= "state_reject_shortconv"_s + completion<Reject>(RejectRuntime<'dispatch>) [guard_reject_queries_absent] / effect_reject_ok,
        "state_building"_s <= "state_reject_attention"_s + completion<Reject>(RejectRuntime<'dispatch>) [guard_reject_queries_absent] / effect_reject_ok,
        "state_building"_s <= "state_reject_shortconv"_s + completion<Reject>(RejectRuntime<'dispatch>) [guard_reject_queries_present] / effect_reject_present,
        "state_building"_s <= "state_reject_attention"_s + completion<Reject>(RejectRuntime<'dispatch>) [guard_reject_queries_present] / effect_reject_present,
        "state_building"_s <= "state_reject_shortconv"_s + completion<Reject>(RejectRuntime<'dispatch>) [guard_reject_queries_dependency_error] / effect_reject_dependency_error,
        "state_building"_s <= "state_reject_attention"_s + completion<Reject>(RejectRuntime<'dispatch>) [guard_reject_queries_dependency_error] / effect_reject_dependency_error,

        // Topology and plan.
        "state_building"_s <= "state_building"_s + Topology(TopologyRuntime<'dispatch>) [guard_topology_valid] / effect_topology,
        "state_building"_s <= "state_building"_s + Topology(TopologyRuntime<'dispatch>) [guard_topology_invalid] / effect_invalid_topology,
        "state_building"_s <= "state_building"_s + Plan(PlanRuntime<'dispatch>) [guard_plan_valid] / effect_plan,
        "state_building"_s <= "state_building"_s + Plan(PlanRuntime<'dispatch>) [guard_plan_invalid] / effect_invalid_plan,

        // Per-block validation.
        "state_building"_s <= "state_building"_s + Validate(ValidateRuntime<'dispatch>) [guard_validate_attention_valid] / effect_validate,
        "state_building"_s <= "state_building"_s + Validate(ValidateRuntime<'dispatch>) [guard_validate_shortconv_valid] / effect_validate,
        "state_building"_s <= "state_building"_s + Validate(ValidateRuntime<'dispatch>) [guard_validate_invalid] / effect_model_invalid_validate,

        // Per-block audit dispatches the capability child and joins in the same RTC.
        "state_audit_attention"_s <= "state_building"_s + Audit(AuditRuntime<'dispatch>) [guard_audit_attention_plain] / effect_query_audit_attention_plain,
        "state_audit_attention_qk"_s <= "state_building"_s + Audit(AuditRuntime<'dispatch>) [guard_audit_attention_qk] / effect_query_audit_attention_qk,
        "state_audit_shortconv"_s <= "state_building"_s + Audit(AuditRuntime<'dispatch>) [guard_audit_shortconv] / effect_query_audit_shortconv,
        "state_building"_s <= "state_building"_s + Audit(AuditRuntime<'dispatch>) [guard_audit_invalid] / effect_invalid_audit,
        "state_building"_s <= "state_audit_attention"_s + completion<Audit>(AuditRuntime<'dispatch>) [guard_audit_dependency_valid] / effect_record_audit,
        "state_building"_s <= "state_audit_attention_qk"_s + completion<Audit>(AuditRuntime<'dispatch>) [guard_audit_dependency_valid] / effect_record_audit,
        "state_building"_s <= "state_audit_shortconv"_s + completion<Audit>(AuditRuntime<'dispatch>) [guard_audit_dependency_valid] / effect_record_audit,
        "state_building"_s <= "state_audit_attention"_s + completion<Audit>(AuditRuntime<'dispatch>) [guard_audit_dependency_error] / effect_audit_dependency_error,
        "state_building"_s <= "state_audit_attention_qk"_s + completion<Audit>(AuditRuntime<'dispatch>) [guard_audit_dependency_error] / effect_audit_dependency_error,
        "state_building"_s <= "state_audit_shortconv"_s + completion<Audit>(AuditRuntime<'dispatch>) [guard_audit_dependency_error] / effect_audit_dependency_error,

        // Finalize exactly one stage per dispatch after all block facts exist.
        "state_stage_decision"_s <= "state_building"_s + Stage(StageRuntime<'dispatch>) [guard_stage_global_vector] / effect_query_global_stage_vector,
        "state_stage_decision"_s <= "state_building"_s + Stage(StageRuntime<'dispatch>) [guard_stage_global_matrix] / effect_query_global_stage_matrix,
        "state_stage_decision"_s <= "state_building"_s + Stage(StageRuntime<'dispatch>) [guard_stage_block] / effect_no_stage_query,
        "state_building"_s <= "state_building"_s + Stage(StageRuntime<'dispatch>) [guard_stage_request_invalid] / effect_invalid_stage,
        "state_building"_s <= "state_stage_decision"_s + completion<Stage>(StageRuntime<'dispatch>) [guard_stage_global_vector_dense] / effect_stage_global_dense,
        "state_building"_s <= "state_stage_decision"_s + completion<Stage>(StageRuntime<'dispatch>) [guard_stage_global_vector_no_claim] / effect_stage_global_no_claim,
        "state_building"_s <= "state_stage_decision"_s + completion<Stage>(StageRuntime<'dispatch>) [guard_stage_global_matrix_native] / effect_stage_global_native,
        "state_building"_s <= "state_stage_decision"_s + completion<Stage>(StageRuntime<'dispatch>) [guard_stage_global_matrix_disallowed] / effect_stage_global_disallowed,
        "state_building"_s <= "state_stage_decision"_s + completion<Stage>(StageRuntime<'dispatch>) [guard_stage_global_matrix_no_claim] / effect_stage_global_no_claim,
        "state_stage_block_native"_s <= "state_stage_decision"_s + completion<Stage>(StageRuntime<'dispatch>) [guard_stage_block_native],
        "state_stage_block_dense"_s <= "state_stage_decision"_s + completion<Stage>(StageRuntime<'dispatch>) [guard_stage_block_dense],
        "state_stage_block_disallowed"_s <= "state_stage_decision"_s + completion<Stage>(StageRuntime<'dispatch>) [guard_stage_block_disallowed],
        "state_stage_block_no_claim"_s <= "state_stage_decision"_s + completion<Stage>(StageRuntime<'dispatch>) [guard_stage_block_no_claim],
        "state_building"_s <= "state_stage_decision"_s + completion<Stage>(StageRuntime<'dispatch>) [guard_stage_block_not_applicable] / effect_stage_block_not_applicable,
        "state_building"_s <= "state_stage_block_native"_s + completion<Stage>(StageRuntime<'dispatch>) [guard_stage_block_consistent] / effect_stage_block_native_consistent,
        "state_building"_s <= "state_stage_block_native"_s + completion<Stage>(StageRuntime<'dispatch>) [guard_stage_block_inconsistent] / effect_stage_block_native_inconsistent,
        "state_building"_s <= "state_stage_block_dense"_s + completion<Stage>(StageRuntime<'dispatch>) [guard_stage_block_consistent] / effect_stage_block_dense_consistent,
        "state_building"_s <= "state_stage_block_dense"_s + completion<Stage>(StageRuntime<'dispatch>) [guard_stage_block_inconsistent] / effect_stage_block_dense_inconsistent,
        "state_building"_s <= "state_stage_block_disallowed"_s + completion<Stage>(StageRuntime<'dispatch>) [guard_stage_block_consistent] / effect_stage_block_disallowed_consistent,
        "state_building"_s <= "state_stage_block_disallowed"_s + completion<Stage>(StageRuntime<'dispatch>) [guard_stage_block_inconsistent] / effect_stage_block_disallowed_inconsistent,
        "state_building"_s <= "state_stage_block_no_claim"_s + completion<Stage>(StageRuntime<'dispatch>) [guard_stage_block_consistent] / effect_stage_block_no_claim_consistent,
        "state_building"_s <= "state_stage_block_no_claim"_s + completion<Stage>(StageRuntime<'dispatch>) [guard_stage_block_inconsistent] / effect_stage_block_no_claim_inconsistent,
        "state_building"_s <= "state_stage_decision"_s + completion<Stage>(StageRuntime<'dispatch>) [guard_stage_dependency_error] / effect_stage_dependency_error,

        // Immutable visits.
        "state_building"_s <= "state_building"_s + Visit(VisitRuntime<'dispatch>) [guard_contract_complete] / effect_visit,
        "state_building"_s <= "state_building"_s + Visit(VisitRuntime<'dispatch>) [guard_contract_incomplete] / effect_invalid_visit,
        "state_building"_s <= "state_building"_s + BlockVisit(BlockVisitRuntime<'dispatch>) [guard_block_visit_valid] / effect_block_visit,
        "state_building"_s <= "state_building"_s + BlockVisit(BlockVisitRuntime<'dispatch>) [guard_block_visit_invalid] / effect_invalid_block_visit,

        // Reset and release.
        "state_bound"_s <= "state_building"_s + Reset(ResetRuntime<'dispatch>) / effect_reset,
        "state_bound"_s <= "state_bound"_s + Reset(ResetRuntime<'dispatch>) / effect_reset,
        "state_empty"_s <= "state_empty"_s + Reset(ResetRuntime<'dispatch>) / effect_storage_unavailable_reset,
        "state_empty"_s <= "state_bound"_s + Release(StorageReleaseRuntime<'dispatch>) / effect_release,
        "state_empty"_s <= "state_empty"_s + Release(StorageReleaseRuntime<'dispatch>) / effect_storage_unavailable_release,
        "state_building"_s <= "state_building"_s + Release(StorageReleaseRuntime<'dispatch>) / effect_busy_release,

        "other"_s <= "other"_s + event<UnexpectedRuntime> [guard_never],
        "state_empty"_s <= "state_empty"_s + unexpected_event<_> / effect_unexpected,
        "state_bound"_s <= "state_bound"_s + unexpected_event<_> / effect_unexpected,
        "state_building"_s <= "state_building"_s + unexpected_event<_> / effect_unexpected,
    }
}
