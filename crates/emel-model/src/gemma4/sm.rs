//! Explicit Gemma4 family orchestration and child-outcome routing.

#![allow(clippy::derive_partial_eq_without_eq)]

use sml::sml;

use super::actor::{
    AuditRuntime, BeginRuntime, BlockRuntime, BlockVisitRuntime, PlanRuntime, ReleaseRuntime,
    ResetRuntime, StageRuntime, TopologyRuntime, UnexpectedRuntime, ValidateRuntime, VisitRuntime,
};

sml! {
    Gemma4Machine<'dispatch> {
        // Architecture binding and source-fixed global tensor inclusion.
        "state_begin_child"_s <= *"state_bound"_s + Begin(BeginRuntime<'dispatch>) [guard_begin_valid] / effect_begin_child,
        "state_bound"_s <= "state_bound"_s + Begin(BeginRuntime<'dispatch>) [guard_begin_architecture_invalid] / effect_begin_invalid_request,
        "state_bound"_s <= "state_bound"_s + Begin(BeginRuntime<'dispatch>) [guard_begin_parameters_invalid] / effect_begin_model_invalid,
        "state_global_child"_s <= "state_begin_child"_s + completion<Begin>(BeginRuntime<'dispatch>) [guard_begin_child_ok] / effect_global_child,
        "state_bound"_s <= "state_begin_child"_s + completion<Begin>(BeginRuntime<'dispatch>) [guard_begin_child_error] / effect_begin_child_error,
        "state_building"_s <= "state_global_child"_s + completion<Begin>(BeginRuntime<'dispatch>) [guard_global_child_ok] / effect_begin_complete,
        "state_begin_reset"_s <= "state_global_child"_s + completion<Begin>(BeginRuntime<'dispatch>) [guard_global_child_error] / effect_begin_reset,
        "state_bound"_s <= "state_begin_reset"_s + completion<Begin>(BeginRuntime<'dispatch>) [guard_begin_reset_ok] / effect_global_child_error,
        "state_bound"_s <= "state_begin_reset"_s + completion<Begin>(BeginRuntime<'dispatch>) [guard_begin_reset_error] / effect_begin_reset_error,
        "state_building"_s <= "state_building"_s + Begin(BeginRuntime<'dispatch>) / effect_busy_begin,

        // Exactly one source-fixed Q/K-normalized Gemma4 attention block per dispatch.
        "state_block_child"_s <= "state_building"_s + Block(BlockRuntime<'dispatch>) [guard_block_valid] / effect_block_child,
        "state_building"_s <= "state_building"_s + Block(BlockRuntime<'dispatch>) [guard_block_invalid] / effect_block_invalid,
        "state_building"_s <= "state_block_child"_s + completion<Block>(BlockRuntime<'dispatch>) [guard_block_child_ok] / effect_child_ok_block,
        "state_building"_s <= "state_block_child"_s + completion<Block>(BlockRuntime<'dispatch>) [guard_block_child_error] / effect_child_error_block,

        // Source-fixed topology and common plan construction.
        "state_topology_child"_s <= "state_building"_s + Topology(TopologyRuntime<'dispatch>) [guard_topology_valid] / effect_topology_child,
        "state_building"_s <= "state_building"_s + Topology(TopologyRuntime<'dispatch>) [guard_topology_invalid] / effect_topology_invalid,
        "state_building"_s <= "state_topology_child"_s + completion<Topology>(TopologyRuntime<'dispatch>) [guard_topology_child_ok] / effect_child_ok_topology,
        "state_building"_s <= "state_topology_child"_s + completion<Topology>(TopologyRuntime<'dispatch>) [guard_topology_child_error] / effect_child_error_topology,
        "state_plan_child"_s <= "state_building"_s + Plan(PlanRuntime<'dispatch>) / effect_plan_child,
        "state_building"_s <= "state_plan_child"_s + completion<Plan>(PlanRuntime<'dispatch>) [guard_plan_child_ok] / effect_child_ok_plan,
        "state_building"_s <= "state_plan_child"_s + completion<Plan>(PlanRuntime<'dispatch>) [guard_plan_child_error] / effect_child_error_plan,

        // Per-block validation and audit remain one event per block.
        "state_validate_child"_s <= "state_building"_s + Validate(ValidateRuntime<'dispatch>) / effect_validate_child,
        "state_validate_visit"_s <= "state_validate_child"_s + completion<Validate>(ValidateRuntime<'dispatch>) [guard_validate_child_ok] / effect_validate_visit,
        "state_building"_s <= "state_validate_child"_s + completion<Validate>(ValidateRuntime<'dispatch>) [guard_validate_child_error] / effect_child_error_validate,
        "state_building"_s <= "state_validate_visit"_s + completion<Validate>(ValidateRuntime<'dispatch>) [guard_validate_visit_ok] / effect_cache_validated_block,
        "state_building"_s <= "state_validate_visit"_s + completion<Validate>(ValidateRuntime<'dispatch>) [guard_validate_visit_error] / effect_validate_visit_error,
        "state_audit_child"_s <= "state_building"_s + Audit(AuditRuntime<'dispatch>) / effect_audit_child,
        "state_building"_s <= "state_audit_child"_s + completion<Audit>(AuditRuntime<'dispatch>) [guard_audit_child_ok] / effect_child_ok_audit,
        "state_building"_s <= "state_audit_child"_s + completion<Audit>(AuditRuntime<'dispatch>) [guard_audit_child_error] / effect_child_error_audit,

        // One quantized stage finalization per dispatch.
        "state_stage_child"_s <= "state_building"_s + Stage(StageRuntime<'dispatch>) / effect_stage_child,
        "state_building"_s <= "state_stage_child"_s + completion<Stage>(StageRuntime<'dispatch>) [guard_stage_child_ok] / effect_child_ok_stage,
        "state_building"_s <= "state_stage_child"_s + completion<Stage>(StageRuntime<'dispatch>) [guard_stage_child_error] / effect_child_error_stage,

        // Immutable family façade visits.
        "state_visit_child"_s <= "state_building"_s + Visit(VisitRuntime<'dispatch>) / effect_visit_child,
        "state_building"_s <= "state_visit_child"_s + completion<Visit>(VisitRuntime<'dispatch>) [guard_visit_child_ok] / effect_visit_ok,
        "state_building"_s <= "state_visit_child"_s + completion<Visit>(VisitRuntime<'dispatch>) [guard_visit_child_error] / effect_visit_error,
        "state_building"_s <= "state_building"_s + BlockVisit(BlockVisitRuntime<'dispatch>) [guard_block_visit_valid] / effect_block_visit,
        "state_building"_s <= "state_building"_s + BlockVisit(BlockVisitRuntime<'dispatch>) [guard_block_visit_invalid] / effect_invalid_block_visit,
        // Reset and caller-owned storage release.
        "state_reset_child"_s <= "state_building"_s + Reset(ResetRuntime<'dispatch>) / effect_reset_child,
        "state_bound"_s <= "state_reset_child"_s + completion<Reset>(ResetRuntime<'dispatch>) [guard_reset_child_ok] / effect_reset_ok,
        "state_building"_s <= "state_reset_child"_s + completion<Reset>(ResetRuntime<'dispatch>) [guard_reset_child_error] / effect_reset_error,
        "state_reset_bound_child"_s <= "state_bound"_s + Reset(ResetRuntime<'dispatch>) / effect_reset_child,
        "state_bound"_s <= "state_reset_bound_child"_s + completion<Reset>(ResetRuntime<'dispatch>) [guard_reset_child_ok] / effect_reset_ok,
        "state_bound"_s <= "state_reset_bound_child"_s + completion<Reset>(ResetRuntime<'dispatch>) [guard_reset_child_error] / effect_reset_error,
        "state_release_child"_s <= "state_bound"_s + Release(ReleaseRuntime<'dispatch>) / effect_release_child,
        "state_empty"_s <= "state_release_child"_s + completion<Release>(ReleaseRuntime<'dispatch>) [guard_release_child_ok] / effect_release_ok,
        "state_bound"_s <= "state_release_child"_s + completion<Release>(ReleaseRuntime<'dispatch>) [guard_release_child_error] / effect_release_error,
        "state_building"_s <= "state_building"_s + Release(ReleaseRuntime<'dispatch>) / effect_busy_release,

        "other"_s <= "other"_s + event<UnexpectedRuntime> [guard_never],
        "state_empty"_s <= "state_empty"_s + unexpected_event<_> / effect_unexpected,
        "state_bound"_s <= "state_bound"_s + unexpected_event<_> / effect_unexpected,
        "state_building"_s <= "state_building"_s + unexpected_event<_> / effect_unexpected,
    }
}
