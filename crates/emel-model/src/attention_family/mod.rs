//! Private compile-time mechanics shared by source-owned attention families.

pub mod context;
pub mod event;

macro_rules! forward_guards {
    ($context:ty; $($name:ident: $event:ty),+ $(,)?) => {
        $(
            fn $name(&self, event: &$event) -> Result<bool, ()> {
                <$context>::$name(self, event)
            }
        )+
    };
}

macro_rules! forward_effects {
    ($context:ty; $($name:ident: $event:ty),+ $(,)?) => {
        $(
            fn $name(&mut self, event: $event) -> Result<(), ()> {
                <$context>::$name(self, event)
            }
        )+
    };
}

/// Implements only the generated callback adapter; transition rows remain local.
macro_rules! impl_machine_context {
    ($trait_name:ident, $policy:ty) => {
        impl $trait_name for $crate::attention_family::context::Context<$policy> {
            $crate::attention_family::forward_guards!($crate::attention_family::context::Context<$policy>;
                guard_begin_valid: $crate::attention_family::context::BeginRuntime<'_>,
                guard_begin_architecture_invalid: $crate::attention_family::context::BeginRuntime<'_>,
                guard_begin_parameters_invalid: $crate::attention_family::context::BeginRuntime<'_>,
                guard_begin_child_ok: $crate::attention_family::context::BeginRuntime<'_>,
                guard_begin_child_error: $crate::attention_family::context::BeginRuntime<'_>,
                guard_global_child_ok: $crate::attention_family::context::BeginRuntime<'_>,
                guard_global_child_error: $crate::attention_family::context::BeginRuntime<'_>,
                guard_begin_reset_ok: $crate::attention_family::context::BeginRuntime<'_>,
                guard_begin_reset_error: $crate::attention_family::context::BeginRuntime<'_>,
                guard_block_valid: $crate::attention_family::context::BlockRuntime<'_>,
                guard_block_invalid: $crate::attention_family::context::BlockRuntime<'_>,
                guard_block_child_ok: $crate::attention_family::context::BlockRuntime<'_>,
                guard_block_child_error: $crate::attention_family::context::BlockRuntime<'_>,
                guard_topology_valid: $crate::attention_family::context::TopologyRuntime<'_>,
                guard_topology_invalid: $crate::attention_family::context::TopologyRuntime<'_>,
                guard_topology_child_ok: $crate::attention_family::context::TopologyRuntime<'_>,
                guard_topology_child_error: $crate::attention_family::context::TopologyRuntime<'_>,
                guard_plan_child_ok: $crate::attention_family::context::PlanRuntime<'_>,
                guard_plan_child_error: $crate::attention_family::context::PlanRuntime<'_>,
                guard_validate_child_ok: $crate::attention_family::context::ValidateRuntime<'_>,
                guard_validate_child_error: $crate::attention_family::context::ValidateRuntime<'_>,
                guard_validate_visit_ok: $crate::attention_family::context::ValidateRuntime<'_>,
                guard_validate_visit_error: $crate::attention_family::context::ValidateRuntime<'_>,
                guard_audit_child_ok: $crate::attention_family::context::AuditRuntime<'_>,
                guard_audit_child_error: $crate::attention_family::context::AuditRuntime<'_>,
                guard_stage_child_ok: $crate::attention_family::context::StageRuntime<'_>,
                guard_stage_child_error: $crate::attention_family::context::StageRuntime<'_>,
                guard_visit_child_ok: $crate::attention_family::context::VisitRuntime<'_>,
                guard_visit_child_error: $crate::attention_family::context::VisitRuntime<'_>,
                guard_block_visit_valid: $crate::attention_family::context::BlockVisitRuntime<'_>,
                guard_block_visit_invalid: $crate::attention_family::context::BlockVisitRuntime<'_>,
                guard_reset_child_ok: $crate::attention_family::context::ResetRuntime<'_>,
                guard_reset_child_error: $crate::attention_family::context::ResetRuntime<'_>,
                guard_release_child_ok: $crate::attention_family::context::ReleaseRuntime<'_>,
                guard_release_child_error: $crate::attention_family::context::ReleaseRuntime<'_>,
                guard_never: $crate::attention_family::context::UnexpectedRuntime,
            );
            $crate::attention_family::forward_effects!($crate::attention_family::context::Context<$policy>;
                effect_begin_child: $crate::attention_family::context::BeginRuntime<'_>,
                effect_global_child: $crate::attention_family::context::BeginRuntime<'_>,
                effect_begin_child_error: $crate::attention_family::context::BeginRuntime<'_>,
                effect_begin_complete: $crate::attention_family::context::BeginRuntime<'_>,
                effect_begin_reset: $crate::attention_family::context::BeginRuntime<'_>,
                effect_global_child_error: $crate::attention_family::context::BeginRuntime<'_>,
                effect_begin_reset_error: $crate::attention_family::context::BeginRuntime<'_>,
                effect_begin_invalid_request: $crate::attention_family::context::BeginRuntime<'_>,
                effect_begin_model_invalid: $crate::attention_family::context::BeginRuntime<'_>,
                effect_busy_begin: $crate::attention_family::context::BeginRuntime<'_>,
                effect_block_child: $crate::attention_family::context::BlockRuntime<'_>,
                effect_block_invalid: $crate::attention_family::context::BlockRuntime<'_>,
                effect_child_ok_block: $crate::attention_family::context::BlockRuntime<'_>,
                effect_child_error_block: $crate::attention_family::context::BlockRuntime<'_>,
                effect_topology_child: $crate::attention_family::context::TopologyRuntime<'_>,
                effect_topology_invalid: $crate::attention_family::context::TopologyRuntime<'_>,
                effect_child_ok_topology: $crate::attention_family::context::TopologyRuntime<'_>,
                effect_child_error_topology: $crate::attention_family::context::TopologyRuntime<'_>,
                effect_plan_child: $crate::attention_family::context::PlanRuntime<'_>,
                effect_child_ok_plan: $crate::attention_family::context::PlanRuntime<'_>,
                effect_child_error_plan: $crate::attention_family::context::PlanRuntime<'_>,
                effect_validate_child: $crate::attention_family::context::ValidateRuntime<'_>,
                effect_validate_visit: $crate::attention_family::context::ValidateRuntime<'_>,
                effect_cache_validated_block: $crate::attention_family::context::ValidateRuntime<'_>,
                effect_validate_visit_error: $crate::attention_family::context::ValidateRuntime<'_>,
                effect_child_error_validate: $crate::attention_family::context::ValidateRuntime<'_>,
                effect_audit_child: $crate::attention_family::context::AuditRuntime<'_>,
                effect_child_ok_audit: $crate::attention_family::context::AuditRuntime<'_>,
                effect_child_error_audit: $crate::attention_family::context::AuditRuntime<'_>,
                effect_stage_child: $crate::attention_family::context::StageRuntime<'_>,
                effect_child_ok_stage: $crate::attention_family::context::StageRuntime<'_>,
                effect_child_error_stage: $crate::attention_family::context::StageRuntime<'_>,
                effect_visit_child: $crate::attention_family::context::VisitRuntime<'_>,
                effect_visit_ok: $crate::attention_family::context::VisitRuntime<'_>,
                effect_visit_error: $crate::attention_family::context::VisitRuntime<'_>,
                effect_block_visit: $crate::attention_family::context::BlockVisitRuntime<'_>,
                effect_invalid_block_visit: $crate::attention_family::context::BlockVisitRuntime<'_>,
                effect_reset_child: $crate::attention_family::context::ResetRuntime<'_>,
                effect_reset_ok: $crate::attention_family::context::ResetRuntime<'_>,
                effect_reset_error: $crate::attention_family::context::ResetRuntime<'_>,
                effect_release_child: $crate::attention_family::context::ReleaseRuntime<'_>,
                effect_release_ok: $crate::attention_family::context::ReleaseRuntime<'_>,
                effect_release_error: $crate::attention_family::context::ReleaseRuntime<'_>,
                effect_busy_release: $crate::attention_family::context::ReleaseRuntime<'_>,
            );
            fn effect_unexpected(&mut self) -> Result<(), ()> {
                <$crate::attention_family::context::Context<$policy>>::effect_unexpected(self)
            }
        }
    };
}

pub(crate) use forward_effects;
pub(crate) use forward_guards;
pub(crate) use impl_machine_context;
