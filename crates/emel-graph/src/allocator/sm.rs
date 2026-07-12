//! State machine scaffold port — not a stable public API.
//! Bodies are stubs (`todo!`) until contexts/guards/actions are ported from C++.

#![allow(
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::empty_structs_with_brackets,
    clippy::missing_const_for_fn,
    dead_code,
    unused_imports,
    missing_docs
)]

use sml::sml;

// --- machine GraphAllocator from emel.cpp/src/emel/graph/allocator/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventAllocateGraphPlan;

sml! {
    GraphAllocator {
        "liveness_pass_model"_s <= *"ready"_s + event<EventAllocateGraphPlan> [valid_allocate] / begin_allocate,
        "ready"_s <= "ready"_s + event<EventAllocateGraphPlan> [invalid_allocate_with_dispatchable_output] / reject_invalid_allocate_with_dispatch,
        "ready"_s <= "ready"_s + event<EventAllocateGraphPlan> [invalid_allocate_with_output_only] / reject_invalid_allocate_with_output_only,
        "ready"_s <= "ready"_s + event<EventAllocateGraphPlan> [invalid_allocate_without_output] / reject_invalid_allocate_without_output,
        "liveness_decision"_s <= "liveness_pass_model"_s + completion<EventAllocateGraphPlan>,
        "ordering_pass_model"_s <= "liveness_decision"_s + completion<EventAllocateGraphPlan> [liveness_done],
        "allocation_decision"_s <= "liveness_decision"_s + completion<EventAllocateGraphPlan> [liveness_failed],
        "ordering_decision"_s <= "ordering_pass_model"_s + completion<EventAllocateGraphPlan>,
        "placement_pass_model"_s <= "ordering_decision"_s + completion<EventAllocateGraphPlan> [ordering_done],
        "allocation_decision"_s <= "ordering_decision"_s + completion<EventAllocateGraphPlan> [ordering_failed],
        "placement_decision"_s <= "placement_pass_model"_s + completion<EventAllocateGraphPlan>,
        "allocation_decision"_s <= "placement_decision"_s + completion<EventAllocateGraphPlan> [placement_done] / commit_plan,
        "allocation_decision"_s <= "placement_decision"_s + completion<EventAllocateGraphPlan> [placement_failed],
        "ready"_s <= "allocation_decision"_s + completion<EventAllocateGraphPlan> [allocation_error_none] / dispatch_done,
        "ready"_s <= "allocation_decision"_s + completion<EventAllocateGraphPlan> [allocation_error_invalid_request] / dispatch_error_from_allocation_decision,
        "ready"_s <= "allocation_decision"_s + completion<EventAllocateGraphPlan> [allocation_error_capacity] / dispatch_error_from_allocation_decision,
        "ready"_s <= "allocation_decision"_s + completion<EventAllocateGraphPlan> [allocation_error_internal_error] / dispatch_error_from_allocation_decision,
        "ready"_s <= "allocation_decision"_s + completion<EventAllocateGraphPlan> [allocation_error_untracked] / dispatch_error_from_allocation_decision,
        "ready"_s <= "allocation_decision"_s + completion<EventAllocateGraphPlan> [allocation_error_unknown] / dispatch_error_from_allocation_decision,
        "ready"_s <= "ready"_s + unexpected_event<_> / on_unexpected_from_ready,
        "allocation_decision"_s <= "liveness_decision"_s + unexpected_event<_> / on_unexpected_from_liveness_decision,
        "allocation_decision"_s <= "ordering_decision"_s + unexpected_event<_> / on_unexpected_from_ordering_decision,
        "allocation_decision"_s <= "placement_decision"_s + unexpected_event<_> / on_unexpected_from_placement_decision,
        "ready"_s <= "allocation_decision"_s + unexpected_event<_> / on_unexpected_from_allocation_decision,
    }
}

/// Context for `GraphAllocator` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct GraphAllocatorContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl GraphAllocatorStateMachineContext for GraphAllocatorContext {
    fn allocation_error_capacity(&self, _event: &EventAllocateGraphPlan) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/allocator/guards.hpp::allocation_error_capacity
        todo!(
            "TODO: port guard `allocation_error_capacity` from emel.cpp/src/emel/graph/allocator/guards.hpp"
        )
    }
    fn allocation_error_internal_error(&self, _event: &EventAllocateGraphPlan) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/allocator/guards.hpp::allocation_error_internal_error
        todo!(
            "TODO: port guard `allocation_error_internal_error` from emel.cpp/src/emel/graph/allocator/guards.hpp"
        )
    }
    fn allocation_error_invalid_request(
        &self,
        _event: &EventAllocateGraphPlan,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/allocator/guards.hpp::allocation_error_invalid_request
        todo!(
            "TODO: port guard `allocation_error_invalid_request` from emel.cpp/src/emel/graph/allocator/guards.hpp"
        )
    }
    fn allocation_error_none(&self, _event: &EventAllocateGraphPlan) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/allocator/guards.hpp::allocation_error_none
        todo!(
            "TODO: port guard `allocation_error_none` from emel.cpp/src/emel/graph/allocator/guards.hpp"
        )
    }
    fn allocation_error_unknown(&self, _event: &EventAllocateGraphPlan) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/allocator/guards.hpp::allocation_error_unknown
        todo!(
            "TODO: port guard `allocation_error_unknown` from emel.cpp/src/emel/graph/allocator/guards.hpp"
        )
    }
    fn allocation_error_untracked(&self, _event: &EventAllocateGraphPlan) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/allocator/guards.hpp::allocation_error_untracked
        todo!(
            "TODO: port guard `allocation_error_untracked` from emel.cpp/src/emel/graph/allocator/guards.hpp"
        )
    }
    fn begin_allocate(&mut self, _event: &EventAllocateGraphPlan) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/allocator/actions.hpp::begin_allocate
        todo!(
            "TODO: port action `begin_allocate` from emel.cpp/src/emel/graph/allocator/actions.hpp"
        )
    }
    fn commit_plan(&mut self, _event: &EventAllocateGraphPlan) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/allocator/actions.hpp::commit_plan
        todo!("TODO: port action `commit_plan` from emel.cpp/src/emel/graph/allocator/actions.hpp")
    }
    fn dispatch_done(&mut self, _event: &EventAllocateGraphPlan) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/allocator/actions.hpp::dispatch_done
        todo!(
            "TODO: port action `dispatch_done` from emel.cpp/src/emel/graph/allocator/actions.hpp"
        )
    }
    fn dispatch_error_from_allocation_decision(
        &mut self,
        _event: &EventAllocateGraphPlan,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/allocator/actions.hpp::dispatch_error
        todo!(
            "TODO: port action `dispatch_error` from emel.cpp/src/emel/graph/allocator/actions.hpp"
        )
    }
    fn invalid_allocate_with_dispatchable_output(
        &self,
        _event: &EventAllocateGraphPlan,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/allocator/guards.hpp::invalid_allocate_with_dispatchable_output
        todo!(
            "TODO: port guard `invalid_allocate_with_dispatchable_output` from emel.cpp/src/emel/graph/allocator/guards.hpp"
        )
    }
    fn invalid_allocate_with_output_only(
        &self,
        _event: &EventAllocateGraphPlan,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/allocator/guards.hpp::invalid_allocate_with_output_only
        todo!(
            "TODO: port guard `invalid_allocate_with_output_only` from emel.cpp/src/emel/graph/allocator/guards.hpp"
        )
    }
    fn invalid_allocate_without_output(&self, _event: &EventAllocateGraphPlan) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/allocator/guards.hpp::invalid_allocate_without_output
        todo!(
            "TODO: port guard `invalid_allocate_without_output` from emel.cpp/src/emel/graph/allocator/guards.hpp"
        )
    }
    fn liveness_done(&self, _event: &EventAllocateGraphPlan) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/allocator/guards.hpp::liveness_done
        todo!("TODO: port guard `liveness_done` from emel.cpp/src/emel/graph/allocator/guards.hpp")
    }
    fn liveness_failed(&self, _event: &EventAllocateGraphPlan) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/allocator/guards.hpp::liveness_failed
        todo!(
            "TODO: port guard `liveness_failed` from emel.cpp/src/emel/graph/allocator/guards.hpp"
        )
    }
    fn on_unexpected_from_allocation_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/allocator/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/allocator/actions.hpp"
        )
    }
    fn on_unexpected_from_liveness_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/allocator/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/allocator/actions.hpp"
        )
    }
    fn on_unexpected_from_ordering_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/allocator/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/allocator/actions.hpp"
        )
    }
    fn on_unexpected_from_placement_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/allocator/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/allocator/actions.hpp"
        )
    }
    fn on_unexpected_from_ready(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/allocator/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/allocator/actions.hpp"
        )
    }
    fn ordering_done(&self, _event: &EventAllocateGraphPlan) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/allocator/guards.hpp::ordering_done
        todo!("TODO: port guard `ordering_done` from emel.cpp/src/emel/graph/allocator/guards.hpp")
    }
    fn ordering_failed(&self, _event: &EventAllocateGraphPlan) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/allocator/guards.hpp::ordering_failed
        todo!(
            "TODO: port guard `ordering_failed` from emel.cpp/src/emel/graph/allocator/guards.hpp"
        )
    }
    fn placement_done(&self, _event: &EventAllocateGraphPlan) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/allocator/guards.hpp::placement_done
        todo!("TODO: port guard `placement_done` from emel.cpp/src/emel/graph/allocator/guards.hpp")
    }
    fn placement_failed(&self, _event: &EventAllocateGraphPlan) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/allocator/guards.hpp::placement_failed
        todo!(
            "TODO: port guard `placement_failed` from emel.cpp/src/emel/graph/allocator/guards.hpp"
        )
    }
    fn reject_invalid_allocate_with_dispatch(
        &mut self,
        _event: &EventAllocateGraphPlan,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/allocator/actions.hpp::reject_invalid_allocate_with_dispatch
        todo!(
            "TODO: port action `reject_invalid_allocate_with_dispatch` from emel.cpp/src/emel/graph/allocator/actions.hpp"
        )
    }
    fn reject_invalid_allocate_with_output_only(
        &mut self,
        _event: &EventAllocateGraphPlan,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/allocator/actions.hpp::reject_invalid_allocate_with_output_only
        todo!(
            "TODO: port action `reject_invalid_allocate_with_output_only` from emel.cpp/src/emel/graph/allocator/actions.hpp"
        )
    }
    fn reject_invalid_allocate_without_output(
        &mut self,
        _event: &EventAllocateGraphPlan,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/allocator/actions.hpp::reject_invalid_allocate_without_output
        todo!(
            "TODO: port action `reject_invalid_allocate_without_output` from emel.cpp/src/emel/graph/allocator/actions.hpp"
        )
    }
    fn valid_allocate(&self, _event: &EventAllocateGraphPlan) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/allocator/guards.hpp::valid_allocate
        todo!("TODO: port guard `valid_allocate` from emel.cpp/src/emel/graph/allocator/guards.hpp")
    }
}
