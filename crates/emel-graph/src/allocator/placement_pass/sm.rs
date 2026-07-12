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

// --- machine GraphAllocatorPlacementPass from emel.cpp/src/emel/graph/allocator/placement_pass/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct AllocatorEventAllocateGraphPlan;

sml! {
    GraphAllocatorPlacementPass {
        "allocate_failed"_s <= *"deciding"_s + completion<AllocatorEventAllocateGraphPlan> [phase_prefailed] / mark_failed_prefailed,
        "allocated"_s <= "deciding"_s + completion<AllocatorEventAllocateGraphPlan> [phase_done] / mark_done,
        "allocate_failed"_s <= "deciding"_s + completion<AllocatorEventAllocateGraphPlan> [phase_prereq_failed] / mark_failed_prereq,
        "allocate_failed"_s <= "deciding"_s + completion<AllocatorEventAllocateGraphPlan> [phase_capacity_exceeded] / mark_failed_capacity,
        "allocate_failed"_s <= "deciding"_s + completion<AllocatorEventAllocateGraphPlan> [phase_invalid_request] / mark_failed_invalid_request,
        "allocate_failed"_s <= "deciding"_s + completion<AllocatorEventAllocateGraphPlan> / mark_failed_internal,
        "unexpected_event"_s <= "deciding"_s + unexpected_event<_> / on_unexpected_from_deciding,
        "unexpected_event"_s <= "allocated"_s + unexpected_event<_> / on_unexpected_from_allocated,
        "unexpected_event"_s <= "allocate_failed"_s + unexpected_event<_> / on_unexpected_from_allocate_failed,
        "unexpected_event"_s <= "unexpected_event"_s + unexpected_event<_> / on_unexpected_from_unexpected_event,
        "allocated"_s = X,
        "allocate_failed"_s = X,
    }
}

/// Context for `GraphAllocatorPlacementPass` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct GraphAllocatorPlacementPassContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl GraphAllocatorPlacementPassStateMachineContext for GraphAllocatorPlacementPassContext {
    fn mark_done(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/allocator/placement_pass/actions.hpp::mark_done
        todo!(
            "TODO: port action `mark_done` from emel.cpp/src/emel/graph/allocator/placement_pass/actions.hpp"
        )
    }
    fn mark_failed_capacity(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/allocator/placement_pass/actions.hpp::mark_failed_capacity
        todo!(
            "TODO: port action `mark_failed_capacity` from emel.cpp/src/emel/graph/allocator/placement_pass/actions.hpp"
        )
    }
    fn mark_failed_internal(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/allocator/placement_pass/actions.hpp::mark_failed_internal
        todo!(
            "TODO: port action `mark_failed_internal` from emel.cpp/src/emel/graph/allocator/placement_pass/actions.hpp"
        )
    }
    fn mark_failed_invalid_request(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/allocator/placement_pass/actions.hpp::mark_failed_invalid_request
        todo!(
            "TODO: port action `mark_failed_invalid_request` from emel.cpp/src/emel/graph/allocator/placement_pass/actions.hpp"
        )
    }
    fn mark_failed_prefailed(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/allocator/placement_pass/actions.hpp::mark_failed_prefailed
        todo!(
            "TODO: port action `mark_failed_prefailed` from emel.cpp/src/emel/graph/allocator/placement_pass/actions.hpp"
        )
    }
    fn mark_failed_prereq(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/allocator/placement_pass/actions.hpp::mark_failed_prereq
        todo!(
            "TODO: port action `mark_failed_prereq` from emel.cpp/src/emel/graph/allocator/placement_pass/actions.hpp"
        )
    }
    fn on_unexpected_from_allocate_failed(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/allocator/placement_pass/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/allocator/placement_pass/actions.hpp"
        )
    }
    fn on_unexpected_from_allocated(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/allocator/placement_pass/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/allocator/placement_pass/actions.hpp"
        )
    }
    fn on_unexpected_from_deciding(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/allocator/placement_pass/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/allocator/placement_pass/actions.hpp"
        )
    }
    fn on_unexpected_from_unexpected_event(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/allocator/placement_pass/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/allocator/placement_pass/actions.hpp"
        )
    }
    fn phase_capacity_exceeded(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/allocator/placement_pass/guards.hpp::phase_capacity_exceeded
        todo!(
            "TODO: port guard `phase_capacity_exceeded` from emel.cpp/src/emel/graph/allocator/placement_pass/guards.hpp"
        )
    }
    fn phase_done(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/allocator/placement_pass/guards.hpp::phase_done
        todo!(
            "TODO: port guard `phase_done` from emel.cpp/src/emel/graph/allocator/placement_pass/guards.hpp"
        )
    }
    fn phase_invalid_request(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/allocator/placement_pass/guards.hpp::phase_invalid_request
        todo!(
            "TODO: port guard `phase_invalid_request` from emel.cpp/src/emel/graph/allocator/placement_pass/guards.hpp"
        )
    }
    fn phase_prefailed(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/allocator/placement_pass/guards.hpp::phase_prefailed
        todo!(
            "TODO: port guard `phase_prefailed` from emel.cpp/src/emel/graph/allocator/placement_pass/guards.hpp"
        )
    }
    fn phase_prereq_failed(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/allocator/placement_pass/guards.hpp::phase_prereq_failed
        todo!(
            "TODO: port guard `phase_prereq_failed` from emel.cpp/src/emel/graph/allocator/placement_pass/guards.hpp"
        )
    }
}
