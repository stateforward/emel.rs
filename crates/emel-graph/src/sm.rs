//! State machine scaffold port — not a stable public API.
//! Bodies are stubs (`todo!`) until contexts/guards/actions are ported from C++.

#![allow(
    clippy::enum_variant_names,
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

// --- machine Graph from emel.cpp/src/emel/graph/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventComputeGraph;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventComputeReservedGraph;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventReserveGraph;

sml! {
    Graph {
        "reserving"_s <= *"uninitialized"_s + event<EventReserveGraph> [valid_reserve] / begin_reserve,
        "uninitialized"_s <= "uninitialized"_s + event<EventReserveGraph> [invalid_reserve_with_dispatchable_output] / reject_invalid_reserve_with_dispatch_from_uninitialized,
        "uninitialized"_s <= "uninitialized"_s + event<EventReserveGraph> [invalid_reserve_with_output_only] / reject_invalid_reserve_with_output_only_from_uninitialized,
        "uninitialized"_s <= "uninitialized"_s + event<EventReserveGraph> [invalid_reserve_without_output] / reject_invalid_reserve_without_output_from_uninitialized,
        "reserved"_s <= "reserved"_s + event<EventReserveGraph> [valid_reserve] / reject_invalid_reserve_with_dispatch_from_reserved,
        "reserved"_s <= "reserved"_s + event<EventReserveGraph> [invalid_reserve_with_dispatchable_output] / reject_invalid_reserve_with_dispatch_from_reserved,
        "reserved"_s <= "reserved"_s + event<EventReserveGraph> [invalid_reserve_with_output_only] / reject_invalid_reserve_with_output_only_from_reserved,
        "reserved"_s <= "reserved"_s + event<EventReserveGraph> [invalid_reserve_without_output] / reject_invalid_reserve_without_output_from_reserved,
        "reserve_decision"_s <= "reserving"_s + completion<EventReserveGraph> / request_reserve,
        "reserve_tensor_decision"_s <= "reserve_decision"_s + completion<EventReserveGraph> [reserve_done] / request_tensor_reserve,
        "uninitialized"_s <= "reserve_decision"_s + completion<EventReserveGraph> [reserve_failed] / dispatch_reserve_error_from_reserve_decision,
        "reserved"_s <= "reserve_tensor_decision"_s + completion<EventReserveGraph> [tensor_reserve_done] / dispatch_reserve_done,
        "uninitialized"_s <= "reserve_tensor_decision"_s + completion<EventReserveGraph> [tensor_reserve_failed] / dispatch_reserve_error_from_reserve_tensor_decision,
        "assembling"_s <= "reserved"_s + event<EventComputeGraph> [valid_compute] / begin_compute,
        "reserved"_s <= "reserved"_s + event<EventComputeGraph> [invalid_compute_with_dispatchable_output] / reject_invalid_compute_with_dispatch_from_reserved,
        "reserved"_s <= "reserved"_s + event<EventComputeGraph> [invalid_compute_with_output_only] / reject_invalid_compute_with_output_only_from_reserved,
        "reserved"_s <= "reserved"_s + event<EventComputeGraph> [invalid_compute_without_output] / reject_invalid_compute_without_output_from_reserved,
        "uninitialized"_s <= "uninitialized"_s + event<EventComputeGraph> [valid_compute] / reject_invalid_compute_with_dispatch_from_uninitialized,
        "uninitialized"_s <= "uninitialized"_s + event<EventComputeGraph> [invalid_compute_with_dispatchable_output] / reject_invalid_compute_with_dispatch_from_uninitialized,
        "uninitialized"_s <= "uninitialized"_s + event<EventComputeGraph> [invalid_compute_with_output_only] / reject_invalid_compute_with_output_only_from_uninitialized,
        "uninitialized"_s <= "uninitialized"_s + event<EventComputeGraph> [invalid_compute_without_output] / reject_invalid_compute_without_output_from_uninitialized,
        "executing"_s <= "reserved"_s + event<EventComputeReservedGraph> [guard_valid_compute_reserved] / effect_begin_reserved_compute,
        "reserved"_s <= "reserved"_s + event<EventComputeReservedGraph> [guard_invalid_compute_reserved_with_dispatchable_output] / effect_reject_invalid_reserved_compute_with_dispatch_from_reserved,
        "reserved"_s <= "reserved"_s + event<EventComputeReservedGraph> [guard_invalid_compute_reserved_with_output_only] / effect_reject_invalid_reserved_compute_with_output_only_from_reserved,
        "reserved"_s <= "reserved"_s + event<EventComputeReservedGraph> [guard_invalid_compute_reserved_without_output] / effect_reject_invalid_reserved_compute_without_output_from_reserved,
        "uninitialized"_s <= "uninitialized"_s + event<EventComputeReservedGraph> [guard_valid_compute_reserved] / effect_reject_invalid_reserved_compute_with_dispatch_from_uninitialized,
        "uninitialized"_s <= "uninitialized"_s + event<EventComputeReservedGraph> [guard_invalid_compute_reserved_with_dispatchable_output] / effect_reject_invalid_reserved_compute_with_dispatch_from_uninitialized,
        "uninitialized"_s <= "uninitialized"_s + event<EventComputeReservedGraph> [guard_invalid_compute_reserved_with_output_only] / effect_reject_invalid_reserved_compute_with_output_only_from_uninitialized,
        "uninitialized"_s <= "uninitialized"_s + event<EventComputeReservedGraph> [guard_invalid_compute_reserved_without_output] / effect_reject_invalid_reserved_compute_without_output_from_uninitialized,
        "assemble_decision"_s <= "assembling"_s + completion<EventComputeGraph> / request_assemble,
        "executing"_s <= "assemble_decision"_s + completion<EventComputeGraph> [assemble_done],
        "compute_decision"_s <= "assemble_decision"_s + completion<EventComputeGraph> [assemble_failed],
        "execute_decision"_s <= "executing"_s + completion<EventComputeGraph> / request_execute,
        "compute_decision"_s <= "execute_decision"_s + completion<EventComputeGraph> [execute_done],
        "compute_decision"_s <= "execute_decision"_s + completion<EventComputeGraph> [execute_failed],
        "execute_decision"_s <= "executing"_s + completion<EventComputeReservedGraph> / effect_request_reserved_execute,
        "compute_decision"_s <= "execute_decision"_s + completion<EventComputeReservedGraph> [guard_reserved_execute_done],
        "compute_decision"_s <= "execute_decision"_s + completion<EventComputeReservedGraph> [guard_reserved_execute_failed],
        "reserved"_s <= "compute_decision"_s + completion<EventComputeGraph> [compute_error_none] / dispatch_compute_done,
        "reserved"_s <= "compute_decision"_s + completion<EventComputeGraph> [compute_error_invalid_request] / dispatch_compute_error_from_compute_decision,
        "reserved"_s <= "compute_decision"_s + completion<EventComputeGraph> [compute_error_assembler_failed] / dispatch_compute_error_from_compute_decision,
        "reserved"_s <= "compute_decision"_s + completion<EventComputeGraph> [compute_error_processor_failed] / dispatch_compute_error_from_compute_decision,
        "reserved"_s <= "compute_decision"_s + completion<EventComputeGraph> [compute_error_busy] / dispatch_compute_error_from_compute_decision,
        "reserved"_s <= "compute_decision"_s + completion<EventComputeGraph> [compute_error_internal_error] / dispatch_compute_error_from_compute_decision,
        "reserved"_s <= "compute_decision"_s + completion<EventComputeGraph> [compute_error_untracked] / dispatch_compute_error_from_compute_decision,
        "reserved"_s <= "compute_decision"_s + completion<EventComputeGraph> [compute_error_unknown] / dispatch_compute_error_from_compute_decision,
        "reserved"_s <= "compute_decision"_s + completion<EventComputeReservedGraph> [guard_reserved_compute_error_none] / effect_dispatch_reserved_compute_done,
        "reserved"_s <= "compute_decision"_s + completion<EventComputeReservedGraph> [guard_reserved_compute_error_invalid_request] / effect_dispatch_reserved_compute_error_from_compute_decision,
        "reserved"_s <= "compute_decision"_s + completion<EventComputeReservedGraph> [guard_reserved_compute_error_assembler_failed] / effect_dispatch_reserved_compute_error_from_compute_decision,
        "reserved"_s <= "compute_decision"_s + completion<EventComputeReservedGraph> [guard_reserved_compute_error_processor_failed] / effect_dispatch_reserved_compute_error_from_compute_decision,
        "reserved"_s <= "compute_decision"_s + completion<EventComputeReservedGraph> [guard_reserved_compute_error_busy] / effect_dispatch_reserved_compute_error_from_compute_decision,
        "reserved"_s <= "compute_decision"_s + completion<EventComputeReservedGraph> [guard_reserved_compute_error_internal_error] / effect_dispatch_reserved_compute_error_from_compute_decision,
        "reserved"_s <= "compute_decision"_s + completion<EventComputeReservedGraph> [guard_reserved_compute_error_untracked] / effect_dispatch_reserved_compute_error_from_compute_decision,
        "reserved"_s <= "compute_decision"_s + completion<EventComputeReservedGraph> [guard_reserved_compute_error_unknown] / effect_dispatch_reserved_compute_error_from_compute_decision,
        "uninitialized"_s <= "uninitialized"_s + unexpected_event<_> / on_unexpected_from_uninitialized,
        "reserved"_s <= "reserved"_s + unexpected_event<_> / on_unexpected_from_reserved,
        "uninitialized"_s <= "reserving"_s + unexpected_event<_> / on_unexpected_from_reserving,
        "uninitialized"_s <= "reserve_decision"_s + unexpected_event<_> / on_unexpected_from_reserve_decision,
        "uninitialized"_s <= "reserve_tensor_decision"_s + unexpected_event<_> / on_unexpected_from_reserve_tensor_decision,
        "reserved"_s <= "assembling"_s + unexpected_event<_> / on_unexpected_from_assembling,
        "reserved"_s <= "assemble_decision"_s + unexpected_event<_> / on_unexpected_from_assemble_decision,
        "reserved"_s <= "executing"_s + unexpected_event<_> / on_unexpected_from_executing,
        "reserved"_s <= "execute_decision"_s + unexpected_event<_> / on_unexpected_from_execute_decision,
        "reserved"_s <= "compute_decision"_s + unexpected_event<_> / on_unexpected_from_compute_decision,
    }
}

/// Context for `Graph` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct GraphContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl GraphStateMachineContext for GraphContext {
    fn assemble_done(&self, _event: &EventComputeGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/guards.hpp::assemble_done
        todo!("TODO: port guard `assemble_done` from emel.cpp/src/emel/graph/guards.hpp")
    }
    fn assemble_failed(&self, _event: &EventComputeGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/guards.hpp::assemble_failed
        todo!("TODO: port guard `assemble_failed` from emel.cpp/src/emel/graph/guards.hpp")
    }
    fn begin_compute(&mut self, _event: &EventComputeGraph) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/actions.hpp::begin_compute
        todo!("TODO: port action `begin_compute` from emel.cpp/src/emel/graph/actions.hpp")
    }
    fn begin_reserve(&mut self, _event: &EventReserveGraph) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/actions.hpp::begin_reserve
        todo!("TODO: port action `begin_reserve` from emel.cpp/src/emel/graph/actions.hpp")
    }
    fn compute_error_assembler_failed(&self, _event: &EventComputeGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/guards.hpp::compute_error_assembler_failed
        todo!(
            "TODO: port guard `compute_error_assembler_failed` from emel.cpp/src/emel/graph/guards.hpp"
        )
    }
    fn compute_error_busy(&self, _event: &EventComputeGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/guards.hpp::compute_error_busy
        todo!("TODO: port guard `compute_error_busy` from emel.cpp/src/emel/graph/guards.hpp")
    }
    fn compute_error_internal_error(&self, _event: &EventComputeGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/guards.hpp::compute_error_internal_error
        todo!(
            "TODO: port guard `compute_error_internal_error` from emel.cpp/src/emel/graph/guards.hpp"
        )
    }
    fn compute_error_invalid_request(&self, _event: &EventComputeGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/guards.hpp::compute_error_invalid_request
        todo!(
            "TODO: port guard `compute_error_invalid_request` from emel.cpp/src/emel/graph/guards.hpp"
        )
    }
    fn compute_error_none(&self, _event: &EventComputeGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/guards.hpp::compute_error_none
        todo!("TODO: port guard `compute_error_none` from emel.cpp/src/emel/graph/guards.hpp")
    }
    fn compute_error_processor_failed(&self, _event: &EventComputeGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/guards.hpp::compute_error_processor_failed
        todo!(
            "TODO: port guard `compute_error_processor_failed` from emel.cpp/src/emel/graph/guards.hpp"
        )
    }
    fn compute_error_unknown(&self, _event: &EventComputeGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/guards.hpp::compute_error_unknown
        todo!("TODO: port guard `compute_error_unknown` from emel.cpp/src/emel/graph/guards.hpp")
    }
    fn compute_error_untracked(&self, _event: &EventComputeGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/guards.hpp::compute_error_untracked
        todo!("TODO: port guard `compute_error_untracked` from emel.cpp/src/emel/graph/guards.hpp")
    }
    fn dispatch_compute_done(&mut self, _event: &EventComputeGraph) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/actions.hpp::dispatch_compute_done
        todo!("TODO: port action `dispatch_compute_done` from emel.cpp/src/emel/graph/actions.hpp")
    }
    fn dispatch_compute_error_from_compute_decision(
        &mut self,
        _event: &EventComputeGraph,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/actions.hpp::dispatch_compute_error
        todo!("TODO: port action `dispatch_compute_error` from emel.cpp/src/emel/graph/actions.hpp")
    }
    fn dispatch_reserve_done(&mut self, _event: &EventReserveGraph) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/actions.hpp::dispatch_reserve_done
        todo!("TODO: port action `dispatch_reserve_done` from emel.cpp/src/emel/graph/actions.hpp")
    }
    fn dispatch_reserve_error_from_reserve_decision(
        &mut self,
        _event: &EventReserveGraph,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/actions.hpp::dispatch_reserve_error
        todo!("TODO: port action `dispatch_reserve_error` from emel.cpp/src/emel/graph/actions.hpp")
    }
    fn dispatch_reserve_error_from_reserve_tensor_decision(
        &mut self,
        _event: &EventReserveGraph,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/actions.hpp::dispatch_reserve_error
        todo!("TODO: port action `dispatch_reserve_error` from emel.cpp/src/emel/graph/actions.hpp")
    }
    fn effect_begin_reserved_compute(
        &mut self,
        _event: &EventComputeReservedGraph,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/actions.hpp::effect_begin_reserved_compute
        todo!(
            "TODO: port action `effect_begin_reserved_compute` from emel.cpp/src/emel/graph/actions.hpp"
        )
    }
    fn effect_dispatch_reserved_compute_done(
        &mut self,
        _event: &EventComputeReservedGraph,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/actions.hpp::effect_dispatch_reserved_compute_done
        todo!(
            "TODO: port action `effect_dispatch_reserved_compute_done` from emel.cpp/src/emel/graph/actions.hpp"
        )
    }
    fn effect_dispatch_reserved_compute_error_from_compute_decision(
        &mut self,
        _event: &EventComputeReservedGraph,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/actions.hpp::effect_dispatch_reserved_compute_error
        todo!(
            "TODO: port action `effect_dispatch_reserved_compute_error` from emel.cpp/src/emel/graph/actions.hpp"
        )
    }
    fn effect_reject_invalid_reserved_compute_with_dispatch_from_reserved(
        &mut self,
        _event: &EventComputeReservedGraph,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/actions.hpp::effect_reject_invalid_reserved_compute_with_dispatch
        todo!(
            "TODO: port action `effect_reject_invalid_reserved_compute_with_dispatch` from emel.cpp/src/emel/graph/actions.hpp"
        )
    }
    fn effect_reject_invalid_reserved_compute_with_dispatch_from_uninitialized(
        &mut self,
        _event: &EventComputeReservedGraph,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/actions.hpp::effect_reject_invalid_reserved_compute_with_dispatch
        todo!(
            "TODO: port action `effect_reject_invalid_reserved_compute_with_dispatch` from emel.cpp/src/emel/graph/actions.hpp"
        )
    }
    fn effect_reject_invalid_reserved_compute_with_output_only_from_reserved(
        &mut self,
        _event: &EventComputeReservedGraph,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/actions.hpp::effect_reject_invalid_reserved_compute_with_output_only
        todo!(
            "TODO: port action `effect_reject_invalid_reserved_compute_with_output_only` from emel.cpp/src/emel/graph/actions.hpp"
        )
    }
    fn effect_reject_invalid_reserved_compute_with_output_only_from_uninitialized(
        &mut self,
        _event: &EventComputeReservedGraph,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/actions.hpp::effect_reject_invalid_reserved_compute_with_output_only
        todo!(
            "TODO: port action `effect_reject_invalid_reserved_compute_with_output_only` from emel.cpp/src/emel/graph/actions.hpp"
        )
    }
    fn effect_reject_invalid_reserved_compute_without_output_from_reserved(
        &mut self,
        _event: &EventComputeReservedGraph,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/actions.hpp::effect_reject_invalid_reserved_compute_without_output
        todo!(
            "TODO: port action `effect_reject_invalid_reserved_compute_without_output` from emel.cpp/src/emel/graph/actions.hpp"
        )
    }
    fn effect_reject_invalid_reserved_compute_without_output_from_uninitialized(
        &mut self,
        _event: &EventComputeReservedGraph,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/actions.hpp::effect_reject_invalid_reserved_compute_without_output
        todo!(
            "TODO: port action `effect_reject_invalid_reserved_compute_without_output` from emel.cpp/src/emel/graph/actions.hpp"
        )
    }
    fn effect_request_reserved_execute(
        &mut self,
        _event: &EventComputeReservedGraph,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/actions.hpp::effect_request_reserved_execute
        todo!(
            "TODO: port action `effect_request_reserved_execute` from emel.cpp/src/emel/graph/actions.hpp"
        )
    }
    fn execute_done(&self, _event: &EventComputeGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/guards.hpp::execute_done
        todo!("TODO: port guard `execute_done` from emel.cpp/src/emel/graph/guards.hpp")
    }
    fn execute_failed(&self, _event: &EventComputeGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/guards.hpp::execute_failed
        todo!("TODO: port guard `execute_failed` from emel.cpp/src/emel/graph/guards.hpp")
    }
    fn guard_invalid_compute_reserved_with_dispatchable_output(
        &self,
        _event: &EventComputeReservedGraph,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/guards.hpp::guard_invalid_compute_reserved_with_dispatchable_output
        todo!(
            "TODO: port guard `guard_invalid_compute_reserved_with_dispatchable_output` from emel.cpp/src/emel/graph/guards.hpp"
        )
    }
    fn guard_invalid_compute_reserved_with_output_only(
        &self,
        _event: &EventComputeReservedGraph,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/guards.hpp::guard_invalid_compute_reserved_with_output_only
        todo!(
            "TODO: port guard `guard_invalid_compute_reserved_with_output_only` from emel.cpp/src/emel/graph/guards.hpp"
        )
    }
    fn guard_invalid_compute_reserved_without_output(
        &self,
        _event: &EventComputeReservedGraph,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/guards.hpp::guard_invalid_compute_reserved_without_output
        todo!(
            "TODO: port guard `guard_invalid_compute_reserved_without_output` from emel.cpp/src/emel/graph/guards.hpp"
        )
    }
    fn guard_reserved_compute_error_assembler_failed(
        &self,
        _event: &EventComputeReservedGraph,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/guards.hpp::guard_reserved_compute_error_assembler_failed
        todo!(
            "TODO: port guard `guard_reserved_compute_error_assembler_failed` from emel.cpp/src/emel/graph/guards.hpp"
        )
    }
    fn guard_reserved_compute_error_busy(
        &self,
        _event: &EventComputeReservedGraph,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/guards.hpp::guard_reserved_compute_error_busy
        todo!(
            "TODO: port guard `guard_reserved_compute_error_busy` from emel.cpp/src/emel/graph/guards.hpp"
        )
    }
    fn guard_reserved_compute_error_internal_error(
        &self,
        _event: &EventComputeReservedGraph,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/guards.hpp::guard_reserved_compute_error_internal_error
        todo!(
            "TODO: port guard `guard_reserved_compute_error_internal_error` from emel.cpp/src/emel/graph/guards.hpp"
        )
    }
    fn guard_reserved_compute_error_invalid_request(
        &self,
        _event: &EventComputeReservedGraph,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/guards.hpp::guard_reserved_compute_error_invalid_request
        todo!(
            "TODO: port guard `guard_reserved_compute_error_invalid_request` from emel.cpp/src/emel/graph/guards.hpp"
        )
    }
    fn guard_reserved_compute_error_none(
        &self,
        _event: &EventComputeReservedGraph,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/guards.hpp::guard_reserved_compute_error_none
        todo!(
            "TODO: port guard `guard_reserved_compute_error_none` from emel.cpp/src/emel/graph/guards.hpp"
        )
    }
    fn guard_reserved_compute_error_processor_failed(
        &self,
        _event: &EventComputeReservedGraph,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/guards.hpp::guard_reserved_compute_error_processor_failed
        todo!(
            "TODO: port guard `guard_reserved_compute_error_processor_failed` from emel.cpp/src/emel/graph/guards.hpp"
        )
    }
    fn guard_reserved_compute_error_unknown(
        &self,
        _event: &EventComputeReservedGraph,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/guards.hpp::guard_reserved_compute_error_unknown
        todo!(
            "TODO: port guard `guard_reserved_compute_error_unknown` from emel.cpp/src/emel/graph/guards.hpp"
        )
    }
    fn guard_reserved_compute_error_untracked(
        &self,
        _event: &EventComputeReservedGraph,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/guards.hpp::guard_reserved_compute_error_untracked
        todo!(
            "TODO: port guard `guard_reserved_compute_error_untracked` from emel.cpp/src/emel/graph/guards.hpp"
        )
    }
    fn guard_reserved_execute_done(&self, _event: &EventComputeReservedGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/guards.hpp::guard_reserved_execute_done
        todo!(
            "TODO: port guard `guard_reserved_execute_done` from emel.cpp/src/emel/graph/guards.hpp"
        )
    }
    fn guard_reserved_execute_failed(
        &self,
        _event: &EventComputeReservedGraph,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/guards.hpp::guard_reserved_execute_failed
        todo!(
            "TODO: port guard `guard_reserved_execute_failed` from emel.cpp/src/emel/graph/guards.hpp"
        )
    }
    fn guard_valid_compute_reserved(&self, _event: &EventComputeReservedGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/guards.hpp::guard_valid_compute_reserved
        todo!(
            "TODO: port guard `guard_valid_compute_reserved` from emel.cpp/src/emel/graph/guards.hpp"
        )
    }
    fn invalid_compute_with_dispatchable_output(
        &self,
        _event: &EventComputeGraph,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/guards.hpp::invalid_compute_with_dispatchable_output
        todo!(
            "TODO: port guard `invalid_compute_with_dispatchable_output` from emel.cpp/src/emel/graph/guards.hpp"
        )
    }
    fn invalid_compute_with_output_only(&self, _event: &EventComputeGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/guards.hpp::invalid_compute_with_output_only
        todo!(
            "TODO: port guard `invalid_compute_with_output_only` from emel.cpp/src/emel/graph/guards.hpp"
        )
    }
    fn invalid_compute_without_output(&self, _event: &EventComputeGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/guards.hpp::invalid_compute_without_output
        todo!(
            "TODO: port guard `invalid_compute_without_output` from emel.cpp/src/emel/graph/guards.hpp"
        )
    }
    fn invalid_reserve_with_dispatchable_output(
        &self,
        _event: &EventReserveGraph,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/guards.hpp::invalid_reserve_with_dispatchable_output
        todo!(
            "TODO: port guard `invalid_reserve_with_dispatchable_output` from emel.cpp/src/emel/graph/guards.hpp"
        )
    }
    fn invalid_reserve_with_output_only(&self, _event: &EventReserveGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/guards.hpp::invalid_reserve_with_output_only
        todo!(
            "TODO: port guard `invalid_reserve_with_output_only` from emel.cpp/src/emel/graph/guards.hpp"
        )
    }
    fn invalid_reserve_without_output(&self, _event: &EventReserveGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/guards.hpp::invalid_reserve_without_output
        todo!(
            "TODO: port guard `invalid_reserve_without_output` from emel.cpp/src/emel/graph/guards.hpp"
        )
    }
    fn on_unexpected_from_assemble_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/actions.hpp")
    }
    fn on_unexpected_from_assembling(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/actions.hpp")
    }
    fn on_unexpected_from_compute_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/actions.hpp")
    }
    fn on_unexpected_from_execute_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/actions.hpp")
    }
    fn on_unexpected_from_executing(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/actions.hpp")
    }
    fn on_unexpected_from_reserve_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/actions.hpp")
    }
    fn on_unexpected_from_reserve_tensor_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/actions.hpp")
    }
    fn on_unexpected_from_reserved(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/actions.hpp")
    }
    fn on_unexpected_from_reserving(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/actions.hpp")
    }
    fn on_unexpected_from_uninitialized(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/actions.hpp::on_unexpected
        todo!("TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/actions.hpp")
    }
    fn reject_invalid_compute_with_dispatch_from_reserved(
        &mut self,
        _event: &EventComputeGraph,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/actions.hpp::reject_invalid_compute_with_dispatch
        todo!(
            "TODO: port action `reject_invalid_compute_with_dispatch` from emel.cpp/src/emel/graph/actions.hpp"
        )
    }
    fn reject_invalid_compute_with_dispatch_from_uninitialized(
        &mut self,
        _event: &EventComputeGraph,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/actions.hpp::reject_invalid_compute_with_dispatch
        todo!(
            "TODO: port action `reject_invalid_compute_with_dispatch` from emel.cpp/src/emel/graph/actions.hpp"
        )
    }
    fn reject_invalid_compute_with_output_only_from_reserved(
        &mut self,
        _event: &EventComputeGraph,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/actions.hpp::reject_invalid_compute_with_output_only
        todo!(
            "TODO: port action `reject_invalid_compute_with_output_only` from emel.cpp/src/emel/graph/actions.hpp"
        )
    }
    fn reject_invalid_compute_with_output_only_from_uninitialized(
        &mut self,
        _event: &EventComputeGraph,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/actions.hpp::reject_invalid_compute_with_output_only
        todo!(
            "TODO: port action `reject_invalid_compute_with_output_only` from emel.cpp/src/emel/graph/actions.hpp"
        )
    }
    fn reject_invalid_compute_without_output_from_reserved(
        &mut self,
        _event: &EventComputeGraph,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/actions.hpp::reject_invalid_compute_without_output
        todo!(
            "TODO: port action `reject_invalid_compute_without_output` from emel.cpp/src/emel/graph/actions.hpp"
        )
    }
    fn reject_invalid_compute_without_output_from_uninitialized(
        &mut self,
        _event: &EventComputeGraph,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/actions.hpp::reject_invalid_compute_without_output
        todo!(
            "TODO: port action `reject_invalid_compute_without_output` from emel.cpp/src/emel/graph/actions.hpp"
        )
    }
    fn reject_invalid_reserve_with_dispatch_from_reserved(
        &mut self,
        _event: &EventReserveGraph,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/actions.hpp::reject_invalid_reserve_with_dispatch
        todo!(
            "TODO: port action `reject_invalid_reserve_with_dispatch` from emel.cpp/src/emel/graph/actions.hpp"
        )
    }
    fn reject_invalid_reserve_with_dispatch_from_uninitialized(
        &mut self,
        _event: &EventReserveGraph,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/actions.hpp::reject_invalid_reserve_with_dispatch
        todo!(
            "TODO: port action `reject_invalid_reserve_with_dispatch` from emel.cpp/src/emel/graph/actions.hpp"
        )
    }
    fn reject_invalid_reserve_with_output_only_from_reserved(
        &mut self,
        _event: &EventReserveGraph,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/actions.hpp::reject_invalid_reserve_with_output_only
        todo!(
            "TODO: port action `reject_invalid_reserve_with_output_only` from emel.cpp/src/emel/graph/actions.hpp"
        )
    }
    fn reject_invalid_reserve_with_output_only_from_uninitialized(
        &mut self,
        _event: &EventReserveGraph,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/actions.hpp::reject_invalid_reserve_with_output_only
        todo!(
            "TODO: port action `reject_invalid_reserve_with_output_only` from emel.cpp/src/emel/graph/actions.hpp"
        )
    }
    fn reject_invalid_reserve_without_output_from_reserved(
        &mut self,
        _event: &EventReserveGraph,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/actions.hpp::reject_invalid_reserve_without_output
        todo!(
            "TODO: port action `reject_invalid_reserve_without_output` from emel.cpp/src/emel/graph/actions.hpp"
        )
    }
    fn reject_invalid_reserve_without_output_from_uninitialized(
        &mut self,
        _event: &EventReserveGraph,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/actions.hpp::reject_invalid_reserve_without_output
        todo!(
            "TODO: port action `reject_invalid_reserve_without_output` from emel.cpp/src/emel/graph/actions.hpp"
        )
    }
    fn request_assemble(&mut self, _event: &EventComputeGraph) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/actions.hpp::request_assemble
        todo!("TODO: port action `request_assemble` from emel.cpp/src/emel/graph/actions.hpp")
    }
    fn request_execute(&mut self, _event: &EventComputeGraph) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/actions.hpp::request_execute
        todo!("TODO: port action `request_execute` from emel.cpp/src/emel/graph/actions.hpp")
    }
    fn request_reserve(&mut self, _event: &EventReserveGraph) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/actions.hpp::request_reserve
        todo!("TODO: port action `request_reserve` from emel.cpp/src/emel/graph/actions.hpp")
    }
    fn request_tensor_reserve(&mut self, _event: &EventReserveGraph) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/actions.hpp::request_tensor_reserve
        todo!("TODO: port action `request_tensor_reserve` from emel.cpp/src/emel/graph/actions.hpp")
    }
    fn reserve_done(&self, _event: &EventReserveGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/guards.hpp::reserve_done
        todo!("TODO: port guard `reserve_done` from emel.cpp/src/emel/graph/guards.hpp")
    }
    fn reserve_failed(&self, _event: &EventReserveGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/guards.hpp::reserve_failed
        todo!("TODO: port guard `reserve_failed` from emel.cpp/src/emel/graph/guards.hpp")
    }
    fn tensor_reserve_done(&self, _event: &EventReserveGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/guards.hpp::tensor_reserve_done
        todo!("TODO: port guard `tensor_reserve_done` from emel.cpp/src/emel/graph/guards.hpp")
    }
    fn tensor_reserve_failed(&self, _event: &EventReserveGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/guards.hpp::tensor_reserve_failed
        todo!("TODO: port guard `tensor_reserve_failed` from emel.cpp/src/emel/graph/guards.hpp")
    }
    fn valid_compute(&self, _event: &EventComputeGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/guards.hpp::valid_compute
        todo!("TODO: port guard `valid_compute` from emel.cpp/src/emel/graph/guards.hpp")
    }
    fn valid_reserve(&self, _event: &EventReserveGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/guards.hpp::valid_reserve
        todo!("TODO: port guard `valid_reserve` from emel.cpp/src/emel/graph/guards.hpp")
    }
}
