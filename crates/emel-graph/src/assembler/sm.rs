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

// --- machine GraphAssembler from emel.cpp/src/emel/graph/assembler/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventAssembleGraph;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventReserveGraph;

sml! {
    GraphAssembler {
        "reserve_validate_pass_model"_s <= *"uninitialized"_s + event<EventReserveGraph> [valid_reserve] / begin_reserve,
        "uninitialized"_s <= "uninitialized"_s + event<EventReserveGraph> [invalid_reserve_with_dispatchable_output] / reject_invalid_reserve_with_dispatch_from_uninitialized,
        "uninitialized"_s <= "uninitialized"_s + event<EventReserveGraph> [invalid_reserve_with_output_only] / reject_invalid_reserve_with_output_only_from_uninitialized,
        "uninitialized"_s <= "uninitialized"_s + event<EventReserveGraph> [invalid_reserve_without_output] / reject_invalid_reserve_without_output_from_uninitialized,
        "reserved"_s <= "reserved"_s + event<EventReserveGraph> [valid_reserve] / reject_invalid_reserve_with_dispatch_from_reserved,
        "reserved"_s <= "reserved"_s + event<EventReserveGraph> [invalid_reserve_with_dispatchable_output] / reject_invalid_reserve_with_dispatch_from_reserved,
        "reserved"_s <= "reserved"_s + event<EventReserveGraph> [invalid_reserve_with_output_only] / reject_invalid_reserve_with_output_only_from_reserved,
        "reserved"_s <= "reserved"_s + event<EventReserveGraph> [invalid_reserve_without_output] / reject_invalid_reserve_without_output_from_reserved,
        "reserve_validate_decision"_s <= "reserve_validate_pass_model"_s + completion<EventReserveGraph>,
        "reserve_build_pass_model"_s <= "reserve_validate_decision"_s + completion<EventReserveGraph> [reserve_validate_done],
        "reserve_dispatch_decision"_s <= "reserve_validate_decision"_s + completion<EventReserveGraph> [reserve_validate_failed],
        "reserve_build_decision"_s <= "reserve_build_pass_model"_s + completion<EventReserveGraph>,
        "reserve_alloc_pass_model"_s <= "reserve_build_decision"_s + completion<EventReserveGraph> [reserve_build_done],
        "reserve_dispatch_decision"_s <= "reserve_build_decision"_s + completion<EventReserveGraph> [reserve_build_failed],
        "reserve_alloc_decision"_s <= "reserve_alloc_pass_model"_s + completion<EventReserveGraph>,
        "reserve_dispatch_decision"_s <= "reserve_alloc_decision"_s + completion<EventReserveGraph> [reserve_alloc_done] / commit_reserve_result,
        "reserve_dispatch_decision"_s <= "reserve_alloc_decision"_s + completion<EventReserveGraph> [reserve_alloc_failed],
        "reserved"_s <= "reserve_dispatch_decision"_s + completion<EventReserveGraph> [reserve_error_none] / dispatch_reserve_done,
        "uninitialized"_s <= "reserve_dispatch_decision"_s + completion<EventReserveGraph> [reserve_error_invalid_request] / dispatch_reserve_error_from_reserve_dispatch_decision,
        "uninitialized"_s <= "reserve_dispatch_decision"_s + completion<EventReserveGraph> [reserve_error_capacity] / dispatch_reserve_error_from_reserve_dispatch_decision,
        "uninitialized"_s <= "reserve_dispatch_decision"_s + completion<EventReserveGraph> [reserve_error_internal_error] / dispatch_reserve_error_from_reserve_dispatch_decision,
        "uninitialized"_s <= "reserve_dispatch_decision"_s + completion<EventReserveGraph> [reserve_error_untracked] / dispatch_reserve_error_from_reserve_dispatch_decision,
        "uninitialized"_s <= "reserve_dispatch_decision"_s + completion<EventReserveGraph> [reserve_error_unknown] / dispatch_reserve_error_from_reserve_dispatch_decision,
        "assemble_validate_pass_model"_s <= "reserved"_s + event<EventAssembleGraph> [valid_assemble] / begin_assemble,
        "reserved"_s <= "reserved"_s + event<EventAssembleGraph> [invalid_assemble_with_dispatchable_output] / reject_invalid_assemble_with_dispatch_from_reserved,
        "reserved"_s <= "reserved"_s + event<EventAssembleGraph> [invalid_assemble_with_output_only] / reject_invalid_assemble_with_output_only_from_reserved,
        "reserved"_s <= "reserved"_s + event<EventAssembleGraph> [invalid_assemble_without_output] / reject_invalid_assemble_without_output_from_reserved,
        "uninitialized"_s <= "uninitialized"_s + event<EventAssembleGraph> [valid_assemble] / reject_invalid_assemble_with_dispatch_from_uninitialized,
        "uninitialized"_s <= "uninitialized"_s + event<EventAssembleGraph> [invalid_assemble_with_dispatchable_output] / reject_invalid_assemble_with_dispatch_from_uninitialized,
        "uninitialized"_s <= "uninitialized"_s + event<EventAssembleGraph> [invalid_assemble_with_output_only] / reject_invalid_assemble_with_output_only_from_uninitialized,
        "uninitialized"_s <= "uninitialized"_s + event<EventAssembleGraph> [invalid_assemble_without_output] / reject_invalid_assemble_without_output_from_uninitialized,
        "assemble_validate_decision"_s <= "assemble_validate_pass_model"_s + completion<EventAssembleGraph>,
        "reuse_decision_pass_model"_s <= "assemble_validate_decision"_s + completion<EventAssembleGraph> [assemble_validate_done],
        "assemble_dispatch_decision"_s <= "assemble_validate_decision"_s + completion<EventAssembleGraph> [assemble_validate_failed],
        "reuse_decision"_s <= "reuse_decision_pass_model"_s + completion<EventAssembleGraph>,
        "assemble_dispatch_decision"_s <= "reuse_decision"_s + completion<EventAssembleGraph> [reuse_decision_reused] / commit_assemble_reuse_result,
        "assemble_build_pass_model"_s <= "reuse_decision"_s + completion<EventAssembleGraph> [reuse_decision_rebuild],
        "assemble_dispatch_decision"_s <= "reuse_decision"_s + completion<EventAssembleGraph> [reuse_decision_failed],
        "assemble_build_decision"_s <= "assemble_build_pass_model"_s + completion<EventAssembleGraph>,
        "assemble_alloc_pass_model"_s <= "assemble_build_decision"_s + completion<EventAssembleGraph> [assemble_build_done],
        "assemble_dispatch_decision"_s <= "assemble_build_decision"_s + completion<EventAssembleGraph> [assemble_build_failed],
        "assemble_alloc_decision"_s <= "assemble_alloc_pass_model"_s + completion<EventAssembleGraph>,
        "assemble_dispatch_decision"_s <= "assemble_alloc_decision"_s + completion<EventAssembleGraph> [assemble_alloc_done] / commit_assemble_rebuild_result,
        "assemble_dispatch_decision"_s <= "assemble_alloc_decision"_s + completion<EventAssembleGraph> [assemble_alloc_failed],
        "reserved"_s <= "assemble_dispatch_decision"_s + completion<EventAssembleGraph> [assemble_error_none] / dispatch_assemble_done,
        "reserved"_s <= "assemble_dispatch_decision"_s + completion<EventAssembleGraph> [assemble_error_invalid_request] / dispatch_assemble_error_from_assemble_dispatch_decision,
        "reserved"_s <= "assemble_dispatch_decision"_s + completion<EventAssembleGraph> [assemble_error_capacity] / dispatch_assemble_error_from_assemble_dispatch_decision,
        "reserved"_s <= "assemble_dispatch_decision"_s + completion<EventAssembleGraph> [assemble_error_internal_error] / dispatch_assemble_error_from_assemble_dispatch_decision,
        "reserved"_s <= "assemble_dispatch_decision"_s + completion<EventAssembleGraph> [assemble_error_untracked] / dispatch_assemble_error_from_assemble_dispatch_decision,
        "reserved"_s <= "assemble_dispatch_decision"_s + completion<EventAssembleGraph> [assemble_error_unknown] / dispatch_assemble_error_from_assemble_dispatch_decision,
        "uninitialized"_s <= "uninitialized"_s + unexpected_event<_> / on_unexpected_from_uninitialized,
        "reserved"_s <= "reserved"_s + unexpected_event<_> / on_unexpected_from_reserved,
        "reserve_dispatch_decision"_s <= "reserve_validate_decision"_s + unexpected_event<_> / on_unexpected_from_reserve_validate_decision,
        "reserve_dispatch_decision"_s <= "reserve_build_decision"_s + unexpected_event<_> / on_unexpected_from_reserve_build_decision,
        "reserve_dispatch_decision"_s <= "reserve_alloc_decision"_s + unexpected_event<_> / on_unexpected_from_reserve_alloc_decision,
        "uninitialized"_s <= "reserve_dispatch_decision"_s + unexpected_event<_> / on_unexpected_from_reserve_dispatch_decision,
        "assemble_dispatch_decision"_s <= "assemble_validate_decision"_s + unexpected_event<_> / on_unexpected_from_assemble_validate_decision,
        "assemble_dispatch_decision"_s <= "reuse_decision"_s + unexpected_event<_> / on_unexpected_from_reuse_decision,
        "assemble_dispatch_decision"_s <= "assemble_build_decision"_s + unexpected_event<_> / on_unexpected_from_assemble_build_decision,
        "assemble_dispatch_decision"_s <= "assemble_alloc_decision"_s + unexpected_event<_> / on_unexpected_from_assemble_alloc_decision,
        "reserved"_s <= "assemble_dispatch_decision"_s + unexpected_event<_> / on_unexpected_from_assemble_dispatch_decision,
    }
}

/// Context for `GraphAssembler` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct GraphAssemblerContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl GraphAssemblerStateMachineContext for GraphAssemblerContext {
    fn assemble_alloc_done(&self, _event: &EventAssembleGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/guards.hpp::assemble_alloc_done
        todo!(
            "TODO: port guard `assemble_alloc_done` from emel.cpp/src/emel/graph/assembler/guards.hpp"
        )
    }
    fn assemble_alloc_failed(&self, _event: &EventAssembleGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/guards.hpp::assemble_alloc_failed
        todo!(
            "TODO: port guard `assemble_alloc_failed` from emel.cpp/src/emel/graph/assembler/guards.hpp"
        )
    }
    fn assemble_build_done(&self, _event: &EventAssembleGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/guards.hpp::assemble_build_done
        todo!(
            "TODO: port guard `assemble_build_done` from emel.cpp/src/emel/graph/assembler/guards.hpp"
        )
    }
    fn assemble_build_failed(&self, _event: &EventAssembleGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/guards.hpp::assemble_build_failed
        todo!(
            "TODO: port guard `assemble_build_failed` from emel.cpp/src/emel/graph/assembler/guards.hpp"
        )
    }
    fn assemble_error_capacity(&self, _event: &EventAssembleGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/guards.hpp::assemble_error_capacity
        todo!(
            "TODO: port guard `assemble_error_capacity` from emel.cpp/src/emel/graph/assembler/guards.hpp"
        )
    }
    fn assemble_error_internal_error(&self, _event: &EventAssembleGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/guards.hpp::assemble_error_internal_error
        todo!(
            "TODO: port guard `assemble_error_internal_error` from emel.cpp/src/emel/graph/assembler/guards.hpp"
        )
    }
    fn assemble_error_invalid_request(&self, _event: &EventAssembleGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/guards.hpp::assemble_error_invalid_request
        todo!(
            "TODO: port guard `assemble_error_invalid_request` from emel.cpp/src/emel/graph/assembler/guards.hpp"
        )
    }
    fn assemble_error_none(&self, _event: &EventAssembleGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/guards.hpp::assemble_error_none
        todo!(
            "TODO: port guard `assemble_error_none` from emel.cpp/src/emel/graph/assembler/guards.hpp"
        )
    }
    fn assemble_error_unknown(&self, _event: &EventAssembleGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/guards.hpp::assemble_error_unknown
        todo!(
            "TODO: port guard `assemble_error_unknown` from emel.cpp/src/emel/graph/assembler/guards.hpp"
        )
    }
    fn assemble_error_untracked(&self, _event: &EventAssembleGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/guards.hpp::assemble_error_untracked
        todo!(
            "TODO: port guard `assemble_error_untracked` from emel.cpp/src/emel/graph/assembler/guards.hpp"
        )
    }
    fn assemble_validate_done(&self, _event: &EventAssembleGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/guards.hpp::assemble_validate_done
        todo!(
            "TODO: port guard `assemble_validate_done` from emel.cpp/src/emel/graph/assembler/guards.hpp"
        )
    }
    fn assemble_validate_failed(&self, _event: &EventAssembleGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/guards.hpp::assemble_validate_failed
        todo!(
            "TODO: port guard `assemble_validate_failed` from emel.cpp/src/emel/graph/assembler/guards.hpp"
        )
    }
    fn begin_assemble(&mut self, _event: &EventAssembleGraph) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/actions.hpp::begin_assemble
        todo!(
            "TODO: port action `begin_assemble` from emel.cpp/src/emel/graph/assembler/actions.hpp"
        )
    }
    fn begin_reserve(&mut self, _event: &EventReserveGraph) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/actions.hpp::begin_reserve
        todo!(
            "TODO: port action `begin_reserve` from emel.cpp/src/emel/graph/assembler/actions.hpp"
        )
    }
    fn commit_assemble_rebuild_result(&mut self, _event: &EventAssembleGraph) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/actions.hpp::commit_assemble_rebuild_result
        todo!(
            "TODO: port action `commit_assemble_rebuild_result` from emel.cpp/src/emel/graph/assembler/actions.hpp"
        )
    }
    fn commit_assemble_reuse_result(&mut self, _event: &EventAssembleGraph) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/actions.hpp::commit_assemble_reuse_result
        todo!(
            "TODO: port action `commit_assemble_reuse_result` from emel.cpp/src/emel/graph/assembler/actions.hpp"
        )
    }
    fn commit_reserve_result(&mut self, _event: &EventReserveGraph) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/actions.hpp::commit_reserve_result
        todo!(
            "TODO: port action `commit_reserve_result` from emel.cpp/src/emel/graph/assembler/actions.hpp"
        )
    }
    fn dispatch_assemble_done(&mut self, _event: &EventAssembleGraph) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/actions.hpp::dispatch_assemble_done
        todo!(
            "TODO: port action `dispatch_assemble_done` from emel.cpp/src/emel/graph/assembler/actions.hpp"
        )
    }
    fn dispatch_assemble_error_from_assemble_dispatch_decision(
        &mut self,
        _event: &EventAssembleGraph,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/actions.hpp::dispatch_assemble_error
        todo!(
            "TODO: port action `dispatch_assemble_error` from emel.cpp/src/emel/graph/assembler/actions.hpp"
        )
    }
    fn dispatch_reserve_done(&mut self, _event: &EventReserveGraph) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/actions.hpp::dispatch_reserve_done
        todo!(
            "TODO: port action `dispatch_reserve_done` from emel.cpp/src/emel/graph/assembler/actions.hpp"
        )
    }
    fn dispatch_reserve_error_from_reserve_dispatch_decision(
        &mut self,
        _event: &EventReserveGraph,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/actions.hpp::dispatch_reserve_error
        todo!(
            "TODO: port action `dispatch_reserve_error` from emel.cpp/src/emel/graph/assembler/actions.hpp"
        )
    }
    fn invalid_assemble_with_dispatchable_output(
        &self,
        _event: &EventAssembleGraph,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/guards.hpp::invalid_assemble_with_dispatchable_output
        todo!(
            "TODO: port guard `invalid_assemble_with_dispatchable_output` from emel.cpp/src/emel/graph/assembler/guards.hpp"
        )
    }
    fn invalid_assemble_with_output_only(&self, _event: &EventAssembleGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/guards.hpp::invalid_assemble_with_output_only
        todo!(
            "TODO: port guard `invalid_assemble_with_output_only` from emel.cpp/src/emel/graph/assembler/guards.hpp"
        )
    }
    fn invalid_assemble_without_output(&self, _event: &EventAssembleGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/guards.hpp::invalid_assemble_without_output
        todo!(
            "TODO: port guard `invalid_assemble_without_output` from emel.cpp/src/emel/graph/assembler/guards.hpp"
        )
    }
    fn invalid_reserve_with_dispatchable_output(
        &self,
        _event: &EventReserveGraph,
    ) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/guards.hpp::invalid_reserve_with_dispatchable_output
        todo!(
            "TODO: port guard `invalid_reserve_with_dispatchable_output` from emel.cpp/src/emel/graph/assembler/guards.hpp"
        )
    }
    fn invalid_reserve_with_output_only(&self, _event: &EventReserveGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/guards.hpp::invalid_reserve_with_output_only
        todo!(
            "TODO: port guard `invalid_reserve_with_output_only` from emel.cpp/src/emel/graph/assembler/guards.hpp"
        )
    }
    fn invalid_reserve_without_output(&self, _event: &EventReserveGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/guards.hpp::invalid_reserve_without_output
        todo!(
            "TODO: port guard `invalid_reserve_without_output` from emel.cpp/src/emel/graph/assembler/guards.hpp"
        )
    }
    fn on_unexpected_from_assemble_alloc_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/assembler/actions.hpp"
        )
    }
    fn on_unexpected_from_assemble_build_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/assembler/actions.hpp"
        )
    }
    fn on_unexpected_from_assemble_dispatch_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/assembler/actions.hpp"
        )
    }
    fn on_unexpected_from_assemble_validate_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/assembler/actions.hpp"
        )
    }
    fn on_unexpected_from_reserve_alloc_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/assembler/actions.hpp"
        )
    }
    fn on_unexpected_from_reserve_build_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/assembler/actions.hpp"
        )
    }
    fn on_unexpected_from_reserve_dispatch_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/assembler/actions.hpp"
        )
    }
    fn on_unexpected_from_reserve_validate_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/assembler/actions.hpp"
        )
    }
    fn on_unexpected_from_reserved(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/assembler/actions.hpp"
        )
    }
    fn on_unexpected_from_reuse_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/assembler/actions.hpp"
        )
    }
    fn on_unexpected_from_uninitialized(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/assembler/actions.hpp"
        )
    }
    fn reject_invalid_assemble_with_dispatch_from_reserved(
        &mut self,
        _event: &EventAssembleGraph,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/actions.hpp::reject_invalid_assemble_with_dispatch
        todo!(
            "TODO: port action `reject_invalid_assemble_with_dispatch` from emel.cpp/src/emel/graph/assembler/actions.hpp"
        )
    }
    fn reject_invalid_assemble_with_dispatch_from_uninitialized(
        &mut self,
        _event: &EventAssembleGraph,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/actions.hpp::reject_invalid_assemble_with_dispatch
        todo!(
            "TODO: port action `reject_invalid_assemble_with_dispatch` from emel.cpp/src/emel/graph/assembler/actions.hpp"
        )
    }
    fn reject_invalid_assemble_with_output_only_from_reserved(
        &mut self,
        _event: &EventAssembleGraph,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/actions.hpp::reject_invalid_assemble_with_output_only
        todo!(
            "TODO: port action `reject_invalid_assemble_with_output_only` from emel.cpp/src/emel/graph/assembler/actions.hpp"
        )
    }
    fn reject_invalid_assemble_with_output_only_from_uninitialized(
        &mut self,
        _event: &EventAssembleGraph,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/actions.hpp::reject_invalid_assemble_with_output_only
        todo!(
            "TODO: port action `reject_invalid_assemble_with_output_only` from emel.cpp/src/emel/graph/assembler/actions.hpp"
        )
    }
    fn reject_invalid_assemble_without_output_from_reserved(
        &mut self,
        _event: &EventAssembleGraph,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/actions.hpp::reject_invalid_assemble_without_output
        todo!(
            "TODO: port action `reject_invalid_assemble_without_output` from emel.cpp/src/emel/graph/assembler/actions.hpp"
        )
    }
    fn reject_invalid_assemble_without_output_from_uninitialized(
        &mut self,
        _event: &EventAssembleGraph,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/actions.hpp::reject_invalid_assemble_without_output
        todo!(
            "TODO: port action `reject_invalid_assemble_without_output` from emel.cpp/src/emel/graph/assembler/actions.hpp"
        )
    }
    fn reject_invalid_reserve_with_dispatch_from_reserved(
        &mut self,
        _event: &EventReserveGraph,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/actions.hpp::reject_invalid_reserve_with_dispatch
        todo!(
            "TODO: port action `reject_invalid_reserve_with_dispatch` from emel.cpp/src/emel/graph/assembler/actions.hpp"
        )
    }
    fn reject_invalid_reserve_with_dispatch_from_uninitialized(
        &mut self,
        _event: &EventReserveGraph,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/actions.hpp::reject_invalid_reserve_with_dispatch
        todo!(
            "TODO: port action `reject_invalid_reserve_with_dispatch` from emel.cpp/src/emel/graph/assembler/actions.hpp"
        )
    }
    fn reject_invalid_reserve_with_output_only_from_reserved(
        &mut self,
        _event: &EventReserveGraph,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/actions.hpp::reject_invalid_reserve_with_output_only
        todo!(
            "TODO: port action `reject_invalid_reserve_with_output_only` from emel.cpp/src/emel/graph/assembler/actions.hpp"
        )
    }
    fn reject_invalid_reserve_with_output_only_from_uninitialized(
        &mut self,
        _event: &EventReserveGraph,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/actions.hpp::reject_invalid_reserve_with_output_only
        todo!(
            "TODO: port action `reject_invalid_reserve_with_output_only` from emel.cpp/src/emel/graph/assembler/actions.hpp"
        )
    }
    fn reject_invalid_reserve_without_output_from_reserved(
        &mut self,
        _event: &EventReserveGraph,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/actions.hpp::reject_invalid_reserve_without_output
        todo!(
            "TODO: port action `reject_invalid_reserve_without_output` from emel.cpp/src/emel/graph/assembler/actions.hpp"
        )
    }
    fn reject_invalid_reserve_without_output_from_uninitialized(
        &mut self,
        _event: &EventReserveGraph,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/actions.hpp::reject_invalid_reserve_without_output
        todo!(
            "TODO: port action `reject_invalid_reserve_without_output` from emel.cpp/src/emel/graph/assembler/actions.hpp"
        )
    }
    fn reserve_alloc_done(&self, _event: &EventReserveGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/guards.hpp::reserve_alloc_done
        todo!(
            "TODO: port guard `reserve_alloc_done` from emel.cpp/src/emel/graph/assembler/guards.hpp"
        )
    }
    fn reserve_alloc_failed(&self, _event: &EventReserveGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/guards.hpp::reserve_alloc_failed
        todo!(
            "TODO: port guard `reserve_alloc_failed` from emel.cpp/src/emel/graph/assembler/guards.hpp"
        )
    }
    fn reserve_build_done(&self, _event: &EventReserveGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/guards.hpp::reserve_build_done
        todo!(
            "TODO: port guard `reserve_build_done` from emel.cpp/src/emel/graph/assembler/guards.hpp"
        )
    }
    fn reserve_build_failed(&self, _event: &EventReserveGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/guards.hpp::reserve_build_failed
        todo!(
            "TODO: port guard `reserve_build_failed` from emel.cpp/src/emel/graph/assembler/guards.hpp"
        )
    }
    fn reserve_error_capacity(&self, _event: &EventReserveGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/guards.hpp::reserve_error_capacity
        todo!(
            "TODO: port guard `reserve_error_capacity` from emel.cpp/src/emel/graph/assembler/guards.hpp"
        )
    }
    fn reserve_error_internal_error(&self, _event: &EventReserveGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/guards.hpp::reserve_error_internal_error
        todo!(
            "TODO: port guard `reserve_error_internal_error` from emel.cpp/src/emel/graph/assembler/guards.hpp"
        )
    }
    fn reserve_error_invalid_request(&self, _event: &EventReserveGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/guards.hpp::reserve_error_invalid_request
        todo!(
            "TODO: port guard `reserve_error_invalid_request` from emel.cpp/src/emel/graph/assembler/guards.hpp"
        )
    }
    fn reserve_error_none(&self, _event: &EventReserveGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/guards.hpp::reserve_error_none
        todo!(
            "TODO: port guard `reserve_error_none` from emel.cpp/src/emel/graph/assembler/guards.hpp"
        )
    }
    fn reserve_error_unknown(&self, _event: &EventReserveGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/guards.hpp::reserve_error_unknown
        todo!(
            "TODO: port guard `reserve_error_unknown` from emel.cpp/src/emel/graph/assembler/guards.hpp"
        )
    }
    fn reserve_error_untracked(&self, _event: &EventReserveGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/guards.hpp::reserve_error_untracked
        todo!(
            "TODO: port guard `reserve_error_untracked` from emel.cpp/src/emel/graph/assembler/guards.hpp"
        )
    }
    fn reserve_validate_done(&self, _event: &EventReserveGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/guards.hpp::reserve_validate_done
        todo!(
            "TODO: port guard `reserve_validate_done` from emel.cpp/src/emel/graph/assembler/guards.hpp"
        )
    }
    fn reserve_validate_failed(&self, _event: &EventReserveGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/guards.hpp::reserve_validate_failed
        todo!(
            "TODO: port guard `reserve_validate_failed` from emel.cpp/src/emel/graph/assembler/guards.hpp"
        )
    }
    fn reuse_decision_failed(&self, _event: &EventAssembleGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/guards.hpp::reuse_decision_failed
        todo!(
            "TODO: port guard `reuse_decision_failed` from emel.cpp/src/emel/graph/assembler/guards.hpp"
        )
    }
    fn reuse_decision_rebuild(&self, _event: &EventAssembleGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/guards.hpp::reuse_decision_rebuild
        todo!(
            "TODO: port guard `reuse_decision_rebuild` from emel.cpp/src/emel/graph/assembler/guards.hpp"
        )
    }
    fn reuse_decision_reused(&self, _event: &EventAssembleGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/guards.hpp::reuse_decision_reused
        todo!(
            "TODO: port guard `reuse_decision_reused` from emel.cpp/src/emel/graph/assembler/guards.hpp"
        )
    }
    fn valid_assemble(&self, _event: &EventAssembleGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/guards.hpp::valid_assemble
        todo!("TODO: port guard `valid_assemble` from emel.cpp/src/emel/graph/assembler/guards.hpp")
    }
    fn valid_reserve(&self, _event: &EventReserveGraph) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/guards.hpp::valid_reserve
        todo!("TODO: port guard `valid_reserve` from emel.cpp/src/emel/graph/assembler/guards.hpp")
    }
}
