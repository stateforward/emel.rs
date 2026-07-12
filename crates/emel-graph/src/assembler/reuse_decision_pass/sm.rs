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

// --- machine GraphAssemblerReuseDecisionPass from emel.cpp/src/emel/graph/assembler/reuse_decision_pass/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct AssemblerEventAssembleGraph;

sml! {
    GraphAssemblerReuseDecisionPass {
        "assemble_failed"_s <= *"deciding"_s + completion<AssemblerEventAssembleGraph> [phase_prefailed] / mark_failed_prefailed,
        "reuse_selected"_s <= "deciding"_s + completion<AssemblerEventAssembleGraph> [phase_reuse] / mark_reuse,
        "rebuild_selected"_s <= "deciding"_s + completion<AssemblerEventAssembleGraph> [phase_rebuild] / mark_rebuild,
        "assemble_failed"_s <= "deciding"_s + completion<AssemblerEventAssembleGraph> [phase_prereq_failed] / mark_failed_prereq,
        "assemble_failed"_s <= "deciding"_s + completion<AssemblerEventAssembleGraph> [phase_invalid_request] / mark_failed_invalid_request,
        "unexpected_event"_s <= "deciding"_s + unexpected_event<_> / on_unexpected_from_deciding,
        "unexpected_event"_s <= "reuse_selected"_s + unexpected_event<_> / on_unexpected_from_reuse_selected,
        "unexpected_event"_s <= "rebuild_selected"_s + unexpected_event<_> / on_unexpected_from_rebuild_selected,
        "unexpected_event"_s <= "assemble_failed"_s + unexpected_event<_> / on_unexpected_from_assemble_failed,
        "unexpected_event"_s <= "unexpected_event"_s + unexpected_event<_> / on_unexpected_from_unexpected_event,
        "reuse_selected"_s = X,
        "rebuild_selected"_s = X,
        "assemble_failed"_s = X,
    }
}

/// Context for `GraphAssemblerReuseDecisionPass` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct GraphAssemblerReuseDecisionPassContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl GraphAssemblerReuseDecisionPassStateMachineContext for GraphAssemblerReuseDecisionPassContext {
    fn mark_failed_invalid_request(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/reuse_decision_pass/actions.hpp::mark_failed_invalid_request
        todo!(
            "TODO: port action `mark_failed_invalid_request` from emel.cpp/src/emel/graph/assembler/reuse_decision_pass/actions.hpp"
        )
    }
    fn mark_failed_prefailed(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/reuse_decision_pass/actions.hpp::mark_failed_prefailed
        todo!(
            "TODO: port action `mark_failed_prefailed` from emel.cpp/src/emel/graph/assembler/reuse_decision_pass/actions.hpp"
        )
    }
    fn mark_failed_prereq(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/reuse_decision_pass/actions.hpp::mark_failed_prereq
        todo!(
            "TODO: port action `mark_failed_prereq` from emel.cpp/src/emel/graph/assembler/reuse_decision_pass/actions.hpp"
        )
    }
    fn mark_rebuild(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/reuse_decision_pass/actions.hpp::mark_rebuild
        todo!(
            "TODO: port action `mark_rebuild` from emel.cpp/src/emel/graph/assembler/reuse_decision_pass/actions.hpp"
        )
    }
    fn mark_reuse(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/reuse_decision_pass/actions.hpp::mark_reuse
        todo!(
            "TODO: port action `mark_reuse` from emel.cpp/src/emel/graph/assembler/reuse_decision_pass/actions.hpp"
        )
    }
    fn on_unexpected_from_assemble_failed(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/reuse_decision_pass/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/assembler/reuse_decision_pass/actions.hpp"
        )
    }
    fn on_unexpected_from_deciding(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/reuse_decision_pass/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/assembler/reuse_decision_pass/actions.hpp"
        )
    }
    fn on_unexpected_from_rebuild_selected(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/reuse_decision_pass/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/assembler/reuse_decision_pass/actions.hpp"
        )
    }
    fn on_unexpected_from_reuse_selected(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/reuse_decision_pass/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/assembler/reuse_decision_pass/actions.hpp"
        )
    }
    fn on_unexpected_from_unexpected_event(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/reuse_decision_pass/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/assembler/reuse_decision_pass/actions.hpp"
        )
    }
    fn phase_invalid_request(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/reuse_decision_pass/guards.hpp::phase_invalid_request
        todo!(
            "TODO: port guard `phase_invalid_request` from emel.cpp/src/emel/graph/assembler/reuse_decision_pass/guards.hpp"
        )
    }
    fn phase_prefailed(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/reuse_decision_pass/guards.hpp::phase_prefailed
        todo!(
            "TODO: port guard `phase_prefailed` from emel.cpp/src/emel/graph/assembler/reuse_decision_pass/guards.hpp"
        )
    }
    fn phase_prereq_failed(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/reuse_decision_pass/guards.hpp::phase_prereq_failed
        todo!(
            "TODO: port guard `phase_prereq_failed` from emel.cpp/src/emel/graph/assembler/reuse_decision_pass/guards.hpp"
        )
    }
    fn phase_rebuild(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/reuse_decision_pass/guards.hpp::phase_rebuild
        todo!(
            "TODO: port guard `phase_rebuild` from emel.cpp/src/emel/graph/assembler/reuse_decision_pass/guards.hpp"
        )
    }
    fn phase_reuse(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/reuse_decision_pass/guards.hpp::phase_reuse
        todo!(
            "TODO: port guard `phase_reuse` from emel.cpp/src/emel/graph/assembler/reuse_decision_pass/guards.hpp"
        )
    }
}
