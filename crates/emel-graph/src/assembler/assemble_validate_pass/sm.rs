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

// --- machine GraphAssemblerAssembleValidatePass from emel.cpp/src/emel/graph/assembler/assemble_validate_pass/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct AssemblerEventAssembleGraph;

sml! {
    GraphAssemblerAssembleValidatePass {
        "assembled"_s <= *"deciding"_s + completion<AssemblerEventAssembleGraph> [phase_done] / mark_done,
        "assemble_failed"_s <= "deciding"_s + completion<AssemblerEventAssembleGraph> [phase_invalid_request] / mark_failed_invalid_request,
        "unexpected_event"_s <= "deciding"_s + unexpected_event<_> / on_unexpected_from_deciding,
        "unexpected_event"_s <= "assembled"_s + unexpected_event<_> / on_unexpected_from_assembled,
        "unexpected_event"_s <= "assemble_failed"_s + unexpected_event<_> / on_unexpected_from_assemble_failed,
        "unexpected_event"_s <= "unexpected_event"_s + unexpected_event<_> / on_unexpected_from_unexpected_event,
        "assembled"_s = X,
        "assemble_failed"_s = X,
    }
}

/// Context for `GraphAssemblerAssembleValidatePass` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct GraphAssemblerAssembleValidatePassContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl GraphAssemblerAssembleValidatePassStateMachineContext
    for GraphAssemblerAssembleValidatePassContext
{
    fn mark_done(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/assemble_validate_pass/actions.hpp::mark_done
        todo!(
            "TODO: port action `mark_done` from emel.cpp/src/emel/graph/assembler/assemble_validate_pass/actions.hpp"
        )
    }
    fn mark_failed_invalid_request(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/assemble_validate_pass/actions.hpp::mark_failed_invalid_request
        todo!(
            "TODO: port action `mark_failed_invalid_request` from emel.cpp/src/emel/graph/assembler/assemble_validate_pass/actions.hpp"
        )
    }
    fn on_unexpected_from_assemble_failed(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/assemble_validate_pass/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/assembler/assemble_validate_pass/actions.hpp"
        )
    }
    fn on_unexpected_from_assembled(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/assemble_validate_pass/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/assembler/assemble_validate_pass/actions.hpp"
        )
    }
    fn on_unexpected_from_deciding(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/assemble_validate_pass/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/assembler/assemble_validate_pass/actions.hpp"
        )
    }
    fn on_unexpected_from_unexpected_event(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/assemble_validate_pass/actions.hpp::on_unexpected
        todo!(
            "TODO: port action `on_unexpected` from emel.cpp/src/emel/graph/assembler/assemble_validate_pass/actions.hpp"
        )
    }
    fn phase_done(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/assemble_validate_pass/guards.hpp::phase_done
        todo!(
            "TODO: port guard `phase_done` from emel.cpp/src/emel/graph/assembler/assemble_validate_pass/guards.hpp"
        )
    }
    fn phase_invalid_request(&self) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/graph/assembler/assemble_validate_pass/guards.hpp::phase_invalid_request
        todo!(
            "TODO: port guard `phase_invalid_request` from emel.cpp/src/emel/graph/assembler/assemble_validate_pass/guards.hpp"
        )
    }
}
