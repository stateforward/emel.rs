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

// --- machine TextGeneratorDecodeWavefront from emel.cpp/src/emel/text/generator/decode_wavefront/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventRun;

sml! {
    TextGeneratorDecodeWavefront {
        "state_validation_decision"_s <= *"state_idle"_s + event<EventRun> / effect_begin_run,
        "state_idle"_s <= "state_validation_decision"_s + completion<EventRun> [guard_invalid_request] / effect_reject_invalid_request,
        "state_idle"_s <= "state_validation_decision"_s + completion<EventRun> [guard_multi_lane_incompatible] / effect_reject_incompatible_lanes,
        "state_group_ready"_s <= "state_validation_decision"_s + completion<EventRun> [guard_single_lane] / effect_mark_single_lane,
        "state_group_ready"_s <= "state_validation_decision"_s + completion<EventRun> [guard_multi_lane_compatible] / effect_mark_grouped_lanes,
        "state_parallel_decision"_s <= "state_group_ready"_s + completion<EventRun> [guard_parallel_dispatch] / effect_dispatch_parallel_lanes,
        "state_lane0_decision"_s <= "state_group_ready"_s + completion<EventRun> [guard_serial_dispatch] / effect_dispatch_lane_0,
        "state_idle"_s <= "state_lane0_decision"_s + completion<EventRun> [guard_lane_rejected_0] / effect_mark_lane_rejected_0_from_state_lane0_decision,
        "state_idle"_s <= "state_lane0_decision"_s + completion<EventRun> [guard_lane_accepted_and_last_0] / effect_commit_done_from_state_lane0_decision,
        "state_lane1_decision"_s <= "state_lane0_decision"_s + completion<EventRun> [guard_lane_accepted_and_more_0] / effect_dispatch_lane_1,
        "state_idle"_s <= "state_lane1_decision"_s + completion<EventRun> [guard_lane_rejected_1] / effect_mark_lane_rejected_1_from_state_lane1_decision,
        "state_idle"_s <= "state_lane1_decision"_s + completion<EventRun> [guard_lane_accepted_and_last_1] / effect_commit_done_from_state_lane1_decision,
        "state_lane2_decision"_s <= "state_lane1_decision"_s + completion<EventRun> [guard_lane_accepted_and_more_1] / effect_dispatch_lane_2,
        "state_idle"_s <= "state_lane2_decision"_s + completion<EventRun> [guard_lane_rejected_2] / effect_mark_lane_rejected_2_from_state_lane2_decision,
        "state_idle"_s <= "state_lane2_decision"_s + completion<EventRun> [guard_lane_accepted_and_last_2] / effect_commit_done_from_state_lane2_decision,
        "state_lane3_decision"_s <= "state_lane2_decision"_s + completion<EventRun> [guard_lane_accepted_and_more_2] / effect_dispatch_lane_3,
        "state_idle"_s <= "state_lane3_decision"_s + completion<EventRun> [guard_lane_rejected_3] / effect_mark_lane_rejected_3_from_state_lane3_decision,
        "state_idle"_s <= "state_lane3_decision"_s + completion<EventRun> [guard_lane_accepted_and_last_3] / effect_commit_done_from_state_lane3_decision,
        "state_lane4_decision"_s <= "state_lane3_decision"_s + completion<EventRun> [guard_lane_accepted_and_more_3] / effect_dispatch_lane_4,
        "state_idle"_s <= "state_lane4_decision"_s + completion<EventRun> [guard_lane_rejected_4] / effect_mark_lane_rejected_4_from_state_lane4_decision,
        "state_idle"_s <= "state_lane4_decision"_s + completion<EventRun> [guard_lane_accepted_and_last_4] / effect_commit_done_from_state_lane4_decision,
        "state_lane5_decision"_s <= "state_lane4_decision"_s + completion<EventRun> [guard_lane_accepted_and_more_4] / effect_dispatch_lane_5,
        "state_idle"_s <= "state_lane5_decision"_s + completion<EventRun> [guard_lane_rejected_5] / effect_mark_lane_rejected_5_from_state_lane5_decision,
        "state_idle"_s <= "state_lane5_decision"_s + completion<EventRun> [guard_lane_accepted_and_last_5] / effect_commit_done_from_state_lane5_decision,
        "state_lane6_decision"_s <= "state_lane5_decision"_s + completion<EventRun> [guard_lane_accepted_and_more_5] / effect_dispatch_lane_6,
        "state_idle"_s <= "state_lane6_decision"_s + completion<EventRun> [guard_lane_rejected_6] / effect_mark_lane_rejected_6_from_state_lane6_decision,
        "state_idle"_s <= "state_lane6_decision"_s + completion<EventRun> [guard_lane_accepted_and_last_6] / effect_commit_done_from_state_lane6_decision,
        "state_lane7_decision"_s <= "state_lane6_decision"_s + completion<EventRun> [guard_lane_accepted_and_more_6] / effect_dispatch_lane_7,
        "state_idle"_s <= "state_lane7_decision"_s + completion<EventRun> [guard_lane_rejected_7] / effect_mark_lane_rejected_7_from_state_lane7_decision,
        "state_idle"_s <= "state_lane7_decision"_s + completion<EventRun> [guard_lane_accepted_and_last_7] / effect_commit_done_from_state_lane7_decision,
        "state_idle"_s <= "state_parallel_decision"_s + completion<EventRun> [guard_parallel_submission_failed] / effect_reject_parallel_scheduler_from_state_parallel_decision,
        "state_idle"_s <= "state_parallel_decision"_s + completion<EventRun> [guard_parallel_join_failed] / effect_reject_parallel_scheduler_from_state_parallel_decision,
        "state_idle"_s <= "state_parallel_decision"_s + completion<EventRun> [guard_parallel_lane_rejected_0] / effect_mark_lane_rejected_0_from_state_parallel_decision,
        "state_idle"_s <= "state_parallel_decision"_s + completion<EventRun> [guard_parallel_lane_rejected_1] / effect_mark_lane_rejected_1_from_state_parallel_decision,
        "state_idle"_s <= "state_parallel_decision"_s + completion<EventRun> [guard_parallel_lane_rejected_2] / effect_mark_lane_rejected_2_from_state_parallel_decision,
        "state_idle"_s <= "state_parallel_decision"_s + completion<EventRun> [guard_parallel_lane_rejected_3] / effect_mark_lane_rejected_3_from_state_parallel_decision,
        "state_idle"_s <= "state_parallel_decision"_s + completion<EventRun> [guard_parallel_lane_rejected_4] / effect_mark_lane_rejected_4_from_state_parallel_decision,
        "state_idle"_s <= "state_parallel_decision"_s + completion<EventRun> [guard_parallel_lane_rejected_5] / effect_mark_lane_rejected_5_from_state_parallel_decision,
        "state_idle"_s <= "state_parallel_decision"_s + completion<EventRun> [guard_parallel_lane_rejected_6] / effect_mark_lane_rejected_6_from_state_parallel_decision,
        "state_idle"_s <= "state_parallel_decision"_s + completion<EventRun> [guard_parallel_lane_rejected_7] / effect_mark_lane_rejected_7_from_state_parallel_decision,
        "state_idle"_s <= "state_parallel_decision"_s + completion<EventRun> [guard_parallel_all_lanes_accepted] / effect_commit_done_from_state_parallel_decision,
        "state_idle"_s <= "state_idle"_s + unexpected_event<_> / effect_on_unexpected_from_state_idle,
        "state_idle"_s <= "state_validation_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_validation_decision,
        "state_idle"_s <= "state_group_ready"_s + unexpected_event<_> / effect_on_unexpected_from_state_group_ready,
        "state_idle"_s <= "state_lane0_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_lane0_decision,
        "state_idle"_s <= "state_lane1_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_lane1_decision,
        "state_idle"_s <= "state_lane2_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_lane2_decision,
        "state_idle"_s <= "state_lane3_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_lane3_decision,
        "state_idle"_s <= "state_lane4_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_lane4_decision,
        "state_idle"_s <= "state_lane5_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_lane5_decision,
        "state_idle"_s <= "state_lane6_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_lane6_decision,
        "state_idle"_s <= "state_lane7_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_lane7_decision,
        "state_idle"_s <= "state_parallel_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_parallel_decision,
    }
}

/// Context for `TextGeneratorDecodeWavefront` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct TextGeneratorDecodeWavefrontContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl TextGeneratorDecodeWavefrontStateMachineContext for TextGeneratorDecodeWavefrontContext {
    fn effect_begin_run(&mut self, _event: &EventRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_begin_run
        todo!(
            "TODO: port action `effect_begin_run` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_commit_done_from_state_lane0_decision(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_commit_done
        todo!(
            "TODO: port action `effect_commit_done` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_commit_done_from_state_lane1_decision(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_commit_done
        todo!(
            "TODO: port action `effect_commit_done` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_commit_done_from_state_lane2_decision(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_commit_done
        todo!(
            "TODO: port action `effect_commit_done` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_commit_done_from_state_lane3_decision(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_commit_done
        todo!(
            "TODO: port action `effect_commit_done` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_commit_done_from_state_lane4_decision(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_commit_done
        todo!(
            "TODO: port action `effect_commit_done` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_commit_done_from_state_lane5_decision(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_commit_done
        todo!(
            "TODO: port action `effect_commit_done` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_commit_done_from_state_lane6_decision(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_commit_done
        todo!(
            "TODO: port action `effect_commit_done` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_commit_done_from_state_lane7_decision(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_commit_done
        todo!(
            "TODO: port action `effect_commit_done` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_commit_done_from_state_parallel_decision(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_commit_done
        todo!(
            "TODO: port action `effect_commit_done` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_dispatch_lane_0(&mut self, _event: &EventRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_dispatch_lane
        todo!(
            "TODO: port action `effect_dispatch_lane` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_dispatch_lane_1(&mut self, _event: &EventRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_dispatch_lane
        todo!(
            "TODO: port action `effect_dispatch_lane` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_dispatch_lane_2(&mut self, _event: &EventRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_dispatch_lane
        todo!(
            "TODO: port action `effect_dispatch_lane` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_dispatch_lane_3(&mut self, _event: &EventRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_dispatch_lane
        todo!(
            "TODO: port action `effect_dispatch_lane` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_dispatch_lane_4(&mut self, _event: &EventRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_dispatch_lane
        todo!(
            "TODO: port action `effect_dispatch_lane` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_dispatch_lane_5(&mut self, _event: &EventRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_dispatch_lane
        todo!(
            "TODO: port action `effect_dispatch_lane` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_dispatch_lane_6(&mut self, _event: &EventRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_dispatch_lane
        todo!(
            "TODO: port action `effect_dispatch_lane` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_dispatch_lane_7(&mut self, _event: &EventRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_dispatch_lane
        todo!(
            "TODO: port action `effect_dispatch_lane` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_dispatch_parallel_lanes(&mut self, _event: &EventRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_dispatch_parallel_lanes
        todo!(
            "TODO: port action `effect_dispatch_parallel_lanes` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_mark_grouped_lanes(&mut self, _event: &EventRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_mark_grouped_lanes
        todo!(
            "TODO: port action `effect_mark_grouped_lanes` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_mark_lane_rejected_0_from_state_lane0_decision(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_mark_lane_rejected
        todo!(
            "TODO: port action `effect_mark_lane_rejected` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_mark_lane_rejected_0_from_state_parallel_decision(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_mark_lane_rejected
        todo!(
            "TODO: port action `effect_mark_lane_rejected` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_mark_lane_rejected_1_from_state_lane1_decision(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_mark_lane_rejected
        todo!(
            "TODO: port action `effect_mark_lane_rejected` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_mark_lane_rejected_1_from_state_parallel_decision(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_mark_lane_rejected
        todo!(
            "TODO: port action `effect_mark_lane_rejected` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_mark_lane_rejected_2_from_state_lane2_decision(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_mark_lane_rejected
        todo!(
            "TODO: port action `effect_mark_lane_rejected` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_mark_lane_rejected_2_from_state_parallel_decision(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_mark_lane_rejected
        todo!(
            "TODO: port action `effect_mark_lane_rejected` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_mark_lane_rejected_3_from_state_lane3_decision(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_mark_lane_rejected
        todo!(
            "TODO: port action `effect_mark_lane_rejected` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_mark_lane_rejected_3_from_state_parallel_decision(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_mark_lane_rejected
        todo!(
            "TODO: port action `effect_mark_lane_rejected` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_mark_lane_rejected_4_from_state_lane4_decision(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_mark_lane_rejected
        todo!(
            "TODO: port action `effect_mark_lane_rejected` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_mark_lane_rejected_4_from_state_parallel_decision(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_mark_lane_rejected
        todo!(
            "TODO: port action `effect_mark_lane_rejected` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_mark_lane_rejected_5_from_state_lane5_decision(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_mark_lane_rejected
        todo!(
            "TODO: port action `effect_mark_lane_rejected` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_mark_lane_rejected_5_from_state_parallel_decision(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_mark_lane_rejected
        todo!(
            "TODO: port action `effect_mark_lane_rejected` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_mark_lane_rejected_6_from_state_lane6_decision(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_mark_lane_rejected
        todo!(
            "TODO: port action `effect_mark_lane_rejected` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_mark_lane_rejected_6_from_state_parallel_decision(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_mark_lane_rejected
        todo!(
            "TODO: port action `effect_mark_lane_rejected` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_mark_lane_rejected_7_from_state_lane7_decision(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_mark_lane_rejected
        todo!(
            "TODO: port action `effect_mark_lane_rejected` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_mark_lane_rejected_7_from_state_parallel_decision(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_mark_lane_rejected
        todo!(
            "TODO: port action `effect_mark_lane_rejected` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_mark_single_lane(&mut self, _event: &EventRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_mark_single_lane
        todo!(
            "TODO: port action `effect_mark_single_lane` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_group_ready(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_idle(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_lane0_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_lane1_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_lane2_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_lane3_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_lane4_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_lane5_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_lane6_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_lane7_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_parallel_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_on_unexpected_from_state_validation_decision(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_on_unexpected
        todo!(
            "TODO: port action `effect_on_unexpected` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_reject_incompatible_lanes(&mut self, _event: &EventRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_reject_incompatible_lanes
        todo!(
            "TODO: port action `effect_reject_incompatible_lanes` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_reject_invalid_request(&mut self, _event: &EventRun) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_reject_invalid_request
        todo!(
            "TODO: port action `effect_reject_invalid_request` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn effect_reject_parallel_scheduler_from_state_parallel_decision(
        &mut self,
        _event: &EventRun,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp::effect_reject_parallel_scheduler
        todo!(
            "TODO: port action `effect_reject_parallel_scheduler` from emel.cpp/src/emel/text/generator/decode_wavefront/actions.hpp"
        )
    }
    fn guard_invalid_request(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp::guard_invalid_request
        todo!(
            "TODO: port guard `guard_invalid_request` from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp"
        )
    }
    fn guard_lane_accepted_and_last_0(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp::guard_lane_accepted_and_last
        todo!(
            "TODO: port guard `guard_lane_accepted_and_last` from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp"
        )
    }
    fn guard_lane_accepted_and_last_1(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp::guard_lane_accepted_and_last
        todo!(
            "TODO: port guard `guard_lane_accepted_and_last` from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp"
        )
    }
    fn guard_lane_accepted_and_last_2(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp::guard_lane_accepted_and_last
        todo!(
            "TODO: port guard `guard_lane_accepted_and_last` from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp"
        )
    }
    fn guard_lane_accepted_and_last_3(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp::guard_lane_accepted_and_last
        todo!(
            "TODO: port guard `guard_lane_accepted_and_last` from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp"
        )
    }
    fn guard_lane_accepted_and_last_4(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp::guard_lane_accepted_and_last
        todo!(
            "TODO: port guard `guard_lane_accepted_and_last` from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp"
        )
    }
    fn guard_lane_accepted_and_last_5(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp::guard_lane_accepted_and_last
        todo!(
            "TODO: port guard `guard_lane_accepted_and_last` from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp"
        )
    }
    fn guard_lane_accepted_and_last_6(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp::guard_lane_accepted_and_last
        todo!(
            "TODO: port guard `guard_lane_accepted_and_last` from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp"
        )
    }
    fn guard_lane_accepted_and_last_7(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp::guard_lane_accepted_and_last
        todo!(
            "TODO: port guard `guard_lane_accepted_and_last` from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp"
        )
    }
    fn guard_lane_accepted_and_more_0(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp::guard_lane_accepted_and_more
        todo!(
            "TODO: port guard `guard_lane_accepted_and_more` from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp"
        )
    }
    fn guard_lane_accepted_and_more_1(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp::guard_lane_accepted_and_more
        todo!(
            "TODO: port guard `guard_lane_accepted_and_more` from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp"
        )
    }
    fn guard_lane_accepted_and_more_2(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp::guard_lane_accepted_and_more
        todo!(
            "TODO: port guard `guard_lane_accepted_and_more` from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp"
        )
    }
    fn guard_lane_accepted_and_more_3(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp::guard_lane_accepted_and_more
        todo!(
            "TODO: port guard `guard_lane_accepted_and_more` from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp"
        )
    }
    fn guard_lane_accepted_and_more_4(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp::guard_lane_accepted_and_more
        todo!(
            "TODO: port guard `guard_lane_accepted_and_more` from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp"
        )
    }
    fn guard_lane_accepted_and_more_5(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp::guard_lane_accepted_and_more
        todo!(
            "TODO: port guard `guard_lane_accepted_and_more` from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp"
        )
    }
    fn guard_lane_accepted_and_more_6(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp::guard_lane_accepted_and_more
        todo!(
            "TODO: port guard `guard_lane_accepted_and_more` from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp"
        )
    }
    fn guard_lane_rejected_0(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp::guard_lane_rejected
        todo!(
            "TODO: port guard `guard_lane_rejected` from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp"
        )
    }
    fn guard_lane_rejected_1(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp::guard_lane_rejected
        todo!(
            "TODO: port guard `guard_lane_rejected` from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp"
        )
    }
    fn guard_lane_rejected_2(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp::guard_lane_rejected
        todo!(
            "TODO: port guard `guard_lane_rejected` from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp"
        )
    }
    fn guard_lane_rejected_3(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp::guard_lane_rejected
        todo!(
            "TODO: port guard `guard_lane_rejected` from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp"
        )
    }
    fn guard_lane_rejected_4(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp::guard_lane_rejected
        todo!(
            "TODO: port guard `guard_lane_rejected` from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp"
        )
    }
    fn guard_lane_rejected_5(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp::guard_lane_rejected
        todo!(
            "TODO: port guard `guard_lane_rejected` from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp"
        )
    }
    fn guard_lane_rejected_6(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp::guard_lane_rejected
        todo!(
            "TODO: port guard `guard_lane_rejected` from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp"
        )
    }
    fn guard_lane_rejected_7(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp::guard_lane_rejected
        todo!(
            "TODO: port guard `guard_lane_rejected` from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp"
        )
    }
    fn guard_multi_lane_compatible(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp::guard_multi_lane_compatible
        todo!(
            "TODO: port guard `guard_multi_lane_compatible` from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp"
        )
    }
    fn guard_multi_lane_incompatible(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp::guard_multi_lane_incompatible
        todo!(
            "TODO: port guard `guard_multi_lane_incompatible` from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp"
        )
    }
    fn guard_parallel_all_lanes_accepted(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp::guard_parallel_all_lanes_accepted
        todo!(
            "TODO: port guard `guard_parallel_all_lanes_accepted` from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp"
        )
    }
    fn guard_parallel_dispatch(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp::guard_parallel_dispatch
        todo!(
            "TODO: port guard `guard_parallel_dispatch` from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp"
        )
    }
    fn guard_parallel_join_failed(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp::guard_parallel_join_failed
        todo!(
            "TODO: port guard `guard_parallel_join_failed` from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp"
        )
    }
    fn guard_parallel_lane_rejected_0(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp::guard_parallel_lane_rejected
        todo!(
            "TODO: port guard `guard_parallel_lane_rejected` from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp"
        )
    }
    fn guard_parallel_lane_rejected_1(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp::guard_parallel_lane_rejected
        todo!(
            "TODO: port guard `guard_parallel_lane_rejected` from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp"
        )
    }
    fn guard_parallel_lane_rejected_2(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp::guard_parallel_lane_rejected
        todo!(
            "TODO: port guard `guard_parallel_lane_rejected` from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp"
        )
    }
    fn guard_parallel_lane_rejected_3(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp::guard_parallel_lane_rejected
        todo!(
            "TODO: port guard `guard_parallel_lane_rejected` from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp"
        )
    }
    fn guard_parallel_lane_rejected_4(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp::guard_parallel_lane_rejected
        todo!(
            "TODO: port guard `guard_parallel_lane_rejected` from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp"
        )
    }
    fn guard_parallel_lane_rejected_5(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp::guard_parallel_lane_rejected
        todo!(
            "TODO: port guard `guard_parallel_lane_rejected` from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp"
        )
    }
    fn guard_parallel_lane_rejected_6(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp::guard_parallel_lane_rejected
        todo!(
            "TODO: port guard `guard_parallel_lane_rejected` from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp"
        )
    }
    fn guard_parallel_lane_rejected_7(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp::guard_parallel_lane_rejected
        todo!(
            "TODO: port guard `guard_parallel_lane_rejected` from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp"
        )
    }
    fn guard_parallel_submission_failed(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp::guard_parallel_submission_failed
        todo!(
            "TODO: port guard `guard_parallel_submission_failed` from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp"
        )
    }
    fn guard_serial_dispatch(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp::guard_serial_dispatch
        todo!(
            "TODO: port guard `guard_serial_dispatch` from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp"
        )
    }
    fn guard_single_lane(&self, _event: &EventRun) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp::guard_single_lane
        todo!(
            "TODO: port guard `guard_single_lane` from emel.cpp/src/emel/text/generator/decode_wavefront/guards.hpp"
        )
    }
}
