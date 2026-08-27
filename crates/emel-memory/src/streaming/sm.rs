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

// --- machine MemoryStreaming from emel.cpp/src/emel/memory/streaming/sm.hpp ---
/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventAdvance;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventCaptureView;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventInitialize;

/// Runtime event shell (TODO: fields from events/detail).
#[derive(Debug, Default, Clone)]
pub struct EventReset;

sml! {
    MemoryStreaming {
        "state_empty"_s <= *"state_uninitialized"_s + event<EventInitialize> [guard_configuration_valid] / effect_initialize_from_state_uninitialized,
        "state_uninitialized"_s <= "state_uninitialized"_s + event<EventInitialize> [guard_configuration_invalid] / effect_reject_error_invalid_configuration_from_state_uninitialized,
        "state_empty"_s <= "state_empty"_s + event<EventInitialize> / effect_reject_error_already_initialized_from_state_empty,
        "state_filling"_s <= "state_filling"_s + event<EventInitialize> / effect_reject_error_already_initialized_from_state_filling,
        "state_full"_s <= "state_full"_s + event<EventInitialize> / effect_reject_error_already_initialized_from_state_full,
        "state_empty"_s <= "state_errored"_s + event<EventInitialize> [guard_configuration_valid] / effect_initialize_from_state_errored,
        "state_errored"_s <= "state_errored"_s + event<EventInitialize> [guard_configuration_invalid] / effect_reject_error_invalid_configuration_from_state_errored,
        "state_full"_s <= "state_empty"_s + event<EventAdvance> [guard_capacity_one],
        "state_filling"_s <= "state_empty"_s + event<EventAdvance> [guard_capacity_many],
        "state_filling"_s <= "state_filling"_s + event<EventAdvance> [guard_filling_remains_partial],
        "state_full"_s <= "state_filling"_s + event<EventAdvance> [guard_filling_becomes_full],
        "state_full"_s <= "state_full"_s + event<EventAdvance> [guard_full_position_available_before_wrap],
        "state_full"_s <= "state_full"_s + event<EventAdvance> [guard_full_position_available_at_wrap],
        "state_full"_s <= "state_full"_s + event<EventAdvance> [guard_full_position_overflow] / effect_reject_error_position_overflow,
        "state_full"_s <= "state_full"_s + event<EventAdvance> [guard_full_cursor_invalid] / effect_reject_error_internal_error_event_advance,
        "state_empty"_s <= "state_empty"_s + event<EventReset> / effect_reset_from_state_empty,
        "state_empty"_s <= "state_filling"_s + event<EventReset> / effect_reset_from_state_filling,
        "state_empty"_s <= "state_full"_s + event<EventReset> / effect_reset_from_state_full,
        "state_empty"_s <= "state_empty"_s + event<EventCaptureView> / effect_capture_view_action_window_mode_empty,
        "state_filling"_s <= "state_filling"_s + event<EventCaptureView> / effect_capture_view_action_window_mode_filling,
        "state_full"_s <= "state_full"_s + event<EventCaptureView> / effect_capture_view_action_window_mode_full,
        "state_uninitialized"_s <= "state_uninitialized"_s + event<EventAdvance> / effect_reject_error_uninitialized_event_advance,
        "state_uninitialized"_s <= "state_uninitialized"_s + event<EventReset> / effect_reject_error_uninitialized_event_reset,
        "state_uninitialized"_s <= "state_uninitialized"_s + event<EventCaptureView> / effect_reject_error_uninitialized_event_capture_view,
        "state_errored"_s <= "state_errored"_s + event<EventAdvance> / effect_reject_error_internal_error_event_advance,
        "state_errored"_s <= "state_errored"_s + event<EventReset> / effect_reject_error_internal_error_event_reset,
        "state_errored"_s <= "state_errored"_s + event<EventCaptureView> / effect_reject_error_internal_error_event_capture_view,
        "state_errored"_s <= "state_uninitialized"_s + unexpected_event<_> / effect_unexpected_from_state_uninitialized,
        "state_errored"_s <= "state_empty"_s + unexpected_event<_> / effect_unexpected_from_state_empty,
        "state_errored"_s <= "state_filling"_s + unexpected_event<_> / effect_unexpected_from_state_filling,
        "state_errored"_s <= "state_full"_s + unexpected_event<_> / effect_unexpected_from_state_full,
        "state_errored"_s <= "state_errored"_s + unexpected_event<_> / effect_unexpected_from_state_errored,
    }
}

/// Context for `MemoryStreaming` (TODO: context.hpp / detail.hpp).
#[derive(Debug, Default)]
pub struct MemoryStreamingContext {
    // TODO: port fields from matching context.hpp / detail.hpp in emel.cpp
}

impl MemoryStreamingStateMachineContext for MemoryStreamingContext {
    fn effect_capture_view_action_window_mode_empty(
        &mut self,
        _event: &EventCaptureView,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/streaming/actions.hpp::effect_capture_view
        todo!(
            "TODO: port action `effect_capture_view` from emel.cpp/src/emel/memory/streaming/actions.hpp"
        )
    }
    fn effect_capture_view_action_window_mode_filling(
        &mut self,
        _event: &EventCaptureView,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/streaming/actions.hpp::effect_capture_view
        todo!(
            "TODO: port action `effect_capture_view` from emel.cpp/src/emel/memory/streaming/actions.hpp"
        )
    }
    fn effect_capture_view_action_window_mode_full(
        &mut self,
        _event: &EventCaptureView,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/streaming/actions.hpp::effect_capture_view
        todo!(
            "TODO: port action `effect_capture_view` from emel.cpp/src/emel/memory/streaming/actions.hpp"
        )
    }
    fn effect_initialize_from_state_errored(&mut self, _event: &EventInitialize) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/streaming/actions.hpp::effect_initialize
        todo!(
            "TODO: port action `effect_initialize` from emel.cpp/src/emel/memory/streaming/actions.hpp"
        )
    }
    fn effect_initialize_from_state_uninitialized(
        &mut self,
        _event: &EventInitialize,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/streaming/actions.hpp::effect_initialize
        todo!(
            "TODO: port action `effect_initialize` from emel.cpp/src/emel/memory/streaming/actions.hpp"
        )
    }
    fn effect_reject_error_already_initialized_from_state_empty(
        &mut self,
        _event: &EventInitialize,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/streaming/actions.hpp::effect_reject
        todo!(
            "TODO: port action `effect_reject` from emel.cpp/src/emel/memory/streaming/actions.hpp"
        )
    }
    fn effect_reject_error_already_initialized_from_state_filling(
        &mut self,
        _event: &EventInitialize,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/streaming/actions.hpp::effect_reject
        todo!(
            "TODO: port action `effect_reject` from emel.cpp/src/emel/memory/streaming/actions.hpp"
        )
    }
    fn effect_reject_error_already_initialized_from_state_full(
        &mut self,
        _event: &EventInitialize,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/streaming/actions.hpp::effect_reject
        todo!(
            "TODO: port action `effect_reject` from emel.cpp/src/emel/memory/streaming/actions.hpp"
        )
    }
    fn effect_reject_error_internal_error_event_advance(
        &mut self,
        _event: &EventAdvance,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/streaming/actions.hpp::effect_reject
        todo!(
            "TODO: port action `effect_reject` from emel.cpp/src/emel/memory/streaming/actions.hpp"
        )
    }
    fn effect_reject_error_internal_error_event_capture_view(
        &mut self,
        _event: &EventCaptureView,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/streaming/actions.hpp::effect_reject
        todo!(
            "TODO: port action `effect_reject` from emel.cpp/src/emel/memory/streaming/actions.hpp"
        )
    }
    fn effect_reject_error_internal_error_event_reset(
        &mut self,
        _event: &EventReset,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/streaming/actions.hpp::effect_reject
        todo!(
            "TODO: port action `effect_reject` from emel.cpp/src/emel/memory/streaming/actions.hpp"
        )
    }
    fn effect_reject_error_invalid_configuration_from_state_errored(
        &mut self,
        _event: &EventInitialize,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/streaming/actions.hpp::effect_reject
        todo!(
            "TODO: port action `effect_reject` from emel.cpp/src/emel/memory/streaming/actions.hpp"
        )
    }
    fn effect_reject_error_invalid_configuration_from_state_uninitialized(
        &mut self,
        _event: &EventInitialize,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/streaming/actions.hpp::effect_reject
        todo!(
            "TODO: port action `effect_reject` from emel.cpp/src/emel/memory/streaming/actions.hpp"
        )
    }
    fn effect_reject_error_position_overflow(&mut self, _event: &EventAdvance) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/streaming/actions.hpp::effect_reject
        todo!(
            "TODO: port action `effect_reject` from emel.cpp/src/emel/memory/streaming/actions.hpp"
        )
    }
    fn effect_reject_error_uninitialized_event_advance(
        &mut self,
        _event: &EventAdvance,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/streaming/actions.hpp::effect_reject
        todo!(
            "TODO: port action `effect_reject` from emel.cpp/src/emel/memory/streaming/actions.hpp"
        )
    }
    fn effect_reject_error_uninitialized_event_capture_view(
        &mut self,
        _event: &EventCaptureView,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/streaming/actions.hpp::effect_reject
        todo!(
            "TODO: port action `effect_reject` from emel.cpp/src/emel/memory/streaming/actions.hpp"
        )
    }
    fn effect_reject_error_uninitialized_event_reset(
        &mut self,
        _event: &EventReset,
    ) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/streaming/actions.hpp::effect_reject
        todo!(
            "TODO: port action `effect_reject` from emel.cpp/src/emel/memory/streaming/actions.hpp"
        )
    }
    fn effect_reset_from_state_empty(&mut self, _event: &EventReset) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/streaming/actions.hpp::effect_reset
        todo!(
            "TODO: port action `effect_reset` from emel.cpp/src/emel/memory/streaming/actions.hpp"
        )
    }
    fn effect_reset_from_state_filling(&mut self, _event: &EventReset) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/streaming/actions.hpp::effect_reset
        todo!(
            "TODO: port action `effect_reset` from emel.cpp/src/emel/memory/streaming/actions.hpp"
        )
    }
    fn effect_reset_from_state_full(&mut self, _event: &EventReset) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/streaming/actions.hpp::effect_reset
        todo!(
            "TODO: port action `effect_reset` from emel.cpp/src/emel/memory/streaming/actions.hpp"
        )
    }
    fn effect_unexpected_from_state_empty(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/streaming/actions.hpp::effect_unexpected
        todo!(
            "TODO: port action `effect_unexpected` from emel.cpp/src/emel/memory/streaming/actions.hpp"
        )
    }
    fn effect_unexpected_from_state_errored(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/streaming/actions.hpp::effect_unexpected
        todo!(
            "TODO: port action `effect_unexpected` from emel.cpp/src/emel/memory/streaming/actions.hpp"
        )
    }
    fn effect_unexpected_from_state_filling(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/streaming/actions.hpp::effect_unexpected
        todo!(
            "TODO: port action `effect_unexpected` from emel.cpp/src/emel/memory/streaming/actions.hpp"
        )
    }
    fn effect_unexpected_from_state_full(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/streaming/actions.hpp::effect_unexpected
        todo!(
            "TODO: port action `effect_unexpected` from emel.cpp/src/emel/memory/streaming/actions.hpp"
        )
    }
    fn effect_unexpected_from_state_uninitialized(&mut self) -> Result<(), ()> {
        // TODO: convert from emel.cpp/src/emel/memory/streaming/actions.hpp::effect_unexpected
        todo!(
            "TODO: port action `effect_unexpected` from emel.cpp/src/emel/memory/streaming/actions.hpp"
        )
    }
    fn guard_capacity_many(&self, _event: &EventAdvance) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/streaming/guards.hpp::guard_capacity_many
        todo!(
            "TODO: port guard `guard_capacity_many` from emel.cpp/src/emel/memory/streaming/guards.hpp"
        )
    }
    fn guard_capacity_one(&self, _event: &EventAdvance) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/streaming/guards.hpp::guard_capacity_one
        todo!(
            "TODO: port guard `guard_capacity_one` from emel.cpp/src/emel/memory/streaming/guards.hpp"
        )
    }
    fn guard_configuration_invalid(&self, _event: &EventInitialize) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/streaming/guards.hpp::guard_configuration_invalid
        todo!(
            "TODO: port guard `guard_configuration_invalid` from emel.cpp/src/emel/memory/streaming/guards.hpp"
        )
    }
    fn guard_configuration_valid(&self, _event: &EventInitialize) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/streaming/guards.hpp::guard_configuration_valid
        todo!(
            "TODO: port guard `guard_configuration_valid` from emel.cpp/src/emel/memory/streaming/guards.hpp"
        )
    }
    fn guard_filling_becomes_full(&self, _event: &EventAdvance) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/streaming/guards.hpp::guard_filling_becomes_full
        todo!(
            "TODO: port guard `guard_filling_becomes_full` from emel.cpp/src/emel/memory/streaming/guards.hpp"
        )
    }
    fn guard_filling_remains_partial(&self, _event: &EventAdvance) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/streaming/guards.hpp::guard_filling_remains_partial
        todo!(
            "TODO: port guard `guard_filling_remains_partial` from emel.cpp/src/emel/memory/streaming/guards.hpp"
        )
    }
    fn guard_full_cursor_invalid(&self, _event: &EventAdvance) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/streaming/guards.hpp::guard_full_cursor_invalid
        todo!(
            "TODO: port guard `guard_full_cursor_invalid` from emel.cpp/src/emel/memory/streaming/guards.hpp"
        )
    }
    fn guard_full_position_available_at_wrap(&self, _event: &EventAdvance) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/streaming/guards.hpp::guard_full_position_available_at_wrap
        todo!(
            "TODO: port guard `guard_full_position_available_at_wrap` from emel.cpp/src/emel/memory/streaming/guards.hpp"
        )
    }
    fn guard_full_position_available_before_wrap(&self, _event: &EventAdvance) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/streaming/guards.hpp::guard_full_position_available_before_wrap
        todo!(
            "TODO: port guard `guard_full_position_available_before_wrap` from emel.cpp/src/emel/memory/streaming/guards.hpp"
        )
    }
    fn guard_full_position_overflow(&self, _event: &EventAdvance) -> Result<bool, ()> {
        // TODO: convert from emel.cpp/src/emel/memory/streaming/guards.hpp::guard_full_position_overflow
        todo!(
            "TODO: port guard `guard_full_position_overflow` from emel.cpp/src/emel/memory/streaming/guards.hpp"
        )
    }
}
