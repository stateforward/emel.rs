//! Run-to-completion state machine for the bounded memory stream.

#![allow(
    clippy::enum_variant_names,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::derive_partial_eq_without_eq,
    dead_code,
    missing_docs
)]

use core::cell::RefCell;

use sml::sml;

/// Error values published by the streaming event boundary.
#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ErrorCode {
    /// No error occurred.
    None = 0,
    /// The injected capacity is not positive.
    InvalidConfiguration = 1,
    /// The stream has not been initialized.
    Uninitialized = 2,
    /// The stream has already been initialized.
    AlreadyInitialized = 4,
    /// The logical cursor cannot advance.
    PositionOverflow = 8,
    /// The stream context is internally inconsistent.
    InternalError = 16,
}

/// The half-open logical and physical window published by the stream.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct WindowView {
    /// Inclusive logical beginning of the view.
    pub logical_begin: i64,
    /// Exclusive logical end of the view.
    pub logical_end: i64,
    /// Physical slot corresponding to `logical_begin`.
    pub physical_begin: i32,
    /// Physical slot selected for the next write.
    pub next_physical_position: i32,
    /// Number of valid logical positions in the view.
    pub valid_positions: i32,
    /// Ring capacity.
    pub capacity: i32,
}

/// Result published after one successful stream advance.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdvanceResult {
    /// Logical position assigned to the newly advanced item.
    pub logical_position: i64,
    /// Physical position assigned to the newly advanced item.
    pub physical_position: i32,
    /// View after the advance.
    pub window: WindowView,
}

impl Default for AdvanceResult {
    fn default() -> Self {
        Self {
            logical_position: -1,
            physical_position: -1,
            window: WindowView::default(),
        }
    }
}

/// Typed result of an initialization event.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum InitializeOutcome {
    /// The caller has not dispatched the event yet.
    #[default]
    Pending,
    /// The stream was initialized or recovered.
    Initialized,
    /// Initialization was rejected with the source-aligned error.
    Rejected(ErrorCode),
}

/// Typed result of an advance event.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum AdvanceOutcome {
    /// The caller has not dispatched the event yet.
    #[default]
    Pending,
    /// The stream advanced and published its new window.
    Advanced(AdvanceResult),
    /// The advance was rejected with the source-aligned error.
    Rejected(ErrorCode),
}

/// Typed result of a reset event.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum ResetOutcome {
    /// The caller has not dispatched the event yet.
    #[default]
    Pending,
    /// The stream was reset to its empty cursors.
    Reset,
    /// The reset was rejected with the source-aligned error.
    Rejected(ErrorCode),
}

/// Typed result of a capture-view event.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CaptureViewOutcome {
    /// The caller has not dispatched the event yet.
    #[default]
    Pending,
    /// The stream published a view.
    Captured(WindowView),
    /// The capture was rejected with the source-aligned error.
    Rejected(ErrorCode),
}

/// Private synchronous bridge from generated shared callbacks to caller-owned output.
struct OutputBridge<'event, T> {
    output: RefCell<&'event mut T>,
}

impl<'event, T> OutputBridge<'event, T> {
    const fn new(output: &'event mut T) -> Self {
        Self {
            output: RefCell::new(output),
        }
    }

    fn publish(&self, value: T) {
        **self.output.borrow_mut() = value;
    }
}

/// Initialization event envelope.
///
/// The mutable output is held only by a private synchronous bridge. The event
/// is consumed by `process_event`; the machine context never retains it.
pub struct EventInitialize<'event> {
    bridge: OutputBridge<'event, InitializeOutcome>,
}

impl<'event> EventInitialize<'event> {
    /// Creates an event that publishes into the caller-owned outcome after dispatch.
    #[must_use]
    pub const fn new(outcome: &'event mut InitializeOutcome) -> Self {
        Self {
            bridge: OutputBridge::new(outcome),
        }
    }
}

impl core::fmt::Debug for EventInitialize<'_> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.debug_struct("EventInitialize").finish()
    }
}

/// Advance event envelope.
///
/// The mutable output is held only by a private synchronous bridge. The event
/// is consumed by `process_event`; the machine context never retains it.
pub struct EventAdvance<'event> {
    bridge: OutputBridge<'event, AdvanceOutcome>,
}

impl<'event> EventAdvance<'event> {
    /// Creates an event that publishes into the caller-owned outcome after dispatch.
    #[must_use]
    pub const fn new(outcome: &'event mut AdvanceOutcome) -> Self {
        Self {
            bridge: OutputBridge::new(outcome),
        }
    }
}

impl core::fmt::Debug for EventAdvance<'_> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.debug_struct("EventAdvance").finish()
    }
}

/// Reset event envelope.
///
/// The mutable output is held only by a private synchronous bridge. The event
/// is consumed by `process_event`; the machine context never retains it.
pub struct EventReset<'event> {
    bridge: OutputBridge<'event, ResetOutcome>,
}

impl<'event> EventReset<'event> {
    /// Creates an event that publishes into the caller-owned outcome after dispatch.
    #[must_use]
    pub const fn new(outcome: &'event mut ResetOutcome) -> Self {
        Self {
            bridge: OutputBridge::new(outcome),
        }
    }
}

impl core::fmt::Debug for EventReset<'_> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.debug_struct("EventReset").finish()
    }
}

/// Capture-view event envelope.
///
/// The mutable output is held only by a private synchronous bridge. The event
/// is consumed by `process_event`; the machine context never retains it.
pub struct EventCaptureView<'event> {
    bridge: OutputBridge<'event, CaptureViewOutcome>,
}

impl<'event> EventCaptureView<'event> {
    /// Creates an event that publishes into the caller-owned outcome after dispatch.
    #[must_use]
    pub const fn new(outcome: &'event mut CaptureViewOutcome) -> Self {
        Self {
            bridge: OutputBridge::new(outcome),
        }
    }
}

impl core::fmt::Debug for EventCaptureView<'_> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.debug_struct("EventCaptureView").finish()
    }
}

sml! {
    MemoryStreaming[custom_error] {
        "state_empty"_s <= *"state_uninitialized"_s + event<EventInitialize>(EventInitialize<'event>) [guard_configuration_valid] / effect_initialize_from_state_uninitialized,
        "state_uninitialized"_s <= "state_uninitialized"_s + event<EventInitialize>(EventInitialize<'event>) [guard_configuration_invalid] / effect_reject_error_invalid_configuration_from_state_uninitialized,
        "state_empty"_s <= "state_empty"_s + event<EventInitialize>(EventInitialize<'event>) / effect_reject_error_already_initialized_from_state_empty,
        "state_filling"_s <= "state_filling"_s + event<EventInitialize>(EventInitialize<'event>) / effect_reject_error_already_initialized_from_state_filling,
        "state_full"_s <= "state_full"_s + event<EventInitialize>(EventInitialize<'event>) / effect_reject_error_already_initialized_from_state_full,
        "state_empty"_s <= "state_errored"_s + event<EventInitialize>(EventInitialize<'event>) [guard_configuration_valid] / effect_initialize_from_state_errored,
        "state_errored"_s <= "state_errored"_s + event<EventInitialize>(EventInitialize<'event>) [guard_configuration_invalid] / effect_reject_error_invalid_configuration_from_state_errored,

        "state_full"_s <= "state_empty"_s + event<EventAdvance>(EventAdvance<'event>) [guard_capacity_one] / effect_advance_empty_capacity_one,
        "state_filling"_s <= "state_empty"_s + event<EventAdvance>(EventAdvance<'event>) [guard_capacity_many] / effect_advance_empty_capacity_many,
        "state_filling"_s <= "state_filling"_s + event<EventAdvance>(EventAdvance<'event>) [guard_filling_remains_partial] / effect_advance_filling_partial,
        "state_full"_s <= "state_filling"_s + event<EventAdvance>(EventAdvance<'event>) [guard_filling_becomes_full] / effect_advance_filling_full,
        "state_full"_s <= "state_full"_s + event<EventAdvance>(EventAdvance<'event>) [guard_full_position_available_before_wrap] / effect_advance_full_before_wrap,
        "state_full"_s <= "state_full"_s + event<EventAdvance>(EventAdvance<'event>) [guard_full_position_available_at_wrap] / effect_advance_full_at_wrap,
        "state_full"_s <= "state_full"_s + event<EventAdvance>(EventAdvance<'event>) [guard_full_position_overflow] / effect_reject_error_position_overflow,
        "state_full"_s <= "state_full"_s + event<EventAdvance>(EventAdvance<'event>) [guard_full_cursor_invalid] / effect_reject_error_internal_error_event_advance,

        "state_empty"_s <= "state_empty"_s + event<EventReset>(EventReset<'event>) / effect_reset_from_state_empty,
        "state_empty"_s <= "state_filling"_s + event<EventReset>(EventReset<'event>) / effect_reset_from_state_filling,
        "state_empty"_s <= "state_full"_s + event<EventReset>(EventReset<'event>) / effect_reset_from_state_full,
        "state_empty"_s <= "state_empty"_s + event<EventCaptureView>(EventCaptureView<'event>) / effect_capture_view_action_window_mode_empty,
        "state_filling"_s <= "state_filling"_s + event<EventCaptureView>(EventCaptureView<'event>) / effect_capture_view_action_window_mode_filling,
        "state_full"_s <= "state_full"_s + event<EventCaptureView>(EventCaptureView<'event>) / effect_capture_view_action_window_mode_full,

        "state_uninitialized"_s <= "state_uninitialized"_s + event<EventAdvance>(EventAdvance<'event>) / effect_reject_error_uninitialized_event_advance,
        "state_uninitialized"_s <= "state_uninitialized"_s + event<EventReset>(EventReset<'event>) / effect_reject_error_uninitialized_event_reset,
        "state_uninitialized"_s <= "state_uninitialized"_s + event<EventCaptureView>(EventCaptureView<'event>) / effect_reject_error_uninitialized_event_capture_view,
        "state_errored"_s <= "state_errored"_s + event<EventAdvance>(EventAdvance<'event>) / effect_reject_error_internal_error_event_advance,
        "state_errored"_s <= "state_errored"_s + event<EventReset>(EventReset<'event>) / effect_reject_error_internal_error_event_reset,
        "state_errored"_s <= "state_errored"_s + event<EventCaptureView>(EventCaptureView<'event>) / effect_reject_error_internal_error_event_capture_view,

        "state_errored"_s <= "state_uninitialized"_s + unexpected_event<_> / effect_unexpected_from_state_uninitialized,
        "state_errored"_s <= "state_empty"_s + unexpected_event<_> / effect_unexpected_from_state_empty,
        "state_errored"_s <= "state_filling"_s + unexpected_event<_> / effect_unexpected_from_state_filling,
        "state_errored"_s <= "state_full"_s + unexpected_event<_> / effect_unexpected_from_state_full,
        "state_errored"_s <= "state_errored"_s + unexpected_event<_> / effect_unexpected_from_state_errored,
    }
}

/// Runtime-owned stream cursors and injected ring capacity.
#[derive(Debug, Default)]
pub struct MemoryStreamingContext {
    capacity: i32,
    next_logical_position: i64,
    next_physical_position: i32,
}

impl MemoryStreamingContext {
    /// Creates an uninitialized stream with the injected capacity.
    pub const fn new(capacity: i32) -> Self {
        Self {
            capacity,
            next_logical_position: 0,
            next_physical_position: 0,
        }
    }

    /// Returns the injected ring capacity.
    pub const fn capacity(&self) -> i32 {
        self.capacity
    }

    /// Returns the next logical cursor.
    pub const fn next_logical_position(&self) -> i64 {
        self.next_logical_position
    }

    /// Returns the next physical cursor.
    pub const fn next_physical_position(&self) -> i32 {
        self.next_physical_position
    }

    const fn reset_cursors(&mut self) {
        self.next_logical_position = 0;
        self.next_physical_position = 0;
    }

    fn initialize(&mut self, event: &EventInitialize<'_>) {
        self.reset_cursors();
        event.bridge.publish(InitializeOutcome::Initialized);
    }

    #[allow(clippy::cast_possible_truncation)]
    fn advance<const FULL: bool, const WRAP: bool>(&mut self, event: &EventAdvance<'_>) {
        let logical_position = self.next_logical_position;
        let physical_position = self.next_physical_position;
        let logical_end = logical_position + 1;
        let next_physical_position = if WRAP { 0 } else { physical_position + 1 };

        let (logical_begin, physical_begin, valid_positions) = if FULL {
            (
                logical_end - i64::from(self.capacity),
                next_physical_position,
                self.capacity,
            )
        } else {
            (0, 0, logical_end as i32)
        };

        event
            .bridge
            .publish(AdvanceOutcome::Advanced(AdvanceResult {
                logical_position,
                physical_position,
                window: WindowView {
                    logical_begin,
                    logical_end,
                    physical_begin,
                    next_physical_position,
                    valid_positions,
                    capacity: self.capacity,
                },
            }));
        self.next_logical_position = logical_end;
        self.next_physical_position = next_physical_position;
    }

    fn reset(&mut self, event: &EventReset<'_>) {
        self.reset_cursors();
        event.bridge.publish(ResetOutcome::Reset);
    }

    fn capture_view(
        &self,
        event: &EventCaptureView<'_>,
        logical_begin: i64,
        physical_begin: i32,
        valid_positions: i32,
    ) {
        event
            .bridge
            .publish(CaptureViewOutcome::Captured(WindowView {
                logical_begin,
                logical_end: self.next_logical_position,
                physical_begin,
                next_physical_position: self.next_physical_position,
                valid_positions,
                capacity: self.capacity,
            }));
    }
}

impl MemoryStreamingStateMachineContext for MemoryStreamingContext {
    type Error = ErrorCode;

    fn effect_capture_view_action_window_mode_empty(
        &mut self,
        event: &EventCaptureView<'_>,
    ) -> Result<(), ErrorCode> {
        self.capture_view(event, 0, 0, 0);
        Ok(())
    }

    #[allow(clippy::cast_possible_truncation)]
    fn effect_capture_view_action_window_mode_filling(
        &mut self,
        event: &EventCaptureView<'_>,
    ) -> Result<(), ErrorCode> {
        self.capture_view(event, 0, 0, self.next_logical_position as i32);
        Ok(())
    }

    fn effect_capture_view_action_window_mode_full(
        &mut self,
        event: &EventCaptureView<'_>,
    ) -> Result<(), ErrorCode> {
        self.capture_view(
            event,
            self.next_logical_position - i64::from(self.capacity),
            self.next_physical_position,
            self.capacity,
        );
        Ok(())
    }

    fn effect_initialize_from_state_errored(
        &mut self,
        event: &EventInitialize<'_>,
    ) -> Result<(), ErrorCode> {
        self.initialize(event);
        Ok(())
    }

    fn effect_initialize_from_state_uninitialized(
        &mut self,
        event: &EventInitialize<'_>,
    ) -> Result<(), ErrorCode> {
        self.initialize(event);
        Ok(())
    }

    fn effect_reject_error_already_initialized_from_state_empty(
        &mut self,
        event: &EventInitialize<'_>,
    ) -> Result<(), ErrorCode> {
        event
            .bridge
            .publish(InitializeOutcome::Rejected(ErrorCode::AlreadyInitialized));
        Ok(())
    }

    fn effect_reject_error_already_initialized_from_state_filling(
        &mut self,
        event: &EventInitialize<'_>,
    ) -> Result<(), ErrorCode> {
        event
            .bridge
            .publish(InitializeOutcome::Rejected(ErrorCode::AlreadyInitialized));
        Ok(())
    }

    fn effect_reject_error_already_initialized_from_state_full(
        &mut self,
        event: &EventInitialize<'_>,
    ) -> Result<(), ErrorCode> {
        event
            .bridge
            .publish(InitializeOutcome::Rejected(ErrorCode::AlreadyInitialized));
        Ok(())
    }

    fn effect_advance_empty_capacity_one(
        &mut self,
        event: &EventAdvance<'_>,
    ) -> Result<(), ErrorCode> {
        self.advance::<true, true>(event);
        Ok(())
    }

    fn effect_advance_empty_capacity_many(
        &mut self,
        event: &EventAdvance<'_>,
    ) -> Result<(), ErrorCode> {
        self.advance::<false, false>(event);
        Ok(())
    }

    fn effect_advance_filling_partial(
        &mut self,
        event: &EventAdvance<'_>,
    ) -> Result<(), ErrorCode> {
        self.advance::<false, false>(event);
        Ok(())
    }

    fn effect_advance_filling_full(&mut self, event: &EventAdvance<'_>) -> Result<(), ErrorCode> {
        self.advance::<true, true>(event);
        Ok(())
    }

    fn effect_advance_full_before_wrap(
        &mut self,
        event: &EventAdvance<'_>,
    ) -> Result<(), ErrorCode> {
        self.advance::<true, false>(event);
        Ok(())
    }

    fn effect_advance_full_at_wrap(&mut self, event: &EventAdvance<'_>) -> Result<(), ErrorCode> {
        self.advance::<true, true>(event);
        Ok(())
    }

    fn effect_reject_error_internal_error_event_advance(
        &mut self,
        event: &EventAdvance<'_>,
    ) -> Result<(), ErrorCode> {
        event
            .bridge
            .publish(AdvanceOutcome::Rejected(ErrorCode::InternalError));
        Ok(())
    }

    fn effect_reject_error_internal_error_event_capture_view(
        &mut self,
        event: &EventCaptureView<'_>,
    ) -> Result<(), ErrorCode> {
        event
            .bridge
            .publish(CaptureViewOutcome::Rejected(ErrorCode::InternalError));
        Ok(())
    }

    fn effect_reject_error_internal_error_event_reset(
        &mut self,
        event: &EventReset<'_>,
    ) -> Result<(), ErrorCode> {
        event
            .bridge
            .publish(ResetOutcome::Rejected(ErrorCode::InternalError));
        Ok(())
    }

    fn effect_reject_error_invalid_configuration_from_state_errored(
        &mut self,
        event: &EventInitialize<'_>,
    ) -> Result<(), ErrorCode> {
        event
            .bridge
            .publish(InitializeOutcome::Rejected(ErrorCode::InvalidConfiguration));
        Ok(())
    }

    fn effect_reject_error_invalid_configuration_from_state_uninitialized(
        &mut self,
        event: &EventInitialize<'_>,
    ) -> Result<(), ErrorCode> {
        event
            .bridge
            .publish(InitializeOutcome::Rejected(ErrorCode::InvalidConfiguration));
        Ok(())
    }

    fn effect_reject_error_position_overflow(
        &mut self,
        event: &EventAdvance<'_>,
    ) -> Result<(), ErrorCode> {
        event
            .bridge
            .publish(AdvanceOutcome::Rejected(ErrorCode::PositionOverflow));
        Ok(())
    }

    fn effect_reject_error_uninitialized_event_advance(
        &mut self,
        event: &EventAdvance<'_>,
    ) -> Result<(), ErrorCode> {
        event
            .bridge
            .publish(AdvanceOutcome::Rejected(ErrorCode::Uninitialized));
        Ok(())
    }

    fn effect_reject_error_uninitialized_event_capture_view(
        &mut self,
        event: &EventCaptureView<'_>,
    ) -> Result<(), ErrorCode> {
        event
            .bridge
            .publish(CaptureViewOutcome::Rejected(ErrorCode::Uninitialized));
        Ok(())
    }

    fn effect_reject_error_uninitialized_event_reset(
        &mut self,
        event: &EventReset<'_>,
    ) -> Result<(), ErrorCode> {
        event
            .bridge
            .publish(ResetOutcome::Rejected(ErrorCode::Uninitialized));
        Ok(())
    }

    fn effect_reset_from_state_empty(&mut self, event: &EventReset<'_>) -> Result<(), ErrorCode> {
        self.reset(event);
        Ok(())
    }

    fn effect_reset_from_state_filling(&mut self, event: &EventReset<'_>) -> Result<(), ErrorCode> {
        self.reset(event);
        Ok(())
    }

    fn effect_reset_from_state_full(&mut self, event: &EventReset<'_>) -> Result<(), ErrorCode> {
        self.reset(event);
        Ok(())
    }

    fn effect_unexpected_from_state_empty(&mut self) -> Result<(), ErrorCode> {
        Ok(())
    }

    fn effect_unexpected_from_state_errored(&mut self) -> Result<(), ErrorCode> {
        Ok(())
    }

    fn effect_unexpected_from_state_filling(&mut self) -> Result<(), ErrorCode> {
        Ok(())
    }

    fn effect_unexpected_from_state_full(&mut self) -> Result<(), ErrorCode> {
        Ok(())
    }

    fn effect_unexpected_from_state_uninitialized(&mut self) -> Result<(), ErrorCode> {
        Ok(())
    }

    fn guard_capacity_many(&self, _event: &EventAdvance<'_>) -> Result<bool, ErrorCode> {
        Ok(self.capacity > 1)
    }

    fn guard_capacity_one(&self, _event: &EventAdvance<'_>) -> Result<bool, ErrorCode> {
        Ok(self.capacity == 1)
    }

    fn guard_configuration_invalid(&self, _event: &EventInitialize<'_>) -> Result<bool, ErrorCode> {
        Ok(self.capacity <= 0)
    }

    fn guard_configuration_valid(&self, _event: &EventInitialize<'_>) -> Result<bool, ErrorCode> {
        Ok(self.capacity > 0)
    }

    fn guard_filling_becomes_full(&self, _event: &EventAdvance<'_>) -> Result<bool, ErrorCode> {
        Ok(self.next_logical_position.checked_add(1) == Some(i64::from(self.capacity)))
    }

    fn guard_filling_remains_partial(&self, _event: &EventAdvance<'_>) -> Result<bool, ErrorCode> {
        Ok(matches!(
            self.next_logical_position.checked_add(1),
            Some(next) if next < i64::from(self.capacity)
        ))
    }

    fn guard_full_cursor_invalid(&self, _event: &EventAdvance<'_>) -> Result<bool, ErrorCode> {
        Ok(self.next_logical_position < i64::MAX
            && (self.next_physical_position < 0 || self.next_physical_position >= self.capacity))
    }

    fn guard_full_position_available_at_wrap(
        &self,
        _event: &EventAdvance<'_>,
    ) -> Result<bool, ErrorCode> {
        Ok(self.next_logical_position < i64::MAX
            && self.next_physical_position >= 0
            && self.capacity.checked_sub(1) == Some(self.next_physical_position))
    }

    fn guard_full_position_available_before_wrap(
        &self,
        _event: &EventAdvance<'_>,
    ) -> Result<bool, ErrorCode> {
        Ok(self.next_logical_position < i64::MAX
            && self.next_physical_position >= 0
            && matches!(
                self.capacity.checked_sub(1),
                Some(last) if self.next_physical_position < last
            ))
    }

    fn guard_full_position_overflow(&self, _event: &EventAdvance<'_>) -> Result<bool, ErrorCode> {
        Ok(self.next_logical_position == i64::MAX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn initialize(
        machine: &mut MemoryStreamingStateMachine<MemoryStreamingContext>,
    ) -> InitializeOutcome {
        let mut outcome = InitializeOutcome::default();
        let event = EventInitialize::new(&mut outcome);
        machine
            .process_event(MemoryStreamingEvents::EventInitialize(event))
            .expect("initialize transition");
        outcome
    }

    fn advance(
        machine: &mut MemoryStreamingStateMachine<MemoryStreamingContext>,
    ) -> AdvanceOutcome {
        let mut outcome = AdvanceOutcome::default();
        let event = EventAdvance::new(&mut outcome);
        machine
            .process_event(MemoryStreamingEvents::EventAdvance(event))
            .expect("advance transition");
        outcome
    }

    fn advanced(
        machine: &mut MemoryStreamingStateMachine<MemoryStreamingContext>,
    ) -> AdvanceResult {
        match advance(machine) {
            AdvanceOutcome::Advanced(result) => result,
            outcome => panic!("expected successful advance, got {outcome:?}"),
        }
    }

    #[test]
    fn invalid_initialize_stays_uninitialized_and_publishes_source_code() {
        let mut machine = MemoryStreamingStateMachine::new(MemoryStreamingContext::new(0));
        let mut outcome = InitializeOutcome::default();
        let event = EventInitialize::new(&mut outcome);
        machine
            .process_event(MemoryStreamingEvents::EventInitialize(event))
            .unwrap();
        assert_eq!(
            outcome,
            InitializeOutcome::Rejected(ErrorCode::InvalidConfiguration)
        );
        assert!(matches!(
            machine.state(),
            MemoryStreamingStates::StateUninitialized
        ));
    }

    #[test]
    fn capacity_one_publishes_full_wrapped_window() {
        let mut machine = MemoryStreamingStateMachine::new(MemoryStreamingContext::new(1));
        assert_eq!(initialize(&mut machine), InitializeOutcome::Initialized);

        let result = advanced(&mut machine);
        assert_eq!(
            result,
            AdvanceResult {
                logical_position: 0,
                physical_position: 0,
                window: WindowView {
                    logical_begin: 0,
                    logical_end: 1,
                    physical_begin: 0,
                    next_physical_position: 0,
                    valid_positions: 1,
                    capacity: 1,
                },
            }
        );
        assert_eq!(machine.context().next_logical_position(), 1);
        assert_eq!(machine.context().next_physical_position(), 0);
    }

    #[test]
    fn capacity_many_fills_then_wraps_and_capture_is_half_open() {
        let mut machine = MemoryStreamingStateMachine::new(MemoryStreamingContext::new(3));
        assert_eq!(initialize(&mut machine), InitializeOutcome::Initialized);

        let first = advanced(&mut machine);
        assert_eq!(first.window.valid_positions, 1);
        let second = advanced(&mut machine);
        assert_eq!(second.window.valid_positions, 2);
        let third = advanced(&mut machine);
        assert_eq!(third.window.logical_begin, 0);
        assert_eq!(third.window.physical_begin, 0);
        assert_eq!(third.window.next_physical_position, 0);

        let fourth = advanced(&mut machine);
        assert_eq!(fourth.logical_position, 3);
        assert_eq!(fourth.physical_position, 0);
        assert_eq!(fourth.window.logical_begin, 1);
        assert_eq!(fourth.window.logical_end, 4);
        assert_eq!(fourth.window.physical_begin, 1);

        let mut outcome = CaptureViewOutcome::default();
        let event = EventCaptureView::new(&mut outcome);
        machine
            .process_event(MemoryStreamingEvents::EventCaptureView(event))
            .unwrap();
        assert_eq!(
            outcome,
            CaptureViewOutcome::Captured(WindowView {
                logical_begin: 1,
                logical_end: 4,
                physical_begin: 1,
                next_physical_position: 1,
                valid_positions: 3,
                capacity: 3,
            })
        );
    }

    #[test]
    fn reset_is_constant_time_and_returns_to_empty_cursors() {
        let mut machine = MemoryStreamingStateMachine::new(MemoryStreamingContext::new(2));
        initialize(&mut machine);
        advanced(&mut machine);
        advanced(&mut machine);

        let mut outcome = ResetOutcome::default();
        let event = EventReset::new(&mut outcome);
        machine
            .process_event(MemoryStreamingEvents::EventReset(event))
            .unwrap();
        assert_eq!(outcome, ResetOutcome::Reset);
        assert!(matches!(machine.state(), MemoryStreamingStates::StateEmpty));
        assert_eq!(machine.context().next_logical_position(), 0);
        assert_eq!(machine.context().next_physical_position(), 0);
    }

    #[test]
    fn pre_initialize_and_duplicate_initialize_preserve_ring_state() {
        let mut machine = MemoryStreamingStateMachine::new(MemoryStreamingContext::new(2));
        let mut outcome = AdvanceOutcome::default();
        let event = EventAdvance::new(&mut outcome);
        machine
            .process_event(MemoryStreamingEvents::EventAdvance(event))
            .unwrap();
        assert_eq!(outcome, AdvanceOutcome::Rejected(ErrorCode::Uninitialized));
        assert_eq!(machine.context().next_logical_position(), 0);

        assert_eq!(initialize(&mut machine), InitializeOutcome::Initialized);
        advanced(&mut machine);
        let logical = machine.context().next_logical_position();
        let physical = machine.context().next_physical_position();

        let mut duplicate_outcome = InitializeOutcome::default();
        let event = EventInitialize::new(&mut duplicate_outcome);
        machine
            .process_event(MemoryStreamingEvents::EventInitialize(event))
            .unwrap();
        assert_eq!(
            duplicate_outcome,
            InitializeOutcome::Rejected(ErrorCode::AlreadyInitialized)
        );
        assert_eq!(machine.context().next_logical_position(), logical);
        assert_eq!(machine.context().next_physical_position(), physical);
    }

    #[test]
    fn pre_initialize_reset_and_capture_publish_source_errors() {
        let mut machine = MemoryStreamingStateMachine::new(MemoryStreamingContext::new(2));

        let mut reset_outcome = ResetOutcome::default();
        let reset_event = EventReset::new(&mut reset_outcome);
        machine
            .process_event(MemoryStreamingEvents::EventReset(reset_event))
            .unwrap();
        assert_eq!(
            reset_outcome,
            ResetOutcome::Rejected(ErrorCode::Uninitialized)
        );

        let mut capture_outcome = CaptureViewOutcome::default();
        let capture_event = EventCaptureView::new(&mut capture_outcome);
        machine
            .process_event(MemoryStreamingEvents::EventCaptureView(capture_event))
            .unwrap();
        assert_eq!(
            capture_outcome,
            CaptureViewOutcome::Rejected(ErrorCode::Uninitialized)
        );
    }

    #[test]
    fn errored_initialize_is_explicitly_invalid_or_recovering() {
        let mut invalid = MemoryStreamingStateMachine::new(MemoryStreamingContext::new(0));
        invalid.set_state(MemoryStreamingStates::StateErrored);
        let mut invalid_outcome = InitializeOutcome::default();
        let invalid_event = EventInitialize::new(&mut invalid_outcome);
        invalid
            .process_event(MemoryStreamingEvents::EventInitialize(invalid_event))
            .unwrap();
        assert_eq!(
            invalid_outcome,
            InitializeOutcome::Rejected(ErrorCode::InvalidConfiguration)
        );
        assert!(matches!(
            invalid.state(),
            MemoryStreamingStates::StateErrored
        ));

        let mut recovering = MemoryStreamingStateMachine::new(MemoryStreamingContext::new(2));
        recovering.set_state(MemoryStreamingStates::StateErrored);
        recovering.context_mut().next_logical_position = 8;
        recovering.context_mut().next_physical_position = 1;
        let mut recovery_outcome = InitializeOutcome::default();
        let recovery_event = EventInitialize::new(&mut recovery_outcome);
        recovering
            .process_event(MemoryStreamingEvents::EventInitialize(recovery_event))
            .unwrap();
        assert_eq!(recovery_outcome, InitializeOutcome::Initialized);
        assert!(matches!(
            recovering.state(),
            MemoryStreamingStates::StateEmpty
        ));
        assert_eq!(recovering.context().next_logical_position(), 0);
        assert_eq!(recovering.context().next_physical_position(), 0);
    }

    #[test]
    fn full_overflow_and_invalid_cursor_publish_errors_without_cursor_changes() {
        let mut machine = MemoryStreamingStateMachine::new(MemoryStreamingContext::new(2));
        initialize(&mut machine);
        advanced(&mut machine);
        advanced(&mut machine);

        machine.context_mut().next_logical_position = i64::MAX;
        machine.context_mut().next_physical_position = 0;
        let before_overflow = machine.context().next_physical_position();
        let mut overflow_outcome = AdvanceOutcome::default();
        let overflow_event = EventAdvance::new(&mut overflow_outcome);
        machine
            .process_event(MemoryStreamingEvents::EventAdvance(overflow_event))
            .unwrap();
        assert_eq!(
            overflow_outcome,
            AdvanceOutcome::Rejected(ErrorCode::PositionOverflow)
        );
        assert_eq!(machine.context().next_physical_position(), before_overflow);

        machine.context_mut().next_logical_position = 2;
        machine.context_mut().next_physical_position = -1;
        let mut cursor_outcome = AdvanceOutcome::default();
        let cursor_event = EventAdvance::new(&mut cursor_outcome);
        machine
            .process_event(MemoryStreamingEvents::EventAdvance(cursor_event))
            .unwrap();
        assert_eq!(
            cursor_outcome,
            AdvanceOutcome::Rejected(ErrorCode::InternalError)
        );
        assert_eq!(machine.context().next_logical_position(), 2);
        assert_eq!(machine.context().next_physical_position(), -1);
    }
}
