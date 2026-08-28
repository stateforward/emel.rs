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

/// Initialization event whose mutable payload is owned by the caller.
///
/// `RefCell` only bridges the shared callback required by the generated SML
/// boundary to the caller-owned mutable output; it does not allocate or
/// retain the output reference in the machine context.
#[derive(Debug)]
pub struct EventInitialize<'event> {
    /// Caller-owned error output.
    pub error_out: RefCell<&'event mut i32>,
}

/// Advance event whose mutable payload is owned by the caller.
#[derive(Debug)]
pub struct EventAdvance<'event> {
    /// Caller-owned result output.
    pub result: RefCell<&'event mut AdvanceResult>,
    /// Caller-owned error output.
    pub error_out: RefCell<&'event mut i32>,
}

/// Reset event whose mutable payload is owned by the caller.
#[derive(Debug)]
pub struct EventReset<'event> {
    /// Caller-owned error output.
    pub error_out: RefCell<&'event mut i32>,
}

/// Capture-view event whose mutable payload is owned by the caller.
#[derive(Debug)]
pub struct EventCaptureView<'event> {
    /// Caller-owned view output.
    pub view_out: RefCell<&'event mut WindowView>,
    /// Caller-owned error output.
    pub error_out: RefCell<&'event mut i32>,
}

sml! {
    MemoryStreaming {
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

    fn reject(event_error: &RefCell<&mut i32>, error: ErrorCode) {
        **event_error.borrow_mut() = error as i32;
    }

    fn initialize(&mut self, event: &EventInitialize<'_>) {
        self.reset_cursors();
        Self::reject(&event.error_out, ErrorCode::None);
    }

    fn advance<const FULL: bool, const WRAP: bool>(
        &mut self,
        event: &EventAdvance<'_>,
    ) -> Result<(), ()> {
        let logical_position = self.next_logical_position;
        let physical_position = self.next_physical_position;
        let logical_end = logical_position.checked_add(1).ok_or(())?;
        let next_physical_position = if WRAP {
            0
        } else {
            physical_position.checked_add(1).ok_or(())?
        };

        let (logical_begin, physical_begin, valid_positions) = if FULL {
            (
                logical_end
                    .checked_sub(i64::from(self.capacity))
                    .ok_or(())?,
                next_physical_position,
                self.capacity,
            )
        } else {
            (0, 0, i32::try_from(logical_end).map_err(|_| ())?)
        };

        **event.result.borrow_mut() = AdvanceResult {
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
        };
        Self::reject(&event.error_out, ErrorCode::None);
        self.next_logical_position = logical_end;
        self.next_physical_position = next_physical_position;
        Ok(())
    }

    fn reset(&mut self, event: &EventReset<'_>) {
        self.reset_cursors();
        Self::reject(&event.error_out, ErrorCode::None);
    }

    fn capture_view(
        &self,
        event: &EventCaptureView<'_>,
        logical_begin: i64,
        physical_begin: i32,
        valid_positions: i32,
    ) {
        **event.view_out.borrow_mut() = WindowView {
            logical_begin,
            logical_end: self.next_logical_position,
            physical_begin,
            next_physical_position: self.next_physical_position,
            valid_positions,
            capacity: self.capacity,
        };
        Self::reject(&event.error_out, ErrorCode::None);
    }
}

impl MemoryStreamingStateMachineContext for MemoryStreamingContext {
    fn effect_capture_view_action_window_mode_empty(
        &mut self,
        event: &EventCaptureView<'_>,
    ) -> Result<(), ()> {
        self.capture_view(event, 0, 0, 0);
        Ok(())
    }

    fn effect_capture_view_action_window_mode_filling(
        &mut self,
        event: &EventCaptureView<'_>,
    ) -> Result<(), ()> {
        self.capture_view(
            event,
            0,
            0,
            i32::try_from(self.next_logical_position).map_err(|_| ())?,
        );
        Ok(())
    }

    fn effect_capture_view_action_window_mode_full(
        &mut self,
        event: &EventCaptureView<'_>,
    ) -> Result<(), ()> {
        self.capture_view(
            event,
            self.next_logical_position
                .checked_sub(i64::from(self.capacity))
                .ok_or(())?,
            self.next_physical_position,
            self.capacity,
        );
        Ok(())
    }

    fn effect_initialize_from_state_errored(
        &mut self,
        event: &EventInitialize<'_>,
    ) -> Result<(), ()> {
        self.initialize(event);
        Ok(())
    }

    fn effect_initialize_from_state_uninitialized(
        &mut self,
        event: &EventInitialize<'_>,
    ) -> Result<(), ()> {
        self.initialize(event);
        Ok(())
    }

    fn effect_reject_error_already_initialized_from_state_empty(
        &mut self,
        event: &EventInitialize<'_>,
    ) -> Result<(), ()> {
        Self::reject(&event.error_out, ErrorCode::AlreadyInitialized);
        Ok(())
    }

    fn effect_reject_error_already_initialized_from_state_filling(
        &mut self,
        event: &EventInitialize<'_>,
    ) -> Result<(), ()> {
        Self::reject(&event.error_out, ErrorCode::AlreadyInitialized);
        Ok(())
    }

    fn effect_reject_error_already_initialized_from_state_full(
        &mut self,
        event: &EventInitialize<'_>,
    ) -> Result<(), ()> {
        Self::reject(&event.error_out, ErrorCode::AlreadyInitialized);
        Ok(())
    }

    fn effect_advance_empty_capacity_one(&mut self, event: &EventAdvance<'_>) -> Result<(), ()> {
        self.advance::<true, true>(event)
    }

    fn effect_advance_empty_capacity_many(&mut self, event: &EventAdvance<'_>) -> Result<(), ()> {
        self.advance::<false, false>(event)
    }

    fn effect_advance_filling_partial(&mut self, event: &EventAdvance<'_>) -> Result<(), ()> {
        self.advance::<false, false>(event)
    }

    fn effect_advance_filling_full(&mut self, event: &EventAdvance<'_>) -> Result<(), ()> {
        self.advance::<true, true>(event)
    }

    fn effect_advance_full_before_wrap(&mut self, event: &EventAdvance<'_>) -> Result<(), ()> {
        self.advance::<true, false>(event)
    }

    fn effect_advance_full_at_wrap(&mut self, event: &EventAdvance<'_>) -> Result<(), ()> {
        self.advance::<true, true>(event)
    }

    fn effect_reject_error_internal_error_event_advance(
        &mut self,
        event: &EventAdvance<'_>,
    ) -> Result<(), ()> {
        Self::reject(&event.error_out, ErrorCode::InternalError);
        Ok(())
    }

    fn effect_reject_error_internal_error_event_capture_view(
        &mut self,
        event: &EventCaptureView<'_>,
    ) -> Result<(), ()> {
        Self::reject(&event.error_out, ErrorCode::InternalError);
        Ok(())
    }

    fn effect_reject_error_internal_error_event_reset(
        &mut self,
        event: &EventReset<'_>,
    ) -> Result<(), ()> {
        Self::reject(&event.error_out, ErrorCode::InternalError);
        Ok(())
    }

    fn effect_reject_error_invalid_configuration_from_state_errored(
        &mut self,
        event: &EventInitialize<'_>,
    ) -> Result<(), ()> {
        Self::reject(&event.error_out, ErrorCode::InvalidConfiguration);
        Ok(())
    }

    fn effect_reject_error_invalid_configuration_from_state_uninitialized(
        &mut self,
        event: &EventInitialize<'_>,
    ) -> Result<(), ()> {
        Self::reject(&event.error_out, ErrorCode::InvalidConfiguration);
        Ok(())
    }

    fn effect_reject_error_position_overflow(
        &mut self,
        event: &EventAdvance<'_>,
    ) -> Result<(), ()> {
        Self::reject(&event.error_out, ErrorCode::PositionOverflow);
        Ok(())
    }

    fn effect_reject_error_uninitialized_event_advance(
        &mut self,
        event: &EventAdvance<'_>,
    ) -> Result<(), ()> {
        Self::reject(&event.error_out, ErrorCode::Uninitialized);
        Ok(())
    }

    fn effect_reject_error_uninitialized_event_capture_view(
        &mut self,
        event: &EventCaptureView<'_>,
    ) -> Result<(), ()> {
        Self::reject(&event.error_out, ErrorCode::Uninitialized);
        Ok(())
    }

    fn effect_reject_error_uninitialized_event_reset(
        &mut self,
        event: &EventReset<'_>,
    ) -> Result<(), ()> {
        Self::reject(&event.error_out, ErrorCode::Uninitialized);
        Ok(())
    }

    fn effect_reset_from_state_empty(&mut self, event: &EventReset<'_>) -> Result<(), ()> {
        self.reset(event);
        Ok(())
    }

    fn effect_reset_from_state_filling(&mut self, event: &EventReset<'_>) -> Result<(), ()> {
        self.reset(event);
        Ok(())
    }

    fn effect_reset_from_state_full(&mut self, event: &EventReset<'_>) -> Result<(), ()> {
        self.reset(event);
        Ok(())
    }

    fn effect_unexpected_from_state_empty(&mut self) -> Result<(), ()> {
        Ok(())
    }

    fn effect_unexpected_from_state_errored(&mut self) -> Result<(), ()> {
        Ok(())
    }

    fn effect_unexpected_from_state_filling(&mut self) -> Result<(), ()> {
        Ok(())
    }

    fn effect_unexpected_from_state_full(&mut self) -> Result<(), ()> {
        Ok(())
    }

    fn effect_unexpected_from_state_uninitialized(&mut self) -> Result<(), ()> {
        Ok(())
    }

    fn guard_capacity_many(&self, _event: &EventAdvance<'_>) -> Result<bool, ()> {
        Ok(self.capacity > 1)
    }

    fn guard_capacity_one(&self, _event: &EventAdvance<'_>) -> Result<bool, ()> {
        Ok(self.capacity == 1)
    }

    fn guard_configuration_invalid(&self, _event: &EventInitialize<'_>) -> Result<bool, ()> {
        Ok(self.capacity <= 0)
    }

    fn guard_configuration_valid(&self, _event: &EventInitialize<'_>) -> Result<bool, ()> {
        Ok(self.capacity > 0)
    }

    fn guard_filling_becomes_full(&self, _event: &EventAdvance<'_>) -> Result<bool, ()> {
        Ok(self.next_logical_position.checked_add(1) == Some(i64::from(self.capacity)))
    }

    fn guard_filling_remains_partial(&self, _event: &EventAdvance<'_>) -> Result<bool, ()> {
        Ok(matches!(
            self.next_logical_position.checked_add(1),
            Some(next) if next < i64::from(self.capacity)
        ))
    }

    fn guard_full_cursor_invalid(&self, _event: &EventAdvance<'_>) -> Result<bool, ()> {
        Ok(self.next_logical_position < i64::MAX
            && (self.next_physical_position < 0 || self.next_physical_position >= self.capacity))
    }

    fn guard_full_position_available_at_wrap(&self, _event: &EventAdvance<'_>) -> Result<bool, ()> {
        Ok(self.next_logical_position < i64::MAX
            && self.next_physical_position >= 0
            && self.capacity.checked_sub(1) == Some(self.next_physical_position))
    }

    fn guard_full_position_available_before_wrap(
        &self,
        _event: &EventAdvance<'_>,
    ) -> Result<bool, ()> {
        Ok(self.next_logical_position < i64::MAX
            && self.next_physical_position >= 0
            && matches!(
                self.capacity.checked_sub(1),
                Some(last) if self.next_physical_position < last
            ))
    }

    fn guard_full_position_overflow(&self, _event: &EventAdvance<'_>) -> Result<bool, ()> {
        Ok(self.next_logical_position == i64::MAX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn initialize(machine: &mut MemoryStreamingStateMachine<MemoryStreamingContext>) -> i32 {
        let mut error = -1;
        let event = EventInitialize {
            error_out: RefCell::new(&mut error),
        };
        machine
            .process_event(MemoryStreamingEvents::EventInitialize(event))
            .expect("initialize transition");
        error
    }

    fn advance(
        machine: &mut MemoryStreamingStateMachine<MemoryStreamingContext>,
    ) -> (AdvanceResult, i32) {
        let mut result = AdvanceResult::default();
        let mut error = -1;
        let event = EventAdvance {
            result: RefCell::new(&mut result),
            error_out: RefCell::new(&mut error),
        };
        machine
            .process_event(MemoryStreamingEvents::EventAdvance(event))
            .expect("advance transition");
        (result, error)
    }

    #[test]
    fn invalid_initialize_stays_uninitialized_and_publishes_source_code() {
        let mut machine = MemoryStreamingStateMachine::new(MemoryStreamingContext::new(0));
        let mut error = -1;
        let event = EventInitialize {
            error_out: RefCell::new(&mut error),
        };
        machine
            .process_event(MemoryStreamingEvents::EventInitialize(event))
            .unwrap();
        assert_eq!(error, ErrorCode::InvalidConfiguration as i32);
        assert!(matches!(
            machine.state(),
            MemoryStreamingStates::StateUninitialized
        ));
    }

    #[test]
    fn capacity_one_publishes_full_wrapped_window() {
        let mut machine = MemoryStreamingStateMachine::new(MemoryStreamingContext::new(1));
        assert_eq!(initialize(&mut machine), ErrorCode::None as i32);

        let (result, error) = advance(&mut machine);
        assert_eq!(error, ErrorCode::None as i32);
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
        assert_eq!(initialize(&mut machine), ErrorCode::None as i32);

        let (first, _) = advance(&mut machine);
        assert_eq!(first.window.valid_positions, 1);
        let (second, _) = advance(&mut machine);
        assert_eq!(second.window.valid_positions, 2);
        let (third, _) = advance(&mut machine);
        assert_eq!(third.window.logical_begin, 0);
        assert_eq!(third.window.physical_begin, 0);
        assert_eq!(third.window.next_physical_position, 0);

        let (fourth, _) = advance(&mut machine);
        assert_eq!(fourth.logical_position, 3);
        assert_eq!(fourth.physical_position, 0);
        assert_eq!(fourth.window.logical_begin, 1);
        assert_eq!(fourth.window.logical_end, 4);
        assert_eq!(fourth.window.physical_begin, 1);

        let mut view = WindowView::default();
        let mut error = -1;
        let event = EventCaptureView {
            view_out: RefCell::new(&mut view),
            error_out: RefCell::new(&mut error),
        };
        machine
            .process_event(MemoryStreamingEvents::EventCaptureView(event))
            .unwrap();
        assert_eq!(error, ErrorCode::None as i32);
        assert_eq!(view.logical_begin, 1);
        assert_eq!(view.logical_end, 4);
        assert_eq!(view.physical_begin, 1);
        assert_eq!(view.next_physical_position, 1);
        assert_eq!(view.valid_positions, 3);
    }

    #[test]
    fn reset_is_constant_time_and_returns_to_empty_cursors() {
        let mut machine = MemoryStreamingStateMachine::new(MemoryStreamingContext::new(2));
        initialize(&mut machine);
        advance(&mut machine);
        advance(&mut machine);

        let mut error = -1;
        let event = EventReset {
            error_out: RefCell::new(&mut error),
        };
        machine
            .process_event(MemoryStreamingEvents::EventReset(event))
            .unwrap();
        assert_eq!(error, ErrorCode::None as i32);
        assert!(matches!(machine.state(), MemoryStreamingStates::StateEmpty));
        assert_eq!(machine.context().next_logical_position(), 0);
        assert_eq!(machine.context().next_physical_position(), 0);
    }

    #[test]
    fn pre_initialize_and_duplicate_initialize_preserve_ring_state() {
        let mut machine = MemoryStreamingStateMachine::new(MemoryStreamingContext::new(2));
        let mut result = AdvanceResult::default();
        let mut error = -1;
        let event = EventAdvance {
            result: RefCell::new(&mut result),
            error_out: RefCell::new(&mut error),
        };
        machine
            .process_event(MemoryStreamingEvents::EventAdvance(event))
            .unwrap();
        assert_eq!(error, ErrorCode::Uninitialized as i32);
        assert_eq!(machine.context().next_logical_position(), 0);

        assert_eq!(initialize(&mut machine), ErrorCode::None as i32);
        advance(&mut machine);
        let logical = machine.context().next_logical_position();
        let physical = machine.context().next_physical_position();

        let mut duplicate_error = -1;
        let event = EventInitialize {
            error_out: RefCell::new(&mut duplicate_error),
        };
        machine
            .process_event(MemoryStreamingEvents::EventInitialize(event))
            .unwrap();
        assert_eq!(duplicate_error, ErrorCode::AlreadyInitialized as i32);
        assert_eq!(machine.context().next_logical_position(), logical);
        assert_eq!(machine.context().next_physical_position(), physical);
    }

    #[test]
    fn pre_initialize_reset_and_capture_publish_source_errors() {
        let mut machine = MemoryStreamingStateMachine::new(MemoryStreamingContext::new(2));

        let mut reset_error = -1;
        let reset_event = EventReset {
            error_out: RefCell::new(&mut reset_error),
        };
        machine
            .process_event(MemoryStreamingEvents::EventReset(reset_event))
            .unwrap();
        assert_eq!(reset_error, ErrorCode::Uninitialized as i32);

        let mut view = WindowView::default();
        let mut capture_error = -1;
        let capture_event = EventCaptureView {
            view_out: RefCell::new(&mut view),
            error_out: RefCell::new(&mut capture_error),
        };
        machine
            .process_event(MemoryStreamingEvents::EventCaptureView(capture_event))
            .unwrap();
        assert_eq!(capture_error, ErrorCode::Uninitialized as i32);
        assert_eq!(view, WindowView::default());
    }

    #[test]
    fn errored_initialize_is_explicitly_invalid_or_recovering() {
        let mut invalid = MemoryStreamingStateMachine::new(MemoryStreamingContext::new(0));
        invalid.set_state(MemoryStreamingStates::StateErrored);
        let mut invalid_error = -1;
        let invalid_event = EventInitialize {
            error_out: RefCell::new(&mut invalid_error),
        };
        invalid
            .process_event(MemoryStreamingEvents::EventInitialize(invalid_event))
            .unwrap();
        assert_eq!(invalid_error, ErrorCode::InvalidConfiguration as i32);
        assert!(matches!(
            invalid.state(),
            MemoryStreamingStates::StateErrored
        ));

        let mut recovering = MemoryStreamingStateMachine::new(MemoryStreamingContext::new(2));
        recovering.set_state(MemoryStreamingStates::StateErrored);
        recovering.context_mut().next_logical_position = 8;
        recovering.context_mut().next_physical_position = 1;
        let mut recovery_error = -1;
        let recovery_event = EventInitialize {
            error_out: RefCell::new(&mut recovery_error),
        };
        recovering
            .process_event(MemoryStreamingEvents::EventInitialize(recovery_event))
            .unwrap();
        assert_eq!(recovery_error, ErrorCode::None as i32);
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
        advance(&mut machine);
        advance(&mut machine);

        machine.context_mut().next_logical_position = i64::MAX;
        machine.context_mut().next_physical_position = 0;
        let before_overflow = machine.context().next_physical_position();
        let mut overflow_result = AdvanceResult::default();
        let mut overflow_error = -1;
        let overflow_event = EventAdvance {
            result: RefCell::new(&mut overflow_result),
            error_out: RefCell::new(&mut overflow_error),
        };
        machine
            .process_event(MemoryStreamingEvents::EventAdvance(overflow_event))
            .unwrap();
        assert_eq!(overflow_error, ErrorCode::PositionOverflow as i32);
        assert_eq!(machine.context().next_physical_position(), before_overflow);

        machine.context_mut().next_logical_position = 2;
        machine.context_mut().next_physical_position = -1;
        let mut cursor_result = AdvanceResult::default();
        let mut cursor_error = -1;
        let cursor_event = EventAdvance {
            result: RefCell::new(&mut cursor_result),
            error_out: RefCell::new(&mut cursor_error),
        };
        machine
            .process_event(MemoryStreamingEvents::EventAdvance(cursor_event))
            .unwrap();
        assert_eq!(cursor_error, ErrorCode::InternalError as i32);
        assert_eq!(machine.context().next_logical_position(), 2);
        assert_eq!(machine.context().next_physical_position(), -1);
    }
}
