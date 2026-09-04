//! Guards for validation, bounded writes, outcomes, and sequencing.

use super::events::FormatRuntime;

/// Request has enough caller-owned output capacity for all message content.
pub fn valid_request(event: &FormatRuntime<'_>) -> bool {
    let request = event.request.borrow();
    let available = request.output.len();
    let required = request.messages.iter().try_fold(0usize, |total, message| {
        total.checked_add(message.content.len())
    });
    matches!(required, Some(required) if required <= available)
}

/// Empty input is always a successful no-write request.
pub fn is_empty(event: &FormatRuntime<'_>) -> bool {
    event.request.borrow().messages.is_empty()
}

/// Input is non-empty but does not fit caller output.
pub fn invalid_capacity(event: &FormatRuntime<'_>) -> bool {
    let empty = event.request.borrow().messages.is_empty();
    !empty && !valid_request(event)
}

/// Completion context indicates successful formatting.
pub fn request_ok(event: &FormatRuntime<'_>) -> bool {
    event.context.borrow().error == crate::ConditionerError::None
}

/// Completion context indicates a typed formatting failure.
pub fn request_failed(event: &FormatRuntime<'_>) -> bool {
    event.context.borrow().error != crate::ConditionerError::None
}
