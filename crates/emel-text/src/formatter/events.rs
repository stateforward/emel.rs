//! Borrowed runtime events and typed outcomes for the bounded formatter actor.

use core::cell::RefCell;

use super::context::Context;
use super::format::FormatRequest;
use crate::ConditionerError;

/// Successful synchronous formatter completion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FormattingDone {
    /// Exact number of bytes written to the caller-owned output.
    pub output_length: usize,
}

/// Failed synchronous formatter completion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FormattingError {
    /// Typed error reported by the formatter boundary.
    pub error: ConditionerError,
}

/// Immediate success callback. The actor never retains this slot.
pub type DoneCallback = fn(FormattingDone);
/// Immediate error callback. The actor never retains this slot.
pub type ErrorCallback = fn(FormattingError);

/// Caller-owned runtime context populated during one dispatch.
#[derive(Clone, Copy, Debug)]
pub struct FormatRuntime<'event> {
    /// Caller-owned borrowed formatting request.
    pub request: &'event RefCell<FormatRequest<'event>>,
    /// Caller-owned result context; it is borrowed only for this dispatch.
    pub context: &'event RefCell<Context>,
    /// Optional immediate success callback.
    pub dispatch_done: Option<DoneCallback>,
    /// Optional immediate error callback.
    pub dispatch_error: Option<ErrorCallback>,
}

impl<'event> FormatRuntime<'event> {
    /// Builds one runtime event with optional immediate callbacks.
    #[must_use]
    pub const fn new(
        request: &'event RefCell<FormatRequest<'event>>,
        context: &'event RefCell<Context>,
        dispatch_done: Option<DoneCallback>,
        dispatch_error: Option<ErrorCallback>,
    ) -> Self {
        Self {
            request,
            context,
            dispatch_done,
            dispatch_error,
        }
    }
}

/// Source-shaped spelling for the formatter runtime event.
pub type EventFormatRuntime<'event> = FormatRuntime<'event>;
