//! Source-aligned, bounded Jinja text formatter actor.
//!
//! This is the Rust projection of the pinned formatter state machine.  The
//! renderer itself is intentionally not exposed here: the formatter copies a
//! caller-provided source buffer into a caller-owned destination and reports
//! the same request/capacity/error outcomes as the source machine.

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

use core::cell::RefCell;
use sml::sml;

/// Error values carried by the pinned formatter runtime context.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(i32)]
pub enum FormatterError {
    /// No error was produced by the request.
    #[default]
    None = 0,
    /// The request was malformed or the source did not fit the destination.
    InvalidRequest = 1,
}

impl FormatterError {
    /// Returns the source-compatible numeric error code.
    #[must_use]
    pub const fn code(self) -> i32 { self as i32 }
}

/// Immediate completion notification for a successful render.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RenderingDone {
    /// Number of source bytes copied to the output.
    pub output_length: usize,
    /// Whether the output was truncated.  The bounded formatter never
    /// truncates successful copies, but retains this source field.
    pub output_truncated: bool,
}

/// Immediate completion notification for a failed render.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RenderingError {
    /// Source-compatible formatter error code.
    pub err: FormatterError,
    /// Position associated with the error, zero for request/capacity errors.
    pub error_pos: usize,
}

/// Synchronous callback used by [`EventRenderRuntime`].
///
/// Function pointers are used instead of heap-backed closures so dispatch is
/// bounded and allocation-free.  The callback's boolean return is ignored,
/// matching the source callback contract.
pub type DoneCallback = fn(RenderingDone) -> bool;
/// Synchronous error callback used by [`EventRenderRuntime`].
pub type ErrorCallback = fn(RenderingError) -> bool;

/// Caller-owned request fields projected from `event::render`.
///
/// The source program and globals are deliberately absent: this target only
/// implements the formatter's bounded source-copy orchestration contract.
#[derive(Clone, Copy, Debug)]
pub struct RenderRequest<'event> {
    /// Source bytes to copy.
    pub source: &'event [u8],
    /// Caller-owned output storage wrapped for synchronous state-machine
    /// completion copies.
    pub output: &'event RefCell<&'event mut [u8]>,
    /// Maximum number of bytes accepted for this request.
    pub output_capacity: usize,
    /// Optional successful completion callback.
    pub dispatch_done: Option<DoneCallback>,
    /// Optional error completion callback.
    pub dispatch_error: Option<ErrorCallback>,
}

impl<'event> RenderRequest<'event> {
    /// Creates a request with both callbacks installed.
    #[must_use]
    pub const fn new(
        source: &'event [u8],
        output: &'event RefCell<&'event mut [u8]>,
        output_capacity: usize,
        dispatch_done: DoneCallback,
        dispatch_error: ErrorCallback,
    ) -> Self {
        Self {
            source,
            output,
            output_capacity,
            dispatch_done: Some(dispatch_done),
            dispatch_error: Some(dispatch_error),
        }
    }

    /// Creates a request with independently optional callbacks.
    #[must_use]
    pub const fn with_callbacks(
        source: &'event [u8],
        output: &'event RefCell<&'event mut [u8]>,
        output_capacity: usize,
        dispatch_done: Option<DoneCallback>,
        dispatch_error: Option<ErrorCallback>,
    ) -> Self {
        Self { source, output, output_capacity, dispatch_done, dispatch_error }
    }
}

/// Mutable result context corresponding to `event::render_ctx`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RenderContext {
    /// Internal formatter error.
    pub err: FormatterError,
    /// Bytes successfully copied.
    pub output_length: usize,
    /// Whether output was truncated.
    pub output_truncated: bool,
    /// Numeric error output corresponding to the source optional sink.
    pub error_out: i32,
    /// Error position corresponding to the source optional sink.
    pub error_pos_out: usize,
}

impl Default for RenderContext {
    fn default() -> Self {
        Self {
            err: FormatterError::None,
            output_length: 0,
            output_truncated: false,
            error_out: FormatterError::None.code(),
            error_pos_out: 0,
        }
    }
}

impl RenderContext {
    fn reset(&mut self) {
        *self = Self::default();
    }

    fn mark_done(&mut self, output_length: usize, output_truncated: bool) {
        self.err = FormatterError::None;
        self.output_length = output_length;
        self.output_truncated = output_truncated;
        self.error_out = FormatterError::None.code();
        self.error_pos_out = 0;
    }

    fn mark_error(&mut self, err: FormatterError, output_truncated: bool, error_pos: usize) {
        self.err = err;
        self.output_length = 0;
        self.output_truncated = output_truncated;
        self.error_out = err.code();
        self.error_pos_out = error_pos;
    }
}

/// Runtime event shell corresponding to `event::render_runtime`.
#[derive(Clone, Copy, Debug)]
pub struct EventRenderRuntime<'event> {
    /// Copied bounded request fields.
    pub request: RenderRequest<'event>,
    /// Synchronous result context.
    pub context: &'event RefCell<RenderContext>,
}

impl<'event> EventRenderRuntime<'event> {
    /// Constructs one bounded runtime event.
    #[must_use]
    pub const fn new(request: RenderRequest<'event>, context: &'event RefCell<RenderContext>) -> Self {
        Self { request, context }
    }
}

// Source mapping: pinned
// `src/emel/text/jinja/formatter/sm.hpp` (destination-first rows retained).
sml! {
    TextJinjaFormatter<'event> {
        "request_decision"_s <= *"initialized"_s + event<EventRenderRuntime<'event>> [valid_render] / begin_render_from_initialized,
        "result_decision"_s <= "initialized"_s + event<EventRenderRuntime<'event>> [invalid_render_with_callbacks] / reject_invalid_render_from_initialized,
        "errored"_s <= "initialized"_s + event<EventRenderRuntime<'event>> [invalid_render_without_callbacks] / reject_invalid_render_from_initialized,
        "request_decision"_s <= "done"_s + event<EventRenderRuntime<'event>> [valid_render] / begin_render_from_done,
        "result_decision"_s <= "done"_s + event<EventRenderRuntime<'event>> [invalid_render_with_callbacks] / reject_invalid_render_from_done,
        "errored"_s <= "done"_s + event<EventRenderRuntime<'event>> [invalid_render_without_callbacks] / reject_invalid_render_from_done,
        "request_decision"_s <= "errored"_s + event<EventRenderRuntime<'event>> [valid_render] / begin_render_from_errored,
        "result_decision"_s <= "errored"_s + event<EventRenderRuntime<'event>> [invalid_render_with_callbacks] / reject_invalid_render_from_errored,
        "errored"_s <= "errored"_s + event<EventRenderRuntime<'event>> [invalid_render_without_callbacks] / reject_invalid_render_from_errored,
        "request_decision"_s <= "unexpected"_s + event<EventRenderRuntime<'event>> [valid_render] / begin_render_from_unexpected,
        "result_decision"_s <= "unexpected"_s + event<EventRenderRuntime<'event>> [invalid_render_with_callbacks] / reject_invalid_render_from_unexpected,
        "errored"_s <= "unexpected"_s + event<EventRenderRuntime<'event>> [invalid_render_without_callbacks] / reject_invalid_render_from_unexpected,
        "result_decision"_s <= "request_decision"_s + completion<EventRenderRuntime>(EventRenderRuntime<'event>) [source_empty] / mark_empty_output,
        "copy_exec"_s <= "request_decision"_s + completion<EventRenderRuntime>(EventRenderRuntime<'event>) [copy_ready] / copy_source_text,
        "result_decision"_s <= "request_decision"_s + completion<EventRenderRuntime>(EventRenderRuntime<'event>) [source_overflow] / mark_capacity_error,
        "result_decision"_s <= "copy_exec"_s + completion<EventRenderRuntime>(EventRenderRuntime<'event>),
        "done"_s <= "result_decision"_s + completion<EventRenderRuntime>(EventRenderRuntime<'event>) [request_ok] / dispatch_done,
        "errored"_s <= "result_decision"_s + completion<EventRenderRuntime>(EventRenderRuntime<'event>) [request_failed] / dispatch_error,
        "unexpected"_s <= "initialized"_s + unexpected_event<_> / on_unexpected_from_initialized,
        "unexpected"_s <= "request_decision"_s + unexpected_event<_> / on_unexpected_from_request_decision,
        "unexpected"_s <= "copy_exec"_s + unexpected_event<_> / on_unexpected_from_copy_exec,
        "unexpected"_s <= "result_decision"_s + unexpected_event<_> / on_unexpected_from_result_decision,
        "unexpected"_s <= "done"_s + unexpected_event<_> / on_unexpected_from_done,
        "unexpected"_s <= "errored"_s + unexpected_event<_> / on_unexpected_from_errored,
        "unexpected"_s <= "unexpected"_s + unexpected_event<_> / on_unexpected_from_unexpected,
    }
}

/// State-machine context retained by the generated machine.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TextJinjaFormatterContext {
    /// Set when an event violates the current sequencing contract.
    pub unexpected: bool,
}

impl TextJinjaFormatterContext {
    fn unexpected(&mut self) -> Result<(), ()> {
        self.unexpected = true;
        Ok(())
    }
}

impl TextJinjaFormatterStateMachineContext for TextJinjaFormatterContext {
    fn begin_render_from_done(&mut self, event: &EventRenderRuntime<'_>) -> Result<(), ()> {
        event.context.borrow_mut().reset();
        self.unexpected = false;
        Ok(())
    }
    fn begin_render_from_errored(&mut self, event: &EventRenderRuntime<'_>) -> Result<(), ()> {
        self.begin_render_from_done(event)
    }
    fn begin_render_from_initialized(&mut self, event: &EventRenderRuntime<'_>) -> Result<(), ()> {
        self.begin_render_from_done(event)
    }
    fn begin_render_from_unexpected(&mut self, event: &EventRenderRuntime<'_>) -> Result<(), ()> {
        self.begin_render_from_done(event)
    }

    fn copy_ready(&self, event: &EventRenderRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.request.source.is_empty() && source_fits(event))
    }
    fn copy_source_text(&mut self, event: &EventRenderRuntime<'_>) -> Result<(), ()> {
        let source = event.request.source;
        event.request.output.borrow_mut()[..source.len()].copy_from_slice(source);
        event.context.borrow_mut().mark_done(source.len(), false);
        Ok(())
    }
    fn dispatch_done(&mut self, event: &EventRenderRuntime<'_>) -> Result<(), ()> {
        let callback = event.request.dispatch_done;
        if let Some(callback) = callback {
            let context = *event.context.borrow();
            let _ = callback(RenderingDone {
                output_length: context.output_length,
                output_truncated: context.output_truncated,
            });
        }
        Ok(())
    }
    fn dispatch_error(&mut self, event: &EventRenderRuntime<'_>) -> Result<(), ()> {
        let callback = event.request.dispatch_error;
        if let Some(callback) = callback {
            let context = *event.context.borrow();
            let _ = callback(RenderingError { err: context.err, error_pos: context.error_pos_out });
        }
        Ok(())
    }

    fn invalid_render_with_callbacks(&self, event: &EventRenderRuntime<'_>) -> Result<bool, ()> {
        Ok(!valid_request(event) && callbacks_present(event))
    }
    fn invalid_render_without_callbacks(&self, event: &EventRenderRuntime<'_>) -> Result<bool, ()> {
        Ok(!callbacks_present(event))
    }
    fn mark_capacity_error(&mut self, event: &EventRenderRuntime<'_>) -> Result<(), ()> {
        event.context.borrow_mut().mark_error(FormatterError::InvalidRequest, true, 0);
        Ok(())
    }
    fn mark_empty_output(&mut self, event: &EventRenderRuntime<'_>) -> Result<(), ()> {
        event.context.borrow_mut().mark_done(0, false);
        Ok(())
    }

    fn on_unexpected_from_copy_exec(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_done(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_errored(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_initialized(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_request_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_result_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn on_unexpected_from_unexpected(&mut self) -> Result<(), ()> { self.unexpected() }

    fn reject_invalid_render_from_done(&mut self, event: &EventRenderRuntime<'_>) -> Result<(), ()> {
        reject_invalid(event)
    }
    fn reject_invalid_render_from_errored(&mut self, event: &EventRenderRuntime<'_>) -> Result<(), ()> {
        reject_invalid(event)
    }
    fn reject_invalid_render_from_initialized(&mut self, event: &EventRenderRuntime<'_>) -> Result<(), ()> {
        reject_invalid(event)
    }
    fn reject_invalid_render_from_unexpected(&mut self, event: &EventRenderRuntime<'_>) -> Result<(), ()> {
        reject_invalid(event)
    }

    fn request_failed(&self, event: &EventRenderRuntime<'_>) -> Result<bool, ()> {
        Ok(event.context.borrow().err != FormatterError::None)
    }
    fn request_ok(&self, event: &EventRenderRuntime<'_>) -> Result<bool, ()> {
        Ok(event.context.borrow().err == FormatterError::None)
    }
    fn source_empty(&self, event: &EventRenderRuntime<'_>) -> Result<bool, ()> {
        Ok(event.request.source.is_empty())
    }
    fn source_overflow(&self, event: &EventRenderRuntime<'_>) -> Result<bool, ()> {
        Ok(!source_fits(event))
    }
    fn valid_render(&self, event: &EventRenderRuntime<'_>) -> Result<bool, ()> {
        Ok(valid_request(event) && callbacks_present(event))
    }
}

fn valid_request(event: &EventRenderRuntime<'_>) -> bool {
    // A Rust slice is always a valid source view; only nonzero capacity
    // remains from the pinned request validation.
    event.request.output_capacity > 0
}

fn callbacks_present(event: &EventRenderRuntime<'_>) -> bool {
    event.request.dispatch_done.is_some() && event.request.dispatch_error.is_some()
}
fn source_fits(event: &EventRenderRuntime<'_>) -> bool {
    let output_capacity = event.request.output.borrow().len();
    event.request.source.len() <= event.request.output_capacity
        && event.request.source.len() <= output_capacity
}

fn reject_invalid(event: &EventRenderRuntime<'_>) -> Result<(), ()> {
    event.context.borrow_mut().mark_error(FormatterError::InvalidRequest, false, 0);
    Ok(())
}

/// Synchronous bounded actor around the generated formatter machine.
pub struct TextJinjaFormatterActor<'event> {
    machine: TextJinjaFormatterStateMachine<'event, TextJinjaFormatterContext>,
}

impl<'event> Default for TextJinjaFormatterActor<'event> {
    fn default() -> Self { Self::new() }
}

impl<'event> TextJinjaFormatterActor<'event> {
    /// Creates an actor in the generated `initialized` state.
    #[must_use]
    pub fn new() -> Self {
        Self { machine: TextJinjaFormatterStateMachine::new(TextJinjaFormatterContext::default()) }
    }

    pub fn process_event(&mut self, event: EventRenderRuntime<'event>) -> bool {
        let context = event.context;
        let accepted = self
            .machine
            .process_event(TextJinjaFormatterEvents::EventRenderRuntime(event))
            .is_ok();
        accepted && context.borrow().err == FormatterError::None
    }

    /// Dispatches an explicit unexpected event and records the violation.
    pub fn process_unexpected(&mut self) -> bool {
        let _ = self.machine.context_mut().unexpected();
        self.machine.set_state(TextJinjaFormatterStates::Unexpected);
        false
    }

    /// Returns generated state inspection data.
    #[must_use]
    pub fn state(&self) -> &TextJinjaFormatterStates { self.machine.state() }

    /// Reports whether the generated machine is in `state`.
    #[must_use]
    pub fn is(&self, state: &TextJinjaFormatterStates) -> bool { self.machine.is(state) }

    /// Returns the generated machine context.
    #[must_use]
    pub fn context(&self) -> &TextJinjaFormatterContext { self.machine.context() }
}

/// Short alias matching the pinned formatter's `Formatter` name.
pub type Formatter<'event> = TextJinjaFormatterActor<'event>;
