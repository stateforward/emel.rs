//! Source-aligned, bounded Jinja formatter actor.
//!
//! The pinned formatter is a synchronous raw renderer: it writes the supplied
//! source bytes to caller-owned output without allocation and reports the
//! bounded result through the supplied callbacks.

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
    pub const fn code(self) -> i32 {
        self as i32
    }
}

/// Immediate completion notification for a successful render.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RenderingDone {
    /// Number of source bytes written to the output.
    pub output_length: usize,
    /// Whether output was truncated. Successful formatting never truncates.
    pub output_truncated: bool,
}

/// Immediate completion notification for a failed render.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RenderingError {
    /// Source-compatible formatter error code.
    pub err: FormatterError,
    /// Source position associated with the error. Request/capacity failures use zero.
    pub error_pos: usize,
}

/// Synchronous successful completion callback.
pub type DoneCallback = fn(RenderingDone) -> bool;
/// Synchronous failed completion callback.
pub type ErrorCallback = fn(RenderingError) -> bool;

/// Caller-owned request fields projected from native `event::render`.
#[derive(Clone, Copy, Debug)]
pub struct RenderRequest<'event> {
    /// Source bytes to format.
    pub source: &'event [u8],
    /// Caller-owned destination storage.
    pub output: &'event RefCell<&'event mut [u8]>,
    /// Declared maximum number of bytes accepted for this request.
    pub output_capacity: usize,
    /// Optional successful completion callback.
    pub dispatch_done: Option<DoneCallback>,
    /// Optional failed completion callback.
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
        Self {
            source,
            output,
            output_capacity,
            dispatch_done,
            dispatch_error,
        }
    }
}

/// Mutable result context corresponding to native `event::render_ctx`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RenderContext {
    /// Internal formatter error.
    pub err: FormatterError,
    /// Number of bytes successfully written.
    pub output_length: usize,
    /// Whether output was truncated.
    pub output_truncated: bool,
    /// Numeric error sink corresponding to native `error_out`.
    pub error_out: i32,
    /// Error position sink corresponding to native `error_pos_out`.
    pub error_pos_out: usize,
}

impl Default for RenderContext {
    fn default() -> Self {
        Self {
            err: FormatterError::None,
            output_length: 0,
            output_truncated: false,
            error_out: 0,
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

/// Runtime event corresponding to native `event::render_runtime`.
#[derive(Clone, Copy, Debug)]
pub struct EventRenderRuntime<'event> {
    /// Caller-owned render request.
    pub request: RenderRequest<'event>,
    /// Synchronous result context.
    pub context: &'event RefCell<RenderContext>,
}

impl<'event> EventRenderRuntime<'event> {
    /// Constructs one runtime render event.
    #[must_use]
    pub const fn new(
        request: RenderRequest<'event>,
        context: &'event RefCell<RenderContext>,
    ) -> Self {
        Self { request, context }
    }
}

// State mapping: `src/emel/text/jinja/formatter/sm.hpp`.
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
        "unexpected"_s <= "initialized"_s + unexpected_event<EventRenderRuntime<'event>> / on_unexpected_runtime,
        "unexpected"_s <= "request_decision"_s + unexpected_event<EventRenderRuntime<'event>> / on_unexpected_runtime,
        "unexpected"_s <= "copy_exec"_s + unexpected_event<EventRenderRuntime<'event>> / on_unexpected_runtime,
        "unexpected"_s <= "result_decision"_s + unexpected_event<EventRenderRuntime<'event>> / on_unexpected_runtime,
        "unexpected"_s <= "done"_s + unexpected_event<EventRenderRuntime<'event>> / on_unexpected_runtime,
        "unexpected"_s <= "errored"_s + unexpected_event<EventRenderRuntime<'event>> / on_unexpected_runtime,
        "unexpected"_s <= "unexpected"_s + unexpected_event<EventRenderRuntime<'event>> / on_unexpected_runtime,
        "unexpected"_s <= "initialized"_s + unexpected_event<_> / on_unexpected_wildcard_from_initialized,
        "unexpected"_s <= "request_decision"_s + unexpected_event<_> / on_unexpected_wildcard_from_request_decision,
        "unexpected"_s <= "copy_exec"_s + unexpected_event<_> / on_unexpected_wildcard_from_copy_exec,
        "unexpected"_s <= "result_decision"_s + unexpected_event<_> / on_unexpected_wildcard_from_result_decision,
        "unexpected"_s <= "done"_s + unexpected_event<_> / on_unexpected_wildcard_from_done,
        "unexpected"_s <= "errored"_s + unexpected_event<_> / on_unexpected_wildcard_from_errored,
        "unexpected"_s <= "unexpected"_s + unexpected_event<_> / on_unexpected_wildcard_from_unexpected,
    }
}

/// State-machine context retained by the generated machine.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TextJinjaFormatterContext {
    /// Set when an event violates the sequencing contract.
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
        if let Some(callback) = event.request.dispatch_done {
            let context = *event.context.borrow();
            let _ = callback(RenderingDone {
                output_length: context.output_length,
                output_truncated: context.output_truncated,
            });
        }
        Ok(())
    }
    fn dispatch_error(&mut self, event: &EventRenderRuntime<'_>) -> Result<(), ()> {
        if let Some(callback) = event.request.dispatch_error {
            let context = *event.context.borrow();
            let _ = callback(RenderingError {
                err: context.err,
                error_pos: context.error_pos_out,
            });
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
        event
            .context
            .borrow_mut()
            .mark_error(FormatterError::InvalidRequest, true, 0);
        Ok(())
    }
    fn mark_empty_output(&mut self, event: &EventRenderRuntime<'_>) -> Result<(), ()> {
        event.context.borrow_mut().mark_done(0, false);
        Ok(())
    }
    fn on_unexpected_runtime(&mut self, event: &EventRenderRuntime<'_>) -> Result<(), ()> {
        event
            .context
            .borrow_mut()
            .mark_error(FormatterError::InvalidRequest, true, 0);
        self.unexpected = true;
        self.dispatch_error(event)
    }
    fn on_unexpected_wildcard_from_copy_exec(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn on_unexpected_wildcard_from_done(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn on_unexpected_wildcard_from_errored(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn on_unexpected_wildcard_from_initialized(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn on_unexpected_wildcard_from_request_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn on_unexpected_wildcard_from_result_decision(&mut self) -> Result<(), ()> {
        self.unexpected()
    }
    fn on_unexpected_wildcard_from_unexpected(&mut self) -> Result<(), ()> {
        self.unexpected()
    }

    fn reject_invalid_render_from_done(
        &mut self,
        event: &EventRenderRuntime<'_>,
    ) -> Result<(), ()> {
        reject_invalid(event)
    }
    fn reject_invalid_render_from_errored(
        &mut self,
        event: &EventRenderRuntime<'_>,
    ) -> Result<(), ()> {
        reject_invalid(event)
    }
    fn reject_invalid_render_from_initialized(
        &mut self,
        event: &EventRenderRuntime<'_>,
    ) -> Result<(), ()> {
        reject_invalid(event)
    }
    fn reject_invalid_render_from_unexpected(
        &mut self,
        event: &EventRenderRuntime<'_>,
    ) -> Result<(), ()> {
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
    event.request.output_capacity > 0
}
fn callbacks_present(event: &EventRenderRuntime<'_>) -> bool {
    event.request.dispatch_done.is_some() && event.request.dispatch_error.is_some()
}
fn source_fits(event: &EventRenderRuntime<'_>) -> bool {
    let actual_capacity = event.request.output.borrow().len();
    event.request.source.len() <= event.request.output_capacity
        && event.request.source.len() <= actual_capacity
}
fn reject_invalid(event: &EventRenderRuntime<'_>) -> Result<(), ()> {
    event
        .context
        .borrow_mut()
        .mark_error(FormatterError::InvalidRequest, false, 0);
    Ok(())
}

/// Synchronous, allocation-free bounded actor around the generated machine.
pub struct TextJinjaFormatterActor {
    machine: TextJinjaFormatterStateMachine<TextJinjaFormatterContext>,
}
impl Default for TextJinjaFormatterActor {
    fn default() -> Self {
        Self::new()
    }
}
impl TextJinjaFormatterActor {
    /// Creates an actor in the generated `initialized` state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: TextJinjaFormatterStateMachine::new(TextJinjaFormatterContext::default()),
        }
    }
    /// Processes one synchronous render request.
    pub fn process_event(&mut self, event: EventRenderRuntime<'_>) -> bool {
        let context = event.context;
        let accepted = self
            .machine
            .process_event(TextJinjaFormatterEvents::EventRenderRuntime(event))
            .is_ok();
        accepted && context.borrow().err == FormatterError::None
    }
    /// Moves the actor to the explicit unexpected state.
    pub fn process_unexpected(&mut self) -> bool {
        let _ = self.machine.context_mut().unexpected();
        self.machine.set_state(TextJinjaFormatterStates::Unexpected);
        false
    }
    /// Returns generated state inspection data.
    #[must_use]
    pub fn state(&self) -> &TextJinjaFormatterStates {
        self.machine.state()
    }
    /// Reports whether the generated machine is in `state`.
    #[must_use]
    pub fn is(&self, state: &TextJinjaFormatterStates) -> bool {
        self.machine.is(state)
    }
    /// Returns the generated machine context.
    #[must_use]
    pub fn context(&self) -> &TextJinjaFormatterContext {
        self.machine.context()
    }
}

/// Short alias matching the pinned formatter's `Formatter` name.
pub type Formatter = TextJinjaFormatterActor;
#[cfg(test)]
#[allow(clippy::cast_sign_loss)]
mod tests {
    use super::*;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Mutex, MutexGuard};

    static TEST_LOCK: Mutex<()> = Mutex::new(());
    fn lock_tests() -> MutexGuard<'static, ()> {
        match TEST_LOCK.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    static ERROR_CALLS: AtomicUsize = AtomicUsize::new(0);
    static LAST_ERROR: AtomicUsize = AtomicUsize::new(usize::MAX);
    static LAST_ERROR_POS: AtomicUsize = AtomicUsize::new(usize::MAX);

    fn done(_: RenderingDone) -> bool {
        true
    }

    fn error(event: RenderingError) -> bool {
        ERROR_CALLS.fetch_add(1, Ordering::SeqCst);
        LAST_ERROR.store(event.err.code() as usize, Ordering::SeqCst);
        LAST_ERROR_POS.store(event.error_pos, Ordering::SeqCst);
        true
    }

    fn reset_callbacks() {
        ERROR_CALLS.store(0, Ordering::SeqCst);
        LAST_ERROR.store(usize::MAX, Ordering::SeqCst);
        LAST_ERROR_POS.store(usize::MAX, Ordering::SeqCst);
    }

    #[test]
    fn invalid_capacity_rejects_without_mutating_output_and_dispatches_error() {
        let _lock = lock_tests();
        reset_callbacks();
        let source = b"ignored";
        let mut bytes = [0xa5; 8];
        let before = bytes;
        let output = RefCell::new(&mut bytes[..]);
        let context = RefCell::new(RenderContext::default());
        let request = RenderRequest::new(source, &output, 0, done, error);
        let event = EventRenderRuntime::new(request, &context);
        let mut actor = TextJinjaFormatterActor::new();

        assert!(!actor.process_event(event));
        assert_eq!(*output.borrow(), before);
        assert_eq!(context.borrow().err, FormatterError::InvalidRequest);
        assert_eq!(context.borrow().output_length, 0);
        assert!(!context.borrow().output_truncated);
        assert_eq!(
            context.borrow().error_out,
            FormatterError::InvalidRequest.code()
        );
        assert_eq!(context.borrow().error_pos_out, 0);
        assert_eq!(ERROR_CALLS.load(Ordering::SeqCst), 1);
        assert_eq!(
            LAST_ERROR.load(Ordering::SeqCst),
            FormatterError::InvalidRequest.code() as usize
        );
        assert_eq!(LAST_ERROR_POS.load(Ordering::SeqCst), 0);
        assert!(actor.is(&TextJinjaFormatterStates::Errored));
    }

    #[test]
    fn source_overflow_marks_truncation_and_preserves_output() {
        let _lock = lock_tests();
        reset_callbacks();
        let source = b"overflow";
        let mut bytes = [0x5a; 4];
        let before = bytes;
        let output = RefCell::new(&mut bytes[..]);
        let context = RefCell::new(RenderContext::default());
        let request = RenderRequest::new(source, &output, 4, done, error);
        let event = EventRenderRuntime::new(request, &context);
        let mut actor = TextJinjaFormatterActor::new();

        assert!(!actor.process_event(event));
        assert_eq!(*output.borrow(), before);
        assert_eq!(context.borrow().err, FormatterError::InvalidRequest);
        assert_eq!(context.borrow().output_length, 0);
        assert!(context.borrow().output_truncated);
        assert_eq!(ERROR_CALLS.load(Ordering::SeqCst), 1);
    }
    #[test]
    fn missing_callbacks_reject_without_dispatch_and_enter_errored() {
        let _lock = lock_tests();
        reset_callbacks();
        let source = b"ignored";
        let mut bytes = [0x3c; 8];
        let before = bytes;
        let output = RefCell::new(&mut bytes[..]);
        let context = RefCell::new(RenderContext::default());
        let request = RenderRequest::with_callbacks(source, &output, 8, None, None);
        let event = EventRenderRuntime::new(request, &context);
        let mut actor = TextJinjaFormatterActor::new();

        assert!(!actor.process_event(event));
        assert_eq!(*output.borrow(), before);
        assert_eq!(ERROR_CALLS.load(Ordering::SeqCst), 0);
        assert_eq!(context.borrow().err, FormatterError::InvalidRequest);
        assert_eq!(context.borrow().output_length, 0);
        assert!(!context.borrow().output_truncated);
        assert!(actor.is(&TextJinjaFormatterStates::Errored));
    }

    #[test]
    fn process_unexpected_preserves_explicit_state_until_valid_request() {
        let mut actor = TextJinjaFormatterActor::new();
        assert!(!actor.process_unexpected());
        assert!(actor.context().unexpected);
        assert!(actor.is(&TextJinjaFormatterStates::Unexpected));

        let source = b"ok";
        let mut bytes = [0; 2];
        let output = RefCell::new(&mut bytes[..]);
        let context = RefCell::new(RenderContext::default());
        let request = RenderRequest::new(source, &output, 2, done, error);
        assert!(actor.process_event(EventRenderRuntime::new(request, &context)));
        assert!(!actor.context().unexpected);
        assert!(actor.is(&TextJinjaFormatterStates::Done));
        assert_eq!(*output.borrow(), *source);
    }
}
