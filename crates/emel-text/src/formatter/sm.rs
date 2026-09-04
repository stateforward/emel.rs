//! Bounded synchronous formatter actor and generated state machine.

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

use super::context::Context;
use super::events::{FormatRuntime, FormattingDone, FormattingError};
use crate::ConditionerError;

sml! {
    TextFormatter<'event> {
        "validate"_s <= *"initialized"_s + event<FormatRuntime<'event>>,
        "validate"_s <= "done"_s + event<FormatRuntime<'event>>,
        "validate"_s <= "errored"_s + event<FormatRuntime<'event>>,
        "validate"_s <= "unexpected"_s + event<FormatRuntime<'event>>,
        "write"_s <= "validate"_s + completion<FormatRuntime>(FormatRuntime<'event>) [valid_request] / write_request,
        "publish_done"_s <= "validate"_s + completion<FormatRuntime>(FormatRuntime<'event>) [is_empty] / write_empty,
        "publish_error"_s <= "validate"_s + completion<FormatRuntime>(FormatRuntime<'event>) [invalid_capacity] / reject_request,
        "done"_s <= "write"_s + completion<FormatRuntime>(FormatRuntime<'event>) / publish_success,
        "done"_s <= "publish_done"_s + completion<FormatRuntime>(FormatRuntime<'event>) / publish_success,
        "errored"_s <= "publish_error"_s + completion<FormatRuntime>(FormatRuntime<'event>) / publish_failure,
        "unexpected"_s <= "initialized"_s + unexpected_event<_> / mark_unexpected,
        "unexpected"_s <= "validate"_s + unexpected_event<_> / mark_unexpected,
        "unexpected"_s <= "write"_s + unexpected_event<_> / mark_unexpected,
        "unexpected"_s <= "publish_done"_s + unexpected_event<_> / mark_unexpected,
        "unexpected"_s <= "publish_error"_s + unexpected_event<_> / mark_unexpected,
        "unexpected"_s <= "done"_s + unexpected_event<_> / mark_unexpected,
        "unexpected"_s <= "errored"_s + unexpected_event<_> / mark_unexpected,
        "unexpected"_s <= "unexpected"_s + unexpected_event<_> / mark_unexpected,
    }
}

impl TextFormatterStateMachineContext for Context {
    fn valid_request(&self, event: &FormatRuntime<'_>) -> Result<bool, ()> {
        Ok(super::guards::valid_request(event))
    }
    fn is_empty(&self, event: &FormatRuntime<'_>) -> Result<bool, ()> {
        Ok(super::guards::is_empty(event))
    }
    fn invalid_capacity(&self, event: &FormatRuntime<'_>) -> Result<bool, ()> {
        Ok(super::guards::invalid_capacity(event))
    }
    fn write_empty(&mut self, event: &FormatRuntime<'_>) -> Result<(), ()> {
        *event.request.borrow_mut().output_length = 0;
        event.context.borrow_mut().done(0);
        Ok(())
    }
    fn write_request(&mut self, event: &FormatRuntime<'_>) -> Result<(), ()> {
        let mut request = event.request.borrow_mut();
        let mut offset = 0;
        for message in request.messages {
            let end = offset + message.content.len();
            request.output[offset..end].copy_from_slice(message.content);
            offset = end;
        }
        *request.output_length = offset;
        event.context.borrow_mut().done(offset);
        Ok(())
    }
    fn publish_success(&mut self, event: &FormatRuntime<'_>) -> Result<(), ()> {
        if let Some(callback) = event.dispatch_done {
            callback(FormattingDone {
                output_length: event.context.borrow().output_length,
            });
        }
        Ok(())
    }
    fn reject_request(&mut self, event: &FormatRuntime<'_>) -> Result<(), ()> {
        *event.request.borrow_mut().output_length = 0;
        event
            .context
            .borrow_mut()
            .error(ConditionerError::InvalidArgument);
        Ok(())
    }
    fn publish_failure(&mut self, event: &FormatRuntime<'_>) -> Result<(), ()> {
        if let Some(callback) = event.dispatch_error {
            callback(FormattingError {
                error: event.context.borrow().error,
            });
        }
        Ok(())
    }
    fn mark_unexpected(&mut self) -> Result<(), ()> {
        self.unexpected = true;
        self.error = ConditionerError::InvalidArgument;
        self.output_length = 0;
        Ok(())
    }
}

/// Synchronous single-writer formatter actor.
pub struct TextFormatterActor {
    machine: TextFormatterStateMachine<Context>,
}
impl Default for TextFormatterActor {
    fn default() -> Self {
        Self::new()
    }
}
impl TextFormatterActor {
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: TextFormatterStateMachine::new(Context::default()),
        }
    }
    pub fn process_event(
        &mut self,
        event: FormatRuntime<'_>,
    ) -> Result<FormattingDone, FormattingError> {
        event.context.borrow_mut().reset();
        *event.request.borrow_mut().output_length = 0;
        let context = event.context;
        let accepted = self
            .machine
            .process_event(TextFormatterEvents::FormatRuntime(event))
            .is_ok();
        let result = *context.borrow();
        if !accepted
            || self.machine.is(&TextFormatterStates::Unexpected)
            || result.error != ConditionerError::None
        {
            return Err(FormattingError {
                error: if result.error == ConditionerError::None {
                    ConditionerError::InvalidArgument
                } else {
                    result.error
                },
            });
        }
        Ok(FormattingDone {
            output_length: result.output_length,
        })
    }
    pub fn process_unexpected_event(&mut self) -> Result<(), FormattingError> {
        let _ = self.machine.context_mut().mark_unexpected();
        self.machine.set_state(TextFormatterStates::Unexpected);
        Err(FormattingError {
            error: ConditionerError::InvalidArgument,
        })
    }
    #[must_use]
    pub fn state(&self) -> &TextFormatterStates {
        self.machine.state()
    }
    #[must_use]
    pub fn is(&self, state: &TextFormatterStates) -> bool {
        self.machine.is(state)
    }
    #[must_use]
    pub fn context(&self) -> &Context {
        self.machine.context()
    }
}
pub type TextFormatter = TextFormatterActor;

#[cfg(test)]
mod tests {
    use super::super::context::Context;
    use super::super::events::{FormatRuntime, FormattingDone, FormattingError};
    use super::super::format::{ChatMessage, FormatRequest};
    use super::*;
    use crate::ConditionerError;
    use core::cell::RefCell;
    use core::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Mutex, MutexGuard};
    static TEST_LOCK: Mutex<()> = Mutex::new(());
    fn lock_tests() -> MutexGuard<'static, ()> {
        TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
    static DONE_CALLS: AtomicUsize = AtomicUsize::new(0);
    static ERROR_CALLS: AtomicUsize = AtomicUsize::new(0);
    static LAST_LENGTH: AtomicUsize = AtomicUsize::new(usize::MAX);
    fn done(outcome: FormattingDone) {
        DONE_CALLS.fetch_add(1, Ordering::SeqCst);
        LAST_LENGTH.store(outcome.output_length, Ordering::SeqCst);
    }
    fn error(outcome: FormattingError) {
        assert_eq!(outcome.error, ConditionerError::InvalidArgument);
        ERROR_CALLS.fetch_add(1, Ordering::SeqCst);
    }
    fn reset() {
        DONE_CALLS.store(0, Ordering::SeqCst);
        ERROR_CALLS.store(0, Ordering::SeqCst);
        LAST_LENGTH.store(usize::MAX, Ordering::SeqCst);
    }
    fn request<'a>(
        messages: &'a [ChatMessage<'a>],
        output: &'a mut [u8],
        length: &'a mut usize,
    ) -> RefCell<FormatRequest<'a>> {
        RefCell::new(FormatRequest {
            messages,
            add_generation_prompt: false,
            enable_thinking: false,
            output,
            output_length: length,
        })
    }
    #[test]
    fn formats_and_publishes_exact_length_synchronously() {
        let _lock = lock_tests();
        reset();
        let messages = [
            ChatMessage {
                role: b"user",
                content: b"hello",
            },
            ChatMessage {
                role: b"assistant",
                content: b" world",
            },
        ];
        let mut output = [0; 11];
        let mut length = 99;
        let request = request(&messages, &mut output, &mut length);
        let context = RefCell::new(Context::default());
        let mut actor = TextFormatterActor::new();
        assert_eq!(
            actor.process_event(FormatRuntime::new(
                &request,
                &context,
                Some(done),
                Some(error)
            )),
            Ok(FormattingDone { output_length: 11 })
        );
        assert_eq!(&output, b"hello world");
        assert_eq!(length, 11);
        assert_eq!(DONE_CALLS.load(Ordering::SeqCst), 1);
        assert_eq!(ERROR_CALLS.load(Ordering::SeqCst), 0);
        assert_eq!(LAST_LENGTH.load(Ordering::SeqCst), 11);
        assert!(actor.is(&TextFormatterStates::Done));
    }
    #[test]
    fn invalid_capacity_resets_length_and_preserves_output() {
        let _lock = lock_tests();
        reset();
        let messages = [ChatMessage {
            role: b"user",
            content: b"hello",
        }];
        let mut output = [0xa5; 4];
        let before = output;
        let mut length = 99;
        let request = request(&messages, &mut output, &mut length);
        let context = RefCell::new(Context::default());
        let mut actor = TextFormatterActor::new();
        assert_eq!(
            actor.process_event(FormatRuntime::new(
                &request,
                &context,
                Some(done),
                Some(error)
            )),
            Err(FormattingError {
                error: ConditionerError::InvalidArgument
            })
        );
        assert_eq!(output, before);
        assert_eq!(length, 0);
        assert_eq!(context.borrow().output_length, 0);
        assert_eq!(ERROR_CALLS.load(Ordering::SeqCst), 1);
        assert!(actor.is(&TextFormatterStates::Errored));
    }
    #[test]
    fn unexpected_event_is_typed_and_state_is_inspectable() {
        let mut actor = TextFormatterActor::new();
        assert_eq!(
            actor.process_unexpected_event(),
            Err(FormattingError {
                error: ConditionerError::InvalidArgument
            })
        );
        assert!(actor.is(&TextFormatterStates::Unexpected));
        assert!(actor.context().unexpected);
    }
}
