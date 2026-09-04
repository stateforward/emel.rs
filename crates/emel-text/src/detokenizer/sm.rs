//! Explicit generated state-machine owner for the stateless text detokenizer.
#![allow(
    clippy::cast_sign_loss,
    clippy::derive_partial_eq_without_eq,
    clippy::empty_structs_with_brackets,
    clippy::enum_variant_names,
    clippy::missing_const_for_fn,
    clippy::missing_errors_doc,
    clippy::module_name_repetitions,
    clippy::needless_pass_by_ref_mut,
    dead_code,
    missing_docs,
    private_interfaces,
    unused_imports
)]

use core::{cell::RefCell, fmt};
use sml::sml;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(i32)]
pub enum TokenType {
    Undefined = 0,
    Normal = 1,
    Unknown = 2,
    Control = 3,
    UserDefined = 4,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TokenEntry<'a> {
    pub piece: &'a [u8],
    pub token_type: TokenType,
}
pub trait VocabularyView {
    fn token_count(&self) -> u32;
    fn token(&self, index: u32) -> Option<TokenEntry<'_>>;
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BindingDone;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BindError {
    InvalidRequest,
    ModelInvalid,
    Backend,
    Internal,
}
pub type BindResult = Result<BindingDone, BindError>;
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Detokenized {
    pub output_length: usize,
    pub pending_length: usize,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DetokenizeError {
    InvalidRequest,
    ModelInvalid,
    Backend,
    Internal,
    Unexpected,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DetokenizeStatus {
    Done(Detokenized),
    Error {
        error: DetokenizeError,
        output_length: usize,
        pending_length: usize,
    },
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DetokenizeResult {
    pub status: DetokenizeStatus,
}
impl DetokenizeResult {
    pub const fn done(output_length: usize, pending_length: usize) -> Self {
        Self {
            status: DetokenizeStatus::Done(Detokenized {
                output_length,
                pending_length,
            }),
        }
    }
    pub const fn error(
        error: DetokenizeError,
        output_length: usize,
        pending_length: usize,
    ) -> Self {
        Self {
            status: DetokenizeStatus::Error {
                error,
                output_length,
                pending_length,
            },
        }
    }
    pub const fn output_length(self) -> usize {
        match self.status {
            DetokenizeStatus::Done(v) => v.output_length,
            DetokenizeStatus::Error { output_length, .. } => output_length,
        }
    }
    pub const fn pending_length(self) -> usize {
        match self.status {
            DetokenizeStatus::Done(v) => v.pending_length,
            DetokenizeStatus::Error { pending_length, .. } => pending_length,
        }
    }
    pub const fn error_kind(self) -> Option<DetokenizeError> {
        match self.status {
            DetokenizeStatus::Done(_) => None,
            DetokenizeStatus::Error { error, .. } => Some(error),
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UnexpectedEvent {
    Bind,
    Detokenize,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct EventBind;
#[derive(Clone, Copy, Debug, Default)]
pub struct EventDetokenize;
#[derive(Clone, Copy, Debug, Default)]
pub struct EventDone;
#[derive(Clone, Copy, Debug, Default)]
pub struct EventError;

#[derive(Clone, Copy, Debug)]
struct EventBindRuntime<'event> {
    result: &'event RefCell<BindResult>,
}
#[derive(Clone, Copy, Debug)]
struct EventDetokenizeRuntime<'event> {
    token: Option<TokenEntry<'event>>,
    emit_special: bool,
    pending: &'event RefCell<&'event mut [u8]>,
    pending_length: &'event RefCell<usize>,
    output: &'event RefCell<&'event mut [u8]>,
    result: &'event RefCell<DetokenizeResult>,
}

sml! {
    TextDetokenizer<'event> {
        "binding"_s <= *"uninitialized"_s + event<EventBindRuntime<'event>>(&'event EventBindRuntime<'event>) [valid_bind] / begin_bind,
        "binding_error_decision"_s <= "uninitialized"_s + event<EventBindRuntime<'event>>(&'event EventBindRuntime<'event>) [invalid_bind] / reject_bind,
        "binding"_s <= "idle"_s + event<EventBindRuntime<'event>>(&'event EventBindRuntime<'event>) [valid_bind] / begin_bind,
        "binding_error_decision"_s <= "idle"_s + event<EventBindRuntime<'event>>(&'event EventBindRuntime<'event>) [invalid_bind] / reject_bind,
        "binding"_s <= "done"_s + event<EventBindRuntime<'event>>(&'event EventBindRuntime<'event>) [valid_bind] / begin_bind,
        "binding_error_decision"_s <= "done"_s + event<EventBindRuntime<'event>>(&'event EventBindRuntime<'event>) [invalid_bind] / reject_bind,
        "binding"_s <= "errored"_s + event<EventBindRuntime<'event>>(&'event EventBindRuntime<'event>) [valid_bind] / begin_bind,
        "binding_error_decision"_s <= "errored"_s + event<EventBindRuntime<'event>>(&'event EventBindRuntime<'event>) [invalid_bind] / reject_bind,
        "binding"_s <= "unexpected"_s + event<EventBindRuntime<'event>>(&'event EventBindRuntime<'event>) [valid_bind] / begin_bind,
        "binding_error_decision"_s <= "unexpected"_s + event<EventBindRuntime<'event>>(&'event EventBindRuntime<'event>) [invalid_bind] / reject_bind,
        "binding_decision"_s <= "binding"_s + completion<EventBindRuntime>(&'event EventBindRuntime<'event>) / commit_bind,
        "binding_done_decision"_s <= "binding_decision"_s + completion<EventBindRuntime>(&'event EventBindRuntime<'event>) [bind_successful] / mark_bind_done,
        "idle"_s <= "binding_done_decision"_s + completion<EventBindRuntime>(&'event EventBindRuntime<'event>),
        "errored"_s <= "binding_error_decision"_s + completion<EventBindRuntime>(&'event EventBindRuntime<'event>) / mark_bind_error,

        "decoding"_s <= "idle"_s + event<EventDetokenizeRuntime<'event>>(&'event EventDetokenizeRuntime<'event>) [valid_detokenize] / begin_detokenize,
        "detokenize_error_decision"_s <= "uninitialized"_s + event<EventDetokenizeRuntime<'event>>(&'event EventDetokenizeRuntime<'event>) / reject_detokenize,
        "detokenize_error_decision"_s <= "idle"_s + event<EventDetokenizeRuntime<'event>>(&'event EventDetokenizeRuntime<'event>) [invalid_detokenize] / reject_detokenize,
        "decoding"_s <= "done"_s + event<EventDetokenizeRuntime<'event>>(&'event EventDetokenizeRuntime<'event>) [valid_detokenize] / begin_detokenize,
        "detokenize_error_decision"_s <= "done"_s + event<EventDetokenizeRuntime<'event>>(&'event EventDetokenizeRuntime<'event>) [invalid_detokenize] / reject_detokenize,
        "decoding"_s <= "errored"_s + event<EventDetokenizeRuntime<'event>>(&'event EventDetokenizeRuntime<'event>) [valid_detokenize] / begin_detokenize,
        "detokenize_error_decision"_s <= "errored"_s + event<EventDetokenizeRuntime<'event>>(&'event EventDetokenizeRuntime<'event>) [invalid_detokenize] / reject_detokenize,
        "decoding"_s <= "unexpected"_s + event<EventDetokenizeRuntime<'event>>(&'event EventDetokenizeRuntime<'event>) [valid_detokenize] / begin_detokenize,
        "detokenize_error_decision"_s <= "unexpected"_s + event<EventDetokenizeRuntime<'event>>(&'event EventDetokenizeRuntime<'event>) [invalid_detokenize] / reject_detokenize,
        "decode_token_validation"_s <= "decoding"_s + completion<EventDetokenizeRuntime>(&'event EventDetokenizeRuntime<'event>),
        "decode_piece_decision"_s <= "decode_token_validation"_s + completion<EventDetokenizeRuntime>(&'event EventDetokenizeRuntime<'event>) [token_in_vocab],
        "detokenize_error_decision"_s <= "decode_token_validation"_s + completion<EventDetokenizeRuntime>(&'event EventDetokenizeRuntime<'event>) [token_out_of_vocab] / mark_model_invalid,
        "detokenize_done_decision"_s <= "decode_piece_decision"_s + completion<EventDetokenizeRuntime>(&'event EventDetokenizeRuntime<'event>) [skip_special_piece] / mark_done,
        "decode_byte_capacity_decision"_s <= "decode_piece_decision"_s + completion<EventDetokenizeRuntime>(&'event EventDetokenizeRuntime<'event>) [byte_piece],
        "decode_text_pending_decision"_s <= "decode_piece_decision"_s + completion<EventDetokenizeRuntime>(&'event EventDetokenizeRuntime<'event>) [text_piece],
        "detokenize_error_decision"_s <= "decode_piece_decision"_s + completion<EventDetokenizeRuntime>(&'event EventDetokenizeRuntime<'event>) / mark_internal_error,
        "decode_byte_pending_decision"_s <= "decode_byte_capacity_decision"_s + completion<EventDetokenizeRuntime>(&'event EventDetokenizeRuntime<'event>) [pending_has_capacity] / append_byte,
        "detokenize_error_decision"_s <= "decode_byte_capacity_decision"_s + completion<EventDetokenizeRuntime>(&'event EventDetokenizeRuntime<'event>) [pending_no_capacity] / mark_invalid,
        "decode_byte_pending_write"_s <= "decode_byte_pending_decision"_s + completion<EventDetokenizeRuntime>(&'event EventDetokenizeRuntime<'event>) [pending_complete] / write_pending,
        "decode_decision"_s <= "decode_byte_pending_decision"_s + completion<EventDetokenizeRuntime>(&'event EventDetokenizeRuntime<'event>) [pending_empty_or_incomplete],
        "detokenize_error_decision"_s <= "decode_byte_pending_decision"_s + completion<EventDetokenizeRuntime>(&'event EventDetokenizeRuntime<'event>) [pending_invalid] / mark_invalid,
        "decode_byte_pending_decision"_s <= "decode_byte_pending_write"_s + completion<EventDetokenizeRuntime>(&'event EventDetokenizeRuntime<'event>),
        "decode_text_pending_write"_s <= "decode_text_pending_decision"_s + completion<EventDetokenizeRuntime>(&'event EventDetokenizeRuntime<'event>) [pending_complete] / write_pending,
        "decode_text_write"_s <= "decode_text_pending_decision"_s + completion<EventDetokenizeRuntime>(&'event EventDetokenizeRuntime<'event>) [pending_empty] / write_text,
        "detokenize_error_decision"_s <= "decode_text_pending_decision"_s + completion<EventDetokenizeRuntime>(&'event EventDetokenizeRuntime<'event>) [pending_invalid] / mark_invalid,
        "detokenize_error_decision"_s <= "decode_text_pending_decision"_s + completion<EventDetokenizeRuntime>(&'event EventDetokenizeRuntime<'event>) / mark_invalid,
        "decode_text_pending_decision"_s <= "decode_text_pending_write"_s + completion<EventDetokenizeRuntime>(&'event EventDetokenizeRuntime<'event>),
        "decode_decision"_s <= "decode_text_write"_s + completion<EventDetokenizeRuntime>(&'event EventDetokenizeRuntime<'event>),
        "detokenize_done_decision"_s <= "decode_decision"_s + completion<EventDetokenizeRuntime>(&'event EventDetokenizeRuntime<'event>) [no_error] / mark_done,
        "detokenize_error_decision"_s <= "decode_decision"_s + completion<EventDetokenizeRuntime>(&'event EventDetokenizeRuntime<'event>) [has_error],
        "done"_s <= "detokenize_done_decision"_s + completion<EventDetokenizeRuntime>(&'event EventDetokenizeRuntime<'event>),
        "errored"_s <= "detokenize_error_decision"_s + completion<EventDetokenizeRuntime>(&'event EventDetokenizeRuntime<'event>),

        "unexpected"_s <= "uninitialized"_s + unexpected_event<_>,
        "unexpected"_s <= "binding"_s + unexpected_event<_>,
        "unexpected"_s <= "binding_decision"_s + unexpected_event<_>,
        "unexpected"_s <= "binding_done_decision"_s + unexpected_event<_>,
        "unexpected"_s <= "binding_error_decision"_s + unexpected_event<_>,
        "unexpected"_s <= "idle"_s + unexpected_event<_>,
        "unexpected"_s <= "decoding"_s + unexpected_event<_>,
        "unexpected"_s <= "decode_token_validation"_s + unexpected_event<_>,
        "unexpected"_s <= "decode_piece_decision"_s + unexpected_event<_>,
        "unexpected"_s <= "decode_byte_capacity_decision"_s + unexpected_event<_>,
        "unexpected"_s <= "decode_byte_pending_decision"_s + unexpected_event<_>,
        "unexpected"_s <= "decode_byte_pending_write"_s + unexpected_event<_>,
        "unexpected"_s <= "decode_text_pending_decision"_s + unexpected_event<_>,
        "unexpected"_s <= "decode_text_pending_write"_s + unexpected_event<_>,
        "unexpected"_s <= "decode_text_write"_s + unexpected_event<_>,
        "unexpected"_s <= "decode_decision"_s + unexpected_event<_>,
        "unexpected"_s <= "detokenize_done_decision"_s + unexpected_event<_>,
        "unexpected"_s <= "detokenize_error_decision"_s + unexpected_event<_>,
        "unexpected"_s <= "done"_s + unexpected_event<_>,
        "unexpected"_s <= "errored"_s + unexpected_event<_>,
        "unexpected"_s <= "unexpected"_s + unexpected_event<_>,
        "unexpected"_s <= "uninitialized"_s + unexpected_event<EventBind>,
        "unexpected"_s <= "uninitialized"_s + unexpected_event<EventDetokenize>,
        "unexpected"_s <= "idle"_s + unexpected_event<EventBind>,
        "unexpected"_s <= "idle"_s + unexpected_event<EventDetokenize>,
        "unexpected"_s <= "done"_s + unexpected_event<EventBind>,
        "unexpected"_s <= "done"_s + unexpected_event<EventDetokenize>,
        "unexpected"_s <= "errored"_s + unexpected_event<EventBind>,
        "unexpected"_s <= "errored"_s + unexpected_event<EventDetokenize>,
        "unexpected"_s <= "unexpected"_s + unexpected_event<EventBind>,
        "unexpected"_s <= "unexpected"_s + unexpected_event<EventDetokenize>,
        "unexpected"_s <= "binding"_s + unexpected_event<EventBind>,
        "unexpected"_s <= "binding"_s + unexpected_event<EventDetokenize>,
        "unexpected"_s <= "binding_decision"_s + unexpected_event<EventBind>,
        "unexpected"_s <= "binding_decision"_s + unexpected_event<EventDetokenize>,
        "unexpected"_s <= "binding_done_decision"_s + unexpected_event<EventBind>,
        "unexpected"_s <= "binding_done_decision"_s + unexpected_event<EventDetokenize>,
        "unexpected"_s <= "binding_error_decision"_s + unexpected_event<EventBind>,
        "unexpected"_s <= "binding_error_decision"_s + unexpected_event<EventDetokenize>,
        "unexpected"_s <= "decoding"_s + unexpected_event<EventBind>,
        "unexpected"_s <= "decoding"_s + unexpected_event<EventDetokenize>,
        "unexpected"_s <= "decode_token_validation"_s + unexpected_event<EventBind>,
        "unexpected"_s <= "decode_token_validation"_s + unexpected_event<EventDetokenize>,
        "unexpected"_s <= "decode_piece_decision"_s + unexpected_event<EventBind>,
        "unexpected"_s <= "decode_piece_decision"_s + unexpected_event<EventDetokenize>,
        "unexpected"_s <= "decode_byte_capacity_decision"_s + unexpected_event<EventBind>,
        "unexpected"_s <= "decode_byte_capacity_decision"_s + unexpected_event<EventDetokenize>,
        "unexpected"_s <= "decode_byte_pending_decision"_s + unexpected_event<EventBind>,
        "unexpected"_s <= "decode_byte_pending_decision"_s + unexpected_event<EventDetokenize>,
        "unexpected"_s <= "decode_byte_pending_write"_s + unexpected_event<EventBind>,
        "unexpected"_s <= "decode_byte_pending_write"_s + unexpected_event<EventDetokenize>,
        "unexpected"_s <= "decode_text_pending_decision"_s + unexpected_event<EventBind>,
        "unexpected"_s <= "decode_text_pending_decision"_s + unexpected_event<EventDetokenize>,
        "unexpected"_s <= "decode_text_pending_write"_s + unexpected_event<EventBind>,
        "unexpected"_s <= "decode_text_pending_write"_s + unexpected_event<EventDetokenize>,
        "unexpected"_s <= "decode_text_write"_s + unexpected_event<EventBind>,
        "unexpected"_s <= "decode_text_write"_s + unexpected_event<EventDetokenize>,
        "unexpected"_s <= "decode_decision"_s + unexpected_event<EventBind>,
        "unexpected"_s <= "decode_decision"_s + unexpected_event<EventDetokenize>,
        "unexpected"_s <= "detokenize_done_decision"_s + unexpected_event<EventBind>,
        "unexpected"_s <= "detokenize_done_decision"_s + unexpected_event<EventDetokenize>,
        "unexpected"_s <= "detokenize_error_decision"_s + unexpected_event<EventBind>,
        "unexpected"_s <= "detokenize_error_decision"_s + unexpected_event<EventDetokenize>,
    }
}

#[derive(Debug)]
pub struct TextDetokenizerContext {
    bound: bool,
    error: Option<DetokenizeError>,
    result: DetokenizeResult,
    unexpected: bool,
}
impl Default for TextDetokenizerContext {
    fn default() -> Self {
        Self {
            bound: false,
            error: None,
            result: DetokenizeResult::done(0, 0),
            unexpected: false,
        }
    }
}
impl TextDetokenizerContext {
    fn reset(&mut self) {
        self.error = None;
        self.result = DetokenizeResult::done(0, 0);
        self.unexpected = false;
    }
}
impl TextDetokenizerStateMachineContext for TextDetokenizerContext {
    fn valid_bind(&self, _: &EventBindRuntime<'_>) -> Result<bool, ()> {
        Ok(true)
    }
    fn invalid_bind(&self, _: &EventBindRuntime<'_>) -> Result<bool, ()> {
        Ok(false)
    }
    fn begin_bind(&mut self, _: &EventBindRuntime<'_>) -> Result<(), ()> {
        self.reset();
        Ok(())
    }
    fn commit_bind(&mut self, _: &EventBindRuntime<'_>) -> Result<(), ()> {
        self.bound = true;
        Ok(())
    }
    fn bind_successful(&self, _: &EventBindRuntime<'_>) -> Result<bool, ()> {
        Ok(self.bound)
    }
    fn mark_bind_done(&mut self, e: &EventBindRuntime<'_>) -> Result<(), ()> {
        *e.result.borrow_mut() = Ok(BindingDone);
        Ok(())
    }
    fn reject_bind(&mut self, e: &EventBindRuntime<'_>) -> Result<(), ()> {
        self.bound = false;
        *e.result.borrow_mut() = Err(BindError::InvalidRequest);
        Ok(())
    }
    fn mark_bind_error(&mut self, e: &EventBindRuntime<'_>) -> Result<(), ()> {
        *e.result.borrow_mut() = Err(BindError::InvalidRequest);
        Ok(())
    }
    fn valid_detokenize(&self, e: &EventDetokenizeRuntime<'_>) -> Result<bool, ()> {
        let p = e.pending.borrow();
        Ok(self.bound && *e.pending_length.borrow() <= p.len() && p.len() == 4)
    }
    fn invalid_detokenize(&self, e: &EventDetokenizeRuntime<'_>) -> Result<bool, ()> {
        let p = e.pending.borrow();
        Ok(!self.bound || *e.pending_length.borrow() > p.len() || p.len() != 4)
    }
    fn reject_detokenize(&mut self, e: &EventDetokenizeRuntime<'_>) -> Result<(), ()> {
        *e.result.borrow_mut() = DetokenizeResult::error(
            DetokenizeError::InvalidRequest,
            0,
            *e.pending_length.borrow(),
        );
        self.error = Some(DetokenizeError::InvalidRequest);
        Ok(())
    }
    fn begin_detokenize(&mut self, e: &EventDetokenizeRuntime<'_>) -> Result<(), ()> {
        self.reset();
        *e.result.borrow_mut() = DetokenizeResult::done(0, *e.pending_length.borrow());
        Ok(())
    }
    fn token_in_vocab(&self, e: &EventDetokenizeRuntime<'_>) -> Result<bool, ()> {
        Ok(e.token.is_some())
    }
    fn token_out_of_vocab(&self, e: &EventDetokenizeRuntime<'_>) -> Result<bool, ()> {
        Ok(e.token.is_none())
    }
    fn skip_special_piece(&self, e: &EventDetokenizeRuntime<'_>) -> Result<bool, ()> {
        Ok(e.token.is_some()
            && !e.emit_special
            && matches!(
                e.token.map(|t| t.token_type),
                Some(TokenType::Unknown | TokenType::Control | TokenType::UserDefined)
            ))
    }
    fn byte_piece(&self, e: &EventDetokenizeRuntime<'_>) -> Result<bool, ()> {
        let Some(t) = e.token else { return Ok(false) };
        let mut value = 0;
        Ok(parse_byte_piece(t.piece, &mut value))
    }
    fn text_piece(&self, e: &EventDetokenizeRuntime<'_>) -> Result<bool, ()> {
        let Some(t) = e.token else { return Ok(false) };
        if !e.emit_special
            && matches!(
                t.token_type,
                TokenType::Unknown | TokenType::Control | TokenType::UserDefined
            )
        {
            return Ok(false);
        }
        let mut value = 0;
        Ok(!parse_byte_piece(t.piece, &mut value))
    }
    fn pending_has_capacity(&self, e: &EventDetokenizeRuntime<'_>) -> Result<bool, ()> {
        Ok(*e.pending_length.borrow() < e.pending.borrow().len())
    }
    fn pending_no_capacity(&self, e: &EventDetokenizeRuntime<'_>) -> Result<bool, ()> {
        let n = *e.pending_length.borrow();
        let len = e.pending.borrow().len();
        Ok(n >= len)
    }
    fn append_byte(&mut self, e: &EventDetokenizeRuntime<'_>) -> Result<(), ()> {
        let Some(t) = e.token else { return Err(()) };
        let mut value = 0;
        if !parse_byte_piece(t.piece, &mut value) {
            return Err(());
        }
        let n = *e.pending_length.borrow();
        e.pending.borrow_mut()[n] = value;
        *e.pending_length.borrow_mut() = n + 1;
        self.result = DetokenizeResult::done(0, n + 1);
        Ok(())
    }
    fn pending_complete(&self, e: &EventDetokenizeRuntime<'_>) -> Result<bool, ()> {
        let p = e.pending.borrow();
        let n = *e.pending_length.borrow();
        let needed = if n == 0 { 0 } else { sequence_length(p[0]) };
        Ok(n != 0 && needed != 0 && n >= needed && continuations_valid(&p, needed))
    }
    fn pending_empty_or_incomplete(&self, e: &EventDetokenizeRuntime<'_>) -> Result<bool, ()> {
        let p = e.pending.borrow();
        let n = *e.pending_length.borrow();
        let needed = if n == 0 { 0 } else { sequence_length(p[0]) };
        Ok(n == 0 || (needed != 0 && n < needed))
    }
    fn pending_empty(&self, e: &EventDetokenizeRuntime<'_>) -> Result<bool, ()> {
        Ok(*e.pending_length.borrow() == 0)
    }
    fn pending_invalid(&self, e: &EventDetokenizeRuntime<'_>) -> Result<bool, ()> {
        let p = e.pending.borrow();
        let n = *e.pending_length.borrow();
        let needed = if n == 0 { 0 } else { sequence_length(p[0]) };
        Ok(n != 0 && (needed == 0 || (n >= needed && !continuations_valid(&p, needed))))
    }
    fn write_pending(&mut self, e: &EventDetokenizeRuntime<'_>) -> Result<(), ()> {
        let mut p = e.pending.borrow_mut();
        let n = *e.pending_length.borrow();
        let needed = sequence_length(p[0]);
        if needed > e.output.borrow().len() {
            self.error = Some(DetokenizeError::InvalidRequest);
            *e.result.borrow_mut() = DetokenizeResult::error(DetokenizeError::InvalidRequest, 0, n);
            return Ok(());
        }
        e.output.borrow_mut()[..needed].copy_from_slice(&p[..needed]);
        let remain = n - needed;
        p.copy_within(needed..n, 0);
        *e.pending_length.borrow_mut() = remain;
        self.result = DetokenizeResult::done(needed, remain);
        Ok(())
    }
    fn write_text(&mut self, e: &EventDetokenizeRuntime<'_>) -> Result<(), ()> {
        let Some(t) = e.token else { return Err(()) };
        if t.piece.len() > e.output.borrow().len() {
            self.error = Some(DetokenizeError::InvalidRequest);
            *e.result.borrow_mut() = DetokenizeResult::error(
                DetokenizeError::InvalidRequest,
                0,
                *e.pending_length.borrow(),
            );
            return Ok(());
        }
        e.output.borrow_mut()[..t.piece.len()].copy_from_slice(t.piece);
        self.result = DetokenizeResult::done(t.piece.len(), 0);
        Ok(())
    }
    fn mark_done(&mut self, e: &EventDetokenizeRuntime<'_>) -> Result<(), ()> {
        *e.result.borrow_mut() = self.result;
        Ok(())
    }
    fn mark_model_invalid(&mut self, e: &EventDetokenizeRuntime<'_>) -> Result<(), ()> {
        self.error = Some(DetokenizeError::ModelInvalid);
        *e.result.borrow_mut() =
            DetokenizeResult::error(DetokenizeError::ModelInvalid, 0, *e.pending_length.borrow());
        Ok(())
    }
    fn mark_invalid(&mut self, e: &EventDetokenizeRuntime<'_>) -> Result<(), ()> {
        self.error = Some(DetokenizeError::InvalidRequest);
        *e.result.borrow_mut() = DetokenizeResult::error(
            DetokenizeError::InvalidRequest,
            0,
            *e.pending_length.borrow(),
        );
        Ok(())
    }
    fn mark_internal_error(&mut self, e: &EventDetokenizeRuntime<'_>) -> Result<(), ()> {
        self.error = Some(DetokenizeError::Internal);
        *e.result.borrow_mut() =
            DetokenizeResult::error(DetokenizeError::Internal, 0, *e.pending_length.borrow());
        Ok(())
    }
    fn no_error(&self, _: &EventDetokenizeRuntime<'_>) -> Result<bool, ()> {
        Ok(self.error.is_none())
    }
    fn has_error(&self, _: &EventDetokenizeRuntime<'_>) -> Result<bool, ()> {
        Ok(self.error.is_some())
    }
}

pub struct TextDetokenizer<'v, V: VocabularyView + ?Sized> {
    machine: TextDetokenizerStateMachine<TextDetokenizerContext>,
    vocabulary: &'v V,
}
impl<V: VocabularyView + ?Sized> fmt::Debug for TextDetokenizer<'_, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TextDetokenizer").finish_non_exhaustive()
    }
}
impl<'v, V: VocabularyView + ?Sized> TextDetokenizer<'v, V> {
    pub fn new(vocabulary: &'v V) -> Self {
        Self {
            machine: TextDetokenizerStateMachine::new(TextDetokenizerContext::default()),
            vocabulary,
        }
    }
    pub fn bind(&mut self, request: super::Bind<'_>) -> BindResult {
        let sink = RefCell::new(Err(BindError::Internal));
        let event = EventBindRuntime { result: &sink };
        let accepted = self
            .machine
            .process_event(TextDetokenizerEvents::EventBindRuntime(&event))
            .is_ok();
        let result = if accepted {
            *sink.borrow()
        } else {
            Err(BindError::Backend)
        };
        *request.result = result;
        result
    }
    pub fn detokenize(&mut self, request: super::Detokenize<'_>) -> DetokenizeResult {
        let result = RefCell::new(DetokenizeResult::error(
            DetokenizeError::Internal,
            0,
            request.pending_length,
        ));
        let pending_length = RefCell::new(request.pending_length);
        let pending = RefCell::new(request.pending);
        let output = RefCell::new(request.output);
        let token = u32::try_from(request.token_id)
            .ok()
            .and_then(|id| self.vocabulary.token(id));
        let event = EventDetokenizeRuntime {
            token,
            emit_special: request.emit_special,
            pending: &pending,
            pending_length: &pending_length,
            output: &output,
            result: &result,
        };
        match self
            .machine
            .process_event(TextDetokenizerEvents::EventDetokenizeRuntime(&event))
        {
            Ok(_) => {}
            Err(TextDetokenizerError::ActionFailed(())) => {
                *result.borrow_mut() =
                    DetokenizeResult::error(DetokenizeError::Backend, 0, *pending_length.borrow());
                self.machine.set_state(TextDetokenizerStates::Errored);
            }
            Err(_) => {
                *result.borrow_mut() = DetokenizeResult::error(
                    DetokenizeError::Unexpected,
                    0,
                    *pending_length.borrow(),
                );
                self.machine.set_state(TextDetokenizerStates::Unexpected);
            }
        }
        let out = *result.borrow();
        *request.result = out;
        out
    }
    pub fn state(&self) -> &'static str {
        match self.machine.state() {
            TextDetokenizerStates::Uninitialized => "uninitialized",
            TextDetokenizerStates::Idle => "idle",
            TextDetokenizerStates::Binding => "binding",
            TextDetokenizerStates::BindingDecision => "binding_decision",
            TextDetokenizerStates::BindingDoneDecision => "binding_done_decision",
            TextDetokenizerStates::BindingErrorDecision => "binding_error_decision",
            TextDetokenizerStates::Decoding => "decoding",
            TextDetokenizerStates::DecodeTokenValidation => "decode_token_validation",
            TextDetokenizerStates::DecodePieceDecision => "decode_piece_decision",
            TextDetokenizerStates::DecodeByteCapacityDecision => "decode_byte_capacity_decision",
            TextDetokenizerStates::DecodeBytePendingDecision => "decode_byte_pending_decision",
            TextDetokenizerStates::DecodeBytePendingWrite => "decode_byte_pending_write",
            TextDetokenizerStates::DecodeTextPendingDecision => "decode_text_pending_decision",
            TextDetokenizerStates::DecodeTextPendingWrite => "decode_text_pending_write",
            TextDetokenizerStates::DecodeTextWrite => "decode_text_write",
            TextDetokenizerStates::DecodeDecision => "decode_decision",
            TextDetokenizerStates::DetokenizeDoneDecision => "detokenize_done_decision",
            TextDetokenizerStates::DetokenizeErrorDecision => "detokenize_error_decision",
            TextDetokenizerStates::Done => "done",
            TextDetokenizerStates::Errored => "errored",
            TextDetokenizerStates::Unexpected => "unexpected",
        }
    }
    pub fn unexpected(&mut self, _: UnexpectedEvent) {
        self.machine.set_state(TextDetokenizerStates::Unexpected);
    }
}
fn parse_byte_piece(piece: &[u8], value: &mut u8) -> bool {
    if piece.len() != 6
        || piece[0] != b'<'
        || piece[1] != b'0'
        || piece[2] != b'x'
        || piece[5] != b'>'
    {
        return false;
    }
    let Some(h) = hex(piece[3]) else { return false };
    let Some(l) = hex(piece[4]) else { return false };
    *value = h << 4 | l;
    true
}
const fn hex(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}
const fn sequence_length(lead: u8) -> usize {
    if lead & 0x80 == 0 {
        1
    } else if lead & 0xe0 == 0xc0 {
        2
    } else if lead & 0xf0 == 0xe0 {
        3
    } else if lead & 0xf8 == 0xf0 {
        4
    } else {
        0
    }
}
fn continuations_valid(bytes: &[u8], needed: usize) -> bool {
    bytes
        .get(1..needed)
        .is_some_and(|tail| tail.iter().all(|b| b & 0xc0 == 0x80))
}
