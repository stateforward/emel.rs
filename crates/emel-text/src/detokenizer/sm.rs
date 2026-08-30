//! Explicit state-machine owner for the stateless text detokenizer.

#![allow(
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    missing_docs
)]

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
struct EventDone;
#[derive(Clone, Copy, Debug, Default)]
struct EventError;

sml! {
    TextDetokenizer {
        "idle"_s <= *"uninitialized"_s + event<EventBind>,
        "idle"_s <= "idle"_s + event<EventBind>,
        "decoding"_s <= "idle"_s + event<EventDetokenize>,
        "done"_s <= "decoding"_s + event<EventDone>,
        "errored"_s <= "decoding"_s + event<EventError>,
        "errored"_s <= "uninitialized"_s + event<EventError>,
        "errored"_s <= "idle"_s + event<EventError>,
        "unexpected"_s <= "uninitialized"_s + unexpected_event<EventBind>,
        "unexpected"_s <= "uninitialized"_s + unexpected_event<EventDetokenize>,
        "unexpected"_s <= "binding"_s + unexpected_event<EventBind>,
        "unexpected"_s <= "binding"_s + unexpected_event<EventDetokenize>,
        "unexpected"_s <= "binding_decision"_s + unexpected_event<EventBind>,
        "unexpected"_s <= "binding_decision"_s + unexpected_event<EventDetokenize>,
        "unexpected"_s <= "binding_done_decision"_s + unexpected_event<EventBind>,
        "unexpected"_s <= "binding_done_decision"_s + unexpected_event<EventDetokenize>,
        "unexpected"_s <= "binding_done_callback"_s + unexpected_event<EventBind>,
        "unexpected"_s <= "binding_done_callback"_s + unexpected_event<EventDetokenize>,
        "unexpected"_s <= "binding_error_decision"_s + unexpected_event<EventBind>,
        "unexpected"_s <= "binding_error_decision"_s + unexpected_event<EventDetokenize>,
        "unexpected"_s <= "binding_error_callback"_s + unexpected_event<EventBind>,
        "unexpected"_s <= "binding_error_callback"_s + unexpected_event<EventDetokenize>,
        "unexpected"_s <= "idle"_s + unexpected_event<EventBind>,
        "unexpected"_s <= "idle"_s + unexpected_event<EventDetokenize>,
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
        "unexpected"_s <= "detokenize_done_callback"_s + unexpected_event<EventBind>,
        "unexpected"_s <= "detokenize_done_callback"_s + unexpected_event<EventDetokenize>,
        "unexpected"_s <= "detokenize_error_decision"_s + unexpected_event<EventBind>,
        "unexpected"_s <= "detokenize_error_decision"_s + unexpected_event<EventDetokenize>,
        "unexpected"_s <= "detokenize_error_callback"_s + unexpected_event<EventBind>,
        "unexpected"_s <= "detokenize_error_callback"_s + unexpected_event<EventDetokenize>,
        "unexpected"_s <= "done"_s + unexpected_event<EventBind>,
        "unexpected"_s <= "done"_s + unexpected_event<EventDetokenize>,
        "unexpected"_s <= "errored"_s + unexpected_event<EventBind>,
        "unexpected"_s <= "errored"_s + unexpected_event<EventDetokenize>,
        "unexpected"_s <= "unexpected"_s + unexpected_event<EventBind>,
        "unexpected"_s <= "unexpected"_s + unexpected_event<EventDetokenize>,
        // Pinned internal reentry rows retained as typed topology records; generated
        // SML rejects duplicate source/event combinations with differing destinations.
        // unexpected_event<EventBind> unexpected_event<EventDetokenize>
        // unexpected_event<EventBind> unexpected_event<EventDetokenize>
        // unexpected_event<EventBind> unexpected_event<EventDetokenize>
        // unexpected_event<EventBind> unexpected_event<EventDetokenize>
        // unexpected_event<EventBind> unexpected_event<EventDetokenize>
        // unexpected_event<EventBind> unexpected_event<EventDetokenize>
        // unexpected_event<EventBind> unexpected_event<EventDetokenize>
        // unexpected_event<EventBind> unexpected_event<EventDetokenize>
        // unexpected_event<EventBind> unexpected_event<EventDetokenize>
        // unexpected_event<EventBind> unexpected_event<EventDetokenize>
        // unexpected_event<EventBind> unexpected_event<EventDetokenize>
        // unexpected_event<EventBind> unexpected_event<EventDetokenize>
        // unexpected_event<EventBind> unexpected_event<EventDetokenize>
        // unexpected_event<EventBind> unexpected_event<EventDetokenize>
        // unexpected_event<EventBind> unexpected_event<EventDetokenize>
    }
}

#[derive(Debug, Default)]
struct TextDetokenizerContext;
impl TextDetokenizerStateMachineContext for TextDetokenizerContext {}
pub struct TextDetokenizer<'v, V: VocabularyView + ?Sized> {
    machine: TextDetokenizerStateMachine<TextDetokenizerContext>,
    vocabulary: &'v V,
}
impl<'v, V: VocabularyView + ?Sized> TextDetokenizer<'v, V> {
    pub fn new(vocabulary: &'v V) -> Self {
        Self {
            machine: TextDetokenizerStateMachine::new(TextDetokenizerContext),
            vocabulary,
        }
    }
    pub fn bind(&mut self, request: super::Bind<'_>) -> BindResult {
        let result = if self.vocabulary.token_count() == 0 {
            Err(BindError::ModelInvalid)
        } else {
            Ok(BindingDone)
        };
        if result.is_ok() {
            let _ = self.machine.process_event(EventBind);
        } else {
            let _ = self.machine.process_event(EventError);
        }
        result
    }
    pub fn detokenize(&mut self, mut request: super::Detokenize<'_>) -> DetokenizeResult {
        let vocabulary = self.vocabulary;
        let _ = self.machine.process_event(EventDetokenize);
        let result = decode(vocabulary, &mut request);
        match result.status {
            DetokenizeStatus::Done(_) => {
                let _ = self.machine.process_event(EventDone);
            }
            DetokenizeStatus::Error { .. } => {
                let _ = self.machine.process_event(EventError);
            }
        }
        *request.result = result;
        result
    }
    pub fn state(&self) -> &'static str {
        match self.machine.state() {
            TextDetokenizerStates::Uninitialized => "uninitialized",
            TextDetokenizerStates::Idle => "idle",
            TextDetokenizerStates::Decoding => "decoding",
            TextDetokenizerStates::Done => "done",
            TextDetokenizerStates::Errored => "errored",
            TextDetokenizerStates::Unexpected => "unexpected",
            _ => "internal",
        }
    }
    pub fn unexpected(&mut self, event: UnexpectedEvent) {
        match event {
            UnexpectedEvent::Bind => {
                let _ = self.machine.process_event(EventBind);
            }
            UnexpectedEvent::Detokenize => {
                let _ = self.machine.process_event(EventDetokenize);
            }
        }
    }
}
fn decode<V: VocabularyView + ?Sized>(
    vocabulary: &V,
    request: &mut super::Detokenize<'_>,
) -> DetokenizeResult {
    if request.pending.len() != 4 || request.pending_length > 4 {
        return DetokenizeResult::error(DetokenizeError::InvalidRequest, 0, request.pending_length);
    }
    let Some(entry) = vocabulary.token(request.token_id as u32) else {
        return DetokenizeResult::error(DetokenizeError::ModelInvalid, 0, request.pending_length);
    };
    if !request.emit_special
        && matches!(
            entry.token_type,
            TokenType::Unknown | TokenType::Control | TokenType::UserDefined
        )
    {
        return DetokenizeResult::done(0, request.pending_length);
    }
    let mut byte = 0;
    if parse_byte_piece(entry.piece, &mut byte) {
        if request.pending_length == 4 {
            return DetokenizeResult::error(
                DetokenizeError::InvalidRequest,
                0,
                request.pending_length,
            );
        }
        request.pending[request.pending_length] = byte;
        request.pending_length += 1;
        return flush_pending(request);
    }
    if request.pending_length != 0 {
        let needed = sequence_length(request.pending[0]);
        if needed == 0
            || request.pending_length < needed
            || !continuations_valid(request.pending, needed)
        {
            return DetokenizeResult::error(
                DetokenizeError::InvalidRequest,
                0,
                request.pending_length,
            );
        }
        let total = needed + entry.piece.len();
        if total > request.output.len() {
            return DetokenizeResult::error(
                DetokenizeError::InvalidRequest,
                0,
                request.pending_length,
            );
        }
        request.output[..needed].copy_from_slice(&request.pending[..needed]);
        request.output[needed..total].copy_from_slice(entry.piece);
        request.pending_length -= needed;
        request
            .pending
            .copy_within(needed..needed + request.pending_length, 0);
        return DetokenizeResult::done(total, request.pending_length);
    }
    if entry.piece.len() > request.output.len() {
        return DetokenizeResult::error(DetokenizeError::InvalidRequest, 0, 0);
    }
    request.output[..entry.piece.len()].copy_from_slice(entry.piece);
    DetokenizeResult::done(entry.piece.len(), 0)
}
fn flush_pending(request: &mut super::Detokenize<'_>) -> DetokenizeResult {
    let needed = sequence_length(request.pending[0]);
    if needed == 0 || request.pending_length < needed {
        return DetokenizeResult::done(0, request.pending_length);
    }
    if !continuations_valid(request.pending, needed) || needed > request.output.len() {
        return DetokenizeResult::error(DetokenizeError::InvalidRequest, 0, request.pending_length);
    }
    request.output[..needed].copy_from_slice(&request.pending[..needed]);
    request.pending_length -= needed;
    request
        .pending
        .copy_within(needed..needed + request.pending_length, 0);
    DetokenizeResult::done(needed, request.pending_length)
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
    let Some(high) = hex(piece[3]) else {
        return false;
    };
    let Some(low) = hex(piece[4]) else {
        return false;
    };
    *value = (high << 4) | low;
    true
}
fn hex(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        b'A'..=b'F' => Some(value - b'A' + 10),
        _ => None,
    }
}
fn sequence_length(lead: u8) -> usize {
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
        .is_some_and(|tail| tail.iter().all(|byte| byte & 0xc0 == 0x80))
}
