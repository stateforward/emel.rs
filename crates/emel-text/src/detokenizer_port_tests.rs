//! Detokenizer topology and behavior coverage.
use super::detokenizer::{
    Bind, BindResult, Detokenize, DetokenizeError, DetokenizeResult, TextDetokenizer, TokenEntry,
    TokenType, VocabularyView,
};

#[test]
fn detokenizer_keeps_typed_unexpected_events() {
    let src = include_str!("detokenizer/sm.rs");
    let n = src.matches("unexpected_event").count();
    assert!(n >= 60, "expected ~65 typed unexpected handlers, got {n}");
    assert!(src.contains("unexpected_event<EventBind>"));
}
struct EmptyVocabulary;
impl VocabularyView for EmptyVocabulary {
    fn token_count(&self) -> u32 {
        0
    }
    fn token(&self, _: u32) -> Option<TokenEntry<'_>> {
        None
    }
}

struct Vocabulary {
    entries: [&'static [u8]; 7],
    types: [TokenType; 7],
}
impl VocabularyView for Vocabulary {
    fn token_count(&self) -> u32 {
        u32::try_from(self.entries.len()).expect("fixed vocabulary fits u32")
    }
    fn token(&self, index: u32) -> Option<TokenEntry<'_>> {
        self.entries.get(index as usize).map(|piece| TokenEntry {
            piece,
            token_type: self.types[index as usize],
        })
    }
}
fn vocabulary() -> Vocabulary {
    Vocabulary {
        entries: [
            b"hello", b"<|eot|>", b"<0xE2>", b"<0x82>", b"<0xAC>", b"ok", b"bad",
        ],
        types: [
            TokenType::Normal,
            TokenType::Control,
            TokenType::Normal,
            TokenType::Normal,
            TokenType::Normal,
            TokenType::Normal,
            TokenType::Normal,
        ],
    }
}
fn bind<V: VocabularyView + ?Sized>(actor: &mut TextDetokenizer<'_, V>) -> BindResult {
    actor.bind(Bind {
        result: &mut Err(super::detokenizer::BindError::Internal),
    })
}
fn decode<V: VocabularyView + ?Sized>(
    actor: &mut TextDetokenizer<'_, V>,
    id: i32,
    special: bool,
    pending: &mut [u8],
    pending_length: usize,
    output: &mut [u8],
) -> DetokenizeResult {
    let mut result = DetokenizeResult::error(DetokenizeError::Internal, 0, pending_length);
    let out = actor.detokenize(Detokenize {
        token_id: id,
        emit_special: special,
        pending,
        pending_length,
        output,
        result: &mut result,
    });
    assert_eq!(out, result);
    out
}
#[test]
fn detokenizer_binds_empty_vocabulary_but_rejects_token() {
    let vocab = EmptyVocabulary;
    let mut actor = TextDetokenizer::new(&vocab);
    assert!(bind(&mut actor).is_ok());
    let mut pending = [0; 4];
    let mut output = [0; 4];
    assert_eq!(
        decode(&mut actor, 0, false, &mut pending, 0, &mut output).error_kind(),
        Some(DetokenizeError::ModelInvalid)
    );
}
#[test]
fn detokenizer_decodes_special_text_and_model_range() {
    let vocab = vocabulary();
    let mut actor = TextDetokenizer::new(&vocab);
    assert!(bind(&mut actor).is_ok());
    let mut pending = [0; 4];
    let mut output = [0; 8];
    assert_eq!(
        decode(&mut actor, 0, false, &mut pending, 0, &mut output).output_length(),
        5
    );
    assert_eq!(&output[..5], b"hello");
    assert_eq!(
        decode(&mut actor, 1, false, &mut pending, 0, &mut output).output_length(),
        0
    );
    assert_eq!(
        decode(&mut actor, 99, false, &mut pending, 0, &mut output).error_kind(),
        Some(DetokenizeError::ModelInvalid)
    );
}
#[test]
fn detokenizer_handles_byte_fallback_utf8_and_capacity() {
    let vocab = vocabulary();
    let mut actor = TextDetokenizer::new(&vocab);
    bind(&mut actor).unwrap();
    let mut pending = [0; 4];
    let mut output = [0xAA; 8];
    assert_eq!(
        decode(&mut actor, 2, false, &mut pending, 0, &mut output).pending_length(),
        1
    );
    assert_eq!(
        decode(&mut actor, 3, false, &mut pending, 1, &mut output).pending_length(),
        2
    );
    assert_eq!(
        decode(&mut actor, 4, false, &mut pending, 2, &mut output).output_length(),
        3
    );
    assert_eq!(&output[..3], "€".as_bytes());
    let before = output;
    assert_eq!(
        decode(&mut actor, 0, false, &mut pending, 0, &mut output[..2]).error_kind(),
        Some(DetokenizeError::InvalidRequest)
    );
    assert_eq!(output, before);
}
#[test]
fn detokenizer_rejects_unbound_invalid_pending_and_recovers_unexpected() {
    let vocab = vocabulary();
    let mut actor = TextDetokenizer::new(&vocab);
    let mut pending = [0; 4];
    let mut output = [0; 4];
    assert_eq!(
        decode(&mut actor, 0, false, &mut pending, 0, &mut output).error_kind(),
        Some(DetokenizeError::InvalidRequest)
    );
    bind(&mut actor).unwrap();
    assert_eq!(
        decode(&mut actor, 0, false, &mut pending[..3], 0, &mut output).error_kind(),
        Some(DetokenizeError::InvalidRequest)
    );
    let before = output;
    let result = decode(&mut actor, 0, false, &mut pending, 5, &mut output);
    assert_eq!(result.error_kind(), Some(DetokenizeError::InvalidRequest));
    assert_eq!(result.pending_length(), 5);
    assert_eq!(output, before);
    actor.unexpected(super::detokenizer::UnexpectedEvent::Detokenize);
    assert_eq!(actor.state(), "unexpected");
}
