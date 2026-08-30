//! Stateless, allocation-free token-to-text decoding.

mod sm;

pub use sm::{
    BindError, BindResult, BindingDone, DetokenizeError, DetokenizeResult, DetokenizeStatus,
    Detokenized, TextDetokenizer, TokenEntry, TokenType, UnexpectedEvent, VocabularyView,
};

/// Synchronous bind result publication request for a pre-bound actor.
#[derive(Debug)]
pub struct Bind<'a> {
    pub result: &'a mut BindResult,
}
/// Borrowed one-token decode request. All storage is supplied by the caller.
pub struct Detokenize<'a> {
    pub token_id: i32,
    pub emit_special: bool,
    pub pending: &'a mut [u8],
    pub pending_length: usize,
    pub output: &'a mut [u8],
    pub result: &'a mut DetokenizeResult,
}
