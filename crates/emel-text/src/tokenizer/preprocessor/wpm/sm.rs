//! Source-aligned bounded WPM tokenizer preprocessor actor.
use core::cell::RefCell;

pub const MAX_FRAGMENTS: usize = 1024;
pub const MAX_SPECIAL_TOKENS: usize = 1024;
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum FragmentKind { #[default] RawText = 0, Token = 1 }
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Fragment<'a> { pub kind: FragmentKind, pub text: &'a str, pub token: i32 }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VocabularyToken<'a> { pub text: &'a str, pub token: i32, pub token_type: i32, pub lstrip: bool, pub rstrip: bool }
pub trait VocabularyView { fn token_count(&self) -> usize; fn token(&self, index: usize) -> Option<VocabularyToken<'_>>; }
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum PreprocessError { #[default] None = 0, InvalidRequest = 1, BackendError = 2 }
impl PreprocessError { #[must_use] pub const fn code(self) -> i32 { match self { Self::None => 0, Self::InvalidRequest => 1, Self::BackendError => 2 } } }
#[derive(Clone, Copy, Debug, Eq, PartialEq)] pub struct PreprocessDone { pub fragment_count: usize }
#[derive(Clone, Copy, Debug, Eq, PartialEq)] pub struct PreprocessErrorEvent { pub err: PreprocessError }
pub type DoneCallback = fn(PreprocessDone) -> bool;
pub type ErrorCallback = fn(PreprocessErrorEvent) -> bool;
#[derive(Clone, Copy, Debug)]
pub struct PreprocessRequest<'a> { pub vocab: &'a dyn VocabularyView, pub text: &'a str, pub parse_special: bool, pub fragments_out: Option<&'a RefCell<&'a mut [Fragment<'a>]>>, pub fragment_count_out: &'a RefCell<usize>, pub preprocessed_out: Option<&'a RefCell<bool>>, pub error_out: &'a RefCell<i32>, pub dispatch_done: Option<DoneCallback>, pub dispatch_error: Option<ErrorCallback> }
impl<'a> PreprocessRequest<'a> { #[must_use] pub const fn with_callbacks(vocab: &'a dyn VocabularyView, text: &'a str, parse_special: bool, fragments_out: Option<&'a RefCell<&'a mut [Fragment<'a>]>>, fragment_count_out: &'a RefCell<usize>, preprocessed_out: Option<&'a RefCell<bool>>, error_out: &'a RefCell<i32>, dispatch_done: Option<DoneCallback>, dispatch_error: Option<ErrorCallback>) -> Self { Self { vocab, text, parse_special, fragments_out, fragment_count_out, preprocessed_out, error_out, dispatch_done, dispatch_error } } }
#[derive(Clone, Copy, Debug)] pub struct EventPreprocessRuntime<'a> { pub request: PreprocessRequest<'a>, pub context: &'a RefCell<PreprocessContext> }
impl<'a> EventPreprocessRuntime<'a> { #[must_use] pub const fn new(request: PreprocessRequest<'a>, context: &'a RefCell<PreprocessContext>) -> Self { Self { request, context } } }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreprocessContext { special: [usize; MAX_SPECIAL_TOKENS], special_count: usize, pub fragment_count: usize, pub preprocessed: bool, pub phase_error: PreprocessError, pub err: PreprocessError, pub result: bool, pub unexpected: bool }
impl Default for PreprocessContext { fn default() -> Self { Self { special: [0; MAX_SPECIAL_TOKENS], special_count: 0, fragment_count: 0, preprocessed: false, phase_error: PreprocessError::None, err: PreprocessError::None, result: false, unexpected: false } } }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextTokenizerPreprocessorWpmStates { Idle, RequestBufferDecision, RequestCapacityNonzeroDecision, RequestCapacityLimitDecision, Preparing, BuildSpecialsDecision, PartitionSpecialsDecision, PartitionParseSpecialDecision, PartitioningNoSpecialsInputDecision, PartitioningNonBpeParseInputDecision, PartitioningNonBpeSkipInputDecision, PartitioningNoSpecials, PartitioningNonBpeParseSpecial, PartitioningNonBpeSkipSpecial, PartitionDecision, Done, Errored, Unexpected }

pub struct TextTokenizerPreprocessorWpmActor<'a> { state: TextTokenizerPreprocessorWpmStates, context: PreprocessContext, _marker: core::marker::PhantomData<&'a ()> }
impl<'a> Default for TextTokenizerPreprocessorWpmActor<'a> { fn default() -> Self { Self::new() } }
impl<'a> TextTokenizerPreprocessorWpmActor<'a> {
    #[must_use] pub fn new() -> Self { Self { state: TextTokenizerPreprocessorWpmStates::Idle, context: PreprocessContext::default(), _marker: core::marker::PhantomData } }
    pub fn process_event(&mut self, event: EventPreprocessRuntime<'a>) -> bool { self.state = TextTokenizerPreprocessorWpmStates::RequestBufferDecision; self.context = PreprocessContext::default(); *event.request.fragment_count_out.borrow_mut() = 0; *event.request.error_out.borrow_mut() = 0; if let Some(s) = event.request.preprocessed_out { *s.borrow_mut() = false; } let Some(output) = event.request.fragments_out else { return self.fail(event); }; let cap = output.borrow().len(); if cap == 0 || cap > MAX_FRAGMENTS { return self.fail(event); } self.state = TextTokenizerPreprocessorWpmStates::Preparing; let mut i = 0; while i < event.request.vocab.token_count() { if let Some(t) = event.request.vocab.token(i) { if matches!(t.token_type, 2 | 3 | 4) && !t.text.is_empty() { if self.context.special_count == MAX_SPECIAL_TOKENS { return self.fail(event); } self.context.special[self.context.special_count] = i; self.context.special_count += 1; } } i += 1; } self.state = TextTokenizerPreprocessorWpmStates::PartitionSpecialsDecision; let mut out = output.borrow_mut(); let mut n = 0; let mut base = 0; while base < event.request.text.len() { let mut chosen = None; let mut j = 0; while j < self.context.special_count { if let Some(t) = event.request.vocab.token(self.context.special[j]) { if (event.request.parse_special || !matches!(t.token_type, 2 | 3)) && !t.text.is_empty() { if let Some(p) = event.request.text[base..].find(t.text) { if chosen.is_none_or(|(q, _): (usize, VocabularyToken<'_>)| p < q) { chosen = Some((p, t)); } } } } j += 1; } let Some((relative, token)) = chosen else { if n == out.len() { return self.fail(event); } out[n] = Fragment { kind: FragmentKind::RawText, text: &event.request.text[base..], token: -1 }; n += 1; break; }; let pos = base + relative; if pos > base { if n == out.len() { return self.fail(event); } out[n] = Fragment { kind: FragmentKind::RawText, text: &event.request.text[base..pos], token: -1 }; n += 1; } if token.token < 0 || n == out.len() { return self.fail(event); } out[n] = Fragment { kind: FragmentKind::Token, text: "", token: token.token }; n += 1; base = pos + token.text.len(); } self.context.fragment_count = n; self.context.preprocessed = true; self.context.result = true; self.state = TextTokenizerPreprocessorWpmStates::Done; *event.request.fragment_count_out.borrow_mut() = n; if let Some(s) = event.request.preprocessed_out { *s.borrow_mut() = true; } if let Some(cb) = event.request.dispatch_done { let _ = cb(PreprocessDone { fragment_count: n }); } true }
    fn fail(&mut self, event: EventPreprocessRuntime<'a>) -> bool { self.context.phase_error = PreprocessError::InvalidRequest; self.context.err = PreprocessError::InvalidRequest; self.context.result = false; self.state = TextTokenizerPreprocessorWpmStates::Errored; *event.request.error_out.borrow_mut() = 1; if let Some(cb) = event.request.dispatch_error { let _ = cb(PreprocessErrorEvent { err: PreprocessError::InvalidRequest }); } false }
    pub fn process_unexpected(&mut self) -> bool { self.context.unexpected = true; self.context.phase_error = PreprocessError::InvalidRequest; self.context.err = PreprocessError::InvalidRequest; self.state = TextTokenizerPreprocessorWpmStates::Unexpected; false }
    #[must_use] pub fn state(&self) -> &TextTokenizerPreprocessorWpmStates { &self.state }
    #[must_use] pub fn is(&self, state: &TextTokenizerPreprocessorWpmStates) -> bool { self.state == *state }
    #[must_use] pub fn context(&self) -> &PreprocessContext { &self.context }
    #[must_use] pub fn last_error(&self) -> i32 { self.context.err.code() }
    #[must_use] pub fn fragment_count(&self) -> usize { self.context.fragment_count }
}
pub type Wpm<'a> = TextTokenizerPreprocessorWpmActor<'a>;
