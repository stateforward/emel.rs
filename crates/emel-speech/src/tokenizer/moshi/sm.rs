//! Source-aligned Moshi speech tokenizer state machine.

#![allow(
    clippy::enum_variant_names,
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

use core::cell::{Cell, RefCell};
use core::convert::TryFrom;
use sml::sml;

/// Errors defined by the pinned Moshi tokenizer contract.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u32)]
pub enum Error {
    #[default]
    None = 0,
    InvalidConfiguration = 1 << 0,
    Uninitialized = 1 << 1,
    AlreadyInitialized = 1 << 2,
    RequestShape = 1 << 3,
    PhaseOrder = 1 << 4,
    PositionOverflow = 1 << 5,
    InternalError = 1 << 6,
}

/// Source-compatible error spelling.
pub type MoshiError = Error;

/// Injected geometry and caller-owned ring storage.
#[derive(Debug)]
pub struct Dependencies<'a> {
    pub delays: &'a [i32],
    pub cache: &'a mut [i32],
    pub codebooks: i32,
    pub generated_audio_codebooks: i32,
    pub delayed_audio_codebooks: i32,
    pub cache_rows: i32,
    pub maximum_delay: i32,
    pub initial_delay_frames: i32,
    pub text_initial_token: i32,
    pub audio_initial_token: i32,
    pub token_zero: i32,
    pub token_ungenerated: i32,
}

impl<'a> Dependencies<'a> {
    #[must_use]
    pub const fn new(
        delays: &'a [i32],
        cache: &'a mut [i32],
        codebooks: i32,
        generated_audio_codebooks: i32,
        delayed_audio_codebooks: i32,
        cache_rows: i32,
        maximum_delay: i32,
        initial_delay_frames: i32,
        text_initial_token: i32,
        audio_initial_token: i32,
        token_zero: i32,
        token_ungenerated: i32,
    ) -> Self {
        Self { delays, cache, codebooks, generated_audio_codebooks, delayed_audio_codebooks, cache_rows, maximum_delay, initial_delay_frames, text_initial_token, audio_initial_token, token_zero, token_ungenerated }
    }
}

/// Initialize request with caller-owned error output.
pub struct EventInitialize<'a> { pub error_out: RefCell<&'a mut Error> }
impl<'a> EventInitialize<'a> {
    #[must_use]
    pub fn new(error_out: &'a mut Error) -> Self { Self { error_out: RefCell::new(error_out) } }
}

/// Tokenization request. Input and output remain caller-owned.
pub struct EventTokenize<'a> {
    pub audio_tokens: &'a [i32],
    pub model_tokens_out: RefCell<&'a mut [i32]>,
    pub error_out: RefCell<&'a mut Error>,
}
impl<'a> EventTokenize<'a> {
    #[must_use]
    pub fn new(audio_tokens: &'a [i32], model_tokens_out: &'a mut [i32], error_out: &'a mut Error) -> Self {
        Self { audio_tokens, model_tokens_out: RefCell::new(model_tokens_out), error_out: RefCell::new(error_out) }
    }
}

/// Detokenization request, including the internal completion handoff state.
pub struct EventDetokenizeRun<'a> {
    pub text_token: i32,
    pub audio_tokens: &'a [i32],
    pub text_token_out: RefCell<&'a mut i32>,
    pub audio_tokens_out: RefCell<&'a mut [i32]>,
    pub produced_out: RefCell<&'a mut bool>,
    pub error_out: RefCell<&'a mut Error>,
    pub source_offset: Cell<i64>,
}
impl<'a> EventDetokenizeRun<'a> {
    #[must_use]
    pub fn new(text_token: i32, audio_tokens: &'a [i32], text_token_out: &'a mut i32, audio_tokens_out: &'a mut [i32], produced_out: &'a mut bool, error_out: &'a mut Error) -> Self {
        Self { text_token, audio_tokens, text_token_out: RefCell::new(text_token_out), audio_tokens_out: RefCell::new(audio_tokens_out), produced_out: RefCell::new(produced_out), error_out: RefCell::new(error_out), source_offset: Cell::new(0) }
    }
}

/// Column-major cache restore request.
pub struct EventRestoreCache<'a> { pub column_major_cache: &'a [i32], pub offset: i64, pub error_out: RefCell<&'a mut Error> }
impl<'a> EventRestoreCache<'a> {
    #[must_use]
    pub fn new(column_major_cache: &'a [i32], offset: i64, error_out: &'a mut Error) -> Self { Self { column_major_cache, offset, error_out: RefCell::new(error_out) } }
}

/// Advance the logical frame offset.
pub struct EventAdvance<'a> { pub error_out: RefCell<&'a mut Error> }
impl<'a> EventAdvance<'a> { #[must_use] pub fn new(error_out: &'a mut Error) -> Self { Self { error_out: RefCell::new(error_out) } } }

/// Reset the cache and logical offset.
pub struct EventReset<'a> { pub error_out: RefCell<&'a mut Error> }
impl<'a> EventReset<'a> { #[must_use] pub fn new(error_out: &'a mut Error) -> Self { Self { error_out: RefCell::new(error_out) } } }

sml! {
    SpeechTokenizerMoshi {
        "state_ready"_s <= *"state_uninitialized"_s + Initialize(&'dispatch EventInitialize<'dispatch>) [guard_configuration_valid] / effect_initialize_from_state_uninitialized,
        "state_uninitialized"_s <= "state_uninitialized"_s + Initialize(&'dispatch EventInitialize<'dispatch>) [guard_configuration_invalid] / effect_reject_error_invalid_configuration_from_state_uninitialized,
        "state_ready"_s <= "state_ready"_s + Initialize(&'dispatch EventInitialize<'dispatch>) / effect_reject_error_already_initialized_from_state_ready,
        "state_prepared_full"_s <= "state_prepared_full"_s + Initialize(&'dispatch EventInitialize<'dispatch>) / effect_reject_error_already_initialized_from_state_prepared_full,
        "state_prepared_generated"_s <= "state_prepared_generated"_s + Initialize(&'dispatch EventInitialize<'dispatch>) / effect_reject_error_already_initialized_from_state_prepared_generated,
        "state_prepared_tail"_s <= "state_prepared_tail"_s + Initialize(&'dispatch EventInitialize<'dispatch>) / effect_reject_error_already_initialized_from_state_prepared_tail,
        "state_ready"_s <= "state_errored"_s + Initialize(&'dispatch EventInitialize<'dispatch>) [guard_configuration_valid] / effect_initialize_from_state_errored,
        "state_errored"_s <= "state_errored"_s + Initialize(&'dispatch EventInitialize<'dispatch>) [guard_configuration_invalid] / effect_reject_error_invalid_configuration_from_state_errored,
        "state_prepared_full"_s <= "state_ready"_s + Tokenize(&'dispatch EventTokenize<'dispatch>) [guard_tokenize_full] / effect_tokenize_full,
        "state_prepared_tail"_s <= "state_ready"_s + Tokenize(&'dispatch EventTokenize<'dispatch>) [guard_tokenize_tail] / effect_tokenize_tail,
        "state_prepared_generated"_s <= "state_ready"_s + Tokenize(&'dispatch EventTokenize<'dispatch>) [guard_tokenize_empty] / effect_tokenize_empty,
        "state_ready"_s <= "state_ready"_s + Tokenize(&'dispatch EventTokenize<'dispatch>) [guard_tokenize_invalid] / effect_reject_error_request_shape_event_tokenize,
        "state_ready"_s <= "state_ready"_s + Tokenize(&'dispatch EventTokenize<'dispatch>) [guard_tokenize_position_overflow] / effect_reject_error_position_overflow_event_tokenize,
        "state_prepared_full"_s <= "state_prepared_full"_s + Tokenize(&'dispatch EventTokenize<'dispatch>) / effect_reject_error_phase_order_event_tokenize,
        "state_prepared_generated"_s <= "state_prepared_generated"_s + Tokenize(&'dispatch EventTokenize<'dispatch>) / effect_reject_error_phase_order_event_tokenize,
        "state_prepared_tail"_s <= "state_prepared_tail"_s + Tokenize(&'dispatch EventTokenize<'dispatch>) / effect_reject_error_phase_order_event_tokenize,
        "state_commit_full_zero"_s <= "state_prepared_full"_s + DetokenizeRun(&'dispatch EventDetokenizeRun<'dispatch>) [guard_detokenize_valid_replace] / effect_begin_detokenize_from_state_prepared_full,
        "state_commit_full_generated"_s <= "state_prepared_full"_s + DetokenizeRun(&'dispatch EventDetokenizeRun<'dispatch>) [guard_detokenize_valid_generated] / effect_begin_detokenize_from_state_prepared_full,
        "state_prepared_full"_s <= "state_prepared_full"_s + DetokenizeRun(&'dispatch EventDetokenizeRun<'dispatch>) [guard_detokenize_request_invalid] / effect_reject_detokenize_error_request_shape_from_state_prepared_full,
        "state_prepared_full"_s <= "state_prepared_full"_s + DetokenizeRun(&'dispatch EventDetokenizeRun<'dispatch>) [guard_position_overflow] / effect_reject_detokenize_error_position_overflow_from_state_prepared_full,
        "state_commit_generated_zero"_s <= "state_prepared_tail"_s + DetokenizeRun(&'dispatch EventDetokenizeRun<'dispatch>) [guard_detokenize_valid_replace] / effect_begin_detokenize_from_state_prepared_tail,
        "state_commit_generated"_s <= "state_prepared_tail"_s + DetokenizeRun(&'dispatch EventDetokenizeRun<'dispatch>) [guard_detokenize_valid_generated] / effect_begin_detokenize_from_state_prepared_tail,
        "state_prepared_tail"_s <= "state_prepared_tail"_s + DetokenizeRun(&'dispatch EventDetokenizeRun<'dispatch>) [guard_detokenize_request_invalid] / effect_reject_detokenize_error_request_shape_from_state_prepared_tail,
        "state_prepared_tail"_s <= "state_prepared_tail"_s + DetokenizeRun(&'dispatch EventDetokenizeRun<'dispatch>) [guard_position_overflow] / effect_reject_detokenize_error_position_overflow_from_state_prepared_tail,
        "state_commit_generated_zero"_s <= "state_prepared_generated"_s + DetokenizeRun(&'dispatch EventDetokenizeRun<'dispatch>) [guard_detokenize_valid_replace] / effect_begin_detokenize_from_state_prepared_generated,
        "state_commit_generated"_s <= "state_prepared_generated"_s + DetokenizeRun(&'dispatch EventDetokenizeRun<'dispatch>) [guard_detokenize_valid_generated] / effect_begin_detokenize_from_state_prepared_generated,
        "state_prepared_generated"_s <= "state_prepared_generated"_s + DetokenizeRun(&'dispatch EventDetokenizeRun<'dispatch>) [guard_detokenize_request_invalid] / effect_reject_detokenize_error_request_shape_from_state_prepared_generated,
        "state_prepared_generated"_s <= "state_prepared_generated"_s + DetokenizeRun(&'dispatch EventDetokenizeRun<'dispatch>) [guard_position_overflow] / effect_reject_detokenize_error_position_overflow_from_state_prepared_generated,
        "state_output_decision"_s <= "state_commit_full_zero"_s + completion<Detokenize>(&'dispatch EventDetokenizeRun<'dispatch>) / effect_commit_full_zero,
        "state_output_decision"_s <= "state_commit_full_generated"_s + completion<Detokenize>(&'dispatch EventDetokenizeRun<'dispatch>) / effect_commit_full_generated,
        "state_output_decision"_s <= "state_commit_generated_zero"_s + completion<Detokenize>(&'dispatch EventDetokenizeRun<'dispatch>) / effect_commit_generated_zero,
        "state_output_decision"_s <= "state_commit_generated"_s + completion<Detokenize>(&'dispatch EventDetokenizeRun<'dispatch>) / effect_commit_generated,
        "state_ready"_s <= "state_output_decision"_s + completion<Detokenize>(&'dispatch EventDetokenizeRun<'dispatch>) [guard_before_output_delay] / effect_publish_no_output_from_state_output_decision,
        "state_output_validation"_s <= "state_output_decision"_s + completion<Detokenize>(&'dispatch EventDetokenizeRun<'dispatch>) [guard_past_output_delay] / effect_collect_output,
        "state_ready"_s <= "state_output_validation"_s + completion<Detokenize>(&'dispatch EventDetokenizeRun<'dispatch>) [guard_output_complete] / effect_publish_output,
        "state_ready"_s <= "state_output_validation"_s + completion<Detokenize>(&'dispatch EventDetokenizeRun<'dispatch>) [guard_output_incomplete] / effect_publish_no_output_from_state_output_validation,
        "state_ready"_s <= "state_ready"_s + RestoreCache(&'dispatch EventRestoreCache<'dispatch>) [guard_restore_valid] / effect_restore_column_major_cache,
        "state_ready"_s <= "state_ready"_s + RestoreCache(&'dispatch EventRestoreCache<'dispatch>) [guard_restore_invalid] / effect_reject_error_request_shape_event_restore_cache,
        "state_prepared_full"_s <= "state_prepared_full"_s + RestoreCache(&'dispatch EventRestoreCache<'dispatch>) / effect_reject_error_phase_order_event_restore_cache,
        "state_prepared_generated"_s <= "state_prepared_generated"_s + RestoreCache(&'dispatch EventRestoreCache<'dispatch>) / effect_reject_error_phase_order_event_restore_cache,
        "state_prepared_tail"_s <= "state_prepared_tail"_s + RestoreCache(&'dispatch EventRestoreCache<'dispatch>) / effect_reject_error_phase_order_event_restore_cache,
        "state_ready"_s <= "state_prepared_full"_s + Advance(&'dispatch EventAdvance<'dispatch>) [guard_advance_position_available] / effect_advance,
        "state_prepared_full"_s <= "state_prepared_full"_s + Advance(&'dispatch EventAdvance<'dispatch>) [guard_advance_position_overflow] / effect_reject_error_position_overflow_event_advance,
        "state_prepared_generated"_s <= "state_prepared_generated"_s + Advance(&'dispatch EventAdvance<'dispatch>) / effect_reject_error_phase_order_event_advance,
        "state_prepared_tail"_s <= "state_prepared_tail"_s + Advance(&'dispatch EventAdvance<'dispatch>) / effect_reject_error_phase_order_event_advance,
        "state_ready"_s <= "state_ready"_s + Advance(&'dispatch EventAdvance<'dispatch>) / effect_reject_error_phase_order_event_advance,
        "state_ready"_s <= "state_ready"_s + Reset(&'dispatch EventReset<'dispatch>) / effect_reset_from_state_ready,
        "state_ready"_s <= "state_prepared_full"_s + Reset(&'dispatch EventReset<'dispatch>) / effect_reset_from_state_prepared_full,
        "state_ready"_s <= "state_prepared_generated"_s + Reset(&'dispatch EventReset<'dispatch>) / effect_reset_from_state_prepared_generated,
        "state_ready"_s <= "state_prepared_tail"_s + Reset(&'dispatch EventReset<'dispatch>) / effect_reset_from_state_prepared_tail,
        "state_ready"_s <= "state_errored"_s + Reset(&'dispatch EventReset<'dispatch>) / effect_reset_from_state_errored,
        "state_uninitialized"_s <= "state_uninitialized"_s + Tokenize(&'dispatch EventTokenize<'dispatch>) / effect_reject_error_uninitialized_event_tokenize,
        "state_uninitialized"_s <= "state_uninitialized"_s + DetokenizeRun(&'dispatch EventDetokenizeRun<'dispatch>) / effect_reject_detokenize_error_uninitialized,
        "state_uninitialized"_s <= "state_uninitialized"_s + RestoreCache(&'dispatch EventRestoreCache<'dispatch>) / effect_reject_error_uninitialized_event_restore_cache,
        "state_uninitialized"_s <= "state_uninitialized"_s + Advance(&'dispatch EventAdvance<'dispatch>) / effect_reject_error_uninitialized_event_advance,
        "state_uninitialized"_s <= "state_uninitialized"_s + Reset(&'dispatch EventReset<'dispatch>) / effect_reject_error_uninitialized_event_reset,
        "state_ready"_s <= "state_ready"_s + DetokenizeRun(&'dispatch EventDetokenizeRun<'dispatch>) / effect_reject_detokenize_error_phase_order,
        "state_errored"_s <= "state_errored"_s + Tokenize(&'dispatch EventTokenize<'dispatch>) / effect_reject_error_internal_error_event_tokenize,
        "state_errored"_s <= "state_errored"_s + DetokenizeRun(&'dispatch EventDetokenizeRun<'dispatch>) / effect_reject_detokenize_error_internal_error,
        "state_errored"_s <= "state_errored"_s + RestoreCache(&'dispatch EventRestoreCache<'dispatch>) / effect_reject_error_internal_error_event_restore_cache,
        "state_errored"_s <= "state_errored"_s + Advance(&'dispatch EventAdvance<'dispatch>) / effect_reject_error_internal_error_event_advance,
        "state_errored"_s <= "state_uninitialized"_s + unexpected_event<_> / effect_unexpected_from_state_uninitialized,
        "state_errored"_s <= "state_ready"_s + unexpected_event<_> / effect_unexpected_from_state_ready,
        "state_errored"_s <= "state_prepared_full"_s + unexpected_event<_> / effect_unexpected_from_state_prepared_full,
        "state_errored"_s <= "state_prepared_generated"_s + unexpected_event<_> / effect_unexpected_from_state_prepared_generated,
        "state_errored"_s <= "state_prepared_tail"_s + unexpected_event<_> / effect_unexpected_from_state_prepared_tail,
        "state_errored"_s <= "state_errored"_s + unexpected_event<_> / effect_unexpected_from_state_errored,
    }
}

#[derive(Debug)]
pub struct SpeechTokenizerMoshiContext<'a> {
    pub config: Dependencies<'a>,
    pub offset: i64,
    pub last_error: Error,
}
impl<'a> SpeechTokenizerMoshiContext<'a> {
    #[must_use]
    pub fn new(config: Dependencies<'a>) -> Self { Self { config, offset: 0, last_error: Error::None } }
    fn row(&self, offset: i64) -> usize { let rows = i64::from(self.config.cache_rows); let position = offset % rows; usize::try_from(position + i64::from(position < 0) * rows).unwrap_or(0) }
    fn cache_index(&self, row: usize, codebook: usize) -> usize { row * usize::try_from(self.config.codebooks).unwrap_or(0) + codebook }
    fn set_error(&mut self, error: Error) { self.last_error = error; }
    fn write_error<'e>(&mut self, out: &RefCell<&'e mut Error>, error: Error) { **out.borrow_mut() = error; self.set_error(error); }
    fn clear_error<'e>(&mut self, out: &RefCell<&'e mut Error>) { **out.borrow_mut() = Error::None; self.set_error(Error::None); }
    fn reset_cache(&mut self) { self.config.cache.fill(self.config.token_ungenerated); self.offset = 0; }
    fn model_tokens(&self, offset: i64, out: &mut [i32]) { let row = self.row(offset); let text_initial = i32::from(offset <= i64::from(self.config.delays[0])); out[0] = text_initial * self.config.text_initial_token + (1 - text_initial) * self.config.cache[self.cache_index(row, 0)]; for codebook in 1..usize::try_from(self.config.codebooks).unwrap_or(0) { let initial = i32::from(offset <= i64::from(self.config.delays[codebook])); out[codebook] = initial * self.config.audio_initial_token + (1 - initial) * self.config.cache[self.cache_index(row, codebook)]; } }
    fn commit<const PRESERVE: bool, const GENERATED: bool>(&mut self, event: &EventDetokenizeRun<'_>) { self.offset = event.source_offset.get() + 1; let row = self.row(self.offset); self.config.cache[self.cache_index(row, 0)] = event.text_token; for audio_codebook in 0..usize::try_from(self.config.generated_audio_codebooks).unwrap_or(0) { let codebook = audio_codebook + 1; let mut token = self.config.token_zero; if GENERATED { let masked = i32::from(self.config.initial_delay_frames > 0 && event.source_offset.get() < i64::from(self.config.delays[codebook]) + i64::from(self.config.initial_delay_frames)); token = masked * self.config.token_zero + (1 - masked) * event.audio_tokens[audio_codebook]; } if PRESERVE { let current = self.config.cache[self.cache_index(row, codebook)]; let missing = i32::from(current == self.config.token_ungenerated); token = missing * token + (1 - missing) * current; } self.config.cache[self.cache_index(row, codebook)] = token; } self.clear_error(&event.error_out); }
}
impl<'a> Default for SpeechTokenizerMoshiContext<'a> { fn default() -> Self { Self::new(Dependencies { delays: &[], cache: &mut [], codebooks: 0, generated_audio_codebooks: 0, delayed_audio_codebooks: 0, cache_rows: 0, maximum_delay: 0, initial_delay_frames: 0, text_initial_token: 0, audio_initial_token: 0, token_zero: -1, token_ungenerated: 0 }) } }

impl SpeechTokenizerMoshiStateMachineContext for SpeechTokenizerMoshiContext<'_> {
    fn effect_advance(&mut self, event: &EventAdvance<'_>) -> Result<(), ()> { self.offset += 1; self.clear_error(&event.error_out); Ok(()) }
    fn effect_begin_detokenize_from_state_prepared_full(&mut self, event: &EventDetokenizeRun<'_>) -> Result<(), ()> { self.begin_detokenize(event); Ok(()) }
    fn effect_begin_detokenize_from_state_prepared_generated(&mut self, event: &EventDetokenizeRun<'_>) -> Result<(), ()> { self.begin_detokenize(event); Ok(()) }
    fn effect_begin_detokenize_from_state_prepared_tail(&mut self, event: &EventDetokenizeRun<'_>) -> Result<(), ()> { self.begin_detokenize(event); Ok(()) }
    fn begin_detokenize(&mut self, event: &EventDetokenizeRun<'_>) { event.source_offset.set(self.offset); **event.text_token_out.borrow_mut() = self.config.token_zero; event.audio_tokens_out.borrow_mut().fill(self.config.token_zero); **event.produced_out.borrow_mut() = false; self.clear_error(&event.error_out); }
    fn effect_commit_full_zero(&mut self, event: &EventDetokenizeRun<'_>) -> Result<(), ()> { self.commit::<true, false>(event); Ok(()) }
    fn effect_commit_full_generated(&mut self, event: &EventDetokenizeRun<'_>) -> Result<(), ()> { self.commit::<true, true>(event); Ok(()) }
    fn effect_commit_generated_zero(&mut self, event: &EventDetokenizeRun<'_>) -> Result<(), ()> { self.commit::<false, false>(event); Ok(()) }
    fn effect_commit_generated(&mut self, event: &EventDetokenizeRun<'_>) -> Result<(), ()> { self.commit::<false, true>(event); Ok(()) }
    fn effect_collect_output(&mut self, event: &EventDetokenizeRun<'_>) -> Result<(), ()> { let row = self.row(self.offset - i64::from(self.config.maximum_delay) + i64::from(self.config.delays[0])); **event.text_token_out.borrow_mut() = self.config.cache[self.cache_index(row, 0)]; for audio_codebook in 0..usize::try_from(self.config.delayed_audio_codebooks).unwrap_or(0) { let codebook = audio_codebook + 1; let row = self.row(self.offset - i64::from(self.config.maximum_delay) + i64::from(self.config.delays[codebook])); event.audio_tokens_out.borrow_mut()[audio_codebook] = self.config.cache[self.cache_index(row, codebook)]; } Ok(()) }
    fn effect_publish_output(&mut self, event: &EventDetokenizeRun<'_>) -> Result<(), ()> { **event.produced_out.borrow_mut() = true; self.clear_error(&event.error_out); Ok(()) }
    fn effect_publish_no_output_from_state_output_decision(&mut self, event: &EventDetokenizeRun<'_>) -> Result<(), ()> { **event.produced_out.borrow_mut() = false; self.clear_error(&event.error_out); Ok(()) }
    fn effect_publish_no_output_from_state_output_validation(&mut self, event: &EventDetokenizeRun<'_>) -> Result<(), ()> { **event.produced_out.borrow_mut() = false; self.clear_error(&event.error_out); Ok(()) }
    fn effect_initialize_from_state_errored(&mut self, event: &EventInitialize<'_>) -> Result<(), ()> { self.reset_cache(); self.clear_error(&event.error_out); Ok(()) }
    fn effect_initialize_from_state_uninitialized(&mut self, event: &EventInitialize<'_>) -> Result<(), ()> { self.reset_cache(); self.clear_error(&event.error_out); Ok(()) }
    fn reject_init(&mut self, event: &EventInitialize<'_>, error: Error) { self.write_error(&event.error_out, error); }
    fn reject_event<'e>(&mut self, event: &RefCell<&'e mut Error>, error: Error) { self.write_error(event, error); }
    fn reject_detokenize(&mut self, event: &EventDetokenizeRun<'_>, error: Error) { self.write_error(&event.error_out, error); }
    fn effect_reject_detokenize_error_internal_error(&mut self, event: &EventDetokenizeRun<'_>) -> Result<(), ()> { self.reject_detokenize(event, Error::InternalError); Ok(()) }
    fn effect_reject_detokenize_error_phase_order(&mut self, event: &EventDetokenizeRun<'_>) -> Result<(), ()> { self.reject_detokenize(event, Error::PhaseOrder); Ok(()) }
    fn effect_reject_detokenize_error_position_overflow_from_state_prepared_full(&mut self, event: &EventDetokenizeRun<'_>) -> Result<(), ()> { self.reject_detokenize(event, Error::PositionOverflow); Ok(()) }
    fn effect_reject_detokenize_error_position_overflow_from_state_prepared_generated(&mut self, event: &EventDetokenizeRun<'_>) -> Result<(), ()> { self.reject_detokenize(event, Error::PositionOverflow); Ok(()) }
    fn effect_reject_detokenize_error_position_overflow_from_state_prepared_tail(&mut self, event: &EventDetokenizeRun<'_>) -> Result<(), ()> { self.reject_detokenize(event, Error::PositionOverflow); Ok(()) }
    fn effect_reject_detokenize_error_request_shape_from_state_prepared_full(&mut self, event: &EventDetokenizeRun<'_>) -> Result<(), ()> { self.reject_detokenize(event, Error::RequestShape); Ok(()) }
    fn effect_reject_detokenize_error_request_shape_from_state_prepared_generated(&mut self, event: &EventDetokenizeRun<'_>) -> Result<(), ()> { self.reject_detokenize(event, Error::RequestShape); Ok(()) }
    fn effect_reject_detokenize_error_request_shape_from_state_prepared_tail(&mut self, event: &EventDetokenizeRun<'_>) -> Result<(), ()> { self.reject_detokenize(event, Error::RequestShape); Ok(()) }
    fn effect_reject_detokenize_error_uninitialized(&mut self, event: &EventDetokenizeRun<'_>) -> Result<(), ()> { self.reject_detokenize(event, Error::Uninitialized); Ok(()) }
    fn reject_initialize_error(&mut self, event: &EventInitialize<'_>, error: Error) { self.reject_init(event, error); }
    fn effect_reject_error_already_initialized_from_state_prepared_full(&mut self, e: &EventInitialize<'_>) -> Result<(), ()> { self.reject_initialize_error(e, Error::AlreadyInitialized); Ok(()) }
    fn effect_reject_error_already_initialized_from_state_prepared_generated(&mut self, e: &EventInitialize<'_>) -> Result<(), ()> { self.reject_initialize_error(e, Error::AlreadyInitialized); Ok(()) }
    fn effect_reject_error_already_initialized_from_state_prepared_tail(&mut self, e: &EventInitialize<'_>) -> Result<(), ()> { self.reject_initialize_error(e, Error::AlreadyInitialized); Ok(()) }
    fn effect_reject_error_already_initialized_from_state_ready(&mut self, e: &EventInitialize<'_>) -> Result<(), ()> { self.reject_initialize_error(e, Error::AlreadyInitialized); Ok(()) }
    fn effect_reject_error_internal_error_event_advance(&mut self, e: &EventAdvance<'_>) -> Result<(), ()> { self.reject_event(&e.error_out, Error::InternalError); Ok(()) }
    fn effect_reject_error_internal_error_event_restore_cache(&mut self, e: &EventRestoreCache<'_>) -> Result<(), ()> { self.reject_event(&e.error_out, Error::InternalError); Ok(()) }
    fn effect_reject_error_internal_error_event_tokenize(&mut self, e: &EventTokenize<'_>) -> Result<(), ()> { self.reject_event(&e.error_out, Error::InternalError); Ok(()) }
    fn effect_reject_error_invalid_configuration_from_state_errored(&mut self, e: &EventInitialize<'_>) -> Result<(), ()> { self.reject_initialize_error(e, Error::InvalidConfiguration); Ok(()) }
    fn effect_reject_error_invalid_configuration_from_state_uninitialized(&mut self, e: &EventInitialize<'_>) -> Result<(), ()> { self.reject_initialize_error(e, Error::InvalidConfiguration); Ok(()) }
    fn effect_reject_error_phase_order_event_advance(&mut self, e: &EventAdvance<'_>) -> Result<(), ()> { self.reject_event(&e.error_out, Error::PhaseOrder); Ok(()) }
    fn effect_reject_error_phase_order_event_restore_cache(&mut self, e: &EventRestoreCache<'_>) -> Result<(), ()> { self.reject_event(&e.error_out, Error::PhaseOrder); Ok(()) }
    fn effect_reject_error_phase_order_event_tokenize(&mut self, e: &EventTokenize<'_>) -> Result<(), ()> { self.reject_event(&e.error_out, Error::PhaseOrder); Ok(()) }
    fn effect_reject_error_position_overflow_event_advance(&mut self, e: &EventAdvance<'_>) -> Result<(), ()> { self.reject_event(&e.error_out, Error::PositionOverflow); Ok(()) }
    fn effect_reject_error_position_overflow_event_tokenize(&mut self, e: &EventTokenize<'_>) -> Result<(), ()> { self.reject_event(&e.error_out, Error::PositionOverflow); Ok(()) }
    fn effect_reject_error_request_shape_event_restore_cache(&mut self, e: &EventRestoreCache<'_>) -> Result<(), ()> { self.reject_event(&e.error_out, Error::RequestShape); Ok(()) }
    fn effect_reject_error_request_shape_event_tokenize(&mut self, e: &EventTokenize<'_>) -> Result<(), ()> { self.reject_event(&e.error_out, Error::RequestShape); Ok(()) }
    fn effect_reject_error_uninitialized_event_advance(&mut self, e: &EventAdvance<'_>) -> Result<(), ()> { self.reject_event(&e.error_out, Error::Uninitialized); Ok(()) }
    fn effect_reject_error_uninitialized_event_reset(&mut self, e: &EventReset<'_>) -> Result<(), ()> { self.reject_event(&e.error_out, Error::Uninitialized); Ok(()) }
    fn effect_reject_error_uninitialized_event_restore_cache(&mut self, e: &EventRestoreCache<'_>) -> Result<(), ()> { self.reject_event(&e.error_out, Error::Uninitialized); Ok(()) }
    fn effect_reject_error_uninitialized_event_tokenize(&mut self, e: &EventTokenize<'_>) -> Result<(), ()> { self.reject_event(&e.error_out, Error::Uninitialized); Ok(()) }
    fn effect_reset_from_state_errored(&mut self, e: &EventReset<'_>) -> Result<(), ()> { self.reset_cache(); self.clear_error(&e.error_out); Ok(()) }
    fn effect_reset_from_state_prepared_full(&mut self, e: &EventReset<'_>) -> Result<(), ()> { self.reset_cache(); self.clear_error(&e.error_out); Ok(()) }
    fn effect_reset_from_state_prepared_generated(&mut self, e: &EventReset<'_>) -> Result<(), ()> { self.reset_cache(); self.clear_error(&e.error_out); Ok(()) }
    fn effect_reset_from_state_prepared_tail(&mut self, e: &EventReset<'_>) -> Result<(), ()> { self.reset_cache(); self.clear_error(&e.error_out); Ok(()) }
    fn effect_reset_from_state_ready(&mut self, e: &EventReset<'_>) -> Result<(), ()> { self.reset_cache(); self.clear_error(&e.error_out); Ok(()) }
    fn effect_restore_column_major_cache(&mut self, e: &EventRestoreCache<'_>) -> Result<(), ()> { let rows = usize::try_from(self.config.cache_rows).unwrap_or(0); let codebooks = usize::try_from(self.config.codebooks).unwrap_or(0); for row in 0..rows { for codebook in 0..codebooks { self.config.cache[self.cache_index(row, codebook)] = e.column_major_cache[row + codebook * rows]; } } self.offset = e.offset; self.clear_error(&e.error_out); Ok(()) }
    fn effect_tokenize_empty(&mut self, e: &EventTokenize<'_>) -> Result<(), ()> { self.model_tokens(self.offset, &mut e.model_tokens_out.borrow_mut()); self.clear_error(&e.error_out); Ok(()) }
    fn effect_tokenize_full(&mut self, e: &EventTokenize<'_>) -> Result<(), ()> { for codebook in 0..usize::try_from(self.config.codebooks).unwrap_or(0) { let row = self.row(self.offset + i64::from(self.config.delays[codebook])); self.config.cache[self.cache_index(row, codebook)] = e.audio_tokens[codebook]; } self.model_tokens(self.offset, &mut e.model_tokens_out.borrow_mut()); self.clear_error(&e.error_out); Ok(()) }
    fn effect_tokenize_tail(&mut self, e: &EventTokenize<'_>) -> Result<(), ()> { let first = usize::try_from(self.config.delayed_audio_codebooks + 1).unwrap_or(0); let needed = usize::try_from(self.config.codebooks - self.config.delayed_audio_codebooks - 1).unwrap_or(0); for tail in 0..needed { let codebook = first + tail; let row = self.row(self.offset + i64::from(self.config.delays[codebook])); self.config.cache[self.cache_index(row, codebook)] = e.audio_tokens[tail]; } self.model_tokens(self.offset, &mut e.model_tokens_out.borrow_mut()); self.clear_error(&e.error_out); Ok(()) }
    fn effect_unexpected_from_state_errored(&mut self) -> Result<(), ()> { self.set_error(Error::InternalError); Ok(()) }
    fn effect_unexpected_from_state_prepared_full(&mut self) -> Result<(), ()> { self.set_error(Error::InternalError); Ok(()) }
    fn effect_unexpected_from_state_prepared_generated(&mut self) -> Result<(), ()> { self.set_error(Error::InternalError); Ok(()) }
    fn effect_unexpected_from_state_prepared_tail(&mut self) -> Result<(), ()> { self.set_error(Error::InternalError); Ok(()) }
    fn effect_unexpected_from_state_ready(&mut self) -> Result<(), ()> { self.set_error(Error::InternalError); Ok(()) }
    fn effect_unexpected_from_state_uninitialized(&mut self) -> Result<(), ()> { self.set_error(Error::InternalError); Ok(()) }

    fn guard_configuration_valid(&self, _: &EventInitialize<'_>) -> Result<bool, ()> { let c = &self.config; if c.codebooks <= 1 || c.generated_audio_codebooks <= 0 || c.generated_audio_codebooks >= c.codebooks || c.delayed_audio_codebooks <= 0 || c.delayed_audio_codebooks > c.generated_audio_codebooks || c.cache_rows <= 0 || c.maximum_delay < 0 || c.initial_delay_frames < 0 || c.text_initial_token <= 0 || c.audio_initial_token <= 0 || c.token_zero >= 0 || c.token_ungenerated >= 0 || c.token_zero == c.token_ungenerated || c.delays.len() < usize::try_from(c.codebooks).unwrap_or(usize::MAX) { return Ok(false); } let needed = c.codebooks - c.delayed_audio_codebooks - 1; if needed < 0 || i64::from(c.cache_rows) < i64::from(c.maximum_delay) + 2 { return Ok(false); } let elements = u64::try_from(c.cache_rows).unwrap_or(0).saturating_mul(u64::try_from(c.codebooks).unwrap_or(0)); if elements > c.cache.len() as u64 { return Ok(false); } let mut observed = 0; for codebook in 0..usize::try_from(c.codebooks).unwrap_or(0) { let delay = c.delays[codebook]; if delay < 0 || delay > c.maximum_delay { return Ok(false); } observed = observed.max(delay); } Ok(observed == c.maximum_delay) }
    fn guard_configuration_invalid(&self, e: &EventInitialize<'_>) -> Result<bool, ()> { Ok(!self.guard_configuration_valid(e)?) }
    fn guard_tokenize_full(&self, e: &EventTokenize<'_>) -> Result<bool, ()> { Ok(self.tokenize_shape(e, true) && self.tokens_valid(e) && self.position_available()) }
    fn guard_tokenize_tail(&self, e: &EventTokenize<'_>) -> Result<bool, ()> { Ok(self.tokenize_shape(e, false) && self.tokens_valid(e) && self.position_available()) }
    fn guard_tokenize_empty(&self, e: &EventTokenize<'_>) -> Result<bool, ()> { let needed = self.config.codebooks - self.config.delayed_audio_codebooks - 1; Ok(e.model_tokens_out.borrow().len() == usize::try_from(self.config.codebooks).unwrap_or(usize::MAX) && needed == 0 && e.audio_tokens.is_empty() && self.tokens_valid(e) && self.position_available()) }
    fn tokenize_shape(&self, e: &EventTokenize<'_>, full: bool) -> bool { let expected = if full { self.config.codebooks } else { self.config.codebooks - self.config.delayed_audio_codebooks - 1 }; e.model_tokens_out.borrow().len() == usize::try_from(self.config.codebooks).unwrap_or(usize::MAX) && expected > 0 && e.audio_tokens.len() == usize::try_from(expected).unwrap_or(usize::MAX) }
    fn tokens_valid(&self, e: &EventTokenize<'_>) -> bool { e.audio_tokens.iter().all(|&token| token >= 0 && token < self.config.audio_initial_token) }
    fn position_available(&self) -> bool { self.offset <= i64::MAX - i64::from(self.config.maximum_delay) }
    fn guard_tokenize_invalid(&self, e: &EventTokenize<'_>) -> Result<bool, ()> { Ok(!((self.tokenize_shape(e, true) || self.tokenize_shape(e, false) || (self.config.codebooks - self.config.delayed_audio_codebooks - 1 == 0 && e.audio_tokens.is_empty() && e.model_tokens_out.borrow().len() == usize::try_from(self.config.codebooks).unwrap_or(usize::MAX))) && self.tokens_valid(e))) }
    fn guard_tokenize_position_overflow(&self, e: &EventTokenize<'_>) -> Result<bool, ()> { Ok(!self.guard_tokenize_invalid(e)? && !self.position_available()) }
    fn guard_restore_valid(&self, e: &EventRestoreCache<'_>) -> Result<bool, ()> { let c = &self.config; let elements = u64::try_from(c.cache_rows).unwrap_or(0).saturating_mul(u64::try_from(c.codebooks).unwrap_or(0)); if elements != e.column_major_cache.len() as u64 || e.offset < 0 || e.offset > i64::MAX - i64::from(c.maximum_delay) { return Ok(false); } for index in 0..e.column_major_cache.len() { let token = e.column_major_cache[index]; let codebook = index / usize::try_from(c.cache_rows).unwrap_or(1); let upper = if codebook == 0 { c.text_initial_token } else { c.audio_initial_token }; if token != c.token_zero && token != c.token_ungenerated && (token < 0 || token >= upper) { return Ok(false); } } Ok(true) }
    fn guard_restore_invalid(&self, e: &EventRestoreCache<'_>) -> Result<bool, ()> { Ok(!self.guard_restore_valid(e)?) }
    fn guard_advance_position_available(&self, _: &EventAdvance<'_>) -> Result<bool, ()> { Ok(self.offset < i64::MAX) }
    fn guard_advance_position_overflow(&self, e: &EventAdvance<'_>) -> Result<bool, ()> { Ok(!self.guard_advance_position_available(e)?) }
    fn detokenize_shape(&self, e: &EventDetokenizeRun<'_>) -> bool { e.audio_tokens.len() == usize::try_from(self.config.generated_audio_codebooks).unwrap_or(usize::MAX) && e.audio_tokens_out.borrow().len() == usize::try_from(self.config.delayed_audio_codebooks).unwrap_or(usize::MAX) }
    fn detokenize_tokens(&self, e: &EventDetokenizeRun<'_>) -> bool { e.text_token >= 0 && e.text_token < self.config.text_initial_token && e.audio_tokens.iter().all(|&token| token >= 0 && token < self.config.audio_initial_token) }
    fn detokenize_valid(&self, e: &EventDetokenizeRun<'_>) -> bool { self.detokenize_shape(e) && self.detokenize_tokens(e) }
    fn guard_detokenize_request_invalid(&self, e: &EventDetokenizeRun<'_>) -> Result<bool, ()> { Ok(!self.detokenize_valid(e)) }
    fn guard_position_available(&self, _: &EventDetokenizeRun<'_>) -> Result<bool, ()> { Ok(self.offset < i64::MAX) }
    fn guard_position_overflow(&self, e: &EventDetokenizeRun<'_>) -> Result<bool, ()> { Ok(self.detokenize_valid(e) && !self.guard_position_available(e)?) }
    fn replace_audio(&self) -> bool { self.config.initial_delay_frames > 0 && self.offset < i64::from(self.config.initial_delay_frames) }
    fn guard_detokenize_valid_replace(&self, e: &EventDetokenizeRun<'_>) -> Result<bool, ()> { Ok(self.detokenize_valid(e) && self.guard_position_available(e)? && self.replace_audio()) }
    fn guard_detokenize_valid_generated(&self, e: &EventDetokenizeRun<'_>) -> Result<bool, ()> { Ok(self.detokenize_valid(e) && self.guard_position_available(e)? && !self.replace_audio()) }
    fn guard_before_output_delay(&self, _: &EventDetokenizeRun<'_>) -> Result<bool, ()> { Ok(self.offset <= i64::from(self.config.maximum_delay)) }
    fn guard_past_output_delay(&self, e: &EventDetokenizeRun<'_>) -> Result<bool, ()> { Ok(!self.guard_before_output_delay(e)?) }
    fn output_incomplete(&self, e: &EventDetokenizeRun<'_>) -> bool { let mut incomplete = **e.text_token_out.borrow() == self.config.token_zero || **e.text_token_out.borrow() == self.config.token_ungenerated; for &token in e.audio_tokens_out.borrow().iter() { incomplete |= token == self.config.token_zero || token == self.config.token_ungenerated; } incomplete }
    fn guard_output_complete(&self, e: &EventDetokenizeRun<'_>) -> Result<bool, ()> { Ok(!self.output_incomplete(e)) }
    fn guard_output_incomplete(&self, e: &EventDetokenizeRun<'_>) -> Result<bool, ()> { Ok(self.output_incomplete(e)) }
}

/// Synchronous single-writer wrapper around the generated machine.
pub struct SpeechTokenizerMoshi<'a> { machine: SpeechTokenizerMoshiStateMachine<SpeechTokenizerMoshiContext<'a>> }
impl<'a> SpeechTokenizerMoshi<'a> {
    #[must_use]
    pub fn new(config: Dependencies<'a>) -> Self { Self { machine: SpeechTokenizerMoshiStateMachine::new(SpeechTokenizerMoshiContext::new(config)) } }
    pub fn process_initialize(&mut self, event: EventInitialize<'_>) -> Result<(), Error> { self.dispatch(SpeechTokenizerMoshiEvents::Initialize(&event)) }
    pub fn process_tokenize(&mut self, event: EventTokenize<'_>) -> Result<(), Error> { self.dispatch(SpeechTokenizerMoshiEvents::Tokenize(&event)) }
    pub fn process_detokenize(&mut self, event: EventDetokenizeRun<'_>) -> Result<(), Error> { self.dispatch(SpeechTokenizerMoshiEvents::Detokenize(&event)) }
    pub fn process_restore_cache(&mut self, event: EventRestoreCache<'_>) -> Result<(), Error> { self.dispatch(SpeechTokenizerMoshiEvents::RestoreCache(&event)) }
    pub fn process_advance(&mut self, event: EventAdvance<'_>) -> Result<(), Error> { self.dispatch(SpeechTokenizerMoshiEvents::Advance(&event)) }
    pub fn process_reset(&mut self, event: EventReset<'_>) -> Result<(), Error> { self.dispatch(SpeechTokenizerMoshiEvents::Reset(&event)) }
    #[must_use] pub fn state(&self) -> &SpeechTokenizerMoshiStates { self.machine.state() }
    #[must_use] pub fn context(&self) -> &SpeechTokenizerMoshiContext<'a> { self.machine.context() }
    #[must_use] pub fn is(&self, state: &SpeechTokenizerMoshiStates) -> bool { self.machine.is(state) }
    pub fn initialize(&mut self, event: EventInitialize<'_>) -> Result<(), Error> { self.process_initialize(event) }
    pub fn tokenize(&mut self, event: EventTokenize<'_>) -> Result<(), Error> { self.process_tokenize(event) }
    pub fn detokenize(&mut self, event: EventDetokenizeRun<'_>) -> Result<(), Error> { self.process_detokenize(event) }
    pub fn restore_cache(&mut self, event: EventRestoreCache<'_>) -> Result<(), Error> { self.process_restore_cache(event) }
    pub fn advance(&mut self, event: EventAdvance<'_>) -> Result<(), Error> { self.process_advance(event) }
    pub fn reset(&mut self, event: EventReset<'_>) -> Result<(), Error> { self.process_reset(event) }
    fn dispatch(&mut self, event: SpeechTokenizerMoshiEvents<'_>) -> Result<(), Error> { if self.machine.process_event(event).is_err() { self.machine.context_mut().set_error(Error::InternalError); return Err(Error::InternalError); } let error = self.machine.context().last_error; if error == Error::None { Ok(()) } else { Err(error) } }
}
