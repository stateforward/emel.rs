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
    missing_debug_implementations,
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
    /// The scalar arguments mirror the source Moshi dependency contract.
    #[allow(clippy::too_many_arguments)]
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
        Self {
            delays,
            cache,
            codebooks,
            generated_audio_codebooks,
            delayed_audio_codebooks,
            cache_rows,
            maximum_delay,
            initial_delay_frames,
            text_initial_token,
            audio_initial_token,
            token_zero,
            token_ungenerated,
        }
    }
}

/// Initialize request with caller-owned error output.
pub struct EventInitialize<'a> {
    pub error_out: RefCell<&'a mut Error>,
}
impl<'a> EventInitialize<'a> {
    #[must_use]
    pub fn new(error_out: &'a mut Error) -> Self {
        Self {
            error_out: RefCell::new(error_out),
        }
    }
}

/// Tokenization request. Input and output remain caller-owned.
pub struct EventTokenize<'a> {
    pub audio_tokens: &'a [i32],
    pub model_tokens_out: RefCell<&'a mut [i32]>,
    pub error_out: RefCell<&'a mut Error>,
}
impl<'a> EventTokenize<'a> {
    #[must_use]
    pub fn new(
        audio_tokens: &'a [i32],
        model_tokens_out: &'a mut [i32],
        error_out: &'a mut Error,
    ) -> Self {
        Self {
            audio_tokens,
            model_tokens_out: RefCell::new(model_tokens_out),
            error_out: RefCell::new(error_out),
        }
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
    pub fn new(
        text_token: i32,
        audio_tokens: &'a [i32],
        text_token_out: &'a mut i32,
        audio_tokens_out: &'a mut [i32],
        produced_out: &'a mut bool,
        error_out: &'a mut Error,
    ) -> Self {
        Self {
            text_token,
            audio_tokens,
            text_token_out: RefCell::new(text_token_out),
            audio_tokens_out: RefCell::new(audio_tokens_out),
            produced_out: RefCell::new(produced_out),
            error_out: RefCell::new(error_out),
            source_offset: Cell::new(0),
        }
    }
}

/// Column-major cache restore request.
pub struct EventRestoreCache<'a> {
    pub column_major_cache: &'a [i32],
    pub offset: i64,
    pub error_out: RefCell<&'a mut Error>,
}
impl<'a> EventRestoreCache<'a> {
    #[must_use]
    pub fn new(column_major_cache: &'a [i32], offset: i64, error_out: &'a mut Error) -> Self {
        Self {
            column_major_cache,
            offset,
            error_out: RefCell::new(error_out),
        }
    }
}

/// Advance the logical frame offset.
pub struct EventAdvance<'a> {
    pub error_out: RefCell<&'a mut Error>,
}
impl<'a> EventAdvance<'a> {
    #[must_use]
    pub fn new(error_out: &'a mut Error) -> Self {
        Self {
            error_out: RefCell::new(error_out),
        }
    }
}

/// Reset the cache and logical offset.
pub struct EventReset<'a> {
    pub error_out: RefCell<&'a mut Error>,
}
impl<'a> EventReset<'a> {
    #[must_use]
    pub fn new(error_out: &'a mut Error) -> Self {
        Self {
            error_out: RefCell::new(error_out),
        }
    }
}

sml! {
    SpeechTokenizerMoshi<'dispatch, 'event>
    where
        'event: 'dispatch,
    {
        "state_ready"_s <= *"state_uninitialized"_s + Initialize(&'dispatch EventInitialize<'event>) [guard_configuration_valid] / effect_initialize_from_state_uninitialized,
        "state_uninitialized"_s <= "state_uninitialized"_s + Initialize(&'dispatch EventInitialize<'event>) [guard_configuration_invalid] / effect_reject_error_invalid_configuration_from_state_uninitialized,
        "state_ready"_s <= "state_ready"_s + Initialize(&'dispatch EventInitialize<'event>) / effect_reject_error_already_initialized_from_state_ready,
        "state_prepared_full"_s <= "state_prepared_full"_s + Initialize(&'dispatch EventInitialize<'event>) / effect_reject_error_already_initialized_from_state_prepared_full,
        "state_prepared_generated"_s <= "state_prepared_generated"_s + Initialize(&'dispatch EventInitialize<'event>) / effect_reject_error_already_initialized_from_state_prepared_generated,
        "state_prepared_tail"_s <= "state_prepared_tail"_s + Initialize(&'dispatch EventInitialize<'event>) / effect_reject_error_already_initialized_from_state_prepared_tail,
        "state_ready"_s <= "state_errored"_s + Initialize(&'dispatch EventInitialize<'event>) [guard_configuration_valid] / effect_initialize_from_state_errored,
        "state_errored"_s <= "state_errored"_s + Initialize(&'dispatch EventInitialize<'event>) [guard_configuration_invalid] / effect_reject_error_invalid_configuration_from_state_errored,
        "state_prepared_full"_s <= "state_ready"_s + Tokenize(&'dispatch EventTokenize<'event>) [guard_tokenize_full] / effect_tokenize_full,
        "state_prepared_tail"_s <= "state_ready"_s + Tokenize(&'dispatch EventTokenize<'event>) [guard_tokenize_tail] / effect_tokenize_tail,
        "state_prepared_generated"_s <= "state_ready"_s + Tokenize(&'dispatch EventTokenize<'event>) [guard_tokenize_empty] / effect_tokenize_empty,
        "state_ready"_s <= "state_ready"_s + Tokenize(&'dispatch EventTokenize<'event>) [guard_tokenize_invalid] / effect_reject_error_request_shape_event_tokenize,
        "state_ready"_s <= "state_ready"_s + Tokenize(&'dispatch EventTokenize<'event>) [guard_tokenize_position_overflow] / effect_reject_error_position_overflow_event_tokenize,
        "state_prepared_full"_s <= "state_prepared_full"_s + Tokenize(&'dispatch EventTokenize<'event>) / effect_reject_error_phase_order_event_tokenize,
        "state_prepared_generated"_s <= "state_prepared_generated"_s + Tokenize(&'dispatch EventTokenize<'event>) / effect_reject_error_phase_order_event_tokenize,
        "state_prepared_tail"_s <= "state_prepared_tail"_s + Tokenize(&'dispatch EventTokenize<'event>) / effect_reject_error_phase_order_event_tokenize,
        "state_commit_full_zero"_s <= "state_prepared_full"_s + DetokenizeRun(&'dispatch EventDetokenizeRun<'event>) [guard_detokenize_valid_replace] / effect_begin_detokenize_from_state_prepared_full,
        "state_commit_full_generated"_s <= "state_prepared_full"_s + DetokenizeRun(&'dispatch EventDetokenizeRun<'event>) [guard_detokenize_valid_generated] / effect_begin_detokenize_from_state_prepared_full,
        "state_prepared_full"_s <= "state_prepared_full"_s + DetokenizeRun(&'dispatch EventDetokenizeRun<'event>) [guard_detokenize_request_invalid] / effect_reject_detokenize_error_request_shape_from_state_prepared_full,
        "state_prepared_full"_s <= "state_prepared_full"_s + DetokenizeRun(&'dispatch EventDetokenizeRun<'event>) [guard_position_overflow] / effect_reject_detokenize_error_position_overflow_from_state_prepared_full,
        "state_commit_generated_zero"_s <= "state_prepared_tail"_s + DetokenizeRun(&'dispatch EventDetokenizeRun<'event>) [guard_detokenize_valid_replace] / effect_begin_detokenize_from_state_prepared_tail,
        "state_commit_generated"_s <= "state_prepared_tail"_s + DetokenizeRun(&'dispatch EventDetokenizeRun<'event>) [guard_detokenize_valid_generated] / effect_begin_detokenize_from_state_prepared_tail,
        "state_prepared_tail"_s <= "state_prepared_tail"_s + DetokenizeRun(&'dispatch EventDetokenizeRun<'event>) [guard_detokenize_request_invalid] / effect_reject_detokenize_error_request_shape_from_state_prepared_tail,
        "state_prepared_tail"_s <= "state_prepared_tail"_s + DetokenizeRun(&'dispatch EventDetokenizeRun<'event>) [guard_position_overflow] / effect_reject_detokenize_error_position_overflow_from_state_prepared_tail,
        "state_commit_generated_zero"_s <= "state_prepared_generated"_s + DetokenizeRun(&'dispatch EventDetokenizeRun<'event>) [guard_detokenize_valid_replace] / effect_begin_detokenize_from_state_prepared_generated,
        "state_commit_generated"_s <= "state_prepared_generated"_s + DetokenizeRun(&'dispatch EventDetokenizeRun<'event>) [guard_detokenize_valid_generated] / effect_begin_detokenize_from_state_prepared_generated,
        "state_prepared_generated"_s <= "state_prepared_generated"_s + DetokenizeRun(&'dispatch EventDetokenizeRun<'event>) [guard_detokenize_request_invalid] / effect_reject_detokenize_error_request_shape_from_state_prepared_generated,
        "state_prepared_generated"_s <= "state_prepared_generated"_s + DetokenizeRun(&'dispatch EventDetokenizeRun<'event>) [guard_position_overflow] / effect_reject_detokenize_error_position_overflow_from_state_prepared_generated,
        "state_output_decision"_s <= "state_commit_full_zero"_s + completion<Detokenize>(&'dispatch EventDetokenizeRun<'event>) / effect_commit_full_zero,
        "state_output_decision"_s <= "state_commit_full_generated"_s + completion<Detokenize>(&'dispatch EventDetokenizeRun<'event>) / effect_commit_full_generated,
        "state_output_decision"_s <= "state_commit_generated_zero"_s + completion<Detokenize>(&'dispatch EventDetokenizeRun<'event>) / effect_commit_generated_zero,
        "state_output_decision"_s <= "state_commit_generated"_s + completion<Detokenize>(&'dispatch EventDetokenizeRun<'event>) / effect_commit_generated,
        "state_ready"_s <= "state_output_decision"_s + completion<Detokenize>(&'dispatch EventDetokenizeRun<'event>) [guard_before_output_delay] / effect_publish_no_output_from_state_output_decision,
        "state_output_validation"_s <= "state_output_decision"_s + completion<Detokenize>(&'dispatch EventDetokenizeRun<'event>) [guard_past_output_delay] / effect_collect_output,
        "state_ready"_s <= "state_output_validation"_s + completion<Detokenize>(&'dispatch EventDetokenizeRun<'event>) [guard_output_complete] / effect_publish_output,
        "state_ready"_s <= "state_output_validation"_s + completion<Detokenize>(&'dispatch EventDetokenizeRun<'event>) [guard_output_incomplete] / effect_publish_no_output_from_state_output_validation,
        "state_ready"_s <= "state_ready"_s + RestoreCache(&'dispatch EventRestoreCache<'event>) [guard_restore_valid] / effect_restore_column_major_cache,
        "state_ready"_s <= "state_ready"_s + RestoreCache(&'dispatch EventRestoreCache<'event>) [guard_restore_invalid] / effect_reject_error_request_shape_event_restore_cache,
        "state_prepared_full"_s <= "state_prepared_full"_s + RestoreCache(&'dispatch EventRestoreCache<'event>) / effect_reject_error_phase_order_event_restore_cache,
        "state_prepared_generated"_s <= "state_prepared_generated"_s + RestoreCache(&'dispatch EventRestoreCache<'event>) / effect_reject_error_phase_order_event_restore_cache,
        "state_prepared_tail"_s <= "state_prepared_tail"_s + RestoreCache(&'dispatch EventRestoreCache<'event>) / effect_reject_error_phase_order_event_restore_cache,
        "state_ready"_s <= "state_prepared_full"_s + Advance(&'dispatch EventAdvance<'event>) [guard_advance_position_available] / effect_advance,
        "state_prepared_full"_s <= "state_prepared_full"_s + Advance(&'dispatch EventAdvance<'event>) [guard_advance_position_overflow] / effect_reject_error_position_overflow_event_advance,
        "state_prepared_generated"_s <= "state_prepared_generated"_s + Advance(&'dispatch EventAdvance<'event>) / effect_reject_error_phase_order_event_advance,
        "state_prepared_tail"_s <= "state_prepared_tail"_s + Advance(&'dispatch EventAdvance<'event>) / effect_reject_error_phase_order_event_advance,
        "state_ready"_s <= "state_ready"_s + Advance(&'dispatch EventAdvance<'event>) / effect_reject_error_phase_order_event_advance,
        "state_ready"_s <= "state_ready"_s + Reset(&'dispatch EventReset<'event>) / effect_reset_from_state_ready,
        "state_ready"_s <= "state_prepared_full"_s + Reset(&'dispatch EventReset<'event>) / effect_reset_from_state_prepared_full,
        "state_ready"_s <= "state_prepared_generated"_s + Reset(&'dispatch EventReset<'event>) / effect_reset_from_state_prepared_generated,
        "state_ready"_s <= "state_prepared_tail"_s + Reset(&'dispatch EventReset<'event>) / effect_reset_from_state_prepared_tail,
        "state_ready"_s <= "state_errored"_s + Reset(&'dispatch EventReset<'event>) / effect_reset_from_state_errored,
        "state_uninitialized"_s <= "state_uninitialized"_s + Tokenize(&'dispatch EventTokenize<'event>) / effect_reject_error_uninitialized_event_tokenize,
        "state_uninitialized"_s <= "state_uninitialized"_s + DetokenizeRun(&'dispatch EventDetokenizeRun<'event>) / effect_reject_detokenize_error_uninitialized,
        "state_uninitialized"_s <= "state_uninitialized"_s + RestoreCache(&'dispatch EventRestoreCache<'event>) / effect_reject_error_uninitialized_event_restore_cache,
        "state_uninitialized"_s <= "state_uninitialized"_s + Advance(&'dispatch EventAdvance<'event>) / effect_reject_error_uninitialized_event_advance,
        "state_uninitialized"_s <= "state_uninitialized"_s + Reset(&'dispatch EventReset<'event>) / effect_reject_error_uninitialized_event_reset,
        "state_ready"_s <= "state_ready"_s + DetokenizeRun(&'dispatch EventDetokenizeRun<'event>) / effect_reject_detokenize_error_phase_order,
        "state_errored"_s <= "state_errored"_s + Tokenize(&'dispatch EventTokenize<'event>) / effect_reject_error_internal_error_event_tokenize,
        "state_errored"_s <= "state_errored"_s + DetokenizeRun(&'dispatch EventDetokenizeRun<'event>) / effect_reject_detokenize_error_internal_error,
        "state_errored"_s <= "state_errored"_s + RestoreCache(&'dispatch EventRestoreCache<'event>) / effect_reject_error_internal_error_event_restore_cache,
        "state_errored"_s <= "state_errored"_s + Advance(&'dispatch EventAdvance<'event>) / effect_reject_error_internal_error_event_advance,
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
    pub fn new(config: Dependencies<'a>) -> Self {
        Self {
            config,
            offset: 0,
            last_error: Error::None,
        }
    }
    fn row(&self, offset: i64) -> usize {
        let rows = i64::from(self.config.cache_rows);
        let position = offset % rows;
        usize::try_from(position + i64::from(position < 0) * rows).unwrap_or(0)
    }
    fn cache_index(&self, row: usize, codebook: usize) -> usize {
        row * usize::try_from(self.config.codebooks).unwrap_or(0) + codebook
    }
    fn set_error(&mut self, error: Error) {
        self.last_error = error;
    }
    fn write_error<'dispatch, 'event>(
        &mut self,
        out: &'dispatch RefCell<&'event mut Error>,
        error: Error,
    ) where
        'event: 'dispatch,
    {
        **out.borrow_mut() = error;
        self.set_error(error);
    }
    fn clear_error<'dispatch, 'event>(&mut self, out: &'dispatch RefCell<&'event mut Error>)
    where
        'event: 'dispatch,
    {
        **out.borrow_mut() = Error::None;
        self.set_error(Error::None);
    }
    fn reset_cache(&mut self) {
        self.config.cache.fill(self.config.token_ungenerated);
        self.offset = 0;
    }
    fn model_tokens(&self, offset: i64, out: &mut [i32]) {
        let row = self.row(offset);
        let text_initial = i32::from(offset <= i64::from(self.config.delays[0]));
        out[0] = text_initial * self.config.text_initial_token
            + (1 - text_initial) * self.config.cache[self.cache_index(row, 0)];
        for (codebook, &delay) in self.config.delays.iter().enumerate().skip(1).take(
            usize::try_from(self.config.codebooks)
                .unwrap_or(0)
                .saturating_sub(1),
        ) {
            let initial = i32::from(offset <= i64::from(delay));
            out[codebook] = initial * self.config.audio_initial_token
                + (1 - initial) * self.config.cache[self.cache_index(row, codebook)];
        }
    }
    fn commit<'dispatch, 'event, const PRESERVE: bool, const GENERATED: bool>(
        &mut self,
        event: &'dispatch EventDetokenizeRun<'event>,
    ) where
        'event: 'dispatch,
    {
        self.offset = event.source_offset.get() + 1;
        let row = self.row(self.offset);
        self.config.cache[self.cache_index(row, 0)] = event.text_token;
        for audio_codebook in 0..usize::try_from(self.config.generated_audio_codebooks).unwrap_or(0)
        {
            let codebook = audio_codebook + 1;
            let mut token = if GENERATED {
                let masked = i32::from(
                    self.config.initial_delay_frames > 0
                        && event.source_offset.get()
                            < i64::from(self.config.delays[codebook])
                                + i64::from(self.config.initial_delay_frames),
                );
                masked * self.config.token_zero + (1 - masked) * event.audio_tokens[audio_codebook]
            } else {
                self.config.token_zero
            };
            if PRESERVE {
                let current = self.config.cache[self.cache_index(row, codebook)];
                let missing = i32::from(current == self.config.token_ungenerated);
                token = missing * token + (1 - missing) * current;
            }
            self.config.cache[self.cache_index(row, codebook)] = token;
        }
        self.clear_error(&event.error_out);
    }
    fn begin_detokenize<'dispatch, 'event>(&mut self, event: &'dispatch EventDetokenizeRun<'event>)
    where
        'event: 'dispatch,
    {
        event.source_offset.set(self.offset);
        **event.text_token_out.borrow_mut() = self.config.token_zero;
        event
            .audio_tokens_out
            .borrow_mut()
            .fill(self.config.token_zero);
        **event.produced_out.borrow_mut() = false;
        self.clear_error(&event.error_out);
    }
    fn reject_init<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventInitialize<'event>,
        error: Error,
    ) where
        'event: 'dispatch,
    {
        self.write_error(&event.error_out, error);
    }
    fn reject_event<'dispatch, 'event>(
        &mut self,
        event: &'dispatch RefCell<&'event mut Error>,
        error: Error,
    ) where
        'event: 'dispatch,
    {
        self.write_error(event, error);
    }
    fn reject_detokenize<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventDetokenizeRun<'event>,
        error: Error,
    ) where
        'event: 'dispatch,
    {
        self.write_error(&event.error_out, error);
    }
    fn reject_initialize_error<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventInitialize<'event>,
        error: Error,
    ) where
        'event: 'dispatch,
    {
        self.reject_init(event, error);
    }
    fn tokenize_shape<'dispatch, 'event>(
        &self,
        e: &'dispatch EventTokenize<'event>,
        full: bool,
    ) -> bool
    where
        'event: 'dispatch,
    {
        let expected = if full {
            self.config.codebooks
        } else {
            self.config.codebooks - self.config.delayed_audio_codebooks - 1
        };
        e.model_tokens_out.borrow().len()
            == usize::try_from(self.config.codebooks).unwrap_or(usize::MAX)
            && expected > 0
            && e.audio_tokens.len() == usize::try_from(expected).unwrap_or(usize::MAX)
    }
    fn tokens_valid<'dispatch, 'event>(&self, e: &'dispatch EventTokenize<'event>) -> bool
    where
        'event: 'dispatch,
    {
        e.audio_tokens
            .iter()
            .all(|&token| token >= 0 && token < self.config.audio_initial_token)
    }
    fn position_available(&self) -> bool {
        self.offset <= i64::MAX - i64::from(self.config.maximum_delay)
    }
    fn detokenize_shape<'dispatch, 'event>(&self, e: &'dispatch EventDetokenizeRun<'event>) -> bool
    where
        'event: 'dispatch,
    {
        e.audio_tokens.len()
            == usize::try_from(self.config.generated_audio_codebooks).unwrap_or(usize::MAX)
            && e.audio_tokens_out.borrow().len()
                == usize::try_from(self.config.delayed_audio_codebooks).unwrap_or(usize::MAX)
    }
    fn detokenize_tokens<'dispatch, 'event>(&self, e: &'dispatch EventDetokenizeRun<'event>) -> bool
    where
        'event: 'dispatch,
    {
        e.text_token >= 0
            && e.text_token < self.config.text_initial_token
            && e.audio_tokens
                .iter()
                .all(|&token| token >= 0 && token < self.config.audio_initial_token)
    }
    fn detokenize_valid<'dispatch, 'event>(&self, e: &'dispatch EventDetokenizeRun<'event>) -> bool
    where
        'event: 'dispatch,
    {
        self.detokenize_shape(e) && self.detokenize_tokens(e)
    }
    /// SML guard callbacks use `Result<bool, ()>` even when this guard cannot fail.
    #[allow(clippy::unnecessary_wraps)]
    fn guard_position_available<'dispatch, 'event>(
        &self,
        _: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(self.offset < i64::MAX)
    }
    fn replace_audio(&self) -> bool {
        self.config.initial_delay_frames > 0
            && self.offset < i64::from(self.config.initial_delay_frames)
    }
    fn output_incomplete<'dispatch, 'event>(&self, e: &'dispatch EventDetokenizeRun<'event>) -> bool
    where
        'event: 'dispatch,
    {
        let mut incomplete = **e.text_token_out.borrow() == self.config.token_zero
            || **e.text_token_out.borrow() == self.config.token_ungenerated;
        for &token in e.audio_tokens_out.borrow().iter() {
            incomplete |= token == self.config.token_zero || token == self.config.token_ungenerated;
        }
        incomplete
    }
}

impl SpeechTokenizerMoshiStateMachineContext for SpeechTokenizerMoshiContext<'_> {
    fn effect_advance<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventAdvance<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.offset += 1;
        self.clear_error(&event.error_out);
        Ok(())
    }
    fn effect_begin_detokenize_from_state_prepared_full<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.begin_detokenize(event);
        Ok(())
    }
    fn effect_begin_detokenize_from_state_prepared_generated<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.begin_detokenize(event);
        Ok(())
    }
    fn effect_begin_detokenize_from_state_prepared_tail<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.begin_detokenize(event);
        Ok(())
    }
    fn effect_commit_full_zero<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.commit::<true, false>(event);
        Ok(())
    }
    fn effect_commit_full_generated<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.commit::<true, true>(event);
        Ok(())
    }
    fn effect_commit_generated_zero<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.commit::<false, false>(event);
        Ok(())
    }
    fn effect_commit_generated<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.commit::<false, true>(event);
        Ok(())
    }
    fn effect_collect_output<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let row = self.row(
            self.offset - i64::from(self.config.maximum_delay) + i64::from(self.config.delays[0]),
        );
        **event.text_token_out.borrow_mut() = self.config.cache[self.cache_index(row, 0)];
        for audio_codebook in 0..usize::try_from(self.config.delayed_audio_codebooks).unwrap_or(0) {
            let codebook = audio_codebook + 1;
            let row = self.row(
                self.offset - i64::from(self.config.maximum_delay)
                    + i64::from(self.config.delays[codebook]),
            );
            event.audio_tokens_out.borrow_mut()[audio_codebook] =
                self.config.cache[self.cache_index(row, codebook)];
        }
        Ok(())
    }
    fn effect_publish_output<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        **event.produced_out.borrow_mut() = true;
        self.clear_error(&event.error_out);
        Ok(())
    }
    fn effect_publish_no_output_from_state_output_decision<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        **event.produced_out.borrow_mut() = false;
        self.clear_error(&event.error_out);
        Ok(())
    }
    fn effect_publish_no_output_from_state_output_validation<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        **event.produced_out.borrow_mut() = false;
        self.clear_error(&event.error_out);
        Ok(())
    }
    fn effect_initialize_from_state_errored<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventInitialize<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reset_cache();
        self.clear_error(&event.error_out);
        Ok(())
    }
    fn effect_initialize_from_state_uninitialized<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventInitialize<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reset_cache();
        self.clear_error(&event.error_out);
        Ok(())
    }
    fn effect_reject_detokenize_error_phase_order<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reject_detokenize(event, Error::PhaseOrder);
        Ok(())
    }
    fn effect_reject_detokenize_error_internal_error<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reject_detokenize(e, Error::InternalError);
        Ok(())
    }
    fn effect_reject_detokenize_error_position_overflow_from_state_prepared_full<
        'dispatch,
        'event,
    >(
        &mut self,
        event: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reject_detokenize(event, Error::PositionOverflow);
        Ok(())
    }
    fn effect_reject_detokenize_error_position_overflow_from_state_prepared_generated<
        'dispatch,
        'event,
    >(
        &mut self,
        event: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reject_detokenize(event, Error::PositionOverflow);
        Ok(())
    }
    fn effect_reject_detokenize_error_position_overflow_from_state_prepared_tail<
        'dispatch,
        'event,
    >(
        &mut self,
        event: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reject_detokenize(event, Error::PositionOverflow);
        Ok(())
    }
    fn effect_reject_detokenize_error_request_shape_from_state_prepared_full<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reject_detokenize(event, Error::RequestShape);
        Ok(())
    }
    fn effect_reject_detokenize_error_request_shape_from_state_prepared_generated<
        'dispatch,
        'event,
    >(
        &mut self,
        event: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reject_detokenize(event, Error::RequestShape);
        Ok(())
    }
    fn effect_reject_detokenize_error_request_shape_from_state_prepared_tail<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reject_detokenize(event, Error::RequestShape);
        Ok(())
    }
    fn effect_reject_detokenize_error_uninitialized<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reject_detokenize(event, Error::Uninitialized);
        Ok(())
    }
    fn effect_reject_error_already_initialized_from_state_prepared_full<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventInitialize<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reject_initialize_error(e, Error::AlreadyInitialized);
        Ok(())
    }
    fn effect_reject_error_already_initialized_from_state_prepared_generated<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventInitialize<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reject_initialize_error(e, Error::AlreadyInitialized);
        Ok(())
    }
    fn effect_reject_error_already_initialized_from_state_prepared_tail<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventInitialize<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reject_initialize_error(e, Error::AlreadyInitialized);
        Ok(())
    }
    fn effect_reject_error_already_initialized_from_state_ready<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventInitialize<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reject_initialize_error(e, Error::AlreadyInitialized);
        Ok(())
    }
    fn effect_reject_error_internal_error_event_advance<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventAdvance<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reject_event(&e.error_out, Error::InternalError);
        Ok(())
    }
    fn effect_reject_error_internal_error_event_restore_cache<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventRestoreCache<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reject_event(&e.error_out, Error::InternalError);
        Ok(())
    }
    fn effect_reject_error_internal_error_event_tokenize<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventTokenize<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reject_event(&e.error_out, Error::InternalError);
        Ok(())
    }
    fn effect_reject_error_invalid_configuration_from_state_errored<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventInitialize<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reject_initialize_error(e, Error::InvalidConfiguration);
        Ok(())
    }
    fn effect_reject_error_invalid_configuration_from_state_uninitialized<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventInitialize<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reject_initialize_error(e, Error::InvalidConfiguration);
        Ok(())
    }
    fn effect_reject_error_phase_order_event_advance<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventAdvance<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reject_event(&e.error_out, Error::PhaseOrder);
        Ok(())
    }
    fn effect_reject_error_phase_order_event_restore_cache<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventRestoreCache<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reject_event(&e.error_out, Error::PhaseOrder);
        Ok(())
    }
    fn effect_reject_error_phase_order_event_tokenize<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventTokenize<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reject_event(&e.error_out, Error::PhaseOrder);
        Ok(())
    }
    fn effect_reject_error_position_overflow_event_advance<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventAdvance<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reject_event(&e.error_out, Error::PositionOverflow);
        Ok(())
    }
    fn effect_reject_error_position_overflow_event_tokenize<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventTokenize<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reject_event(&e.error_out, Error::PositionOverflow);
        Ok(())
    }
    fn effect_reject_error_request_shape_event_restore_cache<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventRestoreCache<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reject_event(&e.error_out, Error::RequestShape);
        Ok(())
    }
    fn effect_reject_error_request_shape_event_tokenize<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventTokenize<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reject_event(&e.error_out, Error::RequestShape);
        Ok(())
    }
    fn effect_reject_error_uninitialized_event_advance<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventAdvance<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reject_event(&e.error_out, Error::Uninitialized);
        Ok(())
    }
    fn effect_reject_error_uninitialized_event_reset<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventReset<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reject_event(&e.error_out, Error::Uninitialized);
        Ok(())
    }
    fn effect_reject_error_uninitialized_event_restore_cache<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventRestoreCache<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reject_event(&e.error_out, Error::Uninitialized);
        Ok(())
    }
    fn effect_reject_error_uninitialized_event_tokenize<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventTokenize<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reject_event(&e.error_out, Error::Uninitialized);
        Ok(())
    }
    fn effect_reset_from_state_errored<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventReset<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reset_cache();
        self.clear_error(&e.error_out);
        Ok(())
    }
    fn effect_reset_from_state_prepared_full<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventReset<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reset_cache();
        self.clear_error(&e.error_out);
        Ok(())
    }
    fn effect_reset_from_state_prepared_generated<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventReset<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reset_cache();
        self.clear_error(&e.error_out);
        Ok(())
    }
    fn effect_reset_from_state_prepared_tail<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventReset<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reset_cache();
        self.clear_error(&e.error_out);
        Ok(())
    }
    fn effect_reset_from_state_ready<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventReset<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.reset_cache();
        self.clear_error(&e.error_out);
        Ok(())
    }
    fn effect_restore_column_major_cache<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventRestoreCache<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let rows = usize::try_from(self.config.cache_rows).unwrap_or(0);
        let codebooks = usize::try_from(self.config.codebooks).unwrap_or(0);
        for row in 0..rows {
            for codebook in 0..codebooks {
                self.config.cache[self.cache_index(row, codebook)] =
                    e.column_major_cache[row + codebook * rows];
            }
        }
        self.offset = e.offset;
        self.clear_error(&e.error_out);
        Ok(())
    }
    fn effect_tokenize_empty<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventTokenize<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.model_tokens(self.offset, &mut e.model_tokens_out.borrow_mut());
        self.clear_error(&e.error_out);
        Ok(())
    }
    fn effect_tokenize_full<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventTokenize<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        for codebook in 0..usize::try_from(self.config.codebooks).unwrap_or(0) {
            let row = self.row(self.offset + i64::from(self.config.delays[codebook]));
            self.config.cache[self.cache_index(row, codebook)] = e.audio_tokens[codebook];
        }
        self.model_tokens(self.offset, &mut e.model_tokens_out.borrow_mut());
        self.clear_error(&e.error_out);
        Ok(())
    }
    fn effect_tokenize_tail<'dispatch, 'event>(
        &mut self,
        e: &'dispatch EventTokenize<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let first = usize::try_from(self.config.delayed_audio_codebooks + 1).unwrap_or(0);
        let needed =
            usize::try_from(self.config.codebooks - self.config.delayed_audio_codebooks - 1)
                .unwrap_or(0);
        for tail in 0..needed {
            let codebook = first + tail;
            let row = self.row(self.offset + i64::from(self.config.delays[codebook]));
            self.config.cache[self.cache_index(row, codebook)] = e.audio_tokens[tail];
        }
        self.model_tokens(self.offset, &mut e.model_tokens_out.borrow_mut());
        self.clear_error(&e.error_out);
        Ok(())
    }
    fn effect_unexpected_from_state_errored(&mut self) -> Result<(), ()> {
        self.set_error(Error::InternalError);
        Ok(())
    }
    fn effect_unexpected_from_state_prepared_full(&mut self) -> Result<(), ()> {
        self.set_error(Error::InternalError);
        Ok(())
    }
    fn effect_unexpected_from_state_prepared_generated(&mut self) -> Result<(), ()> {
        self.set_error(Error::InternalError);
        Ok(())
    }
    fn effect_unexpected_from_state_prepared_tail(&mut self) -> Result<(), ()> {
        self.set_error(Error::InternalError);
        Ok(())
    }
    fn effect_unexpected_from_state_ready(&mut self) -> Result<(), ()> {
        self.set_error(Error::InternalError);
        Ok(())
    }
    fn effect_unexpected_from_state_uninitialized(&mut self) -> Result<(), ()> {
        self.set_error(Error::InternalError);
        Ok(())
    }

    fn guard_configuration_valid<'dispatch, 'event>(
        &self,
        _: &'dispatch EventInitialize<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        let c = &self.config;
        if c.codebooks <= 1
            || c.generated_audio_codebooks <= 0
            || c.generated_audio_codebooks >= c.codebooks
            || c.delayed_audio_codebooks <= 0
            || c.delayed_audio_codebooks > c.generated_audio_codebooks
            || c.cache_rows <= 0
            || c.maximum_delay < 0
            || c.initial_delay_frames < 0
            || c.text_initial_token <= 0
            || c.audio_initial_token <= 0
            || c.token_zero >= 0
            || c.token_ungenerated >= 0
            || c.token_zero == c.token_ungenerated
            || c.delays.len() < usize::try_from(c.codebooks).unwrap_or(usize::MAX)
        {
            return Ok(false);
        }
        let needed = c.codebooks - c.delayed_audio_codebooks - 1;
        if needed < 0 || i64::from(c.cache_rows) < i64::from(c.maximum_delay) + 2 {
            return Ok(false);
        }
        let cache_elements = i64::from(c.cache_rows).saturating_mul(i64::from(c.codebooks));
        Ok(cache_elements > 0 && usize::try_from(cache_elements).is_ok_and(|n| c.cache.len() == n))
    }
    fn guard_configuration_invalid<'dispatch, 'event>(
        &self,
        e: &'dispatch EventInitialize<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!self.guard_configuration_valid(e)?)
    }
    fn guard_tokenize_full<'dispatch, 'event>(
        &self,
        e: &'dispatch EventTokenize<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(self.tokenize_shape(e, true) && self.tokens_valid(e) && self.position_available())
    }
    fn guard_tokenize_tail<'dispatch, 'event>(
        &self,
        e: &'dispatch EventTokenize<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(self.tokenize_shape(e, false) && self.tokens_valid(e) && self.position_available())
    }
    fn guard_tokenize_empty<'dispatch, 'event>(
        &self,
        e: &'dispatch EventTokenize<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        let needed = self.config.codebooks - self.config.delayed_audio_codebooks - 1;
        Ok(e.model_tokens_out.borrow().len()
            == usize::try_from(self.config.codebooks).unwrap_or(usize::MAX)
            && needed == 0
            && e.audio_tokens.is_empty()
            && self.tokens_valid(e)
            && self.position_available())
    }
    fn guard_tokenize_invalid<'dispatch, 'event>(
        &self,
        e: &'dispatch EventTokenize<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!((self.tokenize_shape(e, true)
            || self.tokenize_shape(e, false)
            || (self.config.codebooks - self.config.delayed_audio_codebooks - 1 == 0
                && e.audio_tokens.is_empty()
                && e.model_tokens_out.borrow().len()
                    == usize::try_from(self.config.codebooks).unwrap_or(usize::MAX)))
            && self.tokens_valid(e)))
    }
    fn guard_tokenize_position_overflow<'dispatch, 'event>(
        &self,
        e: &'dispatch EventTokenize<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!self.guard_tokenize_invalid(e)? && !self.position_available())
    }
    fn guard_restore_valid<'dispatch, 'event>(
        &self,
        e: &'dispatch EventRestoreCache<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        let c = &self.config;
        let elements = u64::try_from(c.cache_rows)
            .unwrap_or(0)
            .saturating_mul(u64::try_from(c.codebooks).unwrap_or(0));
        if elements != e.column_major_cache.len() as u64
            || e.offset < 0
            || e.offset > i64::MAX - i64::from(c.maximum_delay)
        {
            return Ok(false);
        }
        for index in 0..e.column_major_cache.len() {
            let token = e.column_major_cache[index];
            let codebook = index / usize::try_from(c.cache_rows).unwrap_or(1);
            let upper = if codebook == 0 {
                c.text_initial_token
            } else {
                c.audio_initial_token
            };
            if token != c.token_zero
                && token != c.token_ungenerated
                && (token < 0 || token >= upper)
            {
                return Ok(false);
            }
        }
        Ok(true)
    }
    fn guard_restore_invalid<'dispatch, 'event>(
        &self,
        e: &'dispatch EventRestoreCache<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!self.guard_restore_valid(e)?)
    }
    fn guard_advance_position_available<'dispatch, 'event>(
        &self,
        _: &'dispatch EventAdvance<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(self.offset < i64::MAX)
    }
    fn guard_advance_position_overflow<'dispatch, 'event>(
        &self,
        e: &'dispatch EventAdvance<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!self.guard_advance_position_available(e)?)
    }
    fn guard_detokenize_request_invalid<'dispatch, 'event>(
        &self,
        e: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!self.detokenize_valid(e))
    }
    fn guard_position_overflow<'dispatch, 'event>(
        &self,
        e: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(self.detokenize_valid(e) && !self.guard_position_available(e)?)
    }
    fn guard_detokenize_valid_replace<'dispatch, 'event>(
        &self,
        e: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(self.detokenize_valid(e) && self.guard_position_available(e)? && self.replace_audio())
    }
    fn guard_detokenize_valid_generated<'dispatch, 'event>(
        &self,
        e: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(self.detokenize_valid(e) && self.guard_position_available(e)? && !self.replace_audio())
    }
    fn guard_before_output_delay<'dispatch, 'event>(
        &self,
        _: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(self.offset <= i64::from(self.config.maximum_delay))
    }
    fn guard_past_output_delay<'dispatch, 'event>(
        &self,
        e: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!self.guard_before_output_delay(e)?)
    }
    fn guard_output_complete<'dispatch, 'event>(
        &self,
        e: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!self.output_incomplete(e))
    }
    fn guard_output_incomplete<'dispatch, 'event>(
        &self,
        e: &'dispatch EventDetokenizeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(self.output_incomplete(e))
    }
}

/// Synchronous single-writer wrapper around the generated machine.
pub struct SpeechTokenizerMoshi<'a> {
    machine: SpeechTokenizerMoshiStateMachine<SpeechTokenizerMoshiContext<'a>>,
}
impl<'a> SpeechTokenizerMoshi<'a> {
    #[must_use]
    pub fn new(config: Dependencies<'a>) -> Self {
        Self {
            machine: SpeechTokenizerMoshiStateMachine::new(SpeechTokenizerMoshiContext::new(
                config,
            )),
        }
    }
    /// Keeps the source-compatible by-value event API while SML dispatch borrows it.
    #[allow(clippy::needless_pass_by_value)]
    pub fn process_initialize(&mut self, event: EventInitialize<'_>) -> Result<(), Error> {
        if self
            .machine
            .process_event(SpeechTokenizerMoshiEvents::Initialize(&event))
            .is_err()
        {
            self.machine.context_mut().set_error(Error::InternalError);
        }
        self.result()
    }
    /// Keeps the source-compatible by-value event API while SML dispatch borrows it.
    #[allow(clippy::needless_pass_by_value)]
    pub fn process_tokenize(&mut self, event: EventTokenize<'_>) -> Result<(), Error> {
        if self
            .machine
            .process_event(SpeechTokenizerMoshiEvents::Tokenize(&event))
            .is_err()
        {
            self.machine.context_mut().set_error(Error::InternalError);
        }
        self.result()
    }
    /// Keeps the source-compatible by-value event API while SML dispatch borrows it.
    #[allow(clippy::needless_pass_by_value)]
    pub fn process_detokenize(&mut self, event: EventDetokenizeRun<'_>) -> Result<(), Error> {
        if self
            .machine
            .process_event(SpeechTokenizerMoshiEvents::Detokenize(&event))
            .is_err()
        {
            self.machine.context_mut().set_error(Error::InternalError);
        }
        self.result()
    }
    /// Keeps the source-compatible by-value event API while SML dispatch borrows it.
    #[allow(clippy::needless_pass_by_value)]
    pub fn process_restore_cache(&mut self, event: EventRestoreCache<'_>) -> Result<(), Error> {
        if self
            .machine
            .process_event(SpeechTokenizerMoshiEvents::RestoreCache(&event))
            .is_err()
        {
            self.machine.context_mut().set_error(Error::InternalError);
        }
        self.result()
    }
    /// Keeps the source-compatible by-value event API while SML dispatch borrows it.
    #[allow(clippy::needless_pass_by_value)]
    pub fn process_advance(&mut self, event: EventAdvance<'_>) -> Result<(), Error> {
        if self
            .machine
            .process_event(SpeechTokenizerMoshiEvents::Advance(&event))
            .is_err()
        {
            self.machine.context_mut().set_error(Error::InternalError);
        }
        self.result()
    }
    /// Keeps the source-compatible by-value event API while SML dispatch borrows it.
    #[allow(clippy::needless_pass_by_value)]
    pub fn process_reset(&mut self, event: EventReset<'_>) -> Result<(), Error> {
        if self
            .machine
            .process_event(SpeechTokenizerMoshiEvents::Reset(&event))
            .is_err()
        {
            self.machine.context_mut().set_error(Error::InternalError);
        }
        self.result()
    }
    fn result(&self) -> Result<(), Error> {
        let error = self.machine.context().last_error;
        if error == Error::None {
            Ok(())
        } else {
            Err(error)
        }
    }
    #[must_use]
    pub fn state(&self) -> &SpeechTokenizerMoshiStates {
        self.machine.state()
    }
    #[must_use]
    pub fn context(&self) -> &SpeechTokenizerMoshiContext<'a> {
        self.machine.context()
    }
    #[must_use]
    pub fn is(&self, state: &SpeechTokenizerMoshiStates) -> bool {
        self.machine.is(state)
    }
    pub fn initialize(&mut self, event: EventInitialize<'_>) -> Result<(), Error> {
        self.process_initialize(event)
    }
    pub fn tokenize(&mut self, event: EventTokenize<'_>) -> Result<(), Error> {
        self.process_tokenize(event)
    }
    pub fn detokenize(&mut self, event: EventDetokenizeRun<'_>) -> Result<(), Error> {
        self.process_detokenize(event)
    }
    pub fn restore_cache(&mut self, event: EventRestoreCache<'_>) -> Result<(), Error> {
        self.process_restore_cache(event)
    }
    pub fn advance(&mut self, event: EventAdvance<'_>) -> Result<(), Error> {
        self.process_advance(event)
    }
    pub fn reset(&mut self, event: EventReset<'_>) -> Result<(), Error> {
        self.process_reset(event)
    }
    fn dispatch<'dispatch, 'event>(
        &mut self,
        event: SpeechTokenizerMoshiEvents<'dispatch, 'event>,
    ) -> Result<(), Error>
    where
        'event: 'dispatch,
    {
        if self.machine.process_event(event).is_err() {
            self.machine.context_mut().set_error(Error::InternalError);
            return Err(Error::InternalError);
        }
        self.result()
    }
}
/// Public typed event union for the maintained Moshi tokenizer owner.
pub enum TokenizerEvent<'event> {
    Initialize(EventInitialize<'event>),
    Tokenize(EventTokenize<'event>),
    Detokenize(EventDetokenizeRun<'event>),
    RestoreCache(EventRestoreCache<'event>),
    Advance(EventAdvance<'event>),
    Reset(EventReset<'event>),
}

/// Dispatches one caller-owned Moshi tokenizer event synchronously.
///
/// This alias is intentionally the existing tokenizer actor: the event union
/// keeps model-domain consumers on a public typed boundary while preserving
/// caller-owned buffers and the generated machine's RTC semantics.
impl SpeechTokenizerMoshi<'_> {
    pub fn process_event(&mut self, event: TokenizerEvent<'_>) -> Result<(), Error> {
        match event {
            TokenizerEvent::Initialize(event) => self.process_initialize(event),
            TokenizerEvent::Tokenize(event) => self.process_tokenize(event),
            TokenizerEvent::Detokenize(event) => self.process_detokenize(event),
            TokenizerEvent::RestoreCache(event) => self.process_restore_cache(event),
            TokenizerEvent::Advance(event) => self.process_advance(event),
            TokenizerEvent::Reset(event) => self.process_reset(event),
        }
    }
}

#[cfg(test)]
mod owner_tests {
    use super::*;

    #[test]
    fn public_event_owner_publishes_uninitialized_error() {
        let delays = [0_i32; 2];
        let mut cache = [0_i32; 4];
        let dependencies = Dependencies::new(&delays, &mut cache, 2, 1, 1, 2, 0, 0, 1, 1, -1, -2);
        let mut actor = SpeechTokenizerMoshi::new(dependencies);
        let audio = [0_i32];
        let mut model = [0_i32; 2];
        let mut error = Error::None;
        assert_eq!(
            actor.process_event(TokenizerEvent::Tokenize(EventTokenize::new(
                &audio, &mut model, &mut error,
            ))),
            Err(Error::Uninitialized)
        );
        assert_eq!(error, Error::Uninitialized);
        assert!(actor.is(&SpeechTokenizerMoshiStates::StateUninitialized));
    }
}
