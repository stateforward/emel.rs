//! Source-aligned bounded fallback tokenizer preprocessor actor.
//!
//! The machine mirrors `fallback/sm.hpp`: request validation, special-token
//! preparation, partitioning, result publication, and explicit unexpected-event
//! handling are all run to completion in one synchronous dispatch.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::derive_partial_eq_without_eq,
    clippy::empty_structs_with_brackets,
    clippy::manual_memcpy,
    clippy::missing_const_for_fn,
    clippy::module_name_repetitions,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::large_stack_arrays,
    clippy::large_stack_frames,
    clippy::too_many_lines,
    clippy::needless_pass_by_value,
    clippy::too_many_arguments,
    clippy::items_after_statements,
    clippy::large_types_passed_by_value,
    dead_code,
    missing_docs,
    private_interfaces,
    unpredictable_function_pointer_comparisons
)]

use sml::sml;

/// Token classes retained by the fallback special-token cache, matching the
/// pinned preprocessor detail policy.
const TOKEN_TYPE_UNKNOWN: i32 = 2;
const TOKEN_TYPE_CONTROL: i32 = 3;
const TOKEN_TYPE_USER_DEFINED: i32 = 4;

fn token_is_special_type(type_id: i32) -> bool {
    matches!(
        type_id,
        TOKEN_TYPE_UNKNOWN | TOKEN_TYPE_CONTROL | TOKEN_TYPE_USER_DEFINED
    )
}

/// Matches the pinned `std::isspace(unsigned char)` checks in the C++ detail
/// implementation without consulting locale-specific Unicode classification.
fn source_is_space(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | b'\n' | b'\x0b' | b'\x0c' | b'\r')
}

/// Maximum number of output fragments accepted by the pinned contract.
pub const MAX_FRAGMENTS: usize = 1024;
/// Maximum number of cached special tokens accepted by the pinned contract.
pub const MAX_SPECIAL_TOKENS: usize = 1024;
/// Maximum copied request text. Oversized requests are rejected before dispatch.
pub const MAX_TEXT_BYTES: usize = 4096;
/// Maximum bytes in one copied special-token spelling.
pub const MAX_SPECIAL_BYTES: usize = 256;

/// Preprocessor result/error codes. Values mirror `preprocessor::error`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum PreprocessError {
    #[default]
    None = 0,
    InvalidRequest = 1,
    BackendError = 2,
    Unknown = 255,
}

impl PreprocessError {
    /// Returns the public integer error code.
    #[must_use]
    pub const fn code(self) -> i32 {
        match self {
            Self::None => 0,
            Self::InvalidRequest => 1,
            Self::BackendError | Self::Unknown => 2,
        }
    }
}

/// Output fragment kind.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum FragmentKind {
    #[default]
    RawText = 0,
    Token = 1,
}

/// A bounded output fragment. Raw text is represented by a byte range into the
/// copied request text; token fragments carry a non-negative token ID.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Fragment {
    pub kind: FragmentKind,
    pub start: u16,
    pub end: u16,
    pub token: i32,
}

/// One bounded special-token cache entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpecialToken {
    pub text: [u8; MAX_SPECIAL_BYTES],
    pub len: u16,
    pub token: i32,
    pub type_id: i32,
    pub lstrip: bool,
    pub rstrip: bool,
}

impl Default for SpecialToken {
    fn default() -> Self {
        Self {
            text: [0; MAX_SPECIAL_BYTES],
            len: 0,
            token: -1,
            type_id: 0,
            lstrip: false,
            rstrip: false,
        }
    }
}

impl SpecialToken {
    /// Creates a cache entry, rejecting spellings that do not fit the bound.
    #[must_use]
    pub fn new(text: &[u8], token: i32, type_id: i32, lstrip: bool, rstrip: bool) -> Option<Self> {
        if text.is_empty() || text.len() > MAX_SPECIAL_BYTES {
            return None;
        }
        let mut out = Self {
            token,
            type_id,
            lstrip,
            rstrip,
            ..Self::default()
        };
        out.text[..text.len()].copy_from_slice(text);
        out.len = text.len() as u16;
        Some(out)
    }

    fn bytes(&self) -> &[u8] {
        &self.text[..self.len as usize]
    }
}

/// Copied, bounded input for one synchronous dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreprocessInput {
    pub text: [u8; MAX_TEXT_BYTES],
    pub text_len: u16,
    pub parse_special: bool,
    pub fragments_capacity: u16,
    pub specials: [SpecialToken; MAX_SPECIAL_TOKENS],
    pub special_count: u16,
    pub on_done: Option<fn(&PreprocessResult)>,
    pub on_error: Option<fn(&PreprocessResult)>,
}

impl Default for PreprocessInput {
    fn default() -> Self {
        Self {
            text: [0; MAX_TEXT_BYTES],
            text_len: 0,
            parse_special: false,
            fragments_capacity: 0,
            specials: [SpecialToken::default(); MAX_SPECIAL_TOKENS],
            special_count: 0,
            on_done: None,
            on_error: None,
        }
    }
}

impl PreprocessInput {
    /// Copies a request text and output capacity into the bounded event.
    #[must_use]
    pub fn new(text: &[u8], fragments_capacity: usize, parse_special: bool) -> Option<Self> {
        if text.len() > MAX_TEXT_BYTES || fragments_capacity > u16::MAX as usize {
            return None;
        }
        let mut out = Self {
            parse_special,
            fragments_capacity: fragments_capacity as u16,
            ..Self::default()
        };
        out.text[..text.len()].copy_from_slice(text);
        out.text_len = text.len() as u16;
        Some(out)
    }

    /// Adds a bounded special token to this request.
    pub fn add_special(&mut self, token: SpecialToken) -> bool {
        let idx = self.special_count as usize;
        if idx >= MAX_SPECIAL_TOKENS {
            return false;
        }
        self.specials[idx] = token;
        self.special_count += 1;
        true
    }

    /// Installs synchronous completion callbacks.
    pub fn callbacks(
        &mut self,
        on_done: Option<fn(&PreprocessResult)>,
        on_error: Option<fn(&PreprocessResult)>,
    ) {
        self.on_done = on_done;
        self.on_error = on_error;
    }
}

/// Bounded output returned by the actor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreprocessResult {
    pub fragments: [Fragment; MAX_FRAGMENTS],
    pub fragment_count: u16,
    pub preprocessed: bool,
    pub error: PreprocessError,
}

impl Default for PreprocessResult {
    fn default() -> Self {
        Self {
            fragments: [Fragment::default(); MAX_FRAGMENTS],
            fragment_count: 0,
            preprocessed: false,
            error: PreprocessError::None,
        }
    }
}

/// Runtime event corresponding to `event::preprocess_runtime`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct EventPreprocessRuntime {
    pub input: PreprocessInput,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CachedSpecials {
    entries: [SpecialToken; MAX_SPECIAL_TOKENS],
    count: usize,
}

impl Default for CachedSpecials {
    fn default() -> Self {
        Self {
            entries: [SpecialToken::default(); MAX_SPECIAL_TOKENS],
            count: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct ScratchFragment {
    kind: FragmentKind,
    start: usize,
    end: usize,
    token: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Context {
    input: PreprocessInput,
    result: PreprocessResult,
    phase_error: PreprocessError,
    specials: CachedSpecials,
    current: [ScratchFragment; MAX_FRAGMENTS],
    next: [ScratchFragment; MAX_FRAGMENTS],
    current_count: usize,
}

impl Default for Context {
    fn default() -> Self {
        Self {
            input: PreprocessInput::default(),
            result: PreprocessResult::default(),
            phase_error: PreprocessError::None,
            specials: CachedSpecials::default(),
            current: [ScratchFragment::default(); MAX_FRAGMENTS],
            next: [ScratchFragment::default(); MAX_FRAGMENTS],
            current_count: 0,
        }
    }
}

impl Context {
    fn set_input(&mut self, input: PreprocessInput) {
        self.input = input;
    }
    fn text(&self) -> &[u8] {
        &self.input.text[..self.input.text_len as usize]
    }
    fn push_current(&mut self, fragment: ScratchFragment) -> bool {
        if self.current_count >= self.input.fragments_capacity as usize
            || self.current_count >= MAX_FRAGMENTS
        {
            return false;
        }
        self.current[self.current_count] = fragment;
        self.current_count += 1;
        true
    }
    fn push_next(&mut self, fragment: ScratchFragment, count: &mut usize) -> bool {
        if *count >= self.input.fragments_capacity as usize || *count >= MAX_FRAGMENTS {
            return false;
        }
        self.next[*count] = fragment;
        *count += 1;
        true
    }
    fn special_allowed(&self, token: &SpecialToken) -> bool {
        self.input.parse_special
            || (token.type_id != TOKEN_TYPE_CONTROL && token.type_id != TOKEN_TYPE_UNKNOWN)
    }
    fn publish(&mut self) {
        self.result.fragments = [Fragment::default(); MAX_FRAGMENTS];
        self.result.fragment_count = self.current_count as u16;
        self.result.preprocessed = true;
        self.result.error = PreprocessError::None;
        for idx in 0..self.current_count {
            let f = self.current[idx];
            self.result.fragments[idx] = Fragment {
                kind: f.kind,
                start: f.start as u16,
                end: f.end as u16,
                token: f.token,
            };
        }
    }
}

sml! {
    TextTokenizerPreprocessorFallback {
        "request_buffer_decision"_s <= *"idle"_s + event<EventPreprocessRuntime>,
        "request_buffer_decision"_s <= "done"_s + event<EventPreprocessRuntime>,
        "request_buffer_decision"_s <= "errored"_s + event<EventPreprocessRuntime>,
        "request_buffer_decision"_s <= "unexpected"_s + event<EventPreprocessRuntime>,
        "request_capacity_nonzero_decision"_s <= "request_buffer_decision"_s + completion<EventPreprocessRuntime> [fragments_buffer_present],
        "errored"_s <= "request_buffer_decision"_s + completion<EventPreprocessRuntime> [fragments_buffer_missing] / reject_invalid_from_request_buffer_decision,
        "errored"_s <= "request_buffer_decision"_s + completion<EventPreprocessRuntime> / reject_invalid_from_request_buffer_decision,
        "request_capacity_limit_decision"_s <= "request_capacity_nonzero_decision"_s + completion<EventPreprocessRuntime> [fragments_capacity_nonzero],
        "errored"_s <= "request_capacity_nonzero_decision"_s + completion<EventPreprocessRuntime> [fragments_capacity_zero] / reject_invalid_from_request_capacity_nonzero_decision,
        "errored"_s <= "request_capacity_nonzero_decision"_s + completion<EventPreprocessRuntime> / reject_invalid_from_request_capacity_nonzero_decision,
        "preparing"_s <= "request_capacity_limit_decision"_s + completion<EventPreprocessRuntime> [fragments_capacity_within_limit] / begin_preprocess,
        "errored"_s <= "request_capacity_limit_decision"_s + completion<EventPreprocessRuntime> [fragments_capacity_exceeds_limit] / reject_invalid_from_request_capacity_limit_decision,
        "errored"_s <= "request_capacity_limit_decision"_s + completion<EventPreprocessRuntime> / reject_invalid_from_request_capacity_limit_decision,
        "build_specials_decision"_s <= "preparing"_s + completion<EventPreprocessRuntime> / build_specials,
        "partition_specials_decision"_s <= "build_specials_decision"_s + completion<EventPreprocessRuntime> [build_specials_ok],
        "errored"_s <= "build_specials_decision"_s + completion<EventPreprocessRuntime> [build_specials_invalid_request_error] / ensure_last_error_from_build_specials_decision,
        "errored"_s <= "build_specials_decision"_s + completion<EventPreprocessRuntime> [build_specials_backend_error] / ensure_last_error_from_build_specials_decision,
        "errored"_s <= "build_specials_decision"_s + completion<EventPreprocessRuntime> [build_specials_unknown_error] / ensure_last_error_from_build_specials_decision,
        "partitioning_no_specials_input_decision"_s <= "partition_specials_decision"_s + completion<EventPreprocessRuntime> [no_specials],
        "partition_parse_special_decision"_s <= "partition_specials_decision"_s + completion<EventPreprocessRuntime> [has_specials],
        "errored"_s <= "partition_specials_decision"_s + completion<EventPreprocessRuntime> / ensure_last_error_from_partition_specials_decision,
        "partitioning_non_bpe_parse_input_decision"_s <= "partition_parse_special_decision"_s + completion<EventPreprocessRuntime> [parse_special_enabled],
        "partitioning_non_bpe_skip_input_decision"_s <= "partition_parse_special_decision"_s + completion<EventPreprocessRuntime> [parse_special_disabled],
        "errored"_s <= "partition_parse_special_decision"_s + completion<EventPreprocessRuntime> / ensure_last_error_from_partition_parse_special_decision,
        "partition_decision"_s <= "partitioning_no_specials_input_decision"_s + completion<EventPreprocessRuntime> [request_text_empty] / set_empty_partition_result_from_partitioning_no_specials_input_decision,
        "partitioning_no_specials"_s <= "partitioning_no_specials_input_decision"_s + completion<EventPreprocessRuntime> [request_text_nonempty],
        "errored"_s <= "partitioning_no_specials_input_decision"_s + completion<EventPreprocessRuntime> / ensure_last_error_from_partitioning_no_specials_input_decision,
        "partition_decision"_s <= "partitioning_non_bpe_parse_input_decision"_s + completion<EventPreprocessRuntime> [request_text_empty] / set_empty_partition_result_from_partitioning_non_bpe_parse_input_decision,
        "partitioning_non_bpe_parse_special"_s <= "partitioning_non_bpe_parse_input_decision"_s + completion<EventPreprocessRuntime> [request_text_nonempty],
        "errored"_s <= "partitioning_non_bpe_parse_input_decision"_s + completion<EventPreprocessRuntime> / ensure_last_error_from_partitioning_non_bpe_parse_input_decision,
        "partition_decision"_s <= "partitioning_non_bpe_skip_input_decision"_s + completion<EventPreprocessRuntime> [request_text_empty] / set_empty_partition_result_from_partitioning_non_bpe_skip_input_decision,
        "partitioning_non_bpe_skip_special"_s <= "partitioning_non_bpe_skip_input_decision"_s + completion<EventPreprocessRuntime> [request_text_nonempty],
        "errored"_s <= "partitioning_non_bpe_skip_input_decision"_s + completion<EventPreprocessRuntime> / ensure_last_error_from_partitioning_non_bpe_skip_input_decision,
        "partition_decision"_s <= "partitioning_no_specials"_s + completion<EventPreprocessRuntime> / partition_no_specials,
        "partition_decision"_s <= "partitioning_non_bpe_parse_special"_s + completion<EventPreprocessRuntime> / partition_non_bpe_parse_special,
        "partition_decision"_s <= "partitioning_non_bpe_skip_special"_s + completion<EventPreprocessRuntime> / partition_non_bpe_skip_special,
        "done"_s <= "partition_decision"_s + completion<EventPreprocessRuntime> [partition_ok] / mark_done,
        "errored"_s <= "partition_decision"_s + completion<EventPreprocessRuntime> [partition_invalid_request_error] / ensure_last_error_from_partition_decision,
        "errored"_s <= "partition_decision"_s + completion<EventPreprocessRuntime> [partition_backend_error] / ensure_last_error_from_partition_decision,
        "errored"_s <= "partition_decision"_s + completion<EventPreprocessRuntime> [partition_unknown_error] / ensure_last_error_from_partition_decision,
        "unexpected"_s <= "idle"_s + unexpected_event<_> / on_unexpected_from_idle,
        "unexpected"_s <= "request_buffer_decision"_s + unexpected_event<_> / on_unexpected_from_request_buffer_decision,
        "unexpected"_s <= "request_capacity_nonzero_decision"_s + unexpected_event<_> / on_unexpected_from_request_capacity_nonzero_decision,
        "unexpected"_s <= "request_capacity_limit_decision"_s + unexpected_event<_> / on_unexpected_from_request_capacity_limit_decision,
        "unexpected"_s <= "preparing"_s + unexpected_event<_> / on_unexpected_from_preparing,
        "unexpected"_s <= "build_specials_decision"_s + unexpected_event<_> / on_unexpected_from_build_specials_decision,
        "unexpected"_s <= "partition_specials_decision"_s + unexpected_event<_> / on_unexpected_from_partition_specials_decision,
        "unexpected"_s <= "partition_parse_special_decision"_s + unexpected_event<_> / on_unexpected_from_partition_parse_special_decision,
        "unexpected"_s <= "partitioning_no_specials_input_decision"_s + unexpected_event<_> / on_unexpected_from_partitioning_no_specials_input_decision,
        "unexpected"_s <= "partitioning_non_bpe_parse_input_decision"_s + unexpected_event<_> / on_unexpected_from_partitioning_non_bpe_parse_input_decision,
        "unexpected"_s <= "partitioning_non_bpe_skip_input_decision"_s + unexpected_event<_> / on_unexpected_from_partitioning_non_bpe_skip_input_decision,
        "unexpected"_s <= "partitioning_no_specials"_s + unexpected_event<_> / on_unexpected_from_partitioning_no_specials,
        "unexpected"_s <= "partitioning_non_bpe_parse_special"_s + unexpected_event<_> / on_unexpected_from_partitioning_non_bpe_parse_special,
        "unexpected"_s <= "partitioning_non_bpe_skip_special"_s + unexpected_event<_> / on_unexpected_from_partitioning_non_bpe_skip_special,
        "unexpected"_s <= "partition_decision"_s + unexpected_event<_> / on_unexpected_from_partition_decision,
        "unexpected"_s <= "done"_s + unexpected_event<_> / on_unexpected_from_done,
        "unexpected"_s <= "errored"_s + unexpected_event<_> / on_unexpected_from_errored,
        "unexpected"_s <= "unexpected"_s + unexpected_event<_> / on_unexpected_from_unexpected,
    }
}

impl TextTokenizerPreprocessorFallbackStateMachineContext for Context {
    fn begin_preprocess(&mut self, _: &EventPreprocessRuntime) -> Result<(), ()> {
        self.result = PreprocessResult::default();
        self.phase_error = if self.input.text_len as usize <= MAX_TEXT_BYTES {
            PreprocessError::None
        } else {
            PreprocessError::InvalidRequest
        };
        self.current_count = 0;
        self.specials = CachedSpecials::default();
        Ok(())
    }
    fn build_specials(&mut self, _: &EventPreprocessRuntime) -> Result<(), ()> {
        if self.phase_error != PreprocessError::None {
            return Ok(());
        }
        let count = self.input.special_count as usize;
        if count > MAX_SPECIAL_TOKENS {
            self.phase_error = PreprocessError::InvalidRequest;
            return Ok(());
        }
        self.specials.count = 0;
        for token in self.input.specials[..count].iter().copied() {
            if token.len == 0 {
                continue;
            }
            if token.len as usize > MAX_SPECIAL_BYTES {
                self.phase_error = PreprocessError::InvalidRequest;
                return Ok(());
            }
            if !token_is_special_type(token.type_id) {
                continue;
            }
            if self.specials.count >= MAX_SPECIAL_TOKENS {
                self.phase_error = PreprocessError::InvalidRequest;
                return Ok(());
            }
            self.specials.entries[self.specials.count] = token;
            self.specials.count += 1;
        }
        for i in 1..self.specials.count {
            let token = self.specials.entries[i];
            let mut j = i;
            while j > 0 && self.specials.entries[j - 1].len < token.len {
                self.specials.entries[j] = self.specials.entries[j - 1];
                j -= 1;
            }
            self.specials.entries[j] = token;
        }
        self.phase_error = PreprocessError::None;
        Ok(())
    }
    fn build_specials_backend_error(&self, _: &EventPreprocessRuntime) -> Result<bool, ()> {
        Ok(self.phase_error == PreprocessError::BackendError)
    }
    fn build_specials_invalid_request_error(&self, _: &EventPreprocessRuntime) -> Result<bool, ()> {
        Ok(self.phase_error == PreprocessError::InvalidRequest)
    }
    fn build_specials_ok(&self, _: &EventPreprocessRuntime) -> Result<bool, ()> {
        Ok(self.phase_error == PreprocessError::None)
    }
    fn build_specials_unknown_error(&self, _: &EventPreprocessRuntime) -> Result<bool, ()> {
        Ok(self.phase_error == PreprocessError::Unknown)
    }
    fn ensure_last_error_from_build_specials_decision(
        &mut self,
        _: &EventPreprocessRuntime,
    ) -> Result<(), ()> {
        self.ensure_error();
        Ok(())
    }
    fn ensure_last_error_from_partition_decision(
        &mut self,
        _: &EventPreprocessRuntime,
    ) -> Result<(), ()> {
        self.ensure_error();
        Ok(())
    }
    fn ensure_last_error_from_partition_parse_special_decision(
        &mut self,
        _: &EventPreprocessRuntime,
    ) -> Result<(), ()> {
        self.ensure_error();
        Ok(())
    }
    fn ensure_last_error_from_partition_specials_decision(
        &mut self,
        _: &EventPreprocessRuntime,
    ) -> Result<(), ()> {
        self.ensure_error();
        Ok(())
    }
    fn ensure_last_error_from_partitioning_no_specials_input_decision(
        &mut self,
        _: &EventPreprocessRuntime,
    ) -> Result<(), ()> {
        self.ensure_error();
        Ok(())
    }
    fn ensure_last_error_from_partitioning_non_bpe_parse_input_decision(
        &mut self,
        _: &EventPreprocessRuntime,
    ) -> Result<(), ()> {
        self.ensure_error();
        Ok(())
    }
    fn ensure_last_error_from_partitioning_non_bpe_skip_input_decision(
        &mut self,
        _: &EventPreprocessRuntime,
    ) -> Result<(), ()> {
        self.ensure_error();
        Ok(())
    }
    fn fragments_buffer_missing(&self, _: &EventPreprocessRuntime) -> Result<bool, ()> {
        Ok(self.input.fragments_capacity == 0)
    }
    fn fragments_buffer_present(&self, _: &EventPreprocessRuntime) -> Result<bool, ()> {
        Ok(self.input.fragments_capacity != 0)
    }
    fn fragments_capacity_exceeds_limit(&self, _: &EventPreprocessRuntime) -> Result<bool, ()> {
        Ok(self.input.fragments_capacity as usize > MAX_FRAGMENTS)
    }
    fn fragments_capacity_nonzero(&self, _: &EventPreprocessRuntime) -> Result<bool, ()> {
        Ok(self.input.fragments_capacity != 0)
    }
    fn fragments_capacity_within_limit(&self, _: &EventPreprocessRuntime) -> Result<bool, ()> {
        Ok((self.input.fragments_capacity as usize) <= MAX_FRAGMENTS)
    }
    fn fragments_capacity_zero(&self, _: &EventPreprocessRuntime) -> Result<bool, ()> {
        Ok(self.input.fragments_capacity == 0)
    }
    fn has_specials(&self, _: &EventPreprocessRuntime) -> Result<bool, ()> {
        Ok(self.specials.count != 0)
    }
    fn no_specials(&self, _: &EventPreprocessRuntime) -> Result<bool, ()> {
        Ok(self.specials.count == 0)
    }
    fn parse_special_disabled(&self, _: &EventPreprocessRuntime) -> Result<bool, ()> {
        Ok(!self.input.parse_special)
    }
    fn parse_special_enabled(&self, _: &EventPreprocessRuntime) -> Result<bool, ()> {
        Ok(self.input.parse_special)
    }
    fn partition_backend_error(&self, _: &EventPreprocessRuntime) -> Result<bool, ()> {
        Ok(self.phase_error == PreprocessError::BackendError)
    }
    fn partition_invalid_request_error(&self, _: &EventPreprocessRuntime) -> Result<bool, ()> {
        Ok(self.phase_error == PreprocessError::InvalidRequest)
    }
    fn partition_unknown_error(&self, _: &EventPreprocessRuntime) -> Result<bool, ()> {
        Ok(self.phase_error == PreprocessError::Unknown)
    }
    fn partition_ok(&self, _: &EventPreprocessRuntime) -> Result<bool, ()> {
        Ok(self.phase_error == PreprocessError::None)
    }
    fn request_text_empty(&self, _: &EventPreprocessRuntime) -> Result<bool, ()> {
        Ok(self.input.text_len == 0)
    }
    fn request_text_nonempty(&self, _: &EventPreprocessRuntime) -> Result<bool, ()> {
        Ok(self.input.text_len != 0)
    }
    fn reject_invalid_from_request_buffer_decision(
        &mut self,
        _: &EventPreprocessRuntime,
    ) -> Result<(), ()> {
        self.fail(PreprocessError::InvalidRequest);
        Ok(())
    }
    fn reject_invalid_from_request_capacity_limit_decision(
        &mut self,
        _: &EventPreprocessRuntime,
    ) -> Result<(), ()> {
        self.fail(PreprocessError::InvalidRequest);
        Ok(())
    }
    fn reject_invalid_from_request_capacity_nonzero_decision(
        &mut self,
        _: &EventPreprocessRuntime,
    ) -> Result<(), ()> {
        self.fail(PreprocessError::InvalidRequest);
        Ok(())
    }
    fn set_empty_partition_result_from_partitioning_no_specials_input_decision(
        &mut self,
        _: &EventPreprocessRuntime,
    ) -> Result<(), ()> {
        self.current_count = 0;
        self.phase_error = PreprocessError::None;
        Ok(())
    }
    fn set_empty_partition_result_from_partitioning_non_bpe_parse_input_decision(
        &mut self,
        _: &EventPreprocessRuntime,
    ) -> Result<(), ()> {
        self.current_count = 0;
        self.phase_error = PreprocessError::None;
        Ok(())
    }
    fn set_empty_partition_result_from_partitioning_non_bpe_skip_input_decision(
        &mut self,
        _: &EventPreprocessRuntime,
    ) -> Result<(), ()> {
        self.current_count = 0;
        self.phase_error = PreprocessError::None;
        Ok(())
    }
    fn partition_no_specials(&mut self, _: &EventPreprocessRuntime) -> Result<(), ()> {
        let len = self.input.text_len as usize;
        if !self.push_current(ScratchFragment {
            kind: FragmentKind::RawText,
            start: 0,
            end: len,
            token: -1,
        }) {
            self.phase_error = PreprocessError::InvalidRequest;
        }
        Ok(())
    }
    fn partition_non_bpe_parse_special(&mut self, _: &EventPreprocessRuntime) -> Result<(), ()> {
        self.partition_with_specials(true);
        Ok(())
    }
    fn partition_non_bpe_skip_special(&mut self, _: &EventPreprocessRuntime) -> Result<(), ()> {
        self.partition_with_specials(false);
        Ok(())
    }
    fn mark_done(&mut self, _: &EventPreprocessRuntime) -> Result<(), ()> {
        self.publish();
        Ok(())
    }
    fn on_unexpected_from_build_specials_decision(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_done(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_errored(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_idle(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_partition_decision(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_partition_parse_special_decision(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_partition_specials_decision(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_partitioning_no_specials(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_partitioning_no_specials_input_decision(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_partitioning_non_bpe_parse_input_decision(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_partitioning_non_bpe_parse_special(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_partitioning_non_bpe_skip_input_decision(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_partitioning_non_bpe_skip_special(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_preparing(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_request_buffer_decision(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_request_capacity_limit_decision(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_request_capacity_nonzero_decision(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
    fn on_unexpected_from_unexpected(&mut self) -> Result<(), ()> {
        self.unexpected();
        Ok(())
    }
}

impl Context {
    fn ensure_error(&mut self) {
        self.result.error = if self.phase_error == PreprocessError::None {
            PreprocessError::BackendError
        } else {
            self.phase_error
        };
        self.result.preprocessed = false;
        self.result.fragment_count = 0;
    }
    fn fail(&mut self, error: PreprocessError) {
        self.phase_error = error;
        self.ensure_error();
    }
    fn unexpected(&mut self) {
        self.fail(PreprocessError::InvalidRequest);
    }
    fn partition_with_specials(&mut self, parse: bool) {
        let len = self.input.text_len as usize;
        self.current_count = 0;
        if !self.push_current(ScratchFragment {
            kind: FragmentKind::RawText,
            start: 0,
            end: len,
            token: -1,
        }) {
            self.phase_error = PreprocessError::InvalidRequest;
            return;
        }
        for token_idx in 0..self.specials.count {
            let token = self.specials.entries[token_idx];
            if !parse && !self.special_allowed(&token) {
                continue;
            }
            let needle = token.bytes();
            if needle.is_empty() {
                continue;
            }
            let prior_count = self.current_count;
            let mut next_count = 0usize;
            for frag_idx in 0..prior_count {
                let frag = self.current[frag_idx];
                if frag.kind == FragmentKind::Token {
                    if frag.token < 0 || !self.push_next(frag, &mut next_count) {
                        self.phase_error = PreprocessError::InvalidRequest;
                        return;
                    }
                    continue;
                }
                let mut base = frag.start;
                while base < frag.end {
                    let mut found = None;
                    let mut pos = base;
                    while pos + needle.len() <= frag.end {
                        if self.text()[pos..pos + needle.len()] == *needle {
                            found = Some(pos);
                            break;
                        }
                        pos += 1;
                    }
                    let Some(match_pos) = found else {
                        if !self.push_next(
                            ScratchFragment {
                                kind: FragmentKind::RawText,
                                start: base,
                                end: frag.end,
                                token: -1,
                            },
                            &mut next_count,
                        ) {
                            self.phase_error = PreprocessError::InvalidRequest;
                            return;
                        }
                        break;
                    };
                    let mut left_end = match_pos;
                    if token.lstrip {
                        while left_end > base && source_is_space(self.text()[left_end - 1]) {
                            left_end -= 1;
                        }
                    }
                    if left_end > base
                        && !self.push_next(
                            ScratchFragment {
                                kind: FragmentKind::RawText,
                                start: base,
                                end: left_end,
                                token: -1,
                            },
                            &mut next_count,
                        )
                    {
                        self.phase_error = PreprocessError::InvalidRequest;
                        return;
                    }
                    if token.token < 0
                        || !self.push_next(
                            ScratchFragment {
                                kind: FragmentKind::Token,
                                start: 0,
                                end: 0,
                                token: token.token,
                            },
                            &mut next_count,
                        )
                    {
                        self.phase_error = PreprocessError::InvalidRequest;
                        return;
                    }
                    base = match_pos + needle.len();
                    if token.rstrip {
                        while base < frag.end && source_is_space(self.text()[base]) {
                            base += 1;
                        }
                    }
                }
            }
            self.current_count = next_count;
            self.current[..next_count].copy_from_slice(&self.next[..next_count]);
        }
        self.phase_error = PreprocessError::None;
    }
}

/// Synchronous bounded actor around the generated fallback machine.
pub struct TextTokenizerPreprocessorFallbackActor {
    machine: TextTokenizerPreprocessorFallbackStateMachine<Context>,
}
impl Default for TextTokenizerPreprocessorFallbackActor {
    fn default() -> Self {
        Self::new()
    }
}
impl TextTokenizerPreprocessorFallbackActor {
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: TextTokenizerPreprocessorFallbackStateMachine::new(Context::default()),
        }
    }
    pub fn process_event(&mut self, input: PreprocessInput) -> PreprocessResult {
        self.machine.context_mut().set_input(input);
        let event = EventPreprocessRuntime { input };
        if self
            .machine
            .process_event(TextTokenizerPreprocessorFallbackEvents::EventPreprocessRuntime(event))
            .is_err()
        {
            self.machine.context_mut().unexpected();
        }
        let result = self.machine.context().result;
        if result.error == PreprocessError::None {
            if let Some(callback) = input.on_done {
                callback(&result);
            }
        } else if let Some(callback) = input.on_error {
            callback(&result);
        }
        result
    }
    pub fn process_unexpected(&mut self) -> PreprocessResult {
        self.machine.context_mut().unexpected();
        self.machine
            .set_state(TextTokenizerPreprocessorFallbackStates::Unexpected);
        self.machine.context().result
    }
    #[must_use]
    pub fn state(&self) -> &TextTokenizerPreprocessorFallbackStates {
        self.machine.state()
    }
    #[must_use]
    pub fn is(&self, state: &TextTokenizerPreprocessorFallbackStates) -> bool {
        self.machine.is(state)
    }
    #[must_use]
    pub fn result(&self) -> &PreprocessResult {
        &self.machine.context().result
    }
    #[must_use]
    pub fn last_error(&self) -> i32 {
        self.machine.context().result.error.code()
    }
    #[must_use]
    pub fn fragment_count(&self) -> usize {
        self.machine.context().result.fragment_count as usize
    }
}
