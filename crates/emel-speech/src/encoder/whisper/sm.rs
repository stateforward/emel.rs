//! Source-aligned synchronous Whisper speech encoder actor.
//!
//! The actor keeps the pinned request graph intact: contract validation,
//! audio validation, bounded output/workspace checks, explicit weight-variant
//! routing, and callback publication are separate run-to-completion phases.
//! Model execution is supplied by a caller-owned synchronous function pointer;
//! the lower-level Rust Whisper encoder owns that implementation boundary.

#![allow(
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::empty_structs_with_brackets,
    elided_lifetimes_in_paths,
    dead_code,
    missing_docs
)]

use core::cell::RefCell;

use sml::sml;

/// Whisper encoder sample-rate contract.
pub const SAMPLE_RATE: i32 = 16_000;
/// Whisper encoder channel-count contract.
pub const CHANNEL_COUNT: i32 = 1;
/// Mel filter-bank width in the pinned model.
pub const MEL_BIN_COUNT: i32 = 80;
/// Encoder embedding width in the pinned model.
pub const EMBEDDING_LENGTH: i32 = 384;
/// Feed-forward width in the pinned model.
pub const FEED_FORWARD_LENGTH: i32 = 1_536;
/// Attention head count in the pinned model.
pub const ATTENTION_HEAD_COUNT: i32 = 6;
/// Number of encoder blocks in the pinned model.
pub const ENCODER_BLOCK_COUNT: i32 = 4;
pub const HOP_LENGTH: usize = 160;
pub const MAX_MEL_FRAME_COUNT: usize = 3_000;
pub const MAX_ENCODER_FRAME_COUNT: usize = 1_500;
pub const MAX_PCM_SAMPLE_COUNT: usize = MAX_MEL_FRAME_COUNT * HOP_LENGTH;
pub const FFT_SIZE: usize = 400;
pub const BLUESTEIN_FFT_SIZE: usize = 1_024;

/// Errors defined by the pinned Whisper encoder contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum Error {
    None = 0,
    ModelInvalid = 1,
    SampleRate = 2,
    ChannelCount = 3,
    PcmShape = 4,
    OutputCapacity = 5,
    WorkspaceCapacity = 6,
    UnsupportedVariant = 7,
    InternalError = 8,
    Unexpected = 9,
}

impl Default for Error {
    fn default() -> Self {
        Self::None
    }
}

/// Source-compatible error alias.
pub type EncoderError = Error;

/// Weight families selected by the source variant decision.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum WeightVariant {
    #[default]
    Unsupported = 0,
    Q8_0F32Aux = 1,
    Q8_0 = 2,
    Q4_0 = 3,
    Q4_1 = 4,
}

/// Bounded summary of the model tensors required by the pinned encoder.
///
/// Each flag represents a complete shape/storage check performed at the model
/// boundary. Keeping the summary bounded avoids retaining loader-owned records
/// in the synchronous actor while preserving the source validation decisions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelAssets {
    pub mel_filters: bool,
    pub conv1_weight: bool,
    pub conv1_bias: bool,
    pub conv2_weight: bool,
    pub conv2_bias: bool,
    pub embed_positions: bool,
    pub layer_norm: bool,
    pub encoder_blocks: u8,
    pub q8_0_f32_aux: bool,
    pub q8_0: bool,
    pub q4_0: bool,
    pub q4_1: bool,
}

impl Default for ModelAssets {
    fn default() -> Self {
        Self {
            mel_filters: false,
            conv1_weight: false,
            conv1_bias: false,
            conv2_weight: false,
            conv2_bias: false,
            embed_positions: false,
            layer_norm: false,
            encoder_blocks: 0,
            q8_0_f32_aux: false,
            q8_0: false,
            q4_0: false,
            q4_1: false,
        }
    }
}

impl ModelAssets {
    #[must_use]
    pub const fn pinned() -> Self {
        Self {
            mel_filters: true,
            conv1_weight: true,
            conv1_bias: true,
            conv2_weight: true,
            conv2_bias: true,
            embed_positions: true,
            layer_norm: true,
            encoder_blocks: ENCODER_BLOCK_COUNT as u8,
            q8_0_f32_aux: true,
            q8_0: true,
            q4_0: true,
            q4_1: true,
        }
    }

    #[must_use]
    pub const fn base_contract_valid(self) -> bool {
        self.mel_filters
            && self.conv1_weight
            && self.conv1_bias
            && self.conv2_weight
            && self.conv2_bias
            && self.embed_positions
            && self.layer_norm
            && self.encoder_blocks == ENCODER_BLOCK_COUNT as u8
    }

    #[must_use]
    pub const fn variant(self) -> WeightVariant {
        if self.q8_0_f32_aux {
            WeightVariant::Q8_0F32Aux
        } else if self.q8_0 {
            WeightVariant::Q8_0
        } else if self.q4_0 {
            WeightVariant::Q4_0
        } else if self.q4_1 {
            WeightVariant::Q4_1
        } else {
            WeightVariant::Unsupported
        }
    }
}

/// Variant-neutral execution contract copied at the speech/model boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExecutionContract {
    pub model_present: bool,
    pub sample_rate: i32,
    pub mel_bin_count: i32,
    pub embedding_length: i32,
    pub feed_forward_length: i32,
    pub attention_head_count: i32,
    pub encoder_block_count: i32,
    pub assets: ModelAssets,
    pub encoder: Option<EncodeFn>,
}

/// Synchronous lower-level encoder boundary.
///
/// The callback receives only borrowed caller-owned data. It returns the
/// bounded frame count and digest produced by the selected encoder route.
pub type EncodeFn = fn(
    &[f32],
    &ExecutionContract,
    &mut [f32],
    &mut [f32],
) -> Option<(i32, u64)>;

impl Default for ExecutionContract {
    fn default() -> Self {
        Self {
            model_present: false,
            sample_rate: 0,
            mel_bin_count: 0,
            embedding_length: 0,
            feed_forward_length: 0,
            attention_head_count: 0,
            encoder_block_count: 0,
            assets: ModelAssets::default(),
            encoder: None,
        }
    }
}

impl ExecutionContract {
    #[must_use]
    pub const fn pinned() -> Self {
        Self {
            model_present: true,
            sample_rate: SAMPLE_RATE,
            mel_bin_count: MEL_BIN_COUNT,
            embedding_length: EMBEDDING_LENGTH,
            feed_forward_length: FEED_FORWARD_LENGTH,
            attention_head_count: ATTENTION_HEAD_COUNT,
            encoder_block_count: ENCODER_BLOCK_COUNT,
            assets: ModelAssets::pinned(),
            encoder: None,
        }
    }

    #[must_use]
    pub const fn model_contract_valid(self) -> bool {
        self.model_present
            && self.sample_rate == SAMPLE_RATE
            && self.mel_bin_count == MEL_BIN_COUNT
            && self.embedding_length == EMBEDDING_LENGTH
            && self.feed_forward_length == FEED_FORWARD_LENGTH
            && self.attention_head_count == ATTENTION_HEAD_COUNT
            && self.encoder_block_count == ENCODER_BLOCK_COUNT
            && self.assets.base_contract_valid()
    }

    #[must_use]
    pub const fn with_encoder(mut self, encoder: EncodeFn) -> Self {
        self.encoder = Some(encoder);
        self
    }
}

/// Successful completion payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EncodeDone {
    pub frame_count: i32,
    pub width: i32,
    pub digest: u64,
}

/// Failed completion payload.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EncodeError {
    pub error: Error,
}

pub type DoneCallback = fn(EncodeDone) -> bool;
pub type ErrorCallback = fn(EncodeError) -> bool;

/// Borrowed request corresponding to the pinned `event::encode` contract.
pub struct EventEncodeRun<'a> {
    pub contract: &'a ExecutionContract,
    pub pcm: &'a [f32],
    pub sample_rate: i32,
    pub channel_count: i32,
    pub workspace: RefCell<&'a mut [f32]>,
    pub encoder_state: RefCell<&'a mut [f32]>,
    pub frame_count_out: RefCell<&'a mut i32>,
    pub width_out: RefCell<&'a mut i32>,
    pub digest_out: RefCell<&'a mut u64>,
    pub error_out: RefCell<Option<&'a mut Error>>,
    pub on_done: Option<DoneCallback>,
    pub on_error: Option<ErrorCallback>,
}

impl<'a> EventEncodeRun<'a> {
    #[must_use]
    pub fn new(
        contract: &'a ExecutionContract,
        pcm: &'a [f32],
        sample_rate: i32,
        channel_count: i32,
        workspace: &'a mut [f32],
        encoder_state: &'a mut [f32],
        frame_count_out: &'a mut i32,
        width_out: &'a mut i32,
        digest_out: &'a mut u64,
    ) -> Self {
        Self {
            contract,
            pcm,
            sample_rate,
            channel_count,
            workspace: RefCell::new(workspace),
            encoder_state: RefCell::new(encoder_state),
            frame_count_out: RefCell::new(frame_count_out),
            width_out: RefCell::new(width_out),
            digest_out: RefCell::new(digest_out),
            error_out: RefCell::new(None),
            on_done: None,
            on_error: None,
        }
    }

    #[must_use]
    pub fn with_callbacks(
        mut self,
        on_done: Option<DoneCallback>,
        on_error: Option<ErrorCallback>,
    ) -> Self {
        self.on_done = on_done;
        self.on_error = on_error;
        self
    }

    #[must_use]
    pub fn with_error_out(mut self, error_out: &'a mut Error) -> Self {
        *self.error_out.get_mut() = Some(error_out);
        self
    }
}

#[must_use]
pub const fn mel_frame_count(sample_count: usize) -> usize {
    sample_count.saturating_add(HOP_LENGTH - 1) / HOP_LENGTH
}

#[must_use]
pub const fn encoder_frame_count(sample_count: usize) -> usize {
    mel_frame_count(sample_count).saturating_add(1) / 2
}

#[must_use]
pub const fn required_encoder_output_floats(sample_count: usize) -> usize {
    encoder_frame_count(sample_count).saturating_mul(EMBEDDING_LENGTH as usize)
}

#[must_use]
pub const fn required_workspace_floats(sample_count: usize) -> usize {
    let mel_frames = mel_frame_count(sample_count);
    let encoder_frames = (mel_frames + 1) / 2;
    MEL_BIN_COUNT as usize * mel_frames
        + EMBEDDING_LENGTH as usize * mel_frames
        + EMBEDDING_LENGTH as usize * encoder_frames * 6
        + EMBEDDING_LENGTH as usize
        + FEED_FORWARD_LENGTH as usize
        + encoder_frames
        + FFT_SIZE * 3
        + BLUESTEIN_FFT_SIZE * 4
}

sml! {
    SpeechEncoderWhisper<'dispatch> {
        "state_model_contract_decision"_s <= *"state_ready"_s + event<&'dispatch EventEncodeRun> / effect_begin_encode,
        "state_sample_rate_decision"_s <= "state_model_contract_decision"_s + completion<&'dispatch EventEncodeRun> [guard_model_contract_valid],
        "state_error_error_out_decision"_s <= "state_model_contract_decision"_s + completion<&'dispatch EventEncodeRun> [guard_model_contract_invalid] / effect_mark_model_invalid,
        "state_channel_count_decision"_s <= "state_sample_rate_decision"_s + completion<&'dispatch EventEncodeRun> [guard_sample_rate_valid],
        "state_error_error_out_decision"_s <= "state_sample_rate_decision"_s + completion<&'dispatch EventEncodeRun> [guard_sample_rate_invalid] / effect_mark_sample_rate_invalid,
        "state_pcm_shape_decision"_s <= "state_channel_count_decision"_s + completion<&'dispatch EventEncodeRun> [guard_channel_count_valid],
        "state_error_error_out_decision"_s <= "state_channel_count_decision"_s + completion<&'dispatch EventEncodeRun> [guard_channel_count_invalid] / effect_mark_channel_count_invalid,
        "state_output_capacity_decision"_s <= "state_pcm_shape_decision"_s + completion<&'dispatch EventEncodeRun> [guard_pcm_shape_valid],
        "state_error_error_out_decision"_s <= "state_pcm_shape_decision"_s + completion<&'dispatch EventEncodeRun> [guard_pcm_shape_invalid] / effect_mark_pcm_shape_invalid,
        "state_workspace_capacity_decision"_s <= "state_output_capacity_decision"_s + completion<&'dispatch EventEncodeRun> [guard_output_capacity_valid],
        "state_error_error_out_decision"_s <= "state_output_capacity_decision"_s + completion<&'dispatch EventEncodeRun> [guard_output_capacity_invalid] / effect_mark_output_capacity_invalid,
        "state_variant_decision"_s <= "state_workspace_capacity_decision"_s + completion<&'dispatch EventEncodeRun> [guard_workspace_capacity_valid],
        "state_error_error_out_decision"_s <= "state_workspace_capacity_decision"_s + completion<&'dispatch EventEncodeRun> [guard_workspace_capacity_invalid] / effect_mark_workspace_capacity_invalid,
        "state_running_q8_0_f32_aux"_s <= "state_variant_decision"_s + completion<&'dispatch EventEncodeRun> [guard_q8_0_f32_aux_variant] / effect_run_encoder_q8_0_f32_aux,
        "state_running_q8_0"_s <= "state_variant_decision"_s + completion<&'dispatch EventEncodeRun> [guard_q8_0_variant] / effect_run_encoder_q8_0,
        "state_running_q4_0"_s <= "state_variant_decision"_s + completion<&'dispatch EventEncodeRun> [guard_q4_0_variant] / effect_run_encoder_q4_0,
        "state_running_q4_1"_s <= "state_variant_decision"_s + completion<&'dispatch EventEncodeRun> [guard_q4_1_variant] / effect_run_encoder_q4_1,
        "state_error_error_out_decision"_s <= "state_variant_decision"_s + completion<&'dispatch EventEncodeRun> [guard_unsupported_variant] / effect_mark_unsupported_variant,
        "state_success_error_out_decision"_s <= "state_running_q8_0_f32_aux"_s + completion<&'dispatch EventEncodeRun>,
        "state_success_error_out_decision"_s <= "state_running_q8_0"_s + completion<&'dispatch EventEncodeRun>,
        "state_success_error_out_decision"_s <= "state_running_q4_0"_s + completion<&'dispatch EventEncodeRun>,
        "state_success_error_out_decision"_s <= "state_running_q4_1"_s + completion<&'dispatch EventEncodeRun>,
        "state_success_callback_decision"_s <= "state_success_error_out_decision"_s + completion<&'dispatch EventEncodeRun> [guard_has_error_out] / effect_store_success_error,
        "state_success_callback_decision"_s <= "state_success_error_out_decision"_s + completion<&'dispatch EventEncodeRun> [guard_no_error_out],
        "state_error_callback_decision"_s <= "state_error_error_out_decision"_s + completion<&'dispatch EventEncodeRun> [guard_has_error_out] / effect_store_error_error,
        "state_error_callback_decision"_s <= "state_error_error_out_decision"_s + completion<&'dispatch EventEncodeRun> [guard_no_error_out],
        "state_done"_s <= "state_success_callback_decision"_s + completion<&'dispatch EventEncodeRun> [guard_has_done_callback] / effect_emit_done,
        "state_done"_s <= "state_success_callback_decision"_s + completion<&'dispatch EventEncodeRun> [guard_no_done_callback],
        "state_errored"_s <= "state_error_callback_decision"_s + completion<&'dispatch EventEncodeRun> [guard_has_error_callback] / effect_emit_error,
        "state_errored"_s <= "state_error_callback_decision"_s + completion<&'dispatch EventEncodeRun> [guard_no_error_callback],
        "state_ready"_s <= "state_done"_s + completion<&'dispatch EventEncodeRun>,
        "state_ready"_s <= "state_errored"_s + completion<&'dispatch EventEncodeRun>,
        "state_ready"_s <= "state_ready"_s + unexpected_event<_> / effect_on_unexpected_from_state_ready,
        "state_ready"_s <= "state_model_contract_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_model_contract_decision,
        "state_ready"_s <= "state_sample_rate_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_sample_rate_decision,
        "state_ready"_s <= "state_channel_count_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_channel_count_decision,
        "state_ready"_s <= "state_pcm_shape_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_pcm_shape_decision,
        "state_ready"_s <= "state_output_capacity_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_output_capacity_decision,
        "state_ready"_s <= "state_workspace_capacity_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_workspace_capacity_decision,
        "state_ready"_s <= "state_variant_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_variant_decision,
        "state_ready"_s <= "state_running_q8_0"_s + unexpected_event<_> / effect_on_unexpected_from_state_running_q8_0,
        "state_ready"_s <= "state_running_q8_0_f32_aux"_s + unexpected_event<_> / effect_on_unexpected_from_state_running_q8_0_f32_aux,
        "state_ready"_s <= "state_running_q4_0"_s + unexpected_event<_> / effect_on_unexpected_from_state_running_q4_0,
        "state_ready"_s <= "state_running_q4_1"_s + unexpected_event<_> / effect_on_unexpected_from_state_running_q4_1,
        "state_ready"_s <= "state_success_error_out_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_success_error_out_decision,
        "state_ready"_s <= "state_success_callback_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_success_callback_decision,
        "state_ready"_s <= "state_error_error_out_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_error_error_out_decision,
        "state_ready"_s <= "state_error_callback_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_error_callback_decision,
        "state_ready"_s <= "state_done"_s + unexpected_event<_> / effect_on_unexpected_from_state_done,
        "state_ready"_s <= "state_errored"_s + unexpected_event<_> / effect_on_unexpected_from_state_errored,
    }
}

#[derive(Debug, Default)]
pub struct SpeechEncoderWhisperContext {
    pub err: Error,
    pub q8_0_dispatch_count: u64,
    pub q4_0_dispatch_count: u64,
    pub q4_1_dispatch_count: u64,
}

impl SpeechEncoderWhisperStateMachineContext for SpeechEncoderWhisperContext {
    fn effect_begin_encode(&mut self, event: &EventEncodeRun) -> Result<(), ()> {
        self.err = Error::None;
        *event.frame_count_out.borrow_mut() = 0;
        *event.width_out.borrow_mut() = 0;
        *event.digest_out.borrow_mut() = 0;
        Ok(())
    }

    fn effect_emit_done(&mut self, event: &EventEncodeRun) -> Result<(), ()> {
        if let Some(callback) = event.on_done {
            let _ = callback(EncodeDone {
                frame_count: *event.frame_count_out.borrow(),
                width: *event.width_out.borrow(),
                digest: *event.digest_out.borrow(),
            });
        }
        Ok(())
    }

    fn effect_emit_error(&mut self, event: &EventEncodeRun) -> Result<(), ()> {
        if let Some(callback) = event.on_error {
            let _ = callback(EncodeError { error: self.err });
        }
        Ok(())
    }

    fn effect_mark_model_invalid(&mut self, _: &EventEncodeRun) -> Result<(), ()> { self.err = Error::ModelInvalid; Ok(()) }
    fn effect_mark_sample_rate_invalid(&mut self, _: &EventEncodeRun) -> Result<(), ()> { self.err = Error::SampleRate; Ok(()) }
    fn effect_mark_channel_count_invalid(&mut self, _: &EventEncodeRun) -> Result<(), ()> { self.err = Error::ChannelCount; Ok(()) }
    fn effect_mark_pcm_shape_invalid(&mut self, _: &EventEncodeRun) -> Result<(), ()> { self.err = Error::PcmShape; Ok(()) }
    fn effect_mark_output_capacity_invalid(&mut self, _: &EventEncodeRun) -> Result<(), ()> { self.err = Error::OutputCapacity; Ok(()) }
    fn effect_mark_workspace_capacity_invalid(&mut self, _: &EventEncodeRun) -> Result<(), ()> { self.err = Error::WorkspaceCapacity; Ok(()) }
    fn effect_mark_unsupported_variant(&mut self, _: &EventEncodeRun) -> Result<(), ()> { self.err = Error::UnsupportedVariant; Ok(()) }

    fn effect_run_encoder_q8_0_f32_aux(&mut self, event: &EventEncodeRun) -> Result<(), ()> {
        let Some(encoder) = event.contract.encoder else { self.err = Error::InternalError; return Ok(()); };
        let required_output = required_encoder_output_floats(event.pcm.len());
        let required_workspace = required_workspace_floats(event.pcm.len());
        let mut workspace = event.workspace.borrow_mut();
        let mut output = event.encoder_state.borrow_mut();
        let Some((frame_count, digest)) = encoder(event.pcm, event.contract, &mut workspace[..required_workspace], &mut output[..required_output]) else { self.err = Error::InternalError; return Ok(()); };
        if !(0..=MAX_ENCODER_FRAME_COUNT as i32).contains(&frame_count) { self.err = Error::InternalError; return Ok(()); }
        *event.frame_count_out.borrow_mut() = frame_count;
        *event.width_out.borrow_mut() = EMBEDDING_LENGTH;
        *event.digest_out.borrow_mut() = digest;
        self.err = Error::None;
        self.q8_0_dispatch_count += 1;
        Ok(())
    }

    fn effect_run_encoder_q8_0(&mut self, event: &EventEncodeRun) -> Result<(), ()> {
        let Some(encoder) = event.contract.encoder else { self.err = Error::InternalError; return Ok(()); };
        let required_output = required_encoder_output_floats(event.pcm.len());
        let required_workspace = required_workspace_floats(event.pcm.len());
        let mut workspace = event.workspace.borrow_mut();
        let mut output = event.encoder_state.borrow_mut();
        let Some((frame_count, digest)) = encoder(event.pcm, event.contract, &mut workspace[..required_workspace], &mut output[..required_output]) else { self.err = Error::InternalError; return Ok(()); };
        if !(0..=MAX_ENCODER_FRAME_COUNT as i32).contains(&frame_count) { self.err = Error::InternalError; return Ok(()); }
        *event.frame_count_out.borrow_mut() = frame_count;
        *event.width_out.borrow_mut() = EMBEDDING_LENGTH;
        *event.digest_out.borrow_mut() = digest;
        self.err = Error::None;
        self.q8_0_dispatch_count += 1;
        Ok(())
    }

    fn effect_run_encoder_q4_0(&mut self, event: &EventEncodeRun) -> Result<(), ()> {
        let Some(encoder) = event.contract.encoder else { self.err = Error::InternalError; return Ok(()); };
        let required_output = required_encoder_output_floats(event.pcm.len());
        let required_workspace = required_workspace_floats(event.pcm.len());
        let mut workspace = event.workspace.borrow_mut();
        let mut output = event.encoder_state.borrow_mut();
        let Some((frame_count, digest)) = encoder(event.pcm, event.contract, &mut workspace[..required_workspace], &mut output[..required_output]) else { self.err = Error::InternalError; return Ok(()); };
        if !(0..=MAX_ENCODER_FRAME_COUNT as i32).contains(&frame_count) { self.err = Error::InternalError; return Ok(()); }
        *event.frame_count_out.borrow_mut() = frame_count;
        *event.width_out.borrow_mut() = EMBEDDING_LENGTH;
        *event.digest_out.borrow_mut() = digest;
        self.err = Error::None;
        self.q4_0_dispatch_count += 1;
        Ok(())
    }

    fn effect_run_encoder_q4_1(&mut self, event: &EventEncodeRun) -> Result<(), ()> {
        let Some(encoder) = event.contract.encoder else { self.err = Error::InternalError; return Ok(()); };
        let required_output = required_encoder_output_floats(event.pcm.len());
        let required_workspace = required_workspace_floats(event.pcm.len());
        let mut workspace = event.workspace.borrow_mut();
        let mut output = event.encoder_state.borrow_mut();
        let Some((frame_count, digest)) = encoder(event.pcm, event.contract, &mut workspace[..required_workspace], &mut output[..required_output]) else { self.err = Error::InternalError; return Ok(()); };
        if !(0..=MAX_ENCODER_FRAME_COUNT as i32).contains(&frame_count) { self.err = Error::InternalError; return Ok(()); }
        *event.frame_count_out.borrow_mut() = frame_count;
        *event.width_out.borrow_mut() = EMBEDDING_LENGTH;
        *event.digest_out.borrow_mut() = digest;
        self.err = Error::None;
        self.q4_1_dispatch_count += 1;
        Ok(())
    }


    fn effect_store_success_error(&mut self, event: &EventEncodeRun) -> Result<(), ()> {
        if let Some(error_out) = event.error_out.borrow_mut().as_deref_mut() { *error_out = self.err; }
        Ok(())
    }
    fn effect_store_error_error(&mut self, event: &EventEncodeRun) -> Result<(), ()> {
        if let Some(error_out) = event.error_out.borrow_mut().as_deref_mut() { *error_out = self.err; }
        Ok(())
    }

    fn guard_model_contract_valid(&self, event: &EventEncodeRun) -> Result<bool, ()> { Ok(event.contract.model_contract_valid()) }
    fn guard_model_contract_invalid(&self, event: &EventEncodeRun) -> Result<bool, ()> { Ok(!event.contract.model_contract_valid()) }
    fn guard_sample_rate_valid(&self, event: &EventEncodeRun) -> Result<bool, ()> { Ok(event.sample_rate == SAMPLE_RATE) }
    fn guard_sample_rate_invalid(&self, event: &EventEncodeRun) -> Result<bool, ()> { Ok(event.sample_rate != SAMPLE_RATE) }
    fn guard_channel_count_valid(&self, event: &EventEncodeRun) -> Result<bool, ()> { Ok(event.channel_count == CHANNEL_COUNT) }
    fn guard_channel_count_invalid(&self, event: &EventEncodeRun) -> Result<bool, ()> { Ok(event.channel_count != CHANNEL_COUNT) }
    fn guard_pcm_shape_valid(&self, event: &EventEncodeRun) -> Result<bool, ()> {
        Ok(!event.pcm.is_empty() && event.pcm.len() <= MAX_PCM_SAMPLE_COUNT && event.pcm.iter().all(|value| value.is_finite()))
    }
    fn guard_pcm_shape_invalid(&self, event: &EventEncodeRun) -> Result<bool, ()> { Ok(!self.guard_pcm_shape_valid(event)?) }
    fn guard_output_capacity_valid(&self, event: &EventEncodeRun) -> Result<bool, ()> {
        Ok(event.encoder_state.borrow().len() >= required_encoder_output_floats(event.pcm.len()))
    }
    fn guard_output_capacity_invalid(&self, event: &EventEncodeRun) -> Result<bool, ()> { Ok(!self.guard_output_capacity_valid(event)?) }
    fn guard_workspace_capacity_valid(&self, event: &EventEncodeRun) -> Result<bool, ()> {
        Ok(event.workspace.borrow().len() >= required_workspace_floats(event.pcm.len()))
    }
    fn guard_workspace_capacity_invalid(&self, event: &EventEncodeRun) -> Result<bool, ()> { Ok(!self.guard_workspace_capacity_valid(event)?) }
    fn guard_q8_0_f32_aux_variant(&self, event: &EventEncodeRun) -> Result<bool, ()> { Ok(event.contract.assets.variant() == WeightVariant::Q8_0F32Aux) }
    fn guard_q8_0_variant(&self, event: &EventEncodeRun) -> Result<bool, ()> { Ok(event.contract.assets.variant() == WeightVariant::Q8_0) }
    fn guard_q4_0_variant(&self, event: &EventEncodeRun) -> Result<bool, ()> { Ok(event.contract.assets.variant() == WeightVariant::Q4_0) }
    fn guard_q4_1_variant(&self, event: &EventEncodeRun) -> Result<bool, ()> { Ok(event.contract.assets.variant() == WeightVariant::Q4_1) }
    fn guard_unsupported_variant(&self, event: &EventEncodeRun) -> Result<bool, ()> { Ok(event.contract.assets.variant() == WeightVariant::Unsupported) }
    fn guard_has_error_out(&self, event: &EventEncodeRun) -> Result<bool, ()> { Ok(event.error_out.borrow().is_some()) }
    fn guard_no_error_out(&self, event: &EventEncodeRun) -> Result<bool, ()> { Ok(event.error_out.borrow().is_none()) }
    fn guard_has_done_callback(&self, event: &EventEncodeRun) -> Result<bool, ()> { Ok(event.on_done.is_some()) }
    fn guard_no_done_callback(&self, event: &EventEncodeRun) -> Result<bool, ()> { Ok(event.on_done.is_none()) }
    fn guard_has_error_callback(&self, event: &EventEncodeRun) -> Result<bool, ()> { Ok(event.on_error.is_some()) }
    fn guard_no_error_callback(&self, event: &EventEncodeRun) -> Result<bool, ()> { Ok(event.on_error.is_none()) }

    fn effect_on_unexpected_from_state_ready(&mut self) -> Result<(), ()> { self.err = Error::Unexpected; Ok(()) }
    fn effect_on_unexpected_from_state_model_contract_decision(&mut self) -> Result<(), ()> { self.err = Error::Unexpected; Ok(()) }
    fn effect_on_unexpected_from_state_sample_rate_decision(&mut self) -> Result<(), ()> { self.err = Error::Unexpected; Ok(()) }
    fn effect_on_unexpected_from_state_channel_count_decision(&mut self) -> Result<(), ()> { self.err = Error::Unexpected; Ok(()) }
    fn effect_on_unexpected_from_state_pcm_shape_decision(&mut self) -> Result<(), ()> { self.err = Error::Unexpected; Ok(()) }
    fn effect_on_unexpected_from_state_output_capacity_decision(&mut self) -> Result<(), ()> { self.err = Error::Unexpected; Ok(()) }
    fn effect_on_unexpected_from_state_workspace_capacity_decision(&mut self) -> Result<(), ()> { self.err = Error::Unexpected; Ok(()) }
    fn effect_on_unexpected_from_state_variant_decision(&mut self) -> Result<(), ()> { self.err = Error::Unexpected; Ok(()) }
    fn effect_on_unexpected_from_state_running_q8_0(&mut self) -> Result<(), ()> { self.err = Error::Unexpected; Ok(()) }
    fn effect_on_unexpected_from_state_running_q8_0_f32_aux(&mut self) -> Result<(), ()> { self.err = Error::Unexpected; Ok(()) }
    fn effect_on_unexpected_from_state_running_q4_0(&mut self) -> Result<(), ()> { self.err = Error::Unexpected; Ok(()) }
    fn effect_on_unexpected_from_state_running_q4_1(&mut self) -> Result<(), ()> { self.err = Error::Unexpected; Ok(()) }
    fn effect_on_unexpected_from_state_success_error_out_decision(&mut self) -> Result<(), ()> { self.err = Error::Unexpected; Ok(()) }
    fn effect_on_unexpected_from_state_success_callback_decision(&mut self) -> Result<(), ()> { self.err = Error::Unexpected; Ok(()) }
    fn effect_on_unexpected_from_state_error_error_out_decision(&mut self) -> Result<(), ()> { self.err = Error::Unexpected; Ok(()) }
    fn effect_on_unexpected_from_state_error_callback_decision(&mut self) -> Result<(), ()> { self.err = Error::Unexpected; Ok(()) }
    fn effect_on_unexpected_from_state_done(&mut self) -> Result<(), ()> { self.err = Error::Unexpected; Ok(()) }
    fn effect_on_unexpected_from_state_errored(&mut self) -> Result<(), ()> { self.err = Error::Unexpected; Ok(()) }
}

/// Public synchronous actor wrapper around the generated state machine.
pub struct SpeechEncoderWhisperActor {
    machine: SpeechEncoderWhisperStateMachine<'static>,
}

impl Default for SpeechEncoderWhisperActor {
    fn default() -> Self { Self::new() }
}

impl SpeechEncoderWhisperActor {
    #[must_use]
    pub fn new() -> Self {
        Self { machine: SpeechEncoderWhisperStateMachine::new(SpeechEncoderWhisperContext::default()) }
    }

    /// Dispatches one borrowed request through the complete source phase graph.
    pub fn process_event<'a>(&mut self, event: EventEncodeRun<'a>) -> bool {
        self.machine.process_event(event).is_ok() && self.machine.context().err == Error::None
    }

    pub fn encode<'a>(&mut self, event: EventEncodeRun<'a>) -> bool { self.process_event(event) }

    #[must_use]
    pub fn context(&self) -> &SpeechEncoderWhisperContext { self.machine.context() }

    #[must_use]
    pub fn state(&self) -> &SpeechEncoderWhisperStates { self.machine.state() }

    #[must_use]
    pub fn q8_0_dispatch_count(&self) -> u64 { self.machine.context().q8_0_dispatch_count }
    #[must_use]
    pub fn q4_0_dispatch_count(&self) -> u64 { self.machine.context().q4_0_dispatch_count }
    #[must_use]
    pub fn q4_1_dispatch_count(&self) -> u64 { self.machine.context().q4_1_dispatch_count }
}

/// Source-compatible request actor name.
pub type Request = SpeechEncoderWhisperActor;
/// Direct access to the generated state-machine type.
pub type Machine = SpeechEncoderWhisperStateMachine<'static>;
