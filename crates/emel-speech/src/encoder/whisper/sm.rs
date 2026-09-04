//! Source-aligned synchronous Whisper speech encoder actor.
//!
//! The actor keeps the pinned request graph intact: contract validation,
//! audio validation, bounded output/workspace checks, explicit weight-variant
//! routing, and native model execution are separate run-to-completion phases.
//! Model execution is performed synchronously by the component-owned native
//! kernel in [`detail`], using only caller-owned buffers and resident model bytes.

// Error variants retain the source-aligned `<kind>Error` naming contract.
#![allow(
    clippy::enum_variant_names,
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::empty_structs_with_brackets,
    elided_lifetimes_in_paths,
    missing_debug_implementations,
    dead_code,
    missing_docs
)]

use core::cell::RefCell;
use emel_model::bridge::Data;
use emel_tensor::dtype::SerializedType;
use sml::sml;

use super::detail;
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
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u32)]
pub enum Error {
    #[default]
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
#[allow(
    clippy::struct_excessive_bools,
    reason = "source-aligned summary intentionally exposes one flag per tensor check"
)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
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
            encoder_blocks: 4,
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
            && self.encoder_blocks == 4
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
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExecutionContract {
    pub model_present: bool,
    pub sample_rate: i32,
    pub mel_bin_count: i32,
    pub embedding_length: i32,
    pub feed_forward_length: i32,
    pub attention_head_count: i32,
    pub encoder_block_count: i32,
    pub assets: ModelAssets,
}

impl ExecutionContract {
    /// Builds the source-bound execution metadata for one immutable Whisper model.
    ///
    /// The model pointer remains owned by the caller; this value copies only the
    /// pinned dimensions and bounded tensor-family summary. Invalid architecture,
    /// metadata, or tensor layouts remain observable through
    /// [`Self::model_contract_valid`].
    ///
    /// # Panics
    ///
    /// Panics only if the fixed encoder block count cannot fit in `u8`; the
    /// maintained source contract uses a count that is known to fit.
    #[must_use]
    pub fn bind_model(model: &Data) -> Self {
        let binding = model.whisper_binding_input();
        let hparams = binding.hparams();
        let metadata_valid = emel_model::whisper::Any::bind(binding).is_ok()
            && model.architecture_name() == b"whisper"
            && hparams.is_some_and(|h| {
                h.n_mels() == MEL_BIN_COUNT as u32
                    && h.n_vocab() == 51_865
                    && h.n_embd() == EMBEDDING_LENGTH as u32
                    && h.n_ff() == FEED_FORWARD_LENGTH as u32
                    && h.n_head() == ATTENTION_HEAD_COUNT as u32
                    && h.n_head_kv() == ATTENTION_HEAD_COUNT as u32
                    && h.n_ctx() == 448
                    && h.encoder_block_count() == ENCODER_BLOCK_COUNT as u32
                    && h.decoder_block_count() == ENCODER_BLOCK_COUNT as u32
            });
        let base_valid = metadata_valid && model_base_valid(model);
        let variant = if base_valid {
            model_variant(model)
        } else {
            WeightVariant::Unsupported
        };
        let mut assets = ModelAssets {
            mel_filters: base_valid,
            conv1_weight: base_valid,
            conv1_bias: base_valid,
            conv2_weight: base_valid,
            conv2_bias: base_valid,
            embed_positions: base_valid,
            layer_norm: base_valid,
            encoder_blocks: if base_valid {
                u8::try_from(ENCODER_BLOCK_COUNT).expect("pinned encoder block count fits u8")
            } else {
                0
            },
            ..ModelAssets::default()
        };
        match variant {
            WeightVariant::Q8_0F32Aux => assets.q8_0_f32_aux = true,
            WeightVariant::Q8_0 => assets.q8_0 = true,
            WeightVariant::Q4_0 => assets.q4_0 = true,
            WeightVariant::Q4_1 => assets.q4_1 = true,
            WeightVariant::Unsupported => {}
        }
        Self {
            model_present: metadata_valid,
            sample_rate: SAMPLE_RATE,
            mel_bin_count: MEL_BIN_COUNT,
            embedding_length: EMBEDDING_LENGTH,
            feed_forward_length: FEED_FORWARD_LENGTH,
            attention_head_count: ATTENTION_HEAD_COUNT,
            encoder_block_count: ENCODER_BLOCK_COUNT,
            assets,
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
        }
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

pub struct EventEncodeRun<'a> {
    pub contract: &'a ExecutionContract,
    pub model: Option<&'a Data>,
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
    #[allow(
        clippy::too_many_arguments,
        reason = "source-aligned constructor keeps caller-owned buffers separate"
    )]
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
            model: None,
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
    pub const fn with_model(mut self, model: &'a Data) -> Self {
        self.model = Some(model);
        self
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
}

#[must_use]
pub const fn mel_frame_count(sample_count: usize) -> usize {
    let frames = sample_count.saturating_add(HOP_LENGTH - 1) / HOP_LENGTH;
    if frames > MAX_MEL_FRAME_COUNT {
        MAX_MEL_FRAME_COUNT
    } else {
        frames
    }
}

#[must_use]
pub const fn encoder_frame_count(sample_count: usize) -> usize {
    mel_frame_count(sample_count).div_ceil(2)
}

#[must_use]
pub const fn required_encoder_output_floats(sample_count: usize) -> usize {
    encoder_frame_count(sample_count).saturating_mul(384)
}

#[must_use]
pub const fn required_workspace_floats(sample_count: usize) -> usize {
    let mel_frames = mel_frame_count(sample_count);
    let encoder_frames = mel_frames.div_ceil(2);
    80 * mel_frames
        + 384 * mel_frames
        + 384 * encoder_frames * 6
        + 384
        + 1_536
        + encoder_frames
        + FFT_SIZE * 3
        + BLUESTEIN_FFT_SIZE * 4
}

fn has_tensor(model: &Data, name: &[u8], dims: &[u64], kind: SerializedType) -> bool {
    let Some(tensor) = model.tensor_named(name) else {
        return false;
    };
    let Some(metadata) = tensor.metadata() else {
        return false;
    };
    metadata.tensor_type() == kind
        && usize::try_from(metadata.dimension_count()).ok() == Some(dims.len())
        && metadata.dimensions().get(..dims.len()) == Some(dims)
        && tensor.bytes().is_some_and(|bytes| !bytes.is_empty())
}

fn has_aux_vector(model: &Data, name: &[u8], length: u64, kind: SerializedType) -> bool {
    has_tensor(model, name, &[length], kind)
}

fn has_any_aux_vector(model: &Data, name: &[u8], length: u64) -> bool {
    has_aux_vector(model, name, length, SerializedType::Q8_0)
        || has_aux_vector(model, name, length, SerializedType::F32)
}

fn has_any_aux_position_matrix(model: &Data, name: &[u8]) -> bool {
    has_tensor(model, name, &[384, 1500], SerializedType::Q8_0)
        || has_tensor(model, name, &[384, 1500], SerializedType::F32)
}

fn has_encoder_block(
    model: &Data,
    block: usize,
    linear: SerializedType,
    aux: SerializedType,
) -> bool {
    let mut prefix = [0_u8; 64];
    let base = b"model.encoder.layers.";
    let mut used = base.len();
    prefix[..used].copy_from_slice(base);
    let mut digits = [0_u8; 20];
    let mut value = block;
    let mut count = 0;
    loop {
        digits[count] = b'0'
            + u8::try_from(value % 10)
                .expect("decimal digit conversion is bounded to zero through nine");
        count += 1;
        value /= 10;
        if value == 0 {
            break;
        }
    }
    while count != 0 {
        count -= 1;
        prefix[used] = digits[count];
        used += 1;
    }
    prefix[used] = b'.';
    used += 1;
    let mut has = |suffix: &[u8], dims: &[u64], kind: SerializedType| {
        let end = used + suffix.len();
        if end > prefix.len() {
            return false;
        }
        prefix[used..end].copy_from_slice(suffix);
        has_tensor(model, &prefix[..end], dims, kind)
    };
    has(b"self_attn.k_proj.weight", &[384, 384], linear)
        && has(b"self_attn.v_proj.weight", &[384, 384], linear)
        && has(b"self_attn.v_proj.bias", &[384], aux)
        && has(b"self_attn.q_proj.weight", &[384, 384], linear)
        && has(b"self_attn.q_proj.bias", &[384], aux)
        && has(b"self_attn.out_proj.weight", &[384, 384], linear)
        && has(b"self_attn.out_proj.bias", &[384], aux)
        && has(b"self_attn_layer_norm.weight", &[384], aux)
        && has(b"self_attn_layer_norm.bias", &[384], aux)
        && has(b"fc1.weight", &[384, 1536], linear)
        && has(b"fc1.bias", &[1536], aux)
        && has(b"fc2.weight", &[1536, 384], linear)
        && has(b"fc2.bias", &[384], aux)
        && has(b"final_layer_norm.weight", &[384], aux)
        && has(b"final_layer_norm.bias", &[384], aux)
}

fn model_base_valid(model: &Data) -> bool {
    has_tensor(model, b"mel_filters", &[201, 80], SerializedType::F32)
        && has_tensor(
            model,
            b"model.encoder.conv1.weight",
            &[3, 80, 384],
            SerializedType::F16,
        )
        && has_any_aux_vector(model, b"model.encoder.conv1.bias", 384)
        && has_tensor(
            model,
            b"model.encoder.conv2.weight",
            &[3, 384, 384],
            SerializedType::F16,
        )
        && has_any_aux_vector(model, b"model.encoder.conv2.bias", 384)
        && has_any_aux_position_matrix(model, b"model.encoder.embed_positions.weight")
        && has_any_aux_vector(model, b"model.encoder.layer_norm.weight", 384)
        && has_any_aux_vector(model, b"model.encoder.layer_norm.bias", 384)
}

fn model_variant(model: &Data) -> WeightVariant {
    let all = |linear, aux| (0..4).all(|block| has_encoder_block(model, block, linear, aux));
    if all(SerializedType::Q8_0, SerializedType::F32) {
        WeightVariant::Q8_0F32Aux
    } else if all(SerializedType::Q8_0, SerializedType::Q8_0) {
        WeightVariant::Q8_0
    } else if all(SerializedType::Q4_0, SerializedType::Q8_0) {
        WeightVariant::Q4_0
    } else if all(SerializedType::Q4_1, SerializedType::Q8_0) {
        WeightVariant::Q4_1
    } else {
        WeightVariant::Unsupported
    }
}

#[allow(clippy::too_many_lines)]
fn model_contract_valid(event: &EventEncodeRun<'_>) -> bool {
    let Some(model) = event.model else {
        return false;
    };
    event.contract == &ExecutionContract::bind_model(model)
        && event.contract.model_present
        && model.architecture_name() == b"whisper"
        && event.contract.sample_rate == SAMPLE_RATE
        && event.contract.mel_bin_count == MEL_BIN_COUNT
        && event.contract.embedding_length == EMBEDDING_LENGTH
        && event.contract.feed_forward_length == FEED_FORWARD_LENGTH
        && event.contract.attention_head_count == ATTENTION_HEAD_COUNT
        && event.contract.encoder_block_count == ENCODER_BLOCK_COUNT
        && model_base_valid(model)
}

sml! {
    SpeechEncoderWhisper<'dispatch, 'event>
    where
        'event: 'dispatch,
    {
        "state_model_contract_decision"_s <= *"state_ready"_s + EventEncodeRun(&'dispatch EventEncodeRun<'event>) / effect_begin_encode,
        "state_sample_rate_decision"_s <= "state_model_contract_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_model_contract_valid],
        "state_error_error_out_decision"_s <= "state_model_contract_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_model_contract_invalid] / effect_mark_model_invalid,
        "state_channel_count_decision"_s <= "state_sample_rate_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_sample_rate_valid],
        "state_error_error_out_decision"_s <= "state_sample_rate_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_sample_rate_invalid] / effect_mark_sample_rate_invalid,
        "state_pcm_shape_decision"_s <= "state_channel_count_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_channel_count_valid],
        "state_error_error_out_decision"_s <= "state_channel_count_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_channel_count_invalid] / effect_mark_channel_count_invalid,
        "state_output_capacity_decision"_s <= "state_pcm_shape_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_pcm_shape_valid],
        "state_error_error_out_decision"_s <= "state_pcm_shape_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_pcm_shape_invalid] / effect_mark_pcm_shape_invalid,
        "state_workspace_capacity_decision"_s <= "state_output_capacity_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_output_capacity_valid],
        "state_error_error_out_decision"_s <= "state_output_capacity_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_output_capacity_invalid] / effect_mark_output_capacity_invalid,
        "state_variant_decision"_s <= "state_workspace_capacity_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_workspace_capacity_valid],
        "state_error_error_out_decision"_s <= "state_workspace_capacity_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_workspace_capacity_invalid] / effect_mark_workspace_capacity_invalid,
        "state_running_q8_0_f32_aux"_s <= "state_variant_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_q8_0_f32_aux_variant] / effect_run_encoder_q8_0_f32_aux,
        "state_running_q8_0"_s <= "state_variant_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_q8_0_variant] / effect_run_encoder_q8_0,
        "state_running_q4_0"_s <= "state_variant_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_q4_0_variant] / effect_run_encoder_q4_0,
        "state_running_q4_1"_s <= "state_variant_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_q4_1_variant] / effect_run_encoder_q4_1,
        "state_error_error_out_decision"_s <= "state_variant_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_unsupported_variant] / effect_mark_unsupported_variant,
        "state_success_error_out_decision"_s <= "state_running_q8_0_f32_aux"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_encoder_success],
        "state_success_error_out_decision"_s <= "state_running_q8_0"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_encoder_success],
        "state_success_error_out_decision"_s <= "state_running_q4_0"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_encoder_success],
        "state_success_error_out_decision"_s <= "state_running_q4_1"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_encoder_success],
        "state_error_error_out_decision"_s <= "state_running_q8_0_f32_aux"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_encoder_failure] / effect_mark_internal_error,
        "state_error_error_out_decision"_s <= "state_running_q8_0"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_encoder_failure] / effect_mark_internal_error,
        "state_error_error_out_decision"_s <= "state_running_q4_0"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_encoder_failure] / effect_mark_internal_error,
        "state_error_error_out_decision"_s <= "state_running_q4_1"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_encoder_failure] / effect_mark_internal_error,
        "state_success_callback_decision"_s <= "state_success_error_out_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_has_error_out] / effect_store_success_error,
        "state_success_callback_decision"_s <= "state_success_error_out_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_no_error_out],
        "state_error_callback_decision"_s <= "state_error_error_out_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_has_error_out] / effect_store_error_error,
        "state_error_callback_decision"_s <= "state_error_error_out_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_no_error_out],
        "state_done"_s <= "state_success_callback_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_has_done_callback] / effect_emit_done,
        "state_done"_s <= "state_success_callback_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_no_done_callback],
        "state_errored"_s <= "state_error_callback_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_has_error_callback] / effect_emit_error,
        "state_errored"_s <= "state_error_callback_decision"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>) [guard_no_error_callback],
        "state_ready"_s <= "state_done"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>),
        "state_ready"_s <= "state_errored"_s + completion<EventEncodeRun>(&'dispatch EventEncodeRun<'event>),
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

impl SpeechEncoderWhisperContext {
    #[allow(
        clippy::unnecessary_wraps,
        reason = "the SML effect boundary requires Result<(), ()> callbacks"
    )]
    fn run_encoder<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventEncodeRun<'event>,
        variant: WeightVariant,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        let required_output = required_encoder_output_floats(event.pcm.len());
        let required_workspace = required_workspace_floats(event.pcm.len());
        let Some(model) = event.model else {
            self.err = Error::ModelInvalid;
            return Ok(());
        };
        let mut workspace = event.workspace.borrow_mut();
        let mut output = event.encoder_state.borrow_mut();
        let Ok((frame_count, digest)) = detail::run(
            model,
            variant,
            event.pcm,
            &mut workspace[..required_workspace],
            &mut output[..required_output],
        ) else {
            self.err = Error::InternalError;
            return Ok(());
        };
        **event.frame_count_out.borrow_mut() = frame_count;
        **event.width_out.borrow_mut() = EMBEDDING_LENGTH;
        **event.digest_out.borrow_mut() = digest;

        #[cfg(test)]
        #[allow(clippy::items_after_statements)]
        mod tests {
            use super::{
                HOP_LENGTH, MAX_MEL_FRAME_COUNT, MAX_PCM_SAMPLE_COUNT, encoder_frame_count,
                mel_frame_count, required_encoder_output_floats,
            };

            #[test]
            fn mel_frames_are_clamped_to_the_pinned_limit() {
                assert_eq!(mel_frame_count(16_000), 100);
                assert_eq!(mel_frame_count(MAX_PCM_SAMPLE_COUNT), MAX_MEL_FRAME_COUNT);
                assert_eq!(
                    mel_frame_count(MAX_PCM_SAMPLE_COUNT + HOP_LENGTH),
                    MAX_MEL_FRAME_COUNT
                );
            }

            #[test]
            fn derived_encoder_capacity_uses_clamped_mel_frames() {
                assert_eq!(encoder_frame_count(MAX_PCM_SAMPLE_COUNT), 1_500);
                assert_eq!(
                    required_encoder_output_floats(MAX_PCM_SAMPLE_COUNT),
                    1_500 * super::EMBEDDING_LENGTH as usize
                );
                assert_eq!(
                    required_encoder_output_floats(MAX_PCM_SAMPLE_COUNT + HOP_LENGTH),
                    required_encoder_output_floats(MAX_PCM_SAMPLE_COUNT)
                );
            }
        }
        self.err = Error::None;
        match variant {
            WeightVariant::Q8_0F32Aux | WeightVariant::Q8_0 => {
                self.q8_0_dispatch_count = self.q8_0_dispatch_count.saturating_add(1);
            }
            WeightVariant::Q4_0 => {
                self.q4_0_dispatch_count = self.q4_0_dispatch_count.saturating_add(1);
            }
            WeightVariant::Q4_1 => {
                self.q4_1_dispatch_count = self.q4_1_dispatch_count.saturating_add(1);
            }
            WeightVariant::Unsupported => {}
        }
        Ok(())
    }
}

impl SpeechEncoderWhisperStateMachineContext for SpeechEncoderWhisperContext {
    fn effect_begin_encode<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.err = Error::None;
        **event.frame_count_out.borrow_mut() = 0;
        **event.width_out.borrow_mut() = 0;
        **event.digest_out.borrow_mut() = 0;
        Ok(())
    }
    fn effect_emit_done<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        if let Some(callback) = event.on_done
            && !callback(EncodeDone {
                frame_count: **event.frame_count_out.borrow(),
                width: **event.width_out.borrow(),
                digest: **event.digest_out.borrow(),
            })
        {
            self.err = Error::InternalError;
        }
        Ok(())
    }
    fn effect_emit_error<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        if let Some(callback) = event.on_error {
            let _ = callback(EncodeError { error: self.err });
        }
        Ok(())
    }
    fn effect_mark_model_invalid<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventEncodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.err = Error::ModelInvalid;
        Ok(())
    }
    fn effect_mark_sample_rate_invalid<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventEncodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.err = Error::SampleRate;
        Ok(())
    }
    fn effect_mark_channel_count_invalid<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventEncodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.err = Error::ChannelCount;
        Ok(())
    }
    fn effect_mark_pcm_shape_invalid<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventEncodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.err = Error::PcmShape;
        Ok(())
    }
    fn effect_mark_output_capacity_invalid<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventEncodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.err = Error::OutputCapacity;
        Ok(())
    }
    fn effect_mark_workspace_capacity_invalid<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventEncodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.err = Error::WorkspaceCapacity;
        Ok(())
    }
    fn effect_mark_unsupported_variant<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventEncodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.err = Error::UnsupportedVariant;
        Ok(())
    }
    fn effect_mark_internal_error<'dispatch, 'event>(
        &mut self,
        _: &'dispatch EventEncodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.err = Error::InternalError;
        Ok(())
    }
    fn effect_run_encoder_q8_0_f32_aux<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.run_encoder(event, WeightVariant::Q8_0F32Aux)
    }
    fn effect_run_encoder_q8_0<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.run_encoder(event, WeightVariant::Q8_0)
    }
    fn effect_run_encoder_q4_0<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.run_encoder(event, WeightVariant::Q4_0)
    }
    fn effect_run_encoder_q4_1<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        self.run_encoder(event, WeightVariant::Q4_1)
    }
    fn effect_store_success_error<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        if let Some(error_out) = event.error_out.borrow_mut().as_deref_mut() {
            *error_out = self.err;
        }
        Ok(())
    }
    fn effect_store_error_error<'dispatch, 'event>(
        &mut self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<(), ()>
    where
        'event: 'dispatch,
    {
        if let Some(error_out) = event.error_out.borrow_mut().as_deref_mut() {
            *error_out = self.err;
        }
        Ok(())
    }
    fn guard_model_contract_valid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(model_contract_valid(event))
    }
    fn guard_model_contract_invalid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!model_contract_valid(event))
    }
    fn guard_sample_rate_valid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.sample_rate == SAMPLE_RATE)
    }
    fn guard_sample_rate_invalid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.sample_rate != SAMPLE_RATE)
    }
    fn guard_channel_count_valid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.channel_count == CHANNEL_COUNT)
    }
    fn guard_channel_count_invalid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.channel_count != CHANNEL_COUNT)
    }
    fn guard_pcm_shape_valid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!event.pcm.is_empty()
            && event.pcm.len() <= MAX_PCM_SAMPLE_COUNT
            && event.pcm.iter().all(|sample| sample.is_finite()))
    }
    fn guard_pcm_shape_invalid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.pcm.is_empty()
            || event.pcm.len() > MAX_PCM_SAMPLE_COUNT
            || event.pcm.iter().any(|sample| !sample.is_finite()))
    }
    fn guard_output_capacity_valid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.encoder_state.borrow().len() >= required_encoder_output_floats(event.pcm.len()))
    }
    fn guard_output_capacity_invalid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!self.guard_output_capacity_valid(event)?)
    }
    fn guard_workspace_capacity_valid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.workspace.borrow().len() >= required_workspace_floats(event.pcm.len()))
    }
    fn guard_workspace_capacity_invalid<'dispatch, 'event>(
        &self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(!self.guard_workspace_capacity_valid(event)?)
    }
    fn guard_q8_0_f32_aux_variant<'dispatch, 'event>(
        &self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event
            .model
            .is_some_and(|model| model_variant(model) == WeightVariant::Q8_0F32Aux))
    }
    fn guard_q8_0_variant<'dispatch, 'event>(
        &self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event
            .model
            .is_some_and(|model| model_variant(model) == WeightVariant::Q8_0))
    }
    fn guard_q4_0_variant<'dispatch, 'event>(
        &self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event
            .model
            .is_some_and(|model| model_variant(model) == WeightVariant::Q4_0))
    }
    fn guard_q4_1_variant<'dispatch, 'event>(
        &self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event
            .model
            .is_some_and(|model| model_variant(model) == WeightVariant::Q4_1))
    }
    fn guard_unsupported_variant<'dispatch, 'event>(
        &self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event
            .model
            .is_none_or(|model| model_variant(model) == WeightVariant::Unsupported))
    }
    fn guard_encoder_success<'dispatch, 'event>(
        &self,
        _: &'dispatch EventEncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(self.err == Error::None)
    }
    fn guard_encoder_failure<'dispatch, 'event>(
        &self,
        _: &'dispatch EventEncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(self.err != Error::None)
    }
    fn guard_has_error_out<'dispatch, 'event>(
        &self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.error_out.borrow().is_some())
    }
    fn guard_no_error_out<'dispatch, 'event>(
        &self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.error_out.borrow().is_none())
    }
    fn guard_has_done_callback<'dispatch, 'event>(
        &self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.on_done.is_some())
    }
    fn guard_no_done_callback<'dispatch, 'event>(
        &self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.on_done.is_none())
    }
    fn guard_has_error_callback<'dispatch, 'event>(
        &self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.on_error.is_some())
    }
    fn guard_no_error_callback<'dispatch, 'event>(
        &self,
        event: &'dispatch EventEncodeRun<'event>,
    ) -> Result<bool, ()>
    where
        'event: 'dispatch,
    {
        Ok(event.on_error.is_none())
    }

    fn effect_on_unexpected_from_state_ready(&mut self) -> Result<(), ()> {
        self.err = Error::Unexpected;
        Ok(())
    }
    fn effect_on_unexpected_from_state_model_contract_decision(&mut self) -> Result<(), ()> {
        self.err = Error::Unexpected;
        Ok(())
    }
    fn effect_on_unexpected_from_state_sample_rate_decision(&mut self) -> Result<(), ()> {
        self.err = Error::Unexpected;
        Ok(())
    }
    fn effect_on_unexpected_from_state_channel_count_decision(&mut self) -> Result<(), ()> {
        self.err = Error::Unexpected;
        Ok(())
    }
    fn effect_on_unexpected_from_state_pcm_shape_decision(&mut self) -> Result<(), ()> {
        self.err = Error::Unexpected;
        Ok(())
    }
    fn effect_on_unexpected_from_state_output_capacity_decision(&mut self) -> Result<(), ()> {
        self.err = Error::Unexpected;
        Ok(())
    }
    fn effect_on_unexpected_from_state_workspace_capacity_decision(&mut self) -> Result<(), ()> {
        self.err = Error::Unexpected;
        Ok(())
    }
    fn effect_on_unexpected_from_state_variant_decision(&mut self) -> Result<(), ()> {
        self.err = Error::Unexpected;
        Ok(())
    }
    fn effect_on_unexpected_from_state_running_q8_0(&mut self) -> Result<(), ()> {
        self.err = Error::Unexpected;
        Ok(())
    }
    fn effect_on_unexpected_from_state_running_q8_0_f32_aux(&mut self) -> Result<(), ()> {
        self.err = Error::Unexpected;
        Ok(())
    }
    fn effect_on_unexpected_from_state_running_q4_0(&mut self) -> Result<(), ()> {
        self.err = Error::Unexpected;
        Ok(())
    }
    fn effect_on_unexpected_from_state_running_q4_1(&mut self) -> Result<(), ()> {
        self.err = Error::Unexpected;
        Ok(())
    }
    fn effect_on_unexpected_from_state_success_error_out_decision(&mut self) -> Result<(), ()> {
        self.err = Error::Unexpected;
        Ok(())
    }
    fn effect_on_unexpected_from_state_success_callback_decision(&mut self) -> Result<(), ()> {
        self.err = Error::Unexpected;
        Ok(())
    }
    fn effect_on_unexpected_from_state_error_error_out_decision(&mut self) -> Result<(), ()> {
        self.err = Error::Unexpected;
        Ok(())
    }
    fn effect_on_unexpected_from_state_error_callback_decision(&mut self) -> Result<(), ()> {
        self.err = Error::Unexpected;
        Ok(())
    }
    fn effect_on_unexpected_from_state_done(&mut self) -> Result<(), ()> {
        self.err = Error::Unexpected;
        Ok(())
    }
    fn effect_on_unexpected_from_state_errored(&mut self) -> Result<(), ()> {
        self.err = Error::Unexpected;
        Ok(())
    }
}

/// Public synchronous actor wrapper around the generated state machine.
pub struct SpeechEncoderWhisperActor {
    machine: SpeechEncoderWhisperStateMachine<SpeechEncoderWhisperContext>,
}

impl Default for SpeechEncoderWhisperActor {
    fn default() -> Self {
        Self::new()
    }
}

impl SpeechEncoderWhisperActor {
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: SpeechEncoderWhisperStateMachine::new(SpeechEncoderWhisperContext::default()),
        }
    }

    /// Dispatches one borrowed request through the complete source phase graph.
    #[allow(
        clippy::needless_pass_by_value,
        reason = "source-aligned API consumes the request to preserve borrow ownership"
    )]
    pub fn process_event(&mut self, event: EventEncodeRun<'_>) -> bool {
        self.machine
            .process_event(SpeechEncoderWhisperEvents::EventEncodeRun(&event))
            .is_ok()
            && self.machine.context().err == Error::None
    }

    pub fn encode(&mut self, event: EventEncodeRun<'_>) -> bool {
        self.process_event(event)
    }

    #[must_use]
    pub fn context(&self) -> &SpeechEncoderWhisperContext {
        self.machine.context()
    }

    #[must_use]
    pub fn state(&self) -> &SpeechEncoderWhisperStates {
        self.machine.state()
    }

    #[must_use]
    pub fn q8_0_dispatch_count(&self) -> u64 {
        self.machine.context().q8_0_dispatch_count
    }

    #[must_use]
    pub fn q4_0_dispatch_count(&self) -> u64 {
        self.machine.context().q4_0_dispatch_count
    }

    #[must_use]
    pub fn q4_1_dispatch_count(&self) -> u64 {
        self.machine.context().q4_1_dispatch_count
    }
}

/// Source-compatible request actor name.
pub type Request = SpeechEncoderWhisperActor;
/// Direct access to the generated state-machine type.
// Scope frozen: this file only. Model-bound validation now checks exact resident
// tensor shapes/types, variant guards inspect the supplied model, and execution
// uses the typed kernel handoff before the legacy callback fallback. Kernel or
// callback rejection marks InternalError, preventing success publication.
#[cfg(test)]
mod tests {
    use emel_model::bridge::{
        Data, TensorInput, TensorMetadata, TensorMetadataInput, WhisperDataInput,
        WhisperHParamsInput,
    };
    use emel_tensor::dtype::SerializedType;

    use super::{
        CHANNEL_COUNT, Error, EventEncodeRun, ExecutionContract, HOP_LENGTH, MAX_MEL_FRAME_COUNT,
        SAMPLE_RATE, SpeechEncoderWhisperActor, encoder_frame_count, mel_frame_count,
        required_encoder_output_floats, required_workspace_floats,
    };

    #[allow(clippy::too_many_lines)]
    fn valid_model() -> Data {
        let mut specs = Vec::new();
        let mut add = |name: Vec<u8>, tensor_type, dimensions, dimension_count| {
            specs.push((name, tensor_type, dimensions, dimension_count));
        };
        add(
            b"mel_filters".to_vec(),
            SerializedType::F32,
            [201, 80, 1, 1],
            2,
        );
        add(
            b"model.encoder.conv1.weight".to_vec(),
            SerializedType::F16,
            [3, 80, 384, 1],
            3,
        );
        add(
            b"model.encoder.conv1.bias".to_vec(),
            SerializedType::Q8_0,
            [384, 1, 1, 1],
            1,
        );
        add(
            b"model.encoder.conv2.weight".to_vec(),
            SerializedType::F16,
            [3, 384, 384, 1],
            3,
        );
        add(
            b"model.encoder.conv2.bias".to_vec(),
            SerializedType::Q8_0,
            [384, 1, 1, 1],
            1,
        );
        add(
            b"model.encoder.embed_positions.weight".to_vec(),
            SerializedType::Q8_0,
            [384, 1500, 1, 1],
            2,
        );
        add(
            b"model.encoder.layer_norm.weight".to_vec(),
            SerializedType::Q8_0,
            [384, 1, 1, 1],
            1,
        );
        add(
            b"model.encoder.layer_norm.bias".to_vec(),
            SerializedType::Q8_0,
            [384, 1, 1, 1],
            1,
        );
        add(
            b"model.decoder.embed_tokens.weight".to_vec(),
            SerializedType::Q8_0,
            [384, 51865, 1, 1],
            2,
        );
        add(
            b"model.decoder.embed_positions.weight".to_vec(),
            SerializedType::Q8_0,
            [384, 448, 1, 1],
            2,
        );
        add(
            b"model.decoder.layer_norm.weight".to_vec(),
            SerializedType::Q8_0,
            [384, 1, 1, 1],
            1,
        );
        add(
            b"model.decoder.layer_norm.bias".to_vec(),
            SerializedType::Q8_0,
            [384, 1, 1, 1],
            1,
        );
        for block in 0..4 {
            for suffix in ["q_proj.weight", "q_proj.bias"] {
                let name = format!("model.encoder.layers.{block}.self_attn.{suffix}").into_bytes();
                let (tensor_type, dimensions, dimension_count) = if suffix.ends_with("weight") {
                    (SerializedType::Q4_0, [384, 384, 1, 1], 2)
                } else {
                    (SerializedType::Q8_0, [384, 1, 1, 1], 1)
                };
                add(name, tensor_type, dimensions, dimension_count);
            }
        }
        for block in 0..4 {
            for prefix in ["self_attn", "encoder_attn"] {
                for suffix in ["q_proj.weight", "q_proj.bias"] {
                    let name =
                        format!("model.decoder.layers.{block}.{prefix}.{suffix}").into_bytes();
                    let (tensor_type, dimensions, dimension_count) = if suffix.ends_with("weight") {
                        (SerializedType::Q4_0, [384, 384, 1, 1], 2)
                    } else {
                        (SerializedType::Q8_0, [384, 1, 1, 1], 1)
                    };
                    add(name, tensor_type, dimensions, dimension_count);
                }
            }
        }
        let payloads: Vec<Vec<u8>> = specs
            .iter()
            .map(|(_, tensor_type, dimensions, dimension_count)| {
                vec![
                    0;
                    usize::try_from(
                        tensor_type
                            .data_size(*dimensions, *dimension_count)
                            .unwrap()
                    )
                    .unwrap()
                ]
            })
            .collect();
        let tensors: Vec<TensorInput<'_>> = specs
            .iter()
            .zip(payloads.iter())
            .map(
                |((name, tensor_type, dimensions, dimension_count), bytes)| {
                    TensorInput::with_bytes(
                        name,
                        TensorMetadata::new(TensorMetadataInput {
                            tensor_type: *tensor_type,
                            dimension_count: *dimension_count,
                            dimensions: *dimensions,
                            data_offset: 0,
                            file_offset: 0,
                            data_size: u64::try_from(bytes.len()).unwrap(),
                            file_index: 0,
                            storage: None,
                        }),
                        bytes,
                    )
                },
            )
            .collect();
        Data::try_from_whisper(WhisperDataInput {
            architecture: b"whisper",
            hparams: WhisperHParamsInput::default(),
            tensors: &tensors,
        })
        .expect("complete Whisper model fixture")
    }

    #[test]
    fn model_binding_rejects_unpopulated_data() {
        let model = Data::try_new().expect("bounded model storage");
        let contract = ExecutionContract::bind_model(&model);

        assert!(!contract.model_present);
        assert!(!contract.model_contract_valid());
        assert_eq!(contract.assets.encoder_blocks, 0);
    }

    #[test]
    fn preprocessing_frame_bounds_match_pinned_hop_and_context() {
        assert_eq!(mel_frame_count(1), 1);
        assert_eq!(mel_frame_count(HOP_LENGTH), 1);
        assert_eq!(mel_frame_count(HOP_LENGTH + 1), 2);
        assert_eq!(
            mel_frame_count(MAX_MEL_FRAME_COUNT * HOP_LENGTH),
            MAX_MEL_FRAME_COUNT
        );
        assert_eq!(encoder_frame_count(MAX_MEL_FRAME_COUNT * HOP_LENGTH), 1_500);
    }

    #[test]
    fn public_actor_maps_non_finite_pcm_to_pcm_shape_error() {
        let model = valid_model();
        let contract = ExecutionContract::bind_model(&model);
        assert!(contract.model_contract_valid());
        let pcm = [f32::NAN];
        let mut workspace = vec![0.0; required_workspace_floats(pcm.len())];
        let mut encoder_state = vec![0.0; required_encoder_output_floats(pcm.len())];
        let mut frame_count = -1;
        let mut width = -1;
        let mut digest = u64::MAX;
        let mut error = Error::None;
        let mut event = EventEncodeRun::new(
            &contract,
            &pcm,
            SAMPLE_RATE,
            CHANNEL_COUNT,
            &mut workspace,
            &mut encoder_state,
            &mut frame_count,
            &mut width,
            &mut digest,
        )
        .with_model(&model);
        event.error_out = core::cell::RefCell::new(Some(&mut error));

        let mut actor = SpeechEncoderWhisperActor::new();
        assert!(!actor.process_event(event));
        assert_eq!(error, Error::PcmShape);
        assert_eq!(frame_count, 0);
        assert_eq!(width, 0);
        assert_eq!(digest, 0);
        assert_eq!(actor.q8_0_dispatch_count(), 0);
        assert_eq!(actor.q4_0_dispatch_count(), 0);
        assert_eq!(actor.q4_1_dispatch_count(), 0);
    }
    #[test]
    fn forged_model_contract_is_rejected_before_native_dispatch() {
        let model = valid_model();
        let bound = ExecutionContract::bind_model(&model);
        assert!(bound.model_contract_valid());
        let mut forged = bound;
        forged.model_present = false;
        forged.embedding_length += 1;
        let pcm = [0.0_f32];
        let mut workspace = vec![0.0; required_workspace_floats(pcm.len())];
        let mut encoder_state = vec![0.0; required_encoder_output_floats(pcm.len())];
        let mut frame_count = -1;
        let mut width = -1;
        let mut digest = u64::MAX;
        let mut error = Error::None;
        let mut event = EventEncodeRun::new(
            &forged,
            &pcm,
            SAMPLE_RATE,
            CHANNEL_COUNT,
            &mut workspace,
            &mut encoder_state,
            &mut frame_count,
            &mut width,
            &mut digest,
        )
        .with_model(&model);
        event.error_out = core::cell::RefCell::new(Some(&mut error));

        let mut actor = SpeechEncoderWhisperActor::new();
        assert!(!actor.process_event(event));
        assert_eq!(error, Error::ModelInvalid);
        assert_eq!(frame_count, 0);
        assert_eq!(width, 0);
        assert_eq!(digest, 0);
        assert_eq!(actor.q8_0_dispatch_count(), 0);
        assert_eq!(actor.q4_0_dispatch_count(), 0);
        assert_eq!(actor.q4_1_dispatch_count(), 0);
    }

    #[test]
    fn stale_pinned_contract_is_rejected_before_native_dispatch() {
        let model = valid_model();
        let bound = ExecutionContract::bind_model(&model);
        let stale = ExecutionContract::pinned();
        assert_ne!(stale, bound);
        let pcm = [0.0_f32];
        let mut workspace = vec![0.0; required_workspace_floats(pcm.len())];
        let mut encoder_state = vec![0.0; required_encoder_output_floats(pcm.len())];
        let mut frame_count = -1;
        let mut width = -1;
        let mut digest = u64::MAX;
        let mut error = Error::None;
        let mut event = EventEncodeRun::new(
            &stale,
            &pcm,
            SAMPLE_RATE,
            CHANNEL_COUNT,
            &mut workspace,
            &mut encoder_state,
            &mut frame_count,
            &mut width,
            &mut digest,
        )
        .with_model(&model);
        event.error_out = core::cell::RefCell::new(Some(&mut error));

        let mut actor = SpeechEncoderWhisperActor::new();
        assert!(!actor.process_event(event));
        assert_eq!(error, Error::ModelInvalid);
        assert_eq!(frame_count, 0);
        assert_eq!(width, 0);
        assert_eq!(digest, 0);
        assert_eq!(actor.q8_0_dispatch_count(), 0);
        assert_eq!(actor.q4_0_dispatch_count(), 0);
        assert_eq!(actor.q4_1_dispatch_count(), 0);
    }

    #[test]
    fn matching_bound_contract_preserves_model_validation() {
        let model = valid_model();
        let contract = ExecutionContract::bind_model(&model);
        assert_eq!(contract, ExecutionContract::bind_model(&model));
        assert!(contract.model_contract_valid());
    }
}
