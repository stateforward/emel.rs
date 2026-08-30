//! Source-aligned, bounded Whisper speech decoder actor.
//!
//! The state topology mirrors the pinned
//! `speech/decoder/whisper/{sm,context,events,actions,guards,errors}.hpp`
//! contract.  Requests borrow caller-owned model, encoder, and scratch storage;
//! dispatch is synchronous and does not allocate.  The numeric decoder is
//! supplied through the typed kernel callback because this crate does not yet
//! expose the complete Whisper decoder sequence as a Rust kernel actor.

#![allow(
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::empty_structs_with_brackets,
    clippy::missing_const_for_fn,
    dead_code,
    unused_imports,
    missing_docs,
    private_interfaces
)]

use emel_model::bridge::Data;
use emel_tensor::dtype::SerializedType;
use sml::sml;

const EMBEDDING_LENGTH: usize = 384;
const FEED_FORWARD_LENGTH: usize = 1536;
const DECODER_BLOCK_COUNT: usize = 4;
const VOCAB_SIZE: usize = 51_865;
const MAX_ENCODER_FRAME_COUNT: i32 = 1500;
const DECODER_SEQUENCE_TOKEN_COUNT: usize = 448;
const MAX_GENERATED_TOKEN_COUNT: usize = DECODER_SEQUENCE_TOKEN_COUNT - 4;

/// Stable Whisper decoder failure values from the pinned `error` enum.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum WhisperDecoderError {
    /// No error occurred.
    #[default]
    None = 0,
    /// The model binding or required tensor contract is invalid.
    ModelInvalid = 1,
    /// The encoder state is absent, empty, or too large.
    EncoderState = 2,
    /// A prompt/control token contract is invalid.
    PromptToken = 3,
    /// The logits destination is too small.
    LogitsCapacity = 4,
    /// The transcript destination is too small (reserved for the tokenizer boundary).
    TranscriptCapacity = 5,
    /// The decoder workspace is too small.
    WorkspaceCapacity = 6,
    /// No maintained quantized decoder variant matches the model.
    UnsupportedVariant = 7,
    /// The injected numeric decoder rejected the request or is unavailable.
    InternalError = 8,
    /// The decode policy is not the maintained tiny-ASR policy.
    DecodePolicy = 9,
    /// The generated-token destination is absent or empty.
    GeneratedTokenCapacity = 10,
}

/// Whisper control-token values copied from the maintained tiny tokenizer policy.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ControlTokens {
    pub eot: i32,
    pub sot: i32,
    pub language_en: i32,
    pub translate: i32,
    pub transcribe: i32,
    pub no_speech: i32,
    pub notimestamps: i32,
    pub timestamp_begin: i32,
    pub space: i32,
}

/// Policy roles used by the maintained tiny Whisper ASR route.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum LanguageRole {
    #[default]
    English = 0,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum TaskRole {
    #[default]
    Transcribe = 0,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum TimestampMode {
    #[default]
    TimestampTokens = 0,
}

/// Bounded, copied decode policy.  No policy storage is allocated by dispatch.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DecodePolicy {
    pub tokens: ControlTokens,
    pub language: LanguageRole,
    pub task: TaskRole,
    pub timestamps: TimestampMode,
    pub suppress_translate: bool,
    pub prompt_tokens: [i32; 3],
}

impl DecodePolicy {
    /// The policy used by the maintained tiny Whisper runtime.
    #[must_use]
    pub const fn tiny_asr() -> Self {
        Self {
            tokens: ControlTokens {
                eot: 50_257,
                sot: 50_258,
                language_en: 50_259,
                translate: 50_358,
                transcribe: 50_359,
                no_speech: 50_362,
                notimestamps: 50_363,
                timestamp_begin: 50_364,
                space: 220,
            },
            language: LanguageRole::English,
            task: TaskRole::Transcribe,
            timestamps: TimestampMode::TimestampTokens,
            suppress_translate: true,
            prompt_tokens: [50_258, 50_259, 50_359],
        }
    }

    #[must_use]
    const fn supported(self) -> bool {
        self == Self::tiny_asr()
    }
}

/// Variant-neutral model dimensions bound at the speech owner boundary.
#[derive(Clone, Copy, Debug)]
pub struct WhisperExecutionContract {
    pub vocab_size: i32,
    pub embedding_length: i32,
    pub decoder_block_count: i32,
}

impl WhisperExecutionContract {
    #[must_use]
    pub const fn bind() -> Self {
        Self { vocab_size: VOCAB_SIZE as i32, embedding_length: EMBEDDING_LENGTH as i32, decoder_block_count: DECODER_BLOCK_COUNT as i32 }
    }
}

/// Maintained decoder quantization route selected by the model guard.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodeVariant {
    Q8_0F32Aux,
    Q8_0,
    Q4_0,
    Q4_1,
}

#[derive(Debug)]
pub struct DecodeKernelRequest<'dispatch> {
    pub variant: DecodeVariant,
    pub model: &'dispatch Data,
    pub contract: &'dispatch WhisperExecutionContract,
    pub encoder_state: &'dispatch [f32],
    pub encoder_frame_count: i32,
    pub policy: DecodePolicy,
    pub generated_tokens: &'dispatch mut [i32],
    pub generated_token_count_out: &'dispatch mut i32,
    pub workspace: &'dispatch mut [f32],
    pub logits: &'dispatch mut [f32],
    pub token_out: &'dispatch mut i32,
    pub confidence_out: &'dispatch mut f32,
    pub digest_out: &'dispatch mut u64,
}

pub type DecodeKernelFn = for<'a> fn(DecodeKernelRequest<'a>) -> bool;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeDone { pub token: i32, pub confidence_bits: u32, pub digest: u64 }
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeError { pub error: WhisperDecoderError }
pub type DoneCallback = fn(DecodeDone) -> bool;
pub type ErrorCallback = fn(DecodeError) -> bool;

#[derive(Debug)]
pub struct DecodeRequest<'dispatch> {
    pub model: &'dispatch Data,
    pub contract: WhisperExecutionContract,
    pub encoder_state: &'dispatch [f32],
    pub encoder_frame_count: i32,
    pub policy: DecodePolicy,
    pub generated_tokens: &'dispatch mut [i32],
    pub generated_token_count_out: &'dispatch mut i32,
    pub workspace: &'dispatch mut [f32],
    pub logits: &'dispatch mut [f32],
    pub token_out: &'dispatch mut i32,
    pub confidence_out: &'dispatch mut f32,
    pub digest_out: &'dispatch mut u64,
    pub error_out: Option<&'dispatch mut WhisperDecoderError>,
    pub on_done: Option<DoneCallback>,
    pub on_error: Option<ErrorCallback>,
    pub decode: Option<DecodeKernelFn>,
}

impl<'dispatch> DecodeRequest<'dispatch> {
    #[must_use]
    pub fn new(model: &'dispatch Data, contract: WhisperExecutionContract, encoder_state: &'dispatch [f32], encoder_frame_count: i32, policy: DecodePolicy, generated_tokens: &'dispatch mut [i32], generated_token_count_out: &'dispatch mut i32, workspace: &'dispatch mut [f32], logits: &'dispatch mut [f32], token_out: &'dispatch mut i32, confidence_out: &'dispatch mut f32, digest_out: &'dispatch mut u64) -> Self {
        Self { model, contract, encoder_state, encoder_frame_count, policy, generated_tokens, generated_token_count_out, workspace, logits, token_out, confidence_out, digest_out, error_out: None, on_done: None, on_error: None, decode: None }
    }
    #[must_use] pub const fn with_error_out(mut self, error_out: &'dispatch mut WhisperDecoderError) -> Self { self.error_out = Some(error_out); self }
    #[must_use] pub const fn with_callbacks(mut self, on_done: Option<DoneCallback>, on_error: Option<ErrorCallback>) -> Self { self.on_done = on_done; self }
    #[must_use] pub const fn with_decoder(mut self, decode: DecodeKernelFn) -> Self { self.decode = Some(decode); self }
}

/// Runtime event corresponding to C++ `event::decode_run`.
#[derive(Debug)]
pub struct EventDecodeRun<'dispatch> { pub request: DecodeRequest<'dispatch> }
impl<'dispatch> EventDecodeRun<'dispatch> { #[must_use] pub const fn new(request: DecodeRequest<'dispatch>) -> Self { Self { request } } }
sml! {
    SpeechDecoderWhisper<'dispatch> {
        "state_model_contract_decision"_s <= *"state_ready"_s + event<EventDecodeRun<'dispatch>> / effect_begin_decode,
        "state_encoder_state_decision"_s <= "state_model_contract_decision"_s + completion<EventDecodeRun<'dispatch>> [guard_model_contract_valid],
        "state_error_error_out_decision"_s <= "state_model_contract_decision"_s + completion<EventDecodeRun<'dispatch>> [guard_model_contract_invalid] / effect_mark_model_invalid,
        "state_decode_policy_decision"_s <= "state_encoder_state_decision"_s + completion<EventDecodeRun<'dispatch>> [guard_encoder_state_valid],
        "state_error_error_out_decision"_s <= "state_encoder_state_decision"_s + completion<EventDecodeRun<'dispatch>> [guard_encoder_state_invalid] / effect_mark_encoder_state_invalid,
        "state_generated_token_capacity_decision"_s <= "state_decode_policy_decision"_s + completion<EventDecodeRun<'dispatch>> [guard_decode_policy_supported],
        "state_error_error_out_decision"_s <= "state_decode_policy_decision"_s + completion<EventDecodeRun<'dispatch>> [guard_decode_policy_unsupported] / effect_mark_decode_policy_invalid,
        "state_logits_capacity_decision"_s <= "state_generated_token_capacity_decision"_s + completion<EventDecodeRun<'dispatch>> [guard_generated_token_capacity_valid],
        "state_error_error_out_decision"_s <= "state_generated_token_capacity_decision"_s + completion<EventDecodeRun<'dispatch>> [guard_generated_token_capacity_invalid] / effect_mark_generated_token_capacity_invalid,
        "state_workspace_capacity_decision"_s <= "state_logits_capacity_decision"_s + completion<EventDecodeRun<'dispatch>> [guard_logits_capacity_valid],
        "state_error_error_out_decision"_s <= "state_logits_capacity_decision"_s + completion<EventDecodeRun<'dispatch>> [guard_logits_capacity_invalid] / effect_mark_logits_capacity_invalid,
        "state_variant_decision"_s <= "state_workspace_capacity_decision"_s + completion<EventDecodeRun<'dispatch>> [guard_workspace_capacity_valid],
        "state_error_error_out_decision"_s <= "state_workspace_capacity_decision"_s + completion<EventDecodeRun<'dispatch>> [guard_workspace_capacity_invalid] / effect_mark_workspace_capacity_invalid,
        "state_running_q8_0_f32_aux"_s <= "state_variant_decision"_s + completion<EventDecodeRun<'dispatch>> [guard_q8_0_f32_aux_variant] / effect_run_decoder_q8_0_f32_aux,
        "state_running_q8_0"_s <= "state_variant_decision"_s + completion<EventDecodeRun<'dispatch>> [guard_q8_0_variant] / effect_run_decoder_q8_0,
        "state_running_q4_0"_s <= "state_variant_decision"_s + completion<EventDecodeRun<'dispatch>> [guard_q4_0_variant] / effect_run_decoder_q4_0,
        "state_running_q4_1"_s <= "state_variant_decision"_s + completion<EventDecodeRun<'dispatch>> [guard_q4_1_variant] / effect_run_decoder_q4_1,
        "state_error_error_out_decision"_s <= "state_variant_decision"_s + completion<EventDecodeRun<'dispatch>> [guard_unsupported_variant] / effect_mark_unsupported_variant,
        "state_success_error_out_decision"_s <= "state_running_q8_0_f32_aux"_s + completion<EventDecodeRun<'dispatch>> [guard_decode_execution_success],
        "state_success_error_out_decision"_s <= "state_running_q8_0"_s + completion<EventDecodeRun<'dispatch>> [guard_decode_execution_success],
        "state_success_error_out_decision"_s <= "state_running_q4_0"_s + completion<EventDecodeRun<'dispatch>> [guard_decode_execution_success],
        "state_success_error_out_decision"_s <= "state_running_q4_1"_s + completion<EventDecodeRun<'dispatch>> [guard_decode_execution_success],
        "state_error_error_out_decision"_s <= "state_running_q8_0_f32_aux"_s + completion<EventDecodeRun<'dispatch>> [guard_decode_execution_failure] / effect_mark_internal_error,
        "state_error_error_out_decision"_s <= "state_running_q8_0"_s + completion<EventDecodeRun<'dispatch>> [guard_decode_execution_failure] / effect_mark_internal_error,
        "state_error_error_out_decision"_s <= "state_running_q4_0"_s + completion<EventDecodeRun<'dispatch>> [guard_decode_execution_failure] / effect_mark_internal_error,
        "state_error_error_out_decision"_s <= "state_running_q4_1"_s + completion<EventDecodeRun<'dispatch>> [guard_decode_execution_failure] / effect_mark_internal_error,
        "state_success_callback_decision"_s <= "state_success_error_out_decision"_s + completion<EventDecodeRun<'dispatch>> [guard_has_error_out] / effect_store_success_error,
        "state_success_callback_decision"_s <= "state_success_error_out_decision"_s + completion<EventDecodeRun<'dispatch>> [guard_no_error_out],
        "state_error_callback_decision"_s <= "state_error_error_out_decision"_s + completion<EventDecodeRun<'dispatch>> [guard_has_error_out] / effect_store_error_error,
        "state_error_callback_decision"_s <= "state_error_error_out_decision"_s + completion<EventDecodeRun<'dispatch>> [guard_no_error_out],
        "state_done"_s <= "state_success_callback_decision"_s + completion<EventDecodeRun<'dispatch>> [guard_has_done_callback] / effect_emit_done,
        "state_done"_s <= "state_success_callback_decision"_s + completion<EventDecodeRun<'dispatch>> [guard_no_done_callback],
        "state_errored"_s <= "state_error_callback_decision"_s + completion<EventDecodeRun<'dispatch>> [guard_has_error_callback] / effect_emit_error,
        "state_errored"_s <= "state_error_callback_decision"_s + completion<EventDecodeRun<'dispatch>> [guard_no_error_callback],
        "state_ready"_s <= "state_done"_s + completion<EventDecodeRun<'dispatch>>,
        "state_ready"_s <= "state_errored"_s + completion<EventDecodeRun<'dispatch>>,
        "state_ready"_s <= "state_ready"_s + unexpected_event<_> / effect_on_unexpected_from_state_ready,
        "state_ready"_s <= "state_model_contract_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_model_contract_decision,
        "state_ready"_s <= "state_encoder_state_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_encoder_state_decision,
        "state_ready"_s <= "state_decode_policy_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_decode_policy_decision,
        "state_ready"_s <= "state_generated_token_capacity_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_generated_token_capacity_decision,
        "state_ready"_s <= "state_logits_capacity_decision"_s + unexpected_event<_> / effect_on_unexpected_from_state_logits_capacity_decision,
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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum WhisperDecoderPhase {
    #[default]
    Ready,
    ModelContractDecision,
    EncoderStateDecision,
    DecodePolicyDecision,
    GeneratedTokenCapacityDecision,
    LogitsCapacityDecision,
    WorkspaceCapacityDecision,
    VariantDecision,
    RunningQ8F32,
    RunningQ8,
    RunningQ4_0,
    RunningQ4_1,
    Success,
    Error,
    Done,
    Errored,
    Unexpected,
}

#[derive(Debug, Default)]
pub struct SpeechDecoderWhisperContext {
    error: WhisperDecoderError,
    phase: WhisperDecoderPhase,
    q8_0_dispatch_count: u64,
    q4_0_dispatch_count: u64,
    q4_1_dispatch_count: u64,
    unexpected_count: u64,
}

impl SpeechDecoderWhisperContext {
    #[must_use]
    pub const fn error(&self) -> WhisperDecoderError { self.error }
    #[must_use]
    pub const fn phase(&self) -> WhisperDecoderPhase { self.phase }
    #[must_use]
    pub const fn q8_0_dispatch_count(&self) -> u64 { self.q8_0_dispatch_count }
    #[must_use]
    pub const fn q4_0_dispatch_count(&self) -> u64 { self.q4_0_dispatch_count }
    #[must_use]
    pub const fn q4_1_dispatch_count(&self) -> u64 { self.q4_1_dispatch_count }
    #[must_use]
    pub const fn unexpected_count(&self) -> u64 { self.unexpected_count }
    fn mark(&mut self, error: WhisperDecoderError) -> Result<(), ()> { self.error = error; self.phase = WhisperDecoderPhase::Error; Ok(()) }
    fn unexpected(&mut self) -> Result<(), ()> { self.error = WhisperDecoderError::InternalError; self.phase = WhisperDecoderPhase::Unexpected; self.unexpected_count = self.unexpected_count.saturating_add(1); Ok(()) }
}

fn has_tensor(model: &Data, name: &[u8], dims: &[u64], kind: SerializedType) -> bool {
    let Some(tensor) = model.tensor_named(name) else { return false };
    let Some(metadata) = tensor.metadata() else { return false };
    metadata.tensor_type() == kind
        && usize::try_from(metadata.dimension_count()).ok() == Some(dims.len())
        && metadata.dimensions().get(..dims.len()) == Some(dims)
        && tensor.bytes().is_some_and(|bytes| !bytes.is_empty())
}

fn has_decoder_tensor(model: &Data, block: usize, suffix: &[u8], dims: &[u64], kind: SerializedType) -> bool {
    let prefix = b"model.decoder.layers.";
    let mut name = [0_u8; 96];
    let mut used = prefix.len();
    name[..used].copy_from_slice(prefix);
    let mut digits = [0_u8; 20];
    let mut value = block;
    let mut digit_count = 0;
    loop {
        digits[digit_count] = b'0' + (value % 10) as u8;
        digit_count += 1;
        value /= 10;
        if value == 0 { break; }
    }
    while digit_count != 0 {
        digit_count -= 1;
        name[used] = digits[digit_count];
        used += 1;
    }
    name[used] = b'.';
    used += 1;
    for byte in suffix {
        name[used] = *byte;
        used += 1;
    }
    has_tensor(model, &name[..used], dims, kind)
}

fn has_decoder_block(model: &Data, block: usize, linear: SerializedType, aux: SerializedType) -> bool {
    has_decoder_tensor(model, block, b"self_attn.k_proj.weight", &[384, 384], linear)
        && has_decoder_tensor(model, block, b"self_attn.v_proj.weight", &[384, 384], linear)
        && has_decoder_tensor(model, block, b"self_attn.v_proj.bias", &[384], aux)
        && has_decoder_tensor(model, block, b"self_attn.q_proj.weight", &[384, 384], linear)
        && has_decoder_tensor(model, block, b"self_attn.q_proj.bias", &[384], aux)
        && has_decoder_tensor(model, block, b"self_attn.out_proj.weight", &[384, 384], linear)
        && has_decoder_tensor(model, block, b"self_attn.out_proj.bias", &[384], aux)
        && has_decoder_tensor(model, block, b"self_attn_layer_norm.weight", &[384], aux)
        && has_decoder_tensor(model, block, b"self_attn_layer_norm.bias", &[384], aux)
        && has_decoder_tensor(model, block, b"encoder_attn.k_proj.weight", &[384, 384], linear)
        && has_decoder_tensor(model, block, b"encoder_attn.v_proj.weight", &[384, 384], linear)
        && has_decoder_tensor(model, block, b"encoder_attn.v_proj.bias", &[384], aux)
        && has_decoder_tensor(model, block, b"encoder_attn.q_proj.weight", &[384, 384], linear)
        && has_decoder_tensor(model, block, b"encoder_attn.q_proj.bias", &[384], aux)
        && has_decoder_tensor(model, block, b"encoder_attn.out_proj.weight", &[384, 384], linear)
        && has_decoder_tensor(model, block, b"encoder_attn.out_proj.bias", &[384], aux)
        && has_decoder_tensor(model, block, b"encoder_attn_layer_norm.weight", &[384], aux)
        && has_decoder_tensor(model, block, b"encoder_attn_layer_norm.bias", &[384], aux)
        && has_decoder_tensor(model, block, b"fc1.weight", &[384, 1536], linear)
        && has_decoder_tensor(model, block, b"fc1.bias", &[1536], aux)
        && has_decoder_tensor(model, block, b"fc2.weight", &[1536, 384], linear)
        && has_decoder_tensor(model, block, b"fc2.bias", &[384], aux)
        && has_decoder_tensor(model, block, b"final_layer_norm.weight", &[384], aux)
        && has_decoder_tensor(model, block, b"final_layer_norm.bias", &[384], aux)
}

fn model_valid(request: &EventDecodeRun<'_>) -> bool {
    let model = request.request.model;
    let contract = &request.request.contract;
    contract.vocab_size == VOCAB_SIZE as i32
        && contract.embedding_length == EMBEDDING_LENGTH as i32
        && contract.decoder_block_count == DECODER_BLOCK_COUNT as i32
        && has_tensor(model, b"model.decoder.embed_tokens.weight", &[384, 51865], SerializedType::Q8_0)
        && (has_tensor(model, b"model.decoder.embed_positions.weight", &[384, 448], SerializedType::Q8_0)
            || has_tensor(model, b"model.decoder.embed_positions.weight", &[384, 448], SerializedType::F32))
        && (has_tensor(model, b"model.decoder.layer_norm.weight", &[384], SerializedType::Q8_0)
            || has_tensor(model, b"model.decoder.layer_norm.weight", &[384], SerializedType::F32))
        && (has_tensor(model, b"model.decoder.layer_norm.bias", &[384], SerializedType::Q8_0)
            || has_tensor(model, b"model.decoder.layer_norm.bias", &[384], SerializedType::F32))
}

fn required_workspace(frames: i32) -> Option<usize> {
    let frames = usize::try_from(frames).ok()?;
    EMBEDDING_LENGTH.checked_mul(frames)?.checked_mul(2)?.checked_mul(DECODER_BLOCK_COUNT)?
        .checked_add(EMBEDDING_LENGTH * DECODER_SEQUENCE_TOKEN_COUNT * 6)?
        .checked_add(FEED_FORWARD_LENGTH)?.checked_add(frames.max(DECODER_SEQUENCE_TOKEN_COUNT))
}

impl SpeechDecoderWhisperStateMachineContext for SpeechDecoderWhisperContext {
    fn effect_begin_decode(&mut self, event: &EventDecodeRun<'_>) -> Result<(), ()> {
        self.error = WhisperDecoderError::None;
        self.phase = WhisperDecoderPhase::ModelContractDecision;
        *event.request.token_out = 0;
        *event.request.confidence_out = 0.0;
        *event.request.digest_out = 0;
        *event.request.generated_token_count_out = 0;
        Ok(())
    }
    fn effect_mark_model_invalid(&mut self, _: &EventDecodeRun<'_>) -> Result<(), ()> { self.mark(WhisperDecoderError::ModelInvalid) }
    fn effect_mark_encoder_state_invalid(&mut self, _: &EventDecodeRun<'_>) -> Result<(), ()> { self.mark(WhisperDecoderError::EncoderState) }
    fn effect_mark_decode_policy_invalid(&mut self, _: &EventDecodeRun<'_>) -> Result<(), ()> { self.mark(WhisperDecoderError::DecodePolicy) }
    fn effect_mark_generated_token_capacity_invalid(&mut self, _: &EventDecodeRun<'_>) -> Result<(), ()> { self.mark(WhisperDecoderError::GeneratedTokenCapacity) }
    fn effect_mark_logits_capacity_invalid(&mut self, _: &EventDecodeRun<'_>) -> Result<(), ()> { self.mark(WhisperDecoderError::LogitsCapacity) }
    fn effect_mark_workspace_capacity_invalid(&mut self, _: &EventDecodeRun<'_>) -> Result<(), ()> { self.mark(WhisperDecoderError::WorkspaceCapacity) }
    fn effect_mark_unsupported_variant(&mut self, _: &EventDecodeRun<'_>) -> Result<(), ()> { self.mark(WhisperDecoderError::UnsupportedVariant) }
    fn effect_mark_internal_error(&mut self, _: &EventDecodeRun<'_>) -> Result<(), ()> { self.mark(WhisperDecoderError::InternalError) }
    fn effect_run_decoder_q8_0_f32_aux(&mut self, event: &EventDecodeRun<'_>) -> Result<(), ()> { self.run(event, DecodeVariant::Q8_0F32Aux) }
    fn effect_run_decoder_q8_0(&mut self, event: &EventDecodeRun<'_>) -> Result<(), ()> { self.run(event, DecodeVariant::Q8_0) }
    fn effect_run_decoder_q4_0(&mut self, event: &EventDecodeRun<'_>) -> Result<(), ()> { self.run(event, DecodeVariant::Q4_0) }
    fn effect_run_decoder_q4_1(&mut self, event: &EventDecodeRun<'_>) -> Result<(), ()> { self.run(event, DecodeVariant::Q4_1) }
    fn effect_store_success_error(&mut self, event: &EventDecodeRun<'_>) -> Result<(), ()> { if let Some(error) = event.request.error_out.as_deref_mut() { *error = WhisperDecoderError::None; } Ok(()) }
    fn effect_store_error_error(&mut self, event: &EventDecodeRun<'_>) -> Result<(), ()> { if let Some(error) = event.request.error_out.as_deref_mut() { *error = self.error; } Ok(()) }
    fn effect_emit_done(&mut self, event: &EventDecodeRun<'_>) -> Result<(), ()> { self.phase = WhisperDecoderPhase::Done; if let Some(callback) = event.request.on_done { let _ = callback(DecodeDone { token: *event.request.token_out, confidence_bits: event.request.confidence_out.to_bits(), digest: *event.request.digest_out }); } Ok(()) }
    fn effect_emit_error(&mut self, event: &EventDecodeRun<'_>) -> Result<(), ()> { self.phase = WhisperDecoderPhase::Errored; if let Some(callback) = event.request.on_error { let _ = callback(DecodeError { error: self.error }); } Ok(()) }
    fn guard_model_contract_valid(&self, event: &EventDecodeRun<'_>) -> Result<bool, ()> { Ok(model_valid(event)) }
    fn guard_model_contract_invalid(&self, event: &EventDecodeRun<'_>) -> Result<bool, ()> { Ok(!model_valid(event)) }
    fn guard_encoder_state_valid(&self, event: &EventDecodeRun<'_>) -> Result<bool, ()> { let frames = event.request.encoder_frame_count; Ok(!event.request.encoder_state.is_empty() && (1..=MAX_ENCODER_FRAME_COUNT).contains(&frames) && usize::try_from(frames).ok().and_then(|f| f.checked_mul(EMBEDDING_LENGTH)).is_some_and(|n| event.request.encoder_state.len() >= n)) }
    fn guard_encoder_state_invalid(&self, event: &EventDecodeRun<'_>) -> Result<bool, ()> { Ok(!self.guard_encoder_state_valid(event)?) }
    fn guard_decode_policy_supported(&self, event: &EventDecodeRun<'_>) -> Result<bool, ()> { Ok(event.request.policy.supported()) }
    fn guard_decode_policy_unsupported(&self, event: &EventDecodeRun<'_>) -> Result<bool, ()> { Ok(!event.request.policy.supported()) }
    fn guard_generated_token_capacity_valid(&self, event: &EventDecodeRun<'_>) -> Result<bool, ()> { Ok(!event.request.generated_tokens.is_empty()) }
    fn guard_generated_token_capacity_invalid(&self, event: &EventDecodeRun<'_>) -> Result<bool, ()> { Ok(!self.guard_generated_token_capacity_valid(event)?) }
    fn guard_logits_capacity_valid(&self, event: &EventDecodeRun<'_>) -> Result<bool, ()> { Ok(event.request.logits.len() >= VOCAB_SIZE) }
    fn guard_logits_capacity_invalid(&self, event: &EventDecodeRun<'_>) -> Result<bool, ()> { Ok(!self.guard_logits_capacity_valid(event)?) }
    fn guard_workspace_capacity_valid(&self, event: &EventDecodeRun<'_>) -> Result<bool, ()> { Ok(required_workspace(event.request.encoder_frame_count).is_some_and(|required| event.request.workspace.len() >= required)) }
    fn guard_workspace_capacity_invalid(&self, event: &EventDecodeRun<'_>) -> Result<bool, ()> { Ok(!self.guard_workspace_capacity_valid(event)?) }
    fn guard_q8_0_variant(&self, event: &EventDecodeRun<'_>) -> Result<bool, ()> { Ok((0..DECODER_BLOCK_COUNT).all(|b| has_decoder_block(event.request.model, b, SerializedType::Q8_0, SerializedType::Q8_0))) }
    fn guard_q8_0_f32_aux_variant(&self, event: &EventDecodeRun<'_>) -> Result<bool, ()> { Ok((0..DECODER_BLOCK_COUNT).all(|b| has_decoder_block(event.request.model, b, SerializedType::Q8_0, SerializedType::F32))) }
    fn guard_q4_0_variant(&self, event: &EventDecodeRun<'_>) -> Result<bool, ()> { Ok((0..DECODER_BLOCK_COUNT).all(|b| has_decoder_block(event.request.model, b, SerializedType::Q4_0, SerializedType::Q8_0))) }
    fn guard_q4_1_variant(&self, event: &EventDecodeRun<'_>) -> Result<bool, ()> { Ok((0..DECODER_BLOCK_COUNT).all(|b| has_decoder_block(event.request.model, b, SerializedType::Q4_1, SerializedType::Q8_0))) }
    fn guard_unsupported_variant(&self, event: &EventDecodeRun<'_>) -> Result<bool, ()> { Ok(!self.guard_q8_0_variant(event)? && !self.guard_q8_0_f32_aux_variant(event)? && !self.guard_q4_0_variant(event)? && !self.guard_q4_1_variant(event)?) }
    fn guard_decode_execution_success(&self, _: &EventDecodeRun<'_>) -> Result<bool, ()> { Ok(self.error == WhisperDecoderError::None) }
    fn guard_decode_execution_failure(&self, _: &EventDecodeRun<'_>) -> Result<bool, ()> { Ok(self.error != WhisperDecoderError::None) }
    fn guard_has_error_out(&self, event: &EventDecodeRun<'_>) -> Result<bool, ()> { Ok(event.request.error_out.is_some()) }
    fn guard_no_error_out(&self, event: &EventDecodeRun<'_>) -> Result<bool, ()> { Ok(event.request.error_out.is_none()) }
    fn guard_has_done_callback(&self, event: &EventDecodeRun<'_>) -> Result<bool, ()> { Ok(event.request.on_done.is_some()) }
    fn guard_no_done_callback(&self, event: &EventDecodeRun<'_>) -> Result<bool, ()> { Ok(event.request.on_done.is_none()) }
    fn guard_has_error_callback(&self, event: &EventDecodeRun<'_>) -> Result<bool, ()> { Ok(event.request.on_error.is_some()) }
    fn guard_no_error_callback(&self, event: &EventDecodeRun<'_>) -> Result<bool, ()> { Ok(event.request.on_error.is_none()) }

    fn effect_on_unexpected_from_state_ready(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_model_contract_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_encoder_state_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_decode_policy_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_generated_token_capacity_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_logits_capacity_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_workspace_capacity_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_variant_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_running_q8_0(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_running_q8_0_f32_aux(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_running_q4_0(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_running_q4_1(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_success_error_out_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_success_callback_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_error_error_out_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_error_callback_decision(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_done(&mut self) -> Result<(), ()> { self.unexpected() }
    fn effect_on_unexpected_from_state_errored(&mut self) -> Result<(), ()> { self.unexpected() }
}

impl SpeechDecoderWhisperContext {
    fn run(&mut self, event: &EventDecodeRun<'_>, variant: DecodeVariant) -> Result<(), ()> {
        self.phase = match variant { DecodeVariant::Q8_0F32Aux => WhisperDecoderPhase::RunningQ8F32, DecodeVariant::Q8_0 => WhisperDecoderPhase::RunningQ8, DecodeVariant::Q4_0 => WhisperDecoderPhase::RunningQ4_0, DecodeVariant::Q4_1 => WhisperDecoderPhase::RunningQ4_1 };
        let Some(decode) = event.request.decode else { self.error = WhisperDecoderError::InternalError; return Ok(()) };
        let accepted = decode(DecodeKernelRequest { variant, contract: &event.request.contract, encoder_state: event.request.encoder_state, encoder_frame_count: event.request.encoder_frame_count, policy: event.request.policy, generated_tokens: event.request.generated_tokens, generated_token_count_out: event.request.generated_token_count_out, workspace: event.request.workspace, logits: event.request.logits, token_out: event.request.token_out, confidence_out: event.request.confidence_out, digest_out: event.request.digest_out });
        if !accepted { self.error = WhisperDecoderError::InternalError; }
        if variant == DecodeVariant::Q8_0 || variant == DecodeVariant::Q8_0F32Aux { self.q8_0_dispatch_count = self.q8_0_dispatch_count.saturating_add(1); }
        if variant == DecodeVariant::Q4_0 { self.q4_0_dispatch_count = self.q4_0_dispatch_count.saturating_add(1); }
        if variant == DecodeVariant::Q4_1 { self.q4_1_dispatch_count = self.q4_1_dispatch_count.saturating_add(1); }
        Ok(())
    }
}

/// Single-writer synchronous Whisper decoder actor.
pub struct SpeechDecoderWhisperActor {
    machine: SpeechDecoderWhisperStateMachine<SpeechDecoderWhisperContext>,
}

impl Default for SpeechDecoderWhisperActor { fn default() -> Self { Self::new() } }

impl SpeechDecoderWhisperActor {
    #[must_use]
    pub fn new() -> Self { Self { machine: SpeechDecoderWhisperStateMachine::new(SpeechDecoderWhisperContext::default()) } }
    #[must_use]
    pub fn state(&self) -> &SpeechDecoderWhisperStates { self.machine.state() }
    #[must_use]
    pub fn context(&self) -> &SpeechDecoderWhisperContext { self.machine.context() }
    pub fn process_event<'dispatch, 'model>(&mut self, event: EventDecodeRun<'dispatch, 'model>) -> Result<(), WhisperDecoderError> where 'model: 'dispatch {
        if !self.machine.is(&SpeechDecoderWhisperStates::StateReady) { return Err(WhisperDecoderError::InternalError); }
        if self.machine.process_event(event).is_err() { return Err(WhisperDecoderError::InternalError); }
        match self.context().error() { WhisperDecoderError::None => Ok(()), error => Err(error) }
    }
}

/// Short alias matching the C++ `sm` surface.
pub type Decoder = SpeechDecoderWhisperActor;
