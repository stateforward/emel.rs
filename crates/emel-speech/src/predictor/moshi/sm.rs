//! Source-aligned run-to-completion [`SpeechPredictorMoshiActor`].
#![allow(
    clippy::enum_variant_names,
    clippy::derive_partial_eq_without_eq,
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use,
    clippy::empty_structs_with_brackets,
    clippy::missing_const_for_fn,
    clippy::struct_excessive_bools,
    dead_code,
    unused_imports,
    missing_docs
)]
use core::cell::Cell;
use core::fmt;
use sml::sml;
pub const MAX_CODEBOOKS: usize = 64;
pub const MAX_DELAY_ROWS: usize = 128;
pub const MAX_VOICE_EMBEDDING_DIM: usize = 8192;
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum PredictorError {
    #[default]
    None = 0,
    NotInitialized = 1,
    BindFailed = 2,
    Memory = 3,
    RequestShape = 4,
    GraphRuntimeUnavailable = 5,
    GraphRuntime = 6,
    OutputWaiting = 7,
    VoiceContract = 8,
    VoicePromptPending = 9,
    PersonaPlexPrompt = 10,
    UnexpectedEvent = 11,
}
impl fmt::Display for PredictorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::None => "no error",
            Self::NotInitialized => "predictor is not initialized",
            Self::BindFailed => "predictor binding failed",
            Self::Memory => "predictor memory operation failed",
            Self::RequestShape => "predictor request shape is invalid",
            Self::GraphRuntimeUnavailable => "predictor graph runtime is unavailable",
            Self::GraphRuntime => "predictor graph runtime failed",
            Self::OutputWaiting => "predictor output is waiting",
            Self::VoiceContract => "voice contract is invalid",
            Self::VoicePromptPending => "voice prompt is pending",
            Self::PersonaPlexPrompt => "PersonaPlex prompt is invalid",
            Self::UnexpectedEvent => "unexpected predictor event",
        })
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelContract {
    pub n_q: i32,
    pub dep_q: i32,
    pub inference_dep_q: i32,
    pub text_card: i32,
    pub text_padding_id: i32,
    pub dim: i32,
    pub context: i32,
    pub delay_count: u32,
    pub delays: [i32; MAX_CODEBOOKS],
    pub inference_prompt_token_count: u32,
    pub inference_pre_text_silence_frames: i32,
    pub inference_post_text_silence_frames: i32,
    pub personaplex: bool,
}
impl Default for ModelContract {
    fn default() -> Self {
        Self {
            n_q: 0,
            dep_q: 0,
            inference_dep_q: 0,
            text_card: 0,
            text_padding_id: 0,
            dim: 0,
            context: 0,
            delay_count: 0,
            delays: [0; MAX_CODEBOOKS],
            inference_prompt_token_count: 0,
            inference_pre_text_silence_frames: 0,
            inference_post_text_silence_frames: 0,
            personaplex: false,
        }
    }
}
impl ModelContract {
    fn valid(self, ms: i32, mb: i32, bt: i32, cc: i32, dc: i32) -> bool {
        let count = self.n_q.saturating_add(1);
        let Ok(delay_count) = usize::try_from(self.delay_count) else {
            return false;
        };
        let Ok(count_usize) = usize::try_from(count) else {
            return false;
        };
        let Ok(count_u32) = u32::try_from(count) else {
            return false;
        };
        let mut max_delay = 0;
        if delay_count > MAX_CODEBOOKS
            || self.n_q < 0
            || count <= 0
            || count_usize > MAX_CODEBOOKS
            || self.dep_q <= 0
            || self.dep_q >= count
            || self.inference_dep_q <= 0
            || self.inference_dep_q > self.dep_q
            || self.delay_count < count_u32
        {
            return false;
        }
        for &delay in &self.delays[..delay_count] {
            if delay < 0 {
                return false;
            }
            max_delay = max_delay.max(delay);
        }
        let rows = max_delay + 2 + i32::from(self.personaplex);
        (1..=1_024).contains(&ms)
            && (1..=65_536).contains(&mb)
            && bt > 0
            && bt <= self.context
            && rows > 0
            && usize::try_from(rows).is_ok_and(|rows| rows <= MAX_DELAY_ROWS)
            && cc >= count
            && usize::try_from(cc).is_ok_and(|count| count <= MAX_CODEBOOKS)
            && dc >= rows
            && usize::try_from(dc).is_ok_and(|rows| rows <= MAX_DELAY_ROWS)
            && (!self.personaplex
                || (self.inference_prompt_token_count == count_u32
                    && self.inference_pre_text_silence_frames >= 0
                    && self.inference_post_text_silence_frames >= 0))
    }
}
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct VoiceContract {
    pub embedding_dim: i32,
    pub prompt_frame_count: i32,
    pub cache_rows: i32,
    pub cache_columns: i32,
    pub embedding_format: u8,
    pub cache_format: u8,
}
impl VoiceContract {
    fn valid(self, m: &ModelContract, rows: i32, count: i32) -> bool {
        self.embedding_dim > 0
            && usize::try_from(self.embedding_dim)
                .is_ok_and(|embedding_dim| embedding_dim <= MAX_VOICE_EMBEDDING_DIM)
            && self.prompt_frame_count > 0
            && self.cache_rows == rows
            && self.cache_columns == count
            && matches!(self.embedding_format, 1..=3)
            && self.cache_format == 4
            && m.personaplex
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PredictionWorkspace {
    pub sampled_audio_tokens: [i32; MAX_CODEBOOKS],
    pub sampled_text_token: i32,
}
impl Default for PredictionWorkspace {
    fn default() -> Self {
        Self {
            sampled_audio_tokens: [0; MAX_CODEBOOKS],
            sampled_text_token: 0,
        }
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphPhase {
    Full,
    Prediction,
    Sampling,
    Voice,
    Prompt,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryOp {
    Reserve,
    AllocateSequence,
    AllocateSlots,
    Capture,
    FreeSequence,
}
pub type GraphFn = fn(GraphPhase, &[i32], &Cell<PredictionWorkspace>) -> Result<(), PredictorError>;
pub type MemoryFn = fn(MemoryOp, i32) -> Result<(), PredictorError>;
pub type InitDoneFn = fn(i32, i32) -> bool;
pub type ErrorFn = fn(PredictorError) -> bool;
pub type VoiceDoneFn = fn(i32) -> bool;
pub type PrefillDoneFn = fn(bool, i32) -> bool;
pub type PromptDoneFn = fn(i32) -> bool;
#[derive(Debug, Clone, Default)]
pub struct BeginPromptRun {
    pub text_token_count: i32,
    pub pre_text_silence_frames: i32,
    pub post_text_silence_frames: i32,
    pub error_out: Option<&'static Cell<PredictorError>>,
    pub on_done: Option<PromptDoneFn>,
    pub on_error: Option<ErrorFn>,
}
#[derive(Debug, Clone, Default)]
pub struct EventCaptureTokenizerState {
    pub cache_out: Option<&'static Cell<[i32; MAX_CODEBOOKS * MAX_DELAY_ROWS]>>,
    pub offset_out: Option<&'static Cell<i64>>,
    pub error_out: Option<&'static Cell<PredictorError>>,
}
#[derive(Debug, Clone)]
pub struct ExecuteRun {
    pub model_tokens: [i32; MAX_CODEBOOKS],
    pub model_token_count: usize,
    pub workspace: Option<&'static Cell<PredictionWorkspace>>,
    pub error_out: Option<&'static Cell<PredictorError>>,
    pub graph_error_out: Option<&'static Cell<PredictorError>>,
    pub graph: Option<GraphFn>,
}
impl Default for ExecuteRun {
    fn default() -> Self {
        Self {
            model_tokens: [0; MAX_CODEBOOKS],
            model_token_count: 0,
            workspace: None,
            error_out: None,
            graph_error_out: None,
            graph: None,
        }
    }
}
#[derive(Debug, Clone, Default)]
pub struct InitRun {
    pub model: ModelContract,
    pub max_sequences: i32,
    pub max_blocks: i32,
    pub block_tokens: i32,
    pub sequence_id: i32,
    pub codebook_capacity: i32,
    pub delay_cache_row_capacity: i32,
    pub error_out: Option<&'static Cell<PredictorError>>,
    pub on_done: Option<InitDoneFn>,
    pub on_error: Option<ErrorFn>,
    pub graph: Option<GraphFn>,
    pub memory: Option<MemoryFn>,
}
#[derive(Debug, Clone, Default)]
pub struct LoadVoiceRun {
    pub voice: VoiceContract,
    pub error_out: Option<&'static Cell<PredictorError>>,
    pub on_done: Option<VoiceDoneFn>,
    pub on_error: Option<ErrorFn>,
}
#[derive(Debug, Clone)]
pub struct PredictRun {
    pub model_tokens: [i32; MAX_CODEBOOKS],
    pub model_token_count: usize,
    pub planned_step_size: i32,
    pub planned_output_count: i32,
    pub error_out: Option<&'static Cell<PredictorError>>,
    pub memory: Option<MemoryFn>,
}
impl Default for PredictRun {
    fn default() -> Self {
        Self {
            model_tokens: [0; MAX_CODEBOOKS],
            model_token_count: 0,
            planned_step_size: 0,
            planned_output_count: 0,
            error_out: None,
            memory: None,
        }
    }
}
#[derive(Debug, Clone)]
pub struct PrefillPromptRun {
    pub text_token: i32,
    pub model_tokens: [i32; MAX_CODEBOOKS],
    pub error_out: Option<&'static Cell<PredictorError>>,
    pub graph_error_out: Option<&'static Cell<PredictorError>>,
    pub complete_out: Option<&'static Cell<bool>>,
    pub remaining_frames_out: Option<&'static Cell<i32>>,
    pub on_done: Option<PrefillDoneFn>,
    pub on_error: Option<ErrorFn>,
    pub graph: Option<GraphFn>,
    pub memory: Option<MemoryFn>,
}
impl Default for PrefillPromptRun {
    fn default() -> Self {
        Self {
            text_token: 0,
            model_tokens: [0; MAX_CODEBOOKS],
            error_out: None,
            graph_error_out: None,
            complete_out: None,
            remaining_frames_out: None,
            on_done: None,
            on_error: None,
            graph: None,
            memory: None,
        }
    }
}
#[derive(Debug, Clone)]
pub struct PrefillVoiceRun {
    pub model_tokens: [i32; MAX_CODEBOOKS],
    pub error_out: Option<&'static Cell<PredictorError>>,
    pub graph_error_out: Option<&'static Cell<PredictorError>>,
    pub complete_out: Option<&'static Cell<bool>>,
    pub remaining_frames_out: Option<&'static Cell<i32>>,
    pub on_done: Option<PrefillDoneFn>,
    pub on_error: Option<ErrorFn>,
    pub graph: Option<GraphFn>,
    pub memory: Option<MemoryFn>,
}
impl Default for PrefillVoiceRun {
    fn default() -> Self {
        Self {
            model_tokens: [0; MAX_CODEBOOKS],
            error_out: None,
            graph_error_out: None,
            complete_out: None,
            remaining_frames_out: None,
            on_done: None,
            on_error: None,
            graph: None,
            memory: None,
        }
    }
}
#[derive(Debug, Clone, Default)]
pub struct ResetRun {
    pub error_out: Option<&'static Cell<PredictorError>>,
    pub graph: Option<fn() -> Result<(), PredictorError>>,
    pub memory: Option<MemoryFn>,
}
#[derive(Debug, Clone)]
pub struct SampleRun {
    pub model_tokens: [i32; MAX_CODEBOOKS],
    pub model_token_count: usize,
    pub audio_tokens_out: Option<&'static Cell<[i32; MAX_CODEBOOKS]>>,
    pub text_token_out: Option<&'static Cell<i32>>,
    pub error_out: Option<&'static Cell<PredictorError>>,
    pub graph_error_out: Option<&'static Cell<PredictorError>>,
    pub graph: Option<GraphFn>,
}
impl Default for SampleRun {
    fn default() -> Self {
        Self {
            model_tokens: [0; MAX_CODEBOOKS],
            model_token_count: 0,
            audio_tokens_out: None,
            text_token_out: None,
            error_out: None,
            graph_error_out: None,
            graph: None,
        }
    }
}
sml! {
    SpeechPredictorMoshi {
        "state_bind_contract_decision"_s <= *"state_uninitialized"_s + event<InitRun>,
        "state_binding"_s <= "state_bind_contract_decision"_s + completion<InitRun> [guard_bind_contract_valid] / effect_bind_contract,
        "state_init_failed_error_out_decision"_s <= "state_bind_contract_decision"_s + completion<InitRun> [guard_bind_contract_invalid] / effect_mark_bind_failed,
        "state_lmgen_mode_decision"_s <= "state_binding"_s + completion<InitRun>,
        "state_configure_lmgen"_s <= "state_lmgen_mode_decision"_s + completion<InitRun> [guard_personaplex_lmgen] / effect_configure_personaplex_lmgen,
        "state_configure_lmgen"_s <= "state_lmgen_mode_decision"_s + completion<InitRun> [guard_standard_lmgen] / effect_configure_standard_lmgen,
        "state_initialize_graph"_s <= "state_configure_lmgen"_s + completion<InitRun> / effect_initialize_graph_graph_actor_type,
        "state_initialize_graph_decision"_s <= "state_initialize_graph"_s + completion<InitRun>,
        "state_reserve_memory"_s <= "state_initialize_graph_decision"_s + completion<InitRun> [guard_graph_initialize_succeeded] / effect_reserve_memory,
        "state_init_failed_error_out_decision"_s <= "state_initialize_graph_decision"_s + completion<InitRun> [guard_graph_initialize_failed] / effect_mark_graph_runtime_error_init_run,
        "state_reserve_memory_decision"_s <= "state_reserve_memory"_s + completion<InitRun>,
        "state_allocate_sequence"_s <= "state_reserve_memory_decision"_s + completion<InitRun> [guard_memory_accepted_init_run] / effect_allocate_sequence,
        "state_init_failed_error_out_decision"_s <= "state_reserve_memory_decision"_s + completion<InitRun> [guard_memory_rejected_init_run] / effect_mark_memory_error_init_run_from_state_reserve_memory_decision,
        "state_allocate_sequence_decision"_s <= "state_allocate_sequence"_s + completion<InitRun>,
        "state_init_error_out_decision"_s <= "state_allocate_sequence_decision"_s + completion<InitRun> [guard_memory_accepted_init_run],
        "state_init_failed_error_out_decision"_s <= "state_allocate_sequence_decision"_s + completion<InitRun> [guard_memory_rejected_init_run] / effect_mark_memory_error_init_run_from_state_allocate_sequence_decision,
        "state_init_callback_decision"_s <= "state_init_error_out_decision"_s + completion<InitRun> [guard_has_error_out_init_run] / effect_store_error_out_init_run_from_state_init_error_out_decision,
        "state_init_callback_decision"_s <= "state_init_error_out_decision"_s + completion<InitRun> [guard_no_error_out_init_run],
        "state_init_failed_callback_decision"_s <= "state_init_failed_error_out_decision"_s + completion<InitRun> [guard_has_error_out_init_run] / effect_store_error_out_init_run_from_state_init_failed_error_out_decision,
        "state_init_failed_callback_decision"_s <= "state_init_failed_error_out_decision"_s + completion<InitRun> [guard_no_error_out_init_run],
        "state_session_ready"_s <= "state_init_callback_decision"_s + completion<InitRun> [guard_has_done_callback_init_run] / effect_emit_initialize_done,
        "state_session_ready"_s <= "state_init_callback_decision"_s + completion<InitRun> [guard_no_done_callback_init_run],
        "state_uninitialized"_s <= "state_init_failed_callback_decision"_s + completion<InitRun> [guard_has_error_callback_init_run] / effect_emit_initialize_error,
        "state_uninitialized"_s <= "state_init_failed_callback_decision"_s + completion<InitRun> [guard_no_error_callback_init_run],
        "state_voice_request_decision"_s <= "state_session_ready"_s + event<LoadVoiceRun>,
        "state_voice_error_out_decision"_s <= "state_voice_request_decision"_s + completion<LoadVoiceRun> [guard_voice_contract_valid] / effect_bind_voice_contract,
        "state_voice_failed_error_out_decision"_s <= "state_voice_request_decision"_s + completion<LoadVoiceRun> [guard_voice_contract_invalid] / effect_mark_voice_contract_error_load_voice_run,
        "state_voice_callback_decision"_s <= "state_voice_error_out_decision"_s + completion<LoadVoiceRun> [guard_has_error_out_load_voice_run] / effect_store_error_out_load_voice_run_from_state_voice_error_out_decision,
        "state_voice_callback_decision"_s <= "state_voice_error_out_decision"_s + completion<LoadVoiceRun> [guard_no_error_out_load_voice_run],
        "state_voice_failed_callback_decision"_s <= "state_voice_failed_error_out_decision"_s + completion<LoadVoiceRun> [guard_has_error_out_load_voice_run] / effect_store_error_out_load_voice_run_from_state_voice_failed_error_out_decision,
        "state_voice_failed_callback_decision"_s <= "state_voice_failed_error_out_decision"_s + completion<LoadVoiceRun> [guard_no_error_out_load_voice_run],
        "state_session_ready"_s <= "state_voice_callback_decision"_s + completion<LoadVoiceRun> [guard_has_done_callback_load_voice_run] / effect_emit_load_voice_done,
        "state_session_ready"_s <= "state_voice_callback_decision"_s + completion<LoadVoiceRun> [guard_no_done_callback_load_voice_run],
        "state_session_ready"_s <= "state_voice_failed_callback_decision"_s + completion<LoadVoiceRun> [guard_has_error_callback_load_voice_run] / effect_emit_load_voice_error_from_state_voice_failed_callback_decision,
        "state_session_ready"_s <= "state_voice_failed_callback_decision"_s + completion<LoadVoiceRun> [guard_no_error_callback_load_voice_run],
        "state_prefill_voice_request_decision"_s <= "state_session_ready"_s + event<PrefillVoiceRun>,
        "state_prefill_voice_allocate_slot"_s <= "state_prefill_voice_request_decision"_s + completion<PrefillVoiceRun> [guard_voice_prefill_request_valid] / effect_allocate_voice_prefill_slot,
        "state_prefill_voice_failed_error_out_decision"_s <= "state_prefill_voice_request_decision"_s + completion<PrefillVoiceRun> [guard_voice_prefill_request_invalid] / effect_mark_voice_contract_error_prefill_voice_run_from_state_prefill_voice_request_decision,
        "state_prefill_voice_allocate_slot_decision"_s <= "state_prefill_voice_allocate_slot"_s + completion<PrefillVoiceRun>,
        "state_prefill_voice_capture_memory"_s <= "state_prefill_voice_allocate_slot_decision"_s + completion<PrefillVoiceRun> [guard_memory_accepted_prefill_voice_run] / effect_capture_voice_prefill_memory,
        "state_prefill_voice_failed_error_out_decision"_s <= "state_prefill_voice_allocate_slot_decision"_s + completion<PrefillVoiceRun> [guard_memory_rejected_prefill_voice_run] / effect_mark_memory_error_prefill_voice_run_from_state_prefill_voice_allocate_slot_decision,
        "state_prefill_voice_capture_memory_decision"_s <= "state_prefill_voice_capture_memory"_s + completion<PrefillVoiceRun>,
        "state_prefill_voice_begin"_s <= "state_prefill_voice_capture_memory_decision"_s + completion<PrefillVoiceRun> [guard_memory_accepted_prefill_voice_run] / effect_begin_voice_prefill,
        "state_prefill_voice_failed_error_out_decision"_s <= "state_prefill_voice_capture_memory_decision"_s + completion<PrefillVoiceRun> [guard_memory_rejected_prefill_voice_run] / effect_mark_memory_error_prefill_voice_run_from_state_prefill_voice_capture_memory_decision,
        "state_prefill_voice_embedding_frame"_s <= "state_prefill_voice_begin"_s + completion<PrefillVoiceRun> [guard_voice_embedding_frame_f32] / effect_load_voice_embedding_frame_f32,
        "state_prefill_voice_embedding_frame"_s <= "state_prefill_voice_begin"_s + completion<PrefillVoiceRun> [guard_voice_embedding_frame_f16] / effect_load_voice_embedding_frame_f16,
        "state_prefill_voice_embedding_frame"_s <= "state_prefill_voice_begin"_s + completion<PrefillVoiceRun> [guard_voice_embedding_frame_bf16] / effect_load_voice_embedding_frame_bf16,
        "state_prefill_voice_embedding_frame_decision"_s <= "state_prefill_voice_embedding_frame"_s + completion<PrefillVoiceRun>,
        "state_prefill_voice_graph_runtime_decision"_s <= "state_prefill_voice_embedding_frame_decision"_s + completion<PrefillVoiceRun> [guard_voice_embedding_frame_loaded],
        "state_prefill_voice_failed_error_out_decision"_s <= "state_prefill_voice_embedding_frame_decision"_s + completion<PrefillVoiceRun> [guard_voice_embedding_frame_failed] / effect_mark_voice_contract_error_prefill_voice_run_from_state_prefill_voice_embedding_frame_decision,
        "state_prefill_voice_running_graph"_s <= "state_prefill_voice_graph_runtime_decision"_s + completion<PrefillVoiceRun> / effect_run_voice_graph_runtime_graph_actor_type,
        "state_prefill_voice_graph_error_out_decision"_s <= "state_prefill_voice_running_graph"_s + completion<PrefillVoiceRun>,
        "state_prefill_voice_graph_result_decision"_s <= "state_prefill_voice_graph_error_out_decision"_s + completion<PrefillVoiceRun> [guard_has_graph_error_out_prefill_voice_run] / effect_store_graph_error_out_prefill_voice_run,
        "state_prefill_voice_graph_result_decision"_s <= "state_prefill_voice_graph_error_out_decision"_s + completion<PrefillVoiceRun> [guard_no_graph_error_out_prefill_voice_run],
        "state_prefill_voice_advance"_s <= "state_prefill_voice_graph_result_decision"_s + completion<PrefillVoiceRun> [guard_graph_step_accepted_prefill_voice_run] / effect_advance_voice_prefill,
        "state_prefill_voice_failed_error_out_decision"_s <= "state_prefill_voice_graph_result_decision"_s + completion<PrefillVoiceRun> [guard_graph_step_rejected_prefill_voice_run] / effect_mark_graph_runtime_error_prefill_voice_run,
        "state_prefill_voice_complete_decision"_s <= "state_prefill_voice_advance"_s + completion<PrefillVoiceRun>,
        "state_prefill_voice_output_decision"_s <= "state_prefill_voice_complete_decision"_s + completion<PrefillVoiceRun> [guard_voice_prefill_complete] / effect_copy_voice_cache,
        "state_prefill_voice_output_decision"_s <= "state_prefill_voice_complete_decision"_s + completion<PrefillVoiceRun> [guard_voice_prefill_pending],
        "state_prefill_voice_remaining_out_decision"_s <= "state_prefill_voice_output_decision"_s + completion<PrefillVoiceRun> [guard_has_voice_complete_out] / effect_store_voice_complete_out,
        "state_prefill_voice_remaining_out_decision"_s <= "state_prefill_voice_output_decision"_s + completion<PrefillVoiceRun> [guard_no_voice_complete_out],
        "state_prefill_voice_error_out_decision"_s <= "state_prefill_voice_remaining_out_decision"_s + completion<PrefillVoiceRun> [guard_has_voice_remaining_out] / effect_store_voice_remaining_out,
        "state_prefill_voice_error_out_decision"_s <= "state_prefill_voice_remaining_out_decision"_s + completion<PrefillVoiceRun> [guard_no_voice_remaining_out],
        "state_prefill_voice_callback_decision"_s <= "state_prefill_voice_error_out_decision"_s + completion<PrefillVoiceRun> [guard_has_error_out_prefill_voice_run] / effect_store_error_out_prefill_voice_run_from_state_prefill_voice_error_out_decision,
        "state_prefill_voice_callback_decision"_s <= "state_prefill_voice_error_out_decision"_s + completion<PrefillVoiceRun> [guard_no_error_out_prefill_voice_run],
        "state_prefill_voice_failed_callback_decision"_s <= "state_prefill_voice_failed_error_out_decision"_s + completion<PrefillVoiceRun> [guard_has_error_out_prefill_voice_run] / effect_store_error_out_prefill_voice_run_from_state_prefill_voice_failed_error_out_decision,
        "state_prefill_voice_failed_callback_decision"_s <= "state_prefill_voice_failed_error_out_decision"_s + completion<PrefillVoiceRun> [guard_no_error_out_prefill_voice_run],
        "state_session_ready"_s <= "state_prefill_voice_callback_decision"_s + completion<PrefillVoiceRun> [guard_has_done_callback_prefill_voice_run] / effect_emit_prefill_voice_done,
        "state_session_ready"_s <= "state_prefill_voice_callback_decision"_s + completion<PrefillVoiceRun> [guard_no_done_callback_prefill_voice_run],
        "state_session_ready"_s <= "state_prefill_voice_failed_callback_decision"_s + completion<PrefillVoiceRun> [guard_has_error_callback_prefill_voice_run] / effect_emit_prefill_voice_error_from_state_prefill_voice_failed_callback_decision,
        "state_session_ready"_s <= "state_prefill_voice_failed_callback_decision"_s + completion<PrefillVoiceRun> [guard_no_error_callback_prefill_voice_run],
        "state_personaplex_prompt_begin_decision"_s <= "state_session_ready"_s + event<BeginPromptRun>,
        "state_personaplex_prompt_begin_error_out_decision"_s <= "state_personaplex_prompt_begin_decision"_s + completion<BeginPromptRun> [guard_personaplex_prompt_begin_nonempty_valid] / effect_bind_personaplex_prompt,
        "state_personaplex_prompt_begin_error_out_decision"_s <= "state_personaplex_prompt_begin_decision"_s + completion<BeginPromptRun> [guard_personaplex_prompt_begin_empty_valid] / effect_bind_empty_personaplex_prompt,
        "state_personaplex_prompt_begin_failed_error_out_decision"_s <= "state_personaplex_prompt_begin_decision"_s + completion<BeginPromptRun> [guard_personaplex_prompt_begin_invalid] / effect_mark_personaplex_prompt_error_begin_prompt_run,
        "state_personaplex_prompt_begin_callback_decision"_s <= "state_personaplex_prompt_begin_error_out_decision"_s + completion<BeginPromptRun> [guard_has_error_out_begin_prompt_run] / effect_store_error_out_begin_prompt_run_from_state_personaplex_prompt_begin_error_out_decision,
        "state_personaplex_prompt_begin_callback_decision"_s <= "state_personaplex_prompt_begin_error_out_decision"_s + completion<BeginPromptRun> [guard_no_error_out_begin_prompt_run],
        "state_personaplex_prompt_begin_failed_callback_decision"_s <= "state_personaplex_prompt_begin_failed_error_out_decision"_s + completion<BeginPromptRun> [guard_has_error_out_begin_prompt_run] / effect_store_error_out_begin_prompt_run_from_state_personaplex_prompt_begin_failed_error_out_decision,
        "state_personaplex_prompt_begin_failed_callback_decision"_s <= "state_personaplex_prompt_begin_failed_error_out_decision"_s + completion<BeginPromptRun> [guard_no_error_out_begin_prompt_run],
        "state_session_ready"_s <= "state_personaplex_prompt_begin_callback_decision"_s + completion<BeginPromptRun> [guard_has_done_callback_begin_prompt_run] / effect_emit_begin_personaplex_prompt_done,
        "state_session_ready"_s <= "state_personaplex_prompt_begin_callback_decision"_s + completion<BeginPromptRun> [guard_no_done_callback_begin_prompt_run],
        "state_session_ready"_s <= "state_personaplex_prompt_begin_failed_callback_decision"_s + completion<BeginPromptRun> [guard_has_error_callback_begin_prompt_run] / effect_emit_begin_personaplex_prompt_error_from_state_personaplex_prompt_begin_failed_callback_decision,
        "state_session_ready"_s <= "state_personaplex_prompt_begin_failed_callback_decision"_s + completion<BeginPromptRun> [guard_no_error_callback_begin_prompt_run],
        "state_prefill_personaplex_prompt_request_decision"_s <= "state_session_ready"_s + event<PrefillPromptRun>,
        "state_prefill_personaplex_prompt_allocate_slot"_s <= "state_prefill_personaplex_prompt_request_decision"_s + completion<PrefillPromptRun> [guard_personaplex_prompt_prefill_request_valid] / effect_allocate_personaplex_prompt_slot,
        "state_prefill_personaplex_prompt_failed_error_out_decision"_s <= "state_prefill_personaplex_prompt_request_decision"_s + completion<PrefillPromptRun> [guard_personaplex_prompt_prefill_request_invalid] / effect_mark_personaplex_prompt_error_prefill_prompt_run_from_state_prefill_personaplex_prompt_request_decision,
        "state_prefill_personaplex_prompt_allocate_slot_decision"_s <= "state_prefill_personaplex_prompt_allocate_slot"_s + completion<PrefillPromptRun>,
        "state_prefill_personaplex_prompt_capture_memory"_s <= "state_prefill_personaplex_prompt_allocate_slot_decision"_s + completion<PrefillPromptRun> [guard_memory_accepted_prefill_prompt_run] / effect_capture_personaplex_prompt_memory,
        "state_prefill_personaplex_prompt_failed_error_out_decision"_s <= "state_prefill_personaplex_prompt_allocate_slot_decision"_s + completion<PrefillPromptRun> [guard_memory_rejected_prefill_prompt_run] / effect_mark_memory_error_prefill_prompt_run_from_state_prefill_personaplex_prompt_allocate_slot_decision,
        "state_prefill_personaplex_prompt_capture_memory_decision"_s <= "state_prefill_personaplex_prompt_capture_memory"_s + completion<PrefillPromptRun>,
        "state_prefill_personaplex_prompt_begin"_s <= "state_prefill_personaplex_prompt_capture_memory_decision"_s + completion<PrefillPromptRun> [guard_memory_accepted_prefill_prompt_run] / effect_begin_personaplex_prompt_prefill,
        "state_prefill_personaplex_prompt_failed_error_out_decision"_s <= "state_prefill_personaplex_prompt_capture_memory_decision"_s + completion<PrefillPromptRun> [guard_memory_rejected_prefill_prompt_run] / effect_mark_memory_error_prefill_prompt_run_from_state_prefill_personaplex_prompt_capture_memory_decision,
        "state_prefill_personaplex_prompt_phase_decision"_s <= "state_prefill_personaplex_prompt_begin"_s + completion<PrefillPromptRun>,
        "state_prefill_personaplex_prompt_frame"_s <= "state_prefill_personaplex_prompt_phase_decision"_s + completion<PrefillPromptRun> [guard_personaplex_prompt_pre_silence_pending] / effect_build_personaplex_prompt_silence_frame_from_state_prefill_personaplex_prompt_phase_decision,
        "state_prefill_personaplex_prompt_frame"_s <= "state_prefill_personaplex_prompt_phase_decision"_s + completion<PrefillPromptRun> [guard_personaplex_prompt_text_pending] / effect_build_personaplex_prompt_text_frame,
        "state_prefill_personaplex_prompt_frame"_s <= "state_prefill_personaplex_prompt_phase_decision"_s + completion<PrefillPromptRun> [guard_personaplex_prompt_post_silence_pending] / effect_build_personaplex_prompt_silence_frame_from_state_prefill_personaplex_prompt_phase_decision,
        "state_prefill_personaplex_prompt_failed_error_out_decision"_s <= "state_prefill_personaplex_prompt_phase_decision"_s + completion<PrefillPromptRun> [guard_personaplex_prompt_phase_invalid] / effect_mark_personaplex_prompt_error_prefill_prompt_run_from_state_prefill_personaplex_prompt_phase_decision,
        "state_prefill_personaplex_prompt_write_input"_s <= "state_prefill_personaplex_prompt_frame"_s + completion<PrefillPromptRun> / effect_write_and_build_personaplex_prompt_input,
        "state_prefill_personaplex_prompt_graph_runtime_decision"_s <= "state_prefill_personaplex_prompt_write_input"_s + completion<PrefillPromptRun>,
        "state_prefill_personaplex_prompt_running_graph"_s <= "state_prefill_personaplex_prompt_graph_runtime_decision"_s + completion<PrefillPromptRun> / effect_run_personaplex_prompt_graph_runtime_graph_actor_type,
        "state_prefill_personaplex_prompt_graph_error_out_decision"_s <= "state_prefill_personaplex_prompt_running_graph"_s + completion<PrefillPromptRun>,
        "state_prefill_personaplex_prompt_graph_result_decision"_s <= "state_prefill_personaplex_prompt_graph_error_out_decision"_s + completion<PrefillPromptRun> [guard_has_graph_error_out_prefill_prompt_run] / effect_store_graph_error_out_prefill_prompt_run,
        "state_prefill_personaplex_prompt_graph_result_decision"_s <= "state_prefill_personaplex_prompt_graph_error_out_decision"_s + completion<PrefillPromptRun> [guard_no_graph_error_out_prefill_prompt_run],
        "state_prefill_personaplex_prompt_advance"_s <= "state_prefill_personaplex_prompt_graph_result_decision"_s + completion<PrefillPromptRun> [guard_graph_step_accepted_personaplex_prompt_pre_silence] / effect_advance_personaplex_prompt_pre_silence,
        "state_prefill_personaplex_prompt_advance"_s <= "state_prefill_personaplex_prompt_graph_result_decision"_s + completion<PrefillPromptRun> [guard_graph_step_accepted_personaplex_prompt_text] / effect_advance_personaplex_prompt_text,
        "state_prefill_personaplex_prompt_advance"_s <= "state_prefill_personaplex_prompt_graph_result_decision"_s + completion<PrefillPromptRun> [guard_graph_step_accepted_personaplex_prompt_post_silence] / effect_advance_personaplex_prompt_post_silence,
        "state_prefill_personaplex_prompt_failed_error_out_decision"_s <= "state_prefill_personaplex_prompt_graph_result_decision"_s + completion<PrefillPromptRun> [guard_graph_step_rejected_prefill_prompt_run] / effect_mark_graph_runtime_error_prefill_prompt_run,
        "state_prefill_personaplex_prompt_complete_decision"_s <= "state_prefill_personaplex_prompt_advance"_s + completion<PrefillPromptRun>,
        "state_prefill_personaplex_prompt_output_decision"_s <= "state_prefill_personaplex_prompt_complete_decision"_s + completion<PrefillPromptRun> [guard_personaplex_prompt_complete] / effect_finish_personaplex_prompt,
        "state_prefill_personaplex_prompt_output_decision"_s <= "state_prefill_personaplex_prompt_complete_decision"_s + completion<PrefillPromptRun> [guard_personaplex_prompt_pending] / effect_publish_personaplex_prompt_pending,
        "state_prefill_personaplex_prompt_remaining_out_decision"_s <= "state_prefill_personaplex_prompt_output_decision"_s + completion<PrefillPromptRun> [guard_has_personaplex_prompt_complete_out] / effect_store_personaplex_prompt_complete_out,
        "state_prefill_personaplex_prompt_remaining_out_decision"_s <= "state_prefill_personaplex_prompt_output_decision"_s + completion<PrefillPromptRun> [guard_no_personaplex_prompt_complete_out],
        "state_prefill_personaplex_prompt_error_out_decision"_s <= "state_prefill_personaplex_prompt_remaining_out_decision"_s + completion<PrefillPromptRun> [guard_has_personaplex_prompt_remaining_out] / effect_store_personaplex_prompt_remaining_out,
        "state_prefill_personaplex_prompt_error_out_decision"_s <= "state_prefill_personaplex_prompt_remaining_out_decision"_s + completion<PrefillPromptRun> [guard_no_personaplex_prompt_remaining_out],
        "state_prefill_personaplex_prompt_callback_decision"_s <= "state_prefill_personaplex_prompt_error_out_decision"_s + completion<PrefillPromptRun> [guard_has_error_out_prefill_prompt_run] / effect_store_error_out_prefill_prompt_run_from_state_prefill_personaplex_prompt_error_out_decision,
        "state_prefill_personaplex_prompt_callback_decision"_s <= "state_prefill_personaplex_prompt_error_out_decision"_s + completion<PrefillPromptRun> [guard_no_error_out_prefill_prompt_run],
        "state_prefill_personaplex_prompt_failed_callback_decision"_s <= "state_prefill_personaplex_prompt_failed_error_out_decision"_s + completion<PrefillPromptRun> [guard_has_error_out_prefill_prompt_run] / effect_store_error_out_prefill_prompt_run_from_state_prefill_personaplex_prompt_failed_error_out_decision,
        "state_prefill_personaplex_prompt_failed_callback_decision"_s <= "state_prefill_personaplex_prompt_failed_error_out_decision"_s + completion<PrefillPromptRun> [guard_no_error_out_prefill_prompt_run],
        "state_session_ready"_s <= "state_prefill_personaplex_prompt_callback_decision"_s + completion<PrefillPromptRun> [guard_has_done_callback_prefill_prompt_run] / effect_emit_prefill_personaplex_prompt_done,
        "state_session_ready"_s <= "state_prefill_personaplex_prompt_callback_decision"_s + completion<PrefillPromptRun> [guard_no_done_callback_prefill_prompt_run],
        "state_session_ready"_s <= "state_prefill_personaplex_prompt_failed_callback_decision"_s + completion<PrefillPromptRun> [guard_has_error_callback_prefill_prompt_run] / effect_emit_prefill_personaplex_prompt_error_from_state_prefill_personaplex_prompt_failed_callback_decision,
        "state_session_ready"_s <= "state_prefill_personaplex_prompt_failed_callback_decision"_s + completion<PrefillPromptRun> [guard_no_error_callback_prefill_prompt_run],
        "state_predict_request_decision"_s <= "state_session_ready"_s + event<PredictRun>,
        "state_predict_allocate_slot"_s <= "state_predict_request_decision"_s + completion<PredictRun> [guard_predict_request_valid] / effect_allocate_step_slot,
        "state_predict_failed_error_out_decision"_s <= "state_predict_request_decision"_s + completion<PredictRun> [guard_predict_blocked_by_voice_prompt] / effect_mark_voice_prompt_pending,
        "state_predict_failed_error_out_decision"_s <= "state_predict_request_decision"_s + completion<PredictRun> [guard_predict_request_shape_invalid] / effect_mark_step_request_invalid_predict_run,
        "state_predict_allocate_slot_decision"_s <= "state_predict_allocate_slot"_s + completion<PredictRun>,
        "state_predict_capture_memory"_s <= "state_predict_allocate_slot_decision"_s + completion<PredictRun> [guard_memory_accepted_predict_run] / effect_capture_memory,
        "state_predict_failed_error_out_decision"_s <= "state_predict_allocate_slot_decision"_s + completion<PredictRun> [guard_memory_rejected_predict_run] / effect_mark_memory_error_predict_run_from_state_predict_allocate_slot_decision,
        "state_predict_capture_memory_decision"_s <= "state_predict_capture_memory"_s + completion<PredictRun>,
        "state_predict_begin"_s <= "state_predict_capture_memory_decision"_s + completion<PredictRun> [guard_memory_accepted_predict_run] / effect_begin_predict,
        "state_predict_failed_error_out_decision"_s <= "state_predict_capture_memory_decision"_s + completion<PredictRun> [guard_memory_rejected_predict_run] / effect_mark_memory_error_predict_run_from_state_predict_capture_memory_decision,
        "state_predict_error_out_decision"_s <= "state_predict_begin"_s + completion<PredictRun> / effect_publish_predict,
        "state_prediction_ready"_s <= "state_predict_error_out_decision"_s + completion<PredictRun> [guard_has_error_out_predict_run] / effect_store_error_out_predict_run_from_state_predict_error_out_decision,
        "state_prediction_ready"_s <= "state_predict_error_out_decision"_s + completion<PredictRun> [guard_no_error_out_predict_run],
        "state_session_ready"_s <= "state_predict_failed_error_out_decision"_s + completion<PredictRun> [guard_has_error_out_predict_run] / effect_store_error_out_predict_run_from_state_predict_failed_error_out_decision,
        "state_session_ready"_s <= "state_predict_failed_error_out_decision"_s + completion<PredictRun> [guard_no_error_out_predict_run],
        "state_execute_request_decision"_s <= "state_prediction_ready"_s + event<ExecuteRun>,
        "state_execute_begin"_s <= "state_execute_request_decision"_s + completion<ExecuteRun> [guard_execute_request_valid] / effect_begin_execute,
        "state_execute_failed_error_out_decision"_s <= "state_execute_request_decision"_s + completion<ExecuteRun> [guard_execute_request_invalid] / effect_mark_step_request_invalid_execute_run,
        "state_execute_graph_runtime_decision"_s <= "state_execute_begin"_s + completion<ExecuteRun>,
        "state_execute_running_graph"_s <= "state_execute_graph_runtime_decision"_s + completion<ExecuteRun> / effect_run_prediction_graph_graph_actor_type,
        "state_execute_graph_error_out_decision"_s <= "state_execute_running_graph"_s + completion<ExecuteRun>,
        "state_execute_graph_result_decision"_s <= "state_execute_graph_error_out_decision"_s + completion<ExecuteRun> [guard_has_graph_error_out_execute_run] / effect_store_execute_graph_error_out,
        "state_execute_graph_result_decision"_s <= "state_execute_graph_error_out_decision"_s + completion<ExecuteRun> [guard_no_graph_error_out_execute_run],
        "state_execute_error_out_decision"_s <= "state_execute_graph_result_decision"_s + completion<ExecuteRun> [guard_graph_step_accepted_execute_run] / effect_publish_execute,
        "state_execute_failed_error_out_decision"_s <= "state_execute_graph_result_decision"_s + completion<ExecuteRun> [guard_graph_step_rejected_execute_run] / effect_mark_graph_runtime_error_execute_run,
        "state_execution_ready"_s <= "state_execute_error_out_decision"_s + completion<ExecuteRun> [guard_has_error_out_execute_run] / effect_store_error_out_execute_run_from_state_execute_error_out_decision,
        "state_execution_ready"_s <= "state_execute_error_out_decision"_s + completion<ExecuteRun> [guard_no_error_out_execute_run],
        "state_session_ready"_s <= "state_execute_failed_error_out_decision"_s + completion<ExecuteRun> [guard_has_error_out_execute_run] / effect_store_error_out_execute_run_from_state_execute_failed_error_out_decision,
        "state_session_ready"_s <= "state_execute_failed_error_out_decision"_s + completion<ExecuteRun> [guard_no_error_out_execute_run],
        "state_sample_request_decision"_s <= "state_execution_ready"_s + event<SampleRun>,
        "state_sample_begin"_s <= "state_sample_request_decision"_s + completion<SampleRun> [guard_sample_request_valid] / effect_begin_sample,
        "state_sample_failed_error_out_decision"_s <= "state_sample_request_decision"_s + completion<SampleRun> [guard_sample_request_invalid] / effect_mark_step_request_invalid_sample_run,
        "state_sample_running_graph"_s <= "state_sample_begin"_s + completion<SampleRun> / effect_run_sampling_graph_graph_actor_type,
        "state_sample_graph_error_out_decision"_s <= "state_sample_running_graph"_s + completion<SampleRun>,
        "state_sample_graph_result_decision"_s <= "state_sample_graph_error_out_decision"_s + completion<SampleRun> [guard_has_graph_error_out_sample_run] / effect_store_sample_graph_error_out,
        "state_sample_graph_result_decision"_s <= "state_sample_graph_error_out_decision"_s + completion<SampleRun> [guard_no_graph_error_out_sample_run],
        "state_sample_error_out_decision"_s <= "state_sample_graph_result_decision"_s + completion<SampleRun> [guard_graph_step_accepted_sample_run] / effect_publish_sample,
        "state_sample_graph_failed_error_out_decision"_s <= "state_sample_graph_result_decision"_s + completion<SampleRun> [guard_graph_step_rejected_sample_run] / effect_mark_graph_runtime_error_sample_run,
        "state_session_ready"_s <= "state_sample_error_out_decision"_s + completion<SampleRun> [guard_has_error_out_sample_run] / effect_store_error_out_sample_run_from_state_sample_error_out_decision,
        "state_session_ready"_s <= "state_sample_error_out_decision"_s + completion<SampleRun> [guard_no_error_out_sample_run],
        "state_execution_ready"_s <= "state_sample_failed_error_out_decision"_s + completion<SampleRun> [guard_has_error_out_sample_run] / effect_store_error_out_sample_run_from_state_sample_failed_error_out_decision,
        "state_execution_ready"_s <= "state_sample_failed_error_out_decision"_s + completion<SampleRun> [guard_no_error_out_sample_run],
        "state_session_ready"_s <= "state_sample_graph_failed_error_out_decision"_s + completion<SampleRun> [guard_has_error_out_sample_run] / effect_store_error_out_sample_run_from_state_sample_graph_failed_error_out_decision,
        "state_session_ready"_s <= "state_sample_graph_failed_error_out_decision"_s + completion<SampleRun> [guard_no_error_out_sample_run],
        "state_session_ready"_s <= "state_session_ready"_s + event<EventCaptureTokenizerState> [guard_capture_tokenizer_state_valid] / effect_capture_tokenizer_state,
        "state_session_ready"_s <= "state_session_ready"_s + event<EventCaptureTokenizerState> [guard_capture_tokenizer_state_invalid] / effect_reject_capture_tokenizer_state_error_request_shape_from_state_session_ready,
        "state_prediction_ready"_s <= "state_prediction_ready"_s + event<EventCaptureTokenizerState> / effect_reject_capture_tokenizer_state_error_request_shape_from_state_prediction_ready,
        "state_execution_ready"_s <= "state_execution_ready"_s + event<EventCaptureTokenizerState> / effect_reject_capture_tokenizer_state_error_request_shape_from_state_execution_ready,
        "state_uninitialized"_s <= "state_uninitialized"_s + event<PredictRun> [guard_has_error_out_predict_run] / effect_mark_not_initialized_and_store_predict_run,
        "state_uninitialized"_s <= "state_uninitialized"_s + event<PredictRun> [guard_no_error_out_predict_run] / effect_mark_not_initialized_predict_run,
        "state_uninitialized"_s <= "state_uninitialized"_s + event<ExecuteRun> [guard_has_error_out_execute_run] / effect_mark_not_initialized_and_store_execute_run,
        "state_uninitialized"_s <= "state_uninitialized"_s + event<ExecuteRun> [guard_no_error_out_execute_run] / effect_mark_not_initialized_execute_run,
        "state_uninitialized"_s <= "state_uninitialized"_s + event<SampleRun> [guard_has_error_out_sample_run] / effect_mark_not_initialized_and_store_sample_run,
        "state_uninitialized"_s <= "state_uninitialized"_s + event<SampleRun> [guard_no_error_out_sample_run] / effect_mark_not_initialized_sample_run,
        "state_uninitialized"_s <= "state_uninitialized"_s + event<EventCaptureTokenizerState> / effect_reject_capture_tokenizer_state_error_not_initialized,
        "state_uninit_voice_error_out_decision"_s <= "state_uninitialized"_s + event<LoadVoiceRun> / effect_mark_not_initialized_load_voice_run,
        "state_uninit_voice_callback_decision"_s <= "state_uninit_voice_error_out_decision"_s + completion<LoadVoiceRun> [guard_has_error_out_load_voice_run] / effect_store_error_out_load_voice_run_from_state_uninit_voice_error_out_decision,
        "state_uninit_voice_callback_decision"_s <= "state_uninit_voice_error_out_decision"_s + completion<LoadVoiceRun> [guard_no_error_out_load_voice_run],
        "state_uninitialized"_s <= "state_uninit_voice_callback_decision"_s + completion<LoadVoiceRun> [guard_has_error_callback_load_voice_run] / effect_emit_load_voice_error_from_state_uninit_voice_callback_decision,
        "state_uninitialized"_s <= "state_uninit_voice_callback_decision"_s + completion<LoadVoiceRun> [guard_no_error_callback_load_voice_run],
        "state_uninit_prefill_voice_error_out_decision"_s <= "state_uninitialized"_s + event<PrefillVoiceRun> / effect_mark_not_initialized_prefill_voice_run,
        "state_uninit_prefill_voice_callback_decision"_s <= "state_uninit_prefill_voice_error_out_decision"_s + completion<PrefillVoiceRun> [guard_has_error_out_prefill_voice_run] / effect_store_error_out_prefill_voice_run_from_state_uninit_prefill_voice_error_out_decision,
        "state_uninit_prefill_voice_callback_decision"_s <= "state_uninit_prefill_voice_error_out_decision"_s + completion<PrefillVoiceRun> [guard_no_error_out_prefill_voice_run],
        "state_uninitialized"_s <= "state_uninit_prefill_voice_callback_decision"_s + completion<PrefillVoiceRun> [guard_has_error_callback_prefill_voice_run] / effect_emit_prefill_voice_error_from_state_uninit_prefill_voice_callback_decision,
        "state_uninitialized"_s <= "state_uninit_prefill_voice_callback_decision"_s + completion<PrefillVoiceRun> [guard_no_error_callback_prefill_voice_run],
        "state_uninit_begin_personaplex_prompt_error_out_decision"_s <= "state_uninitialized"_s + event<BeginPromptRun> / effect_mark_not_initialized_begin_prompt_run,
        "state_uninit_begin_personaplex_prompt_callback_decision"_s <= "state_uninit_begin_personaplex_prompt_error_out_decision"_s + completion<BeginPromptRun> [guard_has_error_out_begin_prompt_run] / effect_store_error_out_begin_prompt_run_from_state_uninit_begin_personaplex_prompt_error_out_decision,
        "state_uninit_begin_personaplex_prompt_callback_decision"_s <= "state_uninit_begin_personaplex_prompt_error_out_decision"_s + completion<BeginPromptRun> [guard_no_error_out_begin_prompt_run],
        "state_uninitialized"_s <= "state_uninit_begin_personaplex_prompt_callback_decision"_s + completion<BeginPromptRun> [guard_has_error_callback_begin_prompt_run] / effect_emit_begin_personaplex_prompt_error_from_state_uninit_begin_personaplex_prompt_callback_decision,
        "state_uninitialized"_s <= "state_uninit_begin_personaplex_prompt_callback_decision"_s + completion<BeginPromptRun> [guard_no_error_callback_begin_prompt_run],
        "state_uninit_prefill_personaplex_prompt_error_out_decision"_s <= "state_uninitialized"_s + event<PrefillPromptRun> / effect_mark_not_initialized_prefill_prompt_run,
        "state_uninit_prefill_personaplex_prompt_callback_decision"_s <= "state_uninit_prefill_personaplex_prompt_error_out_decision"_s + completion<PrefillPromptRun> [guard_has_error_out_prefill_prompt_run] / effect_store_error_out_prefill_prompt_run_from_state_uninit_prefill_personaplex_prompt_error_out_decision,
        "state_uninit_prefill_personaplex_prompt_callback_decision"_s <= "state_uninit_prefill_personaplex_prompt_error_out_decision"_s + completion<PrefillPromptRun> [guard_no_error_out_prefill_prompt_run],
        "state_uninitialized"_s <= "state_uninit_prefill_personaplex_prompt_callback_decision"_s + completion<PrefillPromptRun> [guard_has_error_callback_prefill_prompt_run] / effect_emit_prefill_personaplex_prompt_error_from_state_uninit_prefill_personaplex_prompt_callback_decision,
        "state_uninitialized"_s <= "state_uninit_prefill_personaplex_prompt_callback_decision"_s + completion<PrefillPromptRun> [guard_no_error_callback_prefill_prompt_run],
        "state_session_ready"_s <= "state_session_ready"_s + event<ExecuteRun> [guard_has_error_out_execute_run] / effect_mark_step_request_invalid_and_store_execute_run,
        "state_session_ready"_s <= "state_session_ready"_s + event<ExecuteRun> [guard_no_error_out_execute_run] / effect_mark_step_request_invalid_execute_run,
        "state_session_ready"_s <= "state_session_ready"_s + event<SampleRun> [guard_has_error_out_sample_run] / effect_mark_step_request_invalid_and_store_sample_run,
        "state_session_ready"_s <= "state_session_ready"_s + event<SampleRun> [guard_no_error_out_sample_run] / effect_mark_step_request_invalid_sample_run,
        "state_prediction_ready"_s <= "state_prediction_ready"_s + event<SampleRun> [guard_has_error_out_sample_run] / effect_mark_step_request_invalid_and_store_sample_run,
        "state_prediction_ready"_s <= "state_prediction_ready"_s + event<SampleRun> [guard_no_error_out_sample_run] / effect_mark_step_request_invalid_sample_run,
        "state_reset_graph_result_decision"_s <= "state_session_ready"_s + event<ResetRun> / effect_reset_graph_graph_actor_type_from_state_session_ready,
        "state_reset_graph_result_decision"_s <= "state_execution_ready"_s + event<ResetRun> / effect_reset_graph_graph_actor_type_from_state_execution_ready,
        "state_reset_graph_result_decision"_s <= "state_prediction_ready"_s + event<ResetRun> / effect_reset_graph_graph_actor_type_from_state_prediction_ready,
        "state_reset_memory_result_decision"_s <= "state_reset_graph_result_decision"_s + completion<ResetRun> [guard_reset_graph_succeeded] / effect_release_reset_sequence,
        "state_reset_failed"_s <= "state_reset_graph_result_decision"_s + completion<ResetRun> [guard_reset_graph_failed] / effect_mark_reset_graph_failed,
        "state_uninitialized"_s <= "state_reset_memory_result_decision"_s + completion<ResetRun> [guard_reset_memory_succeeded] / effect_reset_session_from_state_reset_memory_result_decision,
        "state_reset_failed"_s <= "state_reset_memory_result_decision"_s + completion<ResetRun> [guard_reset_memory_failed] / effect_mark_reset_memory_failed,
        "state_uninitialized"_s <= "state_reset_failed"_s + completion<ResetRun> / effect_reset_session_from_state_reset_failed,
        "state_uninitialized"_s <= "state_uninitialized"_s + unexpected_event<_> [guard_unexpected_error_out_present] / effect_mark_unexpected_and_store_from_state_uninitialized,
        "state_uninitialized"_s <= "state_uninitialized"_s + unexpected_event<_> [guard_unexpected_error_out_absent] / effect_mark_unexpected_from_state_uninitialized,
        "state_session_ready"_s <= "state_session_ready"_s + unexpected_event<_> [guard_unexpected_error_out_present] / effect_mark_unexpected_and_store_from_state_session_ready,
        "state_session_ready"_s <= "state_session_ready"_s + unexpected_event<_> [guard_unexpected_error_out_absent] / effect_mark_unexpected_from_state_session_ready,
        "state_execution_ready"_s <= "state_execution_ready"_s + unexpected_event<_> [guard_unexpected_error_out_present] / effect_mark_unexpected_and_store_from_state_execution_ready,
        "state_execution_ready"_s <= "state_execution_ready"_s + unexpected_event<_> [guard_unexpected_error_out_absent] / effect_mark_unexpected_from_state_execution_ready,
        "state_prediction_ready"_s <= "state_prediction_ready"_s + unexpected_event<_> [guard_unexpected_error_out_present] / effect_mark_unexpected_and_store_from_state_prediction_ready,
        "state_prediction_ready"_s <= "state_prediction_ready"_s + unexpected_event<_> [guard_unexpected_error_out_absent] / effect_mark_unexpected_from_state_prediction_ready,
    }
}
#[derive(Debug)]
pub struct SpeechPredictorMoshiContext {
    model: ModelContract,
    voice: VoiceContract,
    initialized: bool,
    voice_loaded: bool,
    voice_ready: bool,
    prompt_started: bool,
    prompt_ready: bool,
    voice_frame_index: i32,
    pre_remaining: i32,
    text_remaining: i32,
    post_remaining: i32,
    prompt_offset: i64,
    lmgen_offset: i64,
    codebook_count: i32,
    lmgen_delayed_dep_q: i32,
    lmgen_needed_tokens: i32,
    lmgen_rows: i32,
    max_delay: i32,
    policy_step_size: i32,
    policy_output_count: i32,
    embedding_frame_ok: bool,
    memory_accepted: bool,
    memory_error: PredictorError,
    graph_accepted: bool,
    graph_error: PredictorError,
    last_error: PredictorError,
    pending_error_out: bool,
    prompt_frame_text: i32,
    sequence_id: i32,
    last_complete: bool,
    last_remaining: i32,
}
impl Default for SpeechPredictorMoshiContext {
    fn default() -> Self {
        Self {
            model: ModelContract::default(),
            voice: VoiceContract::default(),
            initialized: false,
            voice_loaded: false,
            voice_ready: false,
            prompt_started: false,
            prompt_ready: false,
            voice_frame_index: 0,
            pre_remaining: 0,
            text_remaining: 0,
            post_remaining: 0,
            prompt_offset: 0,
            lmgen_offset: 0,
            codebook_count: 0,
            lmgen_delayed_dep_q: 0,
            lmgen_needed_tokens: 0,
            lmgen_rows: 0,
            max_delay: 0,
            policy_step_size: 1,
            policy_output_count: 1,
            embedding_frame_ok: false,
            memory_accepted: false,
            memory_error: PredictorError::None,
            graph_accepted: false,
            graph_error: PredictorError::None,
            last_error: PredictorError::None,
            pending_error_out: false,
            prompt_frame_text: 0,
            sequence_id: -1,
            last_complete: false,
            last_remaining: 0,
        }
    }
}
impl SpeechPredictorMoshiContext {
    fn bind_contract(&mut self, e: &InitRun) {
        self.model = e.model;
        self.sequence_id = e.sequence_id;
        self.codebook_count = e.model.n_q.saturating_add(1);
        let delay_count = usize::try_from(e.model.delay_count)
            .unwrap_or(MAX_CODEBOOKS)
            .min(MAX_CODEBOOKS);
        self.max_delay = e.model.delays[..delay_count]
            .iter()
            .copied()
            .max()
            .unwrap_or(0);
        self.lmgen_rows = self.max_delay + 2 + i32::from(e.model.personaplex);
        self.initialized = false;
        self.last_error = PredictorError::None;
    }
    fn invoke_memory(&mut self, f: Option<MemoryFn>, op: MemoryOp, arg: i32) {
        self.memory_error = PredictorError::None;
        self.memory_accepted = match f {
            None => {
                self.memory_error = PredictorError::GraphRuntimeUnavailable;
                false
            }
            Some(callback) => match callback(op, arg) {
                Ok(()) => true,
                Err(error) => {
                    self.memory_error = error;
                    false
                }
            },
        };
    }
    fn invoke_graph(
        &mut self,
        f: Option<GraphFn>,
        phase: GraphPhase,
        tokens: &[i32],
        workspace: Option<&Cell<PredictionWorkspace>>,
    ) {
        self.graph_error = PredictorError::None;
        self.graph_accepted = match (f, workspace) {
            (Some(cb), Some(w)) => match cb(phase, tokens, w) {
                Ok(()) => true,
                Err(e) => {
                    self.graph_error = e;
                    false
                }
            },
            (Some(cb), None) => {
                let w = Cell::new(PredictionWorkspace::default());
                match cb(phase, tokens, &w) {
                    Ok(()) => true,
                    Err(e) => {
                        self.graph_error = e;
                        false
                    }
                }
            }
            _ => {
                self.graph_error = PredictorError::GraphRuntimeUnavailable;
                false
            }
        };
    }
    fn invoke_reset_graph(&mut self, f: Option<fn() -> Result<(), PredictorError>>) {
        self.graph_error = f.map_or(PredictorError::GraphRuntimeUnavailable, |cb| {
            cb().err().unwrap_or(PredictorError::None)
        });
        self.graph_accepted = self.graph_error == PredictorError::None;
    }
    fn phase_for(n: &str) -> GraphPhase {
        if n.contains("voice") {
            GraphPhase::Voice
        } else if n.contains("prompt") {
            GraphPhase::Prompt
        } else if n.contains("sampling") {
            GraphPhase::Sampling
        } else {
            GraphPhase::Prediction
        }
    }
    fn reset(&mut self) {
        self.model = ModelContract::default();
        self.voice = VoiceContract::default();
        self.initialized = false;
        self.voice_loaded = false;
        self.voice_ready = false;
        self.prompt_started = false;
        self.prompt_ready = false;
        self.voice_frame_index = 0;
        self.pre_remaining = 0;
        self.text_remaining = 0;
        self.post_remaining = 0;
        self.last_error = PredictorError::None;
    }
    fn guard_predict_valid(&self, e: &PredictRun) -> bool {
        self.initialized
            && (!self.voice_loaded || (self.voice_ready && self.prompt_ready))
            && usize::try_from(self.codebook_count)
                .is_ok_and(|codebook_count| e.model_token_count == codebook_count)
            && e.planned_step_size == self.policy_step_size
            && e.planned_output_count == self.policy_output_count
    }
    fn predict_blocked(&self) -> bool {
        self.initialized && self.voice_loaded && (!self.voice_ready || !self.prompt_ready)
    }
    fn execute_valid(&self, e: &ExecuteRun) -> bool {
        self.initialized
            && (!self.voice_loaded || (self.voice_ready && self.prompt_ready))
            && usize::try_from(self.codebook_count)
                .is_ok_and(|codebook_count| e.model_token_count == codebook_count)
            && e.workspace.is_some()
    }
    fn sample_valid(&self, e: &SampleRun) -> bool {
        self.initialized
            && usize::try_from(self.codebook_count)
                .is_ok_and(|codebook_count| e.model_token_count == codebook_count)
            && e.audio_tokens_out.is_some()
            && e.text_token_out.is_some()
    }
    fn voice_prefill_valid(&self) -> bool {
        self.initialized
            && self.voice_loaded
            && !self.voice_ready
            && self.voice_frame_index < self.voice.prompt_frame_count
    }
    fn prompt_remaining(&self) -> i32 {
        self.pre_remaining
            .saturating_add(self.text_remaining)
            .saturating_add(self.post_remaining)
    }
    fn prompt_begin_valid(&self, e: &BeginPromptRun) -> bool {
        self.initialized
            && self.voice_loaded
            && self.voice_ready
            && !self.prompt_ready
            && !self.prompt_started
            && e.text_token_count >= 0
            && e.pre_text_silence_frames >= 0
            && e.post_text_silence_frames >= 0
            && self.model.personaplex
            && u32::try_from(self.codebook_count).is_ok_and(|codebook_count| {
                self.model.inference_prompt_token_count == codebook_count
            })
            && i32::try_from(MAX_DELAY_ROWS).is_ok_and(|max_delay_rows| {
                e.text_token_count
                    .saturating_add(e.pre_text_silence_frames)
                    .saturating_add(e.post_text_silence_frames)
                    <= max_delay_rows
            })
    }
    fn prompt_phase_valid(&self, e: &PrefillPromptRun) -> bool {
        self.pre_remaining > 0
            || (self.pre_remaining == 0
                && self.text_remaining > 0
                && (0..self.model.text_card).contains(&e.text_token))
            || (self.pre_remaining == 0 && self.text_remaining == 0 && self.post_remaining > 0)
    }
    fn prompt_prefill_valid(&self, e: &PrefillPromptRun) -> bool {
        self.initialized
            && self.voice_loaded
            && self.voice_ready
            && self.prompt_started
            && !self.prompt_ready
            && self.prompt_remaining() > 0
            && self.prompt_phase_valid(e)
    }
    fn bind_prompt(&mut self, e: &BeginPromptRun) {
        self.pre_remaining = e.pre_text_silence_frames;
        self.text_remaining = e.text_token_count;
        self.post_remaining = e.post_text_silence_frames;
        self.prompt_started = true;
        self.prompt_ready = false;
    }
    fn capture_state(&self, e: &EventCaptureTokenizerState) {
        e.offset_out.inspect(|output| output.set(self.lmgen_offset));
        e.error_out
            .inspect(|output| output.set(PredictorError::None));
    }
    fn publish_prompt_pending(&mut self, e: &PrefillPromptRun) {
        self.last_complete = false;
        self.last_remaining = self.prompt_remaining();
        if let Some(o) = e.complete_out {
            o.set(false);
        }
        if let Some(o) = e.remaining_frames_out {
            o.set(self.last_remaining);
        }
    }
}

impl SpeechPredictorMoshiStateMachineContext for SpeechPredictorMoshiContext {
    fn effect_advance_personaplex_prompt_post_silence(
        &mut self,
        _event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        self.post_remaining = self.post_remaining.saturating_sub(1);
        self.lmgen_offset = self.lmgen_offset.saturating_add(1);
        Ok(())
    }

    fn effect_advance_personaplex_prompt_pre_silence(
        &mut self,
        _event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        self.pre_remaining = self.pre_remaining.saturating_sub(1);
        self.lmgen_offset = self.lmgen_offset.saturating_add(1);
        Ok(())
    }

    fn effect_advance_personaplex_prompt_text(
        &mut self,
        _event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        self.text_remaining = self.text_remaining.saturating_sub(1);
        self.lmgen_offset = self.lmgen_offset.saturating_add(1);
        Ok(())
    }

    fn effect_advance_voice_prefill(&mut self, _event: &PrefillVoiceRun) -> Result<(), ()> {
        self.voice_frame_index = self.voice_frame_index.saturating_add(1);
        self.lmgen_offset = self.lmgen_offset.saturating_add(1);
        Ok(())
    }

    fn effect_allocate_personaplex_prompt_slot(
        &mut self,
        event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        self.invoke_memory(event.memory, MemoryOp::AllocateSlots, 1);
        Ok(())
    }

    fn effect_allocate_sequence(&mut self, event: &InitRun) -> Result<(), ()> {
        self.invoke_memory(event.memory, MemoryOp::AllocateSequence, event.sequence_id);
        Ok(())
    }

    fn effect_allocate_step_slot(&mut self, event: &PredictRun) -> Result<(), ()> {
        self.invoke_memory(
            event.memory,
            MemoryOp::AllocateSlots,
            event.planned_step_size,
        );
        Ok(())
    }

    fn effect_allocate_voice_prefill_slot(&mut self, event: &PrefillVoiceRun) -> Result<(), ()> {
        self.invoke_memory(event.memory, MemoryOp::AllocateSlots, 1);
        Ok(())
    }

    fn effect_begin_personaplex_prompt_prefill(
        &mut self,
        _event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        self.prompt_offset = 0;
        self.last_error = PredictorError::None;
        Ok(())
    }

    fn effect_begin_predict(&mut self, _event: &PredictRun) -> Result<(), ()> {
        self.last_error = PredictorError::None;
        Ok(())
    }

    fn effect_begin_execute(&mut self, _event: &ExecuteRun) -> Result<(), ()> {
        self.last_error = PredictorError::None;
        Ok(())
    }

    fn effect_begin_sample(&mut self, _event: &SampleRun) -> Result<(), ()> {
        self.last_error = PredictorError::None;
        Ok(())
    }

    fn effect_begin_voice_prefill(&mut self, _event: &PrefillVoiceRun) -> Result<(), ()> {
        self.embedding_frame_ok = false;
        self.last_error = PredictorError::None;
        Ok(())
    }

    fn effect_bind_contract(&mut self, event: &InitRun) -> Result<(), ()> {
        self.bind_contract(event);
        Ok(())
    }

    fn effect_bind_empty_personaplex_prompt(&mut self, event: &BeginPromptRun) -> Result<(), ()> {
        self.bind_prompt(event);
        Ok(())
    }

    fn effect_bind_personaplex_prompt(&mut self, event: &BeginPromptRun) -> Result<(), ()> {
        self.bind_prompt(event);
        Ok(())
    }

    fn effect_bind_voice_contract(&mut self, event: &LoadVoiceRun) -> Result<(), ()> {
        self.voice = event.voice;
        self.voice_loaded = true;
        self.voice_ready = false;
        self.prompt_ready = false;
        self.prompt_started = false;
        self.voice_frame_index = 0;
        Ok(())
    }

    fn effect_build_personaplex_prompt_silence_frame_from_state_prefill_personaplex_prompt_phase_decision(
        &mut self,
        event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        self.prompt_frame_text = event.text_token;
        Ok(())
    }

    fn effect_build_personaplex_prompt_text_frame(
        &mut self,
        event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        self.prompt_frame_text = event.text_token;
        Ok(())
    }

    fn effect_capture_memory(&mut self, event: &PredictRun) -> Result<(), ()> {
        self.invoke_memory(event.memory, MemoryOp::Capture, 0);
        Ok(())
    }

    fn effect_capture_personaplex_prompt_memory(
        &mut self,
        event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        self.invoke_memory(event.memory, MemoryOp::Capture, 0);
        Ok(())
    }

    fn effect_capture_tokenizer_state(
        &mut self,
        event: &EventCaptureTokenizerState,
    ) -> Result<(), ()> {
        self.capture_state(event);
        Ok(())
    }

    fn effect_capture_voice_prefill_memory(&mut self, event: &PrefillVoiceRun) -> Result<(), ()> {
        self.invoke_memory(event.memory, MemoryOp::Capture, 0);
        Ok(())
    }

    fn effect_configure_personaplex_lmgen(&mut self, _event: &InitRun) -> Result<(), ()> {
        self.lmgen_delayed_dep_q = self.model.inference_dep_q;
        self.lmgen_needed_tokens = self.codebook_count - self.lmgen_delayed_dep_q - 1;
        self.lmgen_rows = self.max_delay + 3;
        Ok(())
    }

    fn effect_configure_standard_lmgen(&mut self, _event: &InitRun) -> Result<(), ()> {
        self.lmgen_delayed_dep_q = self.model.dep_q;
        self.lmgen_needed_tokens = self.codebook_count - self.lmgen_delayed_dep_q - 1;
        self.lmgen_rows = self.max_delay + 2;
        Ok(())
    }

    fn effect_copy_voice_cache(&mut self, event: &PrefillVoiceRun) -> Result<(), ()> {
        self.voice_ready = true;
        if let Some(o) = event.complete_out {
            o.set(true);
        }
        if let Some(o) = event.remaining_frames_out {
            o.set(0);
        }
        Ok(())
    }

    fn effect_emit_begin_personaplex_prompt_done(
        &mut self,
        event: &BeginPromptRun,
    ) -> Result<(), ()> {
        if let Some(cb) = event.on_done {
            let _ = cb(self.pre_remaining + self.text_remaining + self.post_remaining);
        }
        Ok(())
    }

    fn effect_emit_begin_personaplex_prompt_error_from_state_personaplex_prompt_begin_failed_callback_decision(
        &mut self,
        event: &BeginPromptRun,
    ) -> Result<(), ()> {
        if let Some(cb) = event.on_error {
            let _ = cb(self.last_error);
        }
        Ok(())
    }

    fn effect_emit_begin_personaplex_prompt_error_from_state_uninit_begin_personaplex_prompt_callback_decision(
        &mut self,
        event: &BeginPromptRun,
    ) -> Result<(), ()> {
        if let Some(cb) = event.on_error {
            let _ = cb(self.last_error);
        }
        Ok(())
    }

    fn effect_emit_initialize_done(&mut self, event: &InitRun) -> Result<(), ()> {
        self.initialized = true;
        if let Some(cb) = event.on_done {
            let _ = cb(self.codebook_count, self.model.dep_q);
        }
        Ok(())
    }

    fn effect_emit_initialize_error(&mut self, event: &InitRun) -> Result<(), ()> {
        if let Some(cb) = event.on_error {
            let _ = cb(self.last_error);
        }
        Ok(())
    }

    fn effect_emit_load_voice_done(&mut self, event: &LoadVoiceRun) -> Result<(), ()> {
        if let Some(cb) = event.on_done {
            let _ = cb(self.voice.prompt_frame_count);
        }
        Ok(())
    }

    fn effect_emit_load_voice_error_from_state_uninit_voice_callback_decision(
        &mut self,
        event: &LoadVoiceRun,
    ) -> Result<(), ()> {
        if let Some(cb) = event.on_error {
            let _ = cb(self.last_error);
        }
        Ok(())
    }

    fn effect_emit_load_voice_error_from_state_voice_failed_callback_decision(
        &mut self,
        event: &LoadVoiceRun,
    ) -> Result<(), ()> {
        if let Some(cb) = event.on_error {
            let _ = cb(self.last_error);
        }
        Ok(())
    }

    fn effect_emit_prefill_personaplex_prompt_done(
        &mut self,
        event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        if let Some(cb) = event.on_done {
            let _ = cb(self.prompt_ready, self.prompt_remaining());
        }
        Ok(())
    }

    fn effect_emit_prefill_personaplex_prompt_error_from_state_prefill_personaplex_prompt_failed_callback_decision(
        &mut self,
        event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        if let Some(cb) = event.on_error {
            let _ = cb(self.last_error);
        }
        Ok(())
    }

    fn effect_emit_prefill_personaplex_prompt_error_from_state_uninit_prefill_personaplex_prompt_callback_decision(
        &mut self,
        event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        if let Some(cb) = event.on_error {
            let _ = cb(self.last_error);
        }
        Ok(())
    }

    fn effect_emit_prefill_voice_done(&mut self, event: &PrefillVoiceRun) -> Result<(), ()> {
        if let Some(cb) = event.on_done {
            let _ = cb(
                self.voice_ready,
                self.voice
                    .prompt_frame_count
                    .saturating_sub(self.voice_frame_index),
            );
        }
        Ok(())
    }

    fn effect_emit_prefill_voice_error_from_state_prefill_voice_failed_callback_decision(
        &mut self,
        event: &PrefillVoiceRun,
    ) -> Result<(), ()> {
        if let Some(cb) = event.on_error {
            let _ = cb(self.last_error);
        }
        Ok(())
    }

    fn effect_emit_prefill_voice_error_from_state_uninit_prefill_voice_callback_decision(
        &mut self,
        event: &PrefillVoiceRun,
    ) -> Result<(), ()> {
        if let Some(cb) = event.on_error {
            let _ = cb(self.last_error);
        }
        Ok(())
    }

    fn effect_finish_personaplex_prompt(&mut self, event: &PrefillPromptRun) -> Result<(), ()> {
        self.prompt_ready = true;
        if let Some(o) = event.complete_out {
            o.set(true);
        }
        if let Some(o) = event.remaining_frames_out {
            o.set(0);
        }
        Ok(())
    }

    fn effect_initialize_graph_graph_actor_type(&mut self, event: &InitRun) -> Result<(), ()> {
        self.invoke_graph(event.graph, GraphPhase::Full, &[], None);
        Ok(())
    }

    fn effect_load_voice_embedding_frame_bf16(
        &mut self,
        _event: &PrefillVoiceRun,
    ) -> Result<(), ()> {
        self.embedding_frame_ok = true;
        Ok(())
    }

    fn effect_load_voice_embedding_frame_f16(
        &mut self,
        _event: &PrefillVoiceRun,
    ) -> Result<(), ()> {
        self.embedding_frame_ok = true;
        Ok(())
    }

    fn effect_load_voice_embedding_frame_f32(
        &mut self,
        _event: &PrefillVoiceRun,
    ) -> Result<(), ()> {
        self.embedding_frame_ok = true;
        Ok(())
    }

    fn effect_mark_bind_failed(&mut self, _event: &InitRun) -> Result<(), ()> {
        self.last_error = PredictorError::BindFailed;
        Ok(())
    }

    fn effect_mark_graph_runtime_error_execute_run(
        &mut self,
        _event: &ExecuteRun,
    ) -> Result<(), ()> {
        self.last_error = PredictorError::GraphRuntime;
        Ok(())
    }

    fn effect_mark_graph_runtime_error_init_run(&mut self, _event: &InitRun) -> Result<(), ()> {
        self.last_error = PredictorError::GraphRuntime;
        Ok(())
    }

    fn effect_mark_graph_runtime_error_prefill_prompt_run(
        &mut self,
        _event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        self.last_error = PredictorError::GraphRuntime;
        Ok(())
    }

    fn effect_mark_graph_runtime_error_prefill_voice_run(
        &mut self,
        _event: &PrefillVoiceRun,
    ) -> Result<(), ()> {
        self.last_error = PredictorError::GraphRuntime;
        Ok(())
    }

    fn effect_mark_graph_runtime_error_sample_run(&mut self, _event: &SampleRun) -> Result<(), ()> {
        self.last_error = PredictorError::GraphRuntime;
        Ok(())
    }

    fn effect_mark_memory_error_init_run_from_state_allocate_sequence_decision(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        self.last_error = PredictorError::Memory;
        Ok(())
    }

    fn effect_mark_memory_error_init_run_from_state_reserve_memory_decision(
        &mut self,
        _event: &InitRun,
    ) -> Result<(), ()> {
        self.last_error = PredictorError::Memory;
        Ok(())
    }

    fn effect_mark_memory_error_predict_run_from_state_predict_allocate_slot_decision(
        &mut self,
        _event: &PredictRun,
    ) -> Result<(), ()> {
        self.last_error = PredictorError::Memory;
        Ok(())
    }

    fn effect_mark_memory_error_predict_run_from_state_predict_capture_memory_decision(
        &mut self,
        _event: &PredictRun,
    ) -> Result<(), ()> {
        self.last_error = PredictorError::Memory;
        Ok(())
    }

    fn effect_mark_memory_error_prefill_prompt_run_from_state_prefill_personaplex_prompt_allocate_slot_decision(
        &mut self,
        _event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        self.last_error = PredictorError::Memory;
        Ok(())
    }

    fn effect_mark_memory_error_prefill_prompt_run_from_state_prefill_personaplex_prompt_capture_memory_decision(
        &mut self,
        _event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        self.last_error = PredictorError::Memory;
        Ok(())
    }

    fn effect_mark_memory_error_prefill_voice_run_from_state_prefill_voice_allocate_slot_decision(
        &mut self,
        _event: &PrefillVoiceRun,
    ) -> Result<(), ()> {
        self.last_error = PredictorError::Memory;
        Ok(())
    }

    fn effect_mark_memory_error_prefill_voice_run_from_state_prefill_voice_capture_memory_decision(
        &mut self,
        _event: &PrefillVoiceRun,
    ) -> Result<(), ()> {
        self.last_error = PredictorError::Memory;
        Ok(())
    }

    fn effect_mark_not_initialized_and_store_execute_run(
        &mut self,
        event: &ExecuteRun,
    ) -> Result<(), ()> {
        self.last_error = PredictorError::NotInitialized;
        if let Some(o) = event.error_out {
            o.set(self.last_error);
        }
        Ok(())
    }

    fn effect_mark_not_initialized_and_store_predict_run(
        &mut self,
        event: &PredictRun,
    ) -> Result<(), ()> {
        self.last_error = PredictorError::NotInitialized;
        if let Some(o) = event.error_out {
            o.set(self.last_error);
        }
        Ok(())
    }

    fn effect_mark_not_initialized_and_store_sample_run(
        &mut self,
        event: &SampleRun,
    ) -> Result<(), ()> {
        self.last_error = PredictorError::NotInitialized;
        if let Some(o) = event.error_out {
            o.set(self.last_error);
        }
        Ok(())
    }

    fn effect_mark_not_initialized_begin_prompt_run(
        &mut self,
        event: &BeginPromptRun,
    ) -> Result<(), ()> {
        self.last_error = PredictorError::NotInitialized;
        if let Some(o) = event.error_out {
            o.set(self.last_error);
        }
        Ok(())
    }

    fn effect_mark_not_initialized_execute_run(&mut self, event: &ExecuteRun) -> Result<(), ()> {
        self.last_error = PredictorError::NotInitialized;
        if let Some(o) = event.error_out {
            o.set(self.last_error);
        }
        Ok(())
    }

    fn effect_mark_not_initialized_load_voice_run(
        &mut self,
        event: &LoadVoiceRun,
    ) -> Result<(), ()> {
        self.last_error = PredictorError::NotInitialized;
        if let Some(o) = event.error_out {
            o.set(self.last_error);
        }
        Ok(())
    }

    fn effect_mark_not_initialized_predict_run(&mut self, event: &PredictRun) -> Result<(), ()> {
        self.last_error = PredictorError::NotInitialized;
        if let Some(o) = event.error_out {
            o.set(self.last_error);
        }
        Ok(())
    }

    fn effect_mark_not_initialized_prefill_prompt_run(
        &mut self,
        event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        self.last_error = PredictorError::NotInitialized;
        if let Some(o) = event.error_out {
            o.set(self.last_error);
        }
        Ok(())
    }

    fn effect_mark_not_initialized_prefill_voice_run(
        &mut self,
        event: &PrefillVoiceRun,
    ) -> Result<(), ()> {
        self.last_error = PredictorError::NotInitialized;
        if let Some(o) = event.error_out {
            o.set(self.last_error);
        }
        Ok(())
    }

    fn effect_mark_not_initialized_sample_run(&mut self, event: &SampleRun) -> Result<(), ()> {
        self.last_error = PredictorError::NotInitialized;
        if let Some(o) = event.error_out {
            o.set(self.last_error);
        }
        Ok(())
    }

    fn effect_mark_personaplex_prompt_error_begin_prompt_run(
        &mut self,
        _event: &BeginPromptRun,
    ) -> Result<(), ()> {
        self.last_error = PredictorError::PersonaPlexPrompt;
        Ok(())
    }

    fn effect_mark_personaplex_prompt_error_prefill_prompt_run_from_state_prefill_personaplex_prompt_phase_decision(
        &mut self,
        _event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        self.last_error = PredictorError::PersonaPlexPrompt;
        Ok(())
    }

    fn effect_mark_personaplex_prompt_error_prefill_prompt_run_from_state_prefill_personaplex_prompt_request_decision(
        &mut self,
        _event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        self.last_error = PredictorError::PersonaPlexPrompt;
        Ok(())
    }

    fn effect_mark_reset_graph_failed(&mut self, _event: &ResetRun) -> Result<(), ()> {
        self.last_error = PredictorError::GraphRuntime;
        Ok(())
    }

    fn effect_mark_reset_memory_failed(&mut self, _event: &ResetRun) -> Result<(), ()> {
        self.last_error = PredictorError::GraphRuntime;
        Ok(())
    }

    fn effect_mark_step_request_invalid_and_store_execute_run(
        &mut self,
        event: &ExecuteRun,
    ) -> Result<(), ()> {
        self.last_error = PredictorError::RequestShape;
        if let Some(o) = event.error_out {
            o.set(self.last_error);
        }
        Ok(())
    }

    fn effect_mark_step_request_invalid_and_store_sample_run(
        &mut self,
        event: &SampleRun,
    ) -> Result<(), ()> {
        self.last_error = PredictorError::RequestShape;
        if let Some(o) = event.error_out {
            o.set(self.last_error);
        }
        Ok(())
    }

    fn effect_mark_step_request_invalid_execute_run(
        &mut self,
        event: &ExecuteRun,
    ) -> Result<(), ()> {
        self.last_error = PredictorError::RequestShape;
        if let Some(o) = event.error_out {
            o.set(self.last_error);
        }
        Ok(())
    }

    fn effect_mark_step_request_invalid_predict_run(
        &mut self,
        event: &PredictRun,
    ) -> Result<(), ()> {
        self.last_error = PredictorError::RequestShape;
        if let Some(o) = event.error_out {
            o.set(self.last_error);
        }
        Ok(())
    }

    fn effect_mark_step_request_invalid_sample_run(&mut self, event: &SampleRun) -> Result<(), ()> {
        self.last_error = PredictorError::RequestShape;
        if let Some(o) = event.error_out {
            o.set(self.last_error);
        }
        Ok(())
    }

    fn effect_mark_unexpected_and_store_from_state_execution_ready(&mut self) -> Result<(), ()> {
        self.last_error = PredictorError::UnexpectedEvent;
        Ok(())
    }

    fn effect_mark_unexpected_and_store_from_state_prediction_ready(&mut self) -> Result<(), ()> {
        self.last_error = PredictorError::UnexpectedEvent;
        Ok(())
    }

    fn effect_mark_unexpected_and_store_from_state_session_ready(&mut self) -> Result<(), ()> {
        self.last_error = PredictorError::UnexpectedEvent;
        Ok(())
    }

    fn effect_mark_unexpected_and_store_from_state_uninitialized(&mut self) -> Result<(), ()> {
        self.last_error = PredictorError::UnexpectedEvent;
        Ok(())
    }

    fn effect_mark_unexpected_from_state_execution_ready(&mut self) -> Result<(), ()> {
        self.last_error = PredictorError::UnexpectedEvent;
        Ok(())
    }

    fn effect_mark_unexpected_from_state_prediction_ready(&mut self) -> Result<(), ()> {
        self.last_error = PredictorError::UnexpectedEvent;
        Ok(())
    }

    fn effect_mark_unexpected_from_state_session_ready(&mut self) -> Result<(), ()> {
        self.last_error = PredictorError::UnexpectedEvent;
        Ok(())
    }

    fn effect_mark_unexpected_from_state_uninitialized(&mut self) -> Result<(), ()> {
        self.last_error = PredictorError::UnexpectedEvent;
        Ok(())
    }

    fn effect_mark_voice_contract_error_load_voice_run(
        &mut self,
        _event: &LoadVoiceRun,
    ) -> Result<(), ()> {
        self.last_error = PredictorError::VoiceContract;
        Ok(())
    }

    fn effect_mark_voice_contract_error_prefill_voice_run_from_state_prefill_voice_embedding_frame_decision(
        &mut self,
        _event: &PrefillVoiceRun,
    ) -> Result<(), ()> {
        self.last_error = PredictorError::VoiceContract;
        Ok(())
    }

    fn effect_mark_voice_contract_error_prefill_voice_run_from_state_prefill_voice_request_decision(
        &mut self,
        _event: &PrefillVoiceRun,
    ) -> Result<(), ()> {
        self.last_error = PredictorError::VoiceContract;
        Ok(())
    }

    fn effect_mark_voice_prompt_pending(&mut self, _event: &PredictRun) -> Result<(), ()> {
        self.last_error = PredictorError::VoicePromptPending;
        Ok(())
    }

    fn effect_publish_execute(&mut self, _event: &ExecuteRun) -> Result<(), ()> {
        self.last_error = PredictorError::None;
        Ok(())
    }

    fn effect_publish_personaplex_prompt_pending(
        &mut self,
        event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        self.publish_prompt_pending(event);
        Ok(())
    }

    fn effect_publish_predict(&mut self, _event: &PredictRun) -> Result<(), ()> {
        self.last_error = PredictorError::None;
        Ok(())
    }

    fn effect_publish_sample(&mut self, _event: &SampleRun) -> Result<(), ()> {
        self.last_error = PredictorError::None;
        Ok(())
    }

    fn effect_reject_capture_tokenizer_state_error_not_initialized(
        &mut self,
        event: &EventCaptureTokenizerState,
    ) -> Result<(), ()> {
        self.last_error = PredictorError::RequestShape;
        if let Some(o) = event.error_out {
            o.set(self.last_error);
        }
        Ok(())
    }

    fn effect_reject_capture_tokenizer_state_error_request_shape_from_state_execution_ready(
        &mut self,
        event: &EventCaptureTokenizerState,
    ) -> Result<(), ()> {
        self.last_error = PredictorError::RequestShape;
        if let Some(o) = event.error_out {
            o.set(self.last_error);
        }
        Ok(())
    }

    fn effect_reject_capture_tokenizer_state_error_request_shape_from_state_prediction_ready(
        &mut self,
        event: &EventCaptureTokenizerState,
    ) -> Result<(), ()> {
        self.last_error = PredictorError::RequestShape;
        if let Some(o) = event.error_out {
            o.set(self.last_error);
        }
        Ok(())
    }

    fn effect_reject_capture_tokenizer_state_error_request_shape_from_state_session_ready(
        &mut self,
        event: &EventCaptureTokenizerState,
    ) -> Result<(), ()> {
        self.last_error = PredictorError::RequestShape;
        if let Some(o) = event.error_out {
            o.set(self.last_error);
        }
        Ok(())
    }

    fn effect_release_reset_sequence(&mut self, event: &ResetRun) -> Result<(), ()> {
        self.invoke_memory(event.memory, MemoryOp::FreeSequence, self.sequence_id);
        Ok(())
    }

    fn effect_reserve_memory(&mut self, event: &InitRun) -> Result<(), ()> {
        self.invoke_memory(event.memory, MemoryOp::Reserve, event.max_blocks);
        Ok(())
    }

    fn effect_reset_graph_graph_actor_type_from_state_execution_ready(
        &mut self,
        event: &ResetRun,
    ) -> Result<(), ()> {
        self.invoke_reset_graph(event.graph);
        Ok(())
    }

    fn effect_reset_graph_graph_actor_type_from_state_prediction_ready(
        &mut self,
        event: &ResetRun,
    ) -> Result<(), ()> {
        self.invoke_reset_graph(event.graph);
        Ok(())
    }

    fn effect_reset_graph_graph_actor_type_from_state_session_ready(
        &mut self,
        event: &ResetRun,
    ) -> Result<(), ()> {
        self.invoke_reset_graph(event.graph);
        Ok(())
    }

    fn effect_reset_session_from_state_reset_failed(
        &mut self,
        _event: &ResetRun,
    ) -> Result<(), ()> {
        self.reset();
        Ok(())
    }

    fn effect_reset_session_from_state_reset_memory_result_decision(
        &mut self,
        _event: &ResetRun,
    ) -> Result<(), ()> {
        self.reset();
        Ok(())
    }

    fn effect_run_personaplex_prompt_graph_runtime_graph_actor_type(
        &mut self,
        event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        self.invoke_graph(event.graph, GraphPhase::Prompt, &event.model_tokens, None);
        Ok(())
    }

    fn effect_run_prediction_graph_graph_actor_type(
        &mut self,
        event: &ExecuteRun,
    ) -> Result<(), ()> {
        self.invoke_graph(
            event.graph,
            GraphPhase::Prediction,
            &event.model_tokens,
            event.workspace,
        );
        Ok(())
    }

    fn effect_run_sampling_graph_graph_actor_type(&mut self, event: &SampleRun) -> Result<(), ()> {
        self.invoke_graph(event.graph, GraphPhase::Sampling, &event.model_tokens, None);
        Ok(())
    }

    fn effect_run_voice_graph_runtime_graph_actor_type(
        &mut self,
        event: &PrefillVoiceRun,
    ) -> Result<(), ()> {
        self.invoke_graph(event.graph, GraphPhase::Voice, &event.model_tokens, None);
        Ok(())
    }

    fn effect_store_error_out_begin_prompt_run_from_state_personaplex_prompt_begin_error_out_decision(
        &mut self,
        event: &BeginPromptRun,
    ) -> Result<(), ()> {
        if let Some(o) = event.error_out {
            o.set(self.last_error);
        }
        Ok(())
    }

    fn effect_store_error_out_begin_prompt_run_from_state_personaplex_prompt_begin_failed_error_out_decision(
        &mut self,
        event: &BeginPromptRun,
    ) -> Result<(), ()> {
        if let Some(o) = event.error_out {
            o.set(self.last_error);
        }
        Ok(())
    }

    fn effect_store_error_out_begin_prompt_run_from_state_uninit_begin_personaplex_prompt_error_out_decision(
        &mut self,
        event: &BeginPromptRun,
    ) -> Result<(), ()> {
        if let Some(o) = event.error_out {
            o.set(self.last_error);
        }
        Ok(())
    }

    fn effect_store_error_out_execute_run_from_state_execute_error_out_decision(
        &mut self,
        event: &ExecuteRun,
    ) -> Result<(), ()> {
        if let Some(o) = event.error_out {
            o.set(self.last_error);
        }
        Ok(())
    }

    fn effect_store_error_out_execute_run_from_state_execute_failed_error_out_decision(
        &mut self,
        event: &ExecuteRun,
    ) -> Result<(), ()> {
        if let Some(o) = event.error_out {
            o.set(self.last_error);
        }
        Ok(())
    }

    fn effect_store_error_out_init_run_from_state_init_error_out_decision(
        &mut self,
        event: &InitRun,
    ) -> Result<(), ()> {
        if let Some(o) = event.error_out {
            o.set(self.last_error);
        }
        Ok(())
    }

    fn effect_store_error_out_init_run_from_state_init_failed_error_out_decision(
        &mut self,
        event: &InitRun,
    ) -> Result<(), ()> {
        if let Some(o) = event.error_out {
            o.set(self.last_error);
        }
        Ok(())
    }

    fn effect_store_error_out_load_voice_run_from_state_uninit_voice_error_out_decision(
        &mut self,
        event: &LoadVoiceRun,
    ) -> Result<(), ()> {
        if let Some(o) = event.error_out {
            o.set(self.last_error);
        }
        Ok(())
    }

    fn effect_store_error_out_load_voice_run_from_state_voice_error_out_decision(
        &mut self,
        event: &LoadVoiceRun,
    ) -> Result<(), ()> {
        if let Some(o) = event.error_out {
            o.set(self.last_error);
        }
        Ok(())
    }

    fn effect_store_error_out_load_voice_run_from_state_voice_failed_error_out_decision(
        &mut self,
        event: &LoadVoiceRun,
    ) -> Result<(), ()> {
        if let Some(o) = event.error_out {
            o.set(self.last_error);
        }
        Ok(())
    }

    fn effect_store_error_out_predict_run_from_state_predict_error_out_decision(
        &mut self,
        event: &PredictRun,
    ) -> Result<(), ()> {
        if let Some(o) = event.error_out {
            o.set(self.last_error);
        }
        Ok(())
    }

    fn effect_store_error_out_predict_run_from_state_predict_failed_error_out_decision(
        &mut self,
        event: &PredictRun,
    ) -> Result<(), ()> {
        if let Some(o) = event.error_out {
            o.set(self.last_error);
        }
        Ok(())
    }

    fn effect_store_error_out_prefill_prompt_run_from_state_prefill_personaplex_prompt_error_out_decision(
        &mut self,
        event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        if let Some(o) = event.error_out {
            o.set(self.last_error);
        }
        Ok(())
    }

    fn effect_store_error_out_prefill_prompt_run_from_state_prefill_personaplex_prompt_failed_error_out_decision(
        &mut self,
        event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        if let Some(o) = event.error_out {
            o.set(self.last_error);
        }
        Ok(())
    }

    fn effect_store_error_out_prefill_prompt_run_from_state_uninit_prefill_personaplex_prompt_error_out_decision(
        &mut self,
        event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        if let Some(o) = event.error_out {
            o.set(self.last_error);
        }
        Ok(())
    }

    fn effect_store_error_out_prefill_voice_run_from_state_prefill_voice_error_out_decision(
        &mut self,
        event: &PrefillVoiceRun,
    ) -> Result<(), ()> {
        if let Some(o) = event.error_out {
            o.set(self.last_error);
        }
        Ok(())
    }

    fn effect_store_error_out_prefill_voice_run_from_state_prefill_voice_failed_error_out_decision(
        &mut self,
        event: &PrefillVoiceRun,
    ) -> Result<(), ()> {
        if let Some(o) = event.error_out {
            o.set(self.last_error);
        }
        Ok(())
    }

    fn effect_store_error_out_prefill_voice_run_from_state_uninit_prefill_voice_error_out_decision(
        &mut self,
        event: &PrefillVoiceRun,
    ) -> Result<(), ()> {
        if let Some(o) = event.error_out {
            o.set(self.last_error);
        }
        Ok(())
    }

    fn effect_store_error_out_sample_run_from_state_sample_error_out_decision(
        &mut self,
        event: &SampleRun,
    ) -> Result<(), ()> {
        if let Some(o) = event.error_out {
            o.set(self.last_error);
        }
        Ok(())
    }

    fn effect_store_error_out_sample_run_from_state_sample_failed_error_out_decision(
        &mut self,
        event: &SampleRun,
    ) -> Result<(), ()> {
        if let Some(o) = event.error_out {
            o.set(self.last_error);
        }
        Ok(())
    }

    fn effect_store_error_out_sample_run_from_state_sample_graph_failed_error_out_decision(
        &mut self,
        event: &SampleRun,
    ) -> Result<(), ()> {
        if let Some(o) = event.error_out {
            o.set(self.last_error);
        }
        Ok(())
    }

    fn effect_store_execute_graph_error_out(&mut self, event: &ExecuteRun) -> Result<(), ()> {
        if let Some(o) = event.graph_error_out {
            o.set(self.graph_error);
        }
        Ok(())
    }

    fn effect_store_graph_error_out_prefill_prompt_run(
        &mut self,
        event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        if let Some(o) = event.graph_error_out {
            o.set(self.graph_error);
        }
        Ok(())
    }

    fn effect_store_graph_error_out_prefill_voice_run(
        &mut self,
        event: &PrefillVoiceRun,
    ) -> Result<(), ()> {
        if let Some(o) = event.graph_error_out {
            o.set(self.graph_error);
        }
        Ok(())
    }

    fn effect_store_personaplex_prompt_complete_out(
        &mut self,
        event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        if let Some(o) = event.complete_out {
            o.set(self.last_complete);
        }
        Ok(())
    }

    fn effect_store_personaplex_prompt_remaining_out(
        &mut self,
        event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        if let Some(o) = event.remaining_frames_out {
            o.set(self.last_remaining);
        }
        Ok(())
    }

    fn effect_store_sample_graph_error_out(&mut self, event: &SampleRun) -> Result<(), ()> {
        if let Some(o) = event.graph_error_out {
            o.set(self.graph_error);
        }
        Ok(())
    }

    fn effect_store_voice_complete_out(&mut self, event: &PrefillVoiceRun) -> Result<(), ()> {
        if let Some(o) = event.complete_out {
            o.set(self.last_complete);
        }
        Ok(())
    }

    fn effect_store_voice_remaining_out(&mut self, event: &PrefillVoiceRun) -> Result<(), ()> {
        if let Some(o) = event.remaining_frames_out {
            o.set(self.last_remaining);
        }
        Ok(())
    }

    fn effect_write_and_build_personaplex_prompt_input(
        &mut self,
        _event: &PrefillPromptRun,
    ) -> Result<(), ()> {
        self.prompt_offset = self.prompt_offset.saturating_add(1);
        Ok(())
    }

    fn guard_bind_contract_invalid(&self, event: &InitRun) -> Result<bool, ()> {
        Ok(!event.model.valid(
            event.max_sequences,
            event.max_blocks,
            event.block_tokens,
            event.codebook_capacity,
            event.delay_cache_row_capacity,
        ))
    }

    fn guard_bind_contract_valid(&self, event: &InitRun) -> Result<bool, ()> {
        Ok(event.model.valid(
            event.max_sequences,
            event.max_blocks,
            event.block_tokens,
            event.codebook_capacity,
            event.delay_cache_row_capacity,
        ))
    }

    fn guard_capture_tokenizer_state_invalid(
        &self,
        event: &EventCaptureTokenizerState,
    ) -> Result<bool, ()> {
        Ok(!self.initialized || event.cache_out.is_none() || event.offset_out.is_none())
    }

    fn guard_capture_tokenizer_state_valid(
        &self,
        event: &EventCaptureTokenizerState,
    ) -> Result<bool, ()> {
        Ok(self.initialized && event.cache_out.is_some() && event.offset_out.is_some())
    }

    fn guard_execute_request_invalid(&self, event: &ExecuteRun) -> Result<bool, ()> {
        Ok(!self.execute_valid(event))
    }

    fn guard_execute_request_valid(&self, event: &ExecuteRun) -> Result<bool, ()> {
        Ok(self.execute_valid(event))
    }

    fn guard_graph_initialize_failed(&self, _event: &InitRun) -> Result<bool, ()> {
        Ok(!self.graph_accepted || self.graph_error != PredictorError::None)
    }

    fn guard_graph_initialize_succeeded(&self, _event: &InitRun) -> Result<bool, ()> {
        Ok(self.graph_accepted && self.graph_error == PredictorError::None)
    }

    fn guard_graph_step_accepted_execute_run(&self, _event: &ExecuteRun) -> Result<bool, ()> {
        Ok(self.graph_accepted && self.graph_error == PredictorError::None)
    }

    fn guard_graph_step_accepted_personaplex_prompt_post_silence(
        &self,
        _event: &PrefillPromptRun,
    ) -> Result<bool, ()> {
        Ok(self.graph_accepted && self.graph_error == PredictorError::None)
    }

    fn guard_graph_step_accepted_personaplex_prompt_pre_silence(
        &self,
        _event: &PrefillPromptRun,
    ) -> Result<bool, ()> {
        Ok(self.graph_accepted && self.graph_error == PredictorError::None)
    }

    fn guard_graph_step_accepted_personaplex_prompt_text(
        &self,
        _event: &PrefillPromptRun,
    ) -> Result<bool, ()> {
        Ok(self.graph_accepted && self.graph_error == PredictorError::None)
    }

    fn guard_graph_step_accepted_prefill_voice_run(
        &self,
        _event: &PrefillVoiceRun,
    ) -> Result<bool, ()> {
        Ok(self.graph_accepted && self.graph_error == PredictorError::None)
    }

    fn guard_graph_step_accepted_sample_run(&self, _event: &SampleRun) -> Result<bool, ()> {
        Ok(self.graph_accepted && self.graph_error == PredictorError::None)
    }

    fn guard_graph_step_rejected_execute_run(&self, _event: &ExecuteRun) -> Result<bool, ()> {
        Ok(!self.graph_accepted || self.graph_error != PredictorError::None)
    }

    fn guard_graph_step_rejected_prefill_prompt_run(
        &self,
        _event: &PrefillPromptRun,
    ) -> Result<bool, ()> {
        Ok(!self.graph_accepted || self.graph_error != PredictorError::None)
    }

    fn guard_graph_step_rejected_prefill_voice_run(
        &self,
        _event: &PrefillVoiceRun,
    ) -> Result<bool, ()> {
        Ok(!self.graph_accepted || self.graph_error != PredictorError::None)
    }

    fn guard_graph_step_rejected_sample_run(&self, _event: &SampleRun) -> Result<bool, ()> {
        Ok(!self.graph_accepted || self.graph_error != PredictorError::None)
    }

    fn guard_has_done_callback_begin_prompt_run(&self, event: &BeginPromptRun) -> Result<bool, ()> {
        Ok(event.on_done.is_some())
    }

    fn guard_has_done_callback_init_run(&self, event: &InitRun) -> Result<bool, ()> {
        Ok(event.on_done.is_some())
    }

    fn guard_has_done_callback_load_voice_run(&self, event: &LoadVoiceRun) -> Result<bool, ()> {
        Ok(event.on_done.is_some())
    }

    fn guard_has_done_callback_prefill_prompt_run(
        &self,
        event: &PrefillPromptRun,
    ) -> Result<bool, ()> {
        Ok(event.on_done.is_some())
    }

    fn guard_has_done_callback_prefill_voice_run(
        &self,
        event: &PrefillVoiceRun,
    ) -> Result<bool, ()> {
        Ok(event.on_done.is_some())
    }

    fn guard_has_error_callback_begin_prompt_run(
        &self,
        event: &BeginPromptRun,
    ) -> Result<bool, ()> {
        Ok(event.on_error.is_some())
    }

    fn guard_has_error_callback_init_run(&self, event: &InitRun) -> Result<bool, ()> {
        Ok(event.on_error.is_some())
    }

    fn guard_has_error_callback_load_voice_run(&self, event: &LoadVoiceRun) -> Result<bool, ()> {
        Ok(event.on_error.is_some())
    }

    fn guard_has_error_callback_prefill_prompt_run(
        &self,
        event: &PrefillPromptRun,
    ) -> Result<bool, ()> {
        Ok(event.on_error.is_some())
    }

    fn guard_has_error_callback_prefill_voice_run(
        &self,
        event: &PrefillVoiceRun,
    ) -> Result<bool, ()> {
        Ok(event.on_error.is_some())
    }

    fn guard_has_error_out_begin_prompt_run(&self, event: &BeginPromptRun) -> Result<bool, ()> {
        Ok(event.error_out.is_some())
    }

    fn guard_has_error_out_execute_run(&self, event: &ExecuteRun) -> Result<bool, ()> {
        Ok(event.error_out.is_some())
    }

    fn guard_has_error_out_init_run(&self, event: &InitRun) -> Result<bool, ()> {
        Ok(event.error_out.is_some())
    }

    fn guard_has_error_out_load_voice_run(&self, event: &LoadVoiceRun) -> Result<bool, ()> {
        Ok(event.error_out.is_some())
    }

    fn guard_has_error_out_predict_run(&self, event: &PredictRun) -> Result<bool, ()> {
        Ok(event.error_out.is_some())
    }

    fn guard_has_error_out_prefill_prompt_run(&self, event: &PrefillPromptRun) -> Result<bool, ()> {
        Ok(event.error_out.is_some())
    }

    fn guard_has_error_out_prefill_voice_run(&self, event: &PrefillVoiceRun) -> Result<bool, ()> {
        Ok(event.error_out.is_some())
    }

    fn guard_has_error_out_sample_run(&self, event: &SampleRun) -> Result<bool, ()> {
        Ok(event.error_out.is_some())
    }

    fn guard_has_graph_error_out_execute_run(&self, event: &ExecuteRun) -> Result<bool, ()> {
        Ok(event.graph_error_out.is_some())
    }

    fn guard_has_graph_error_out_prefill_prompt_run(
        &self,
        event: &PrefillPromptRun,
    ) -> Result<bool, ()> {
        Ok(event.graph_error_out.is_some())
    }

    fn guard_has_graph_error_out_prefill_voice_run(
        &self,
        event: &PrefillVoiceRun,
    ) -> Result<bool, ()> {
        Ok(event.graph_error_out.is_some())
    }

    fn guard_has_graph_error_out_sample_run(&self, event: &SampleRun) -> Result<bool, ()> {
        Ok(event.graph_error_out.is_some())
    }

    fn guard_has_personaplex_prompt_complete_out(
        &self,
        _event: &PrefillPromptRun,
    ) -> Result<bool, ()> {
        Ok(self.prompt_remaining() == 0)
    }

    fn guard_has_personaplex_prompt_remaining_out(
        &self,
        event: &PrefillPromptRun,
    ) -> Result<bool, ()> {
        Ok(event.remaining_frames_out.is_some())
    }

    fn guard_has_voice_complete_out(&self, event: &PrefillVoiceRun) -> Result<bool, ()> {
        Ok(event.complete_out.is_some())
    }

    fn guard_has_voice_remaining_out(&self, event: &PrefillVoiceRun) -> Result<bool, ()> {
        Ok(event.remaining_frames_out.is_some())
    }

    fn guard_memory_accepted_init_run(&self, _event: &InitRun) -> Result<bool, ()> {
        Ok(self.memory_accepted && self.memory_error == PredictorError::None)
    }

    fn guard_memory_accepted_predict_run(&self, _event: &PredictRun) -> Result<bool, ()> {
        Ok(self.memory_accepted && self.memory_error == PredictorError::None)
    }

    fn guard_memory_accepted_prefill_prompt_run(
        &self,
        _event: &PrefillPromptRun,
    ) -> Result<bool, ()> {
        Ok(self.memory_accepted && self.memory_error == PredictorError::None)
    }

    fn guard_memory_accepted_prefill_voice_run(
        &self,
        _event: &PrefillVoiceRun,
    ) -> Result<bool, ()> {
        Ok(self.memory_accepted && self.memory_error == PredictorError::None)
    }

    fn guard_memory_rejected_init_run(&self, _event: &InitRun) -> Result<bool, ()> {
        Ok(!self.memory_accepted || self.memory_error != PredictorError::None)
    }

    fn guard_memory_rejected_predict_run(&self, _event: &PredictRun) -> Result<bool, ()> {
        Ok(!self.memory_accepted || self.memory_error != PredictorError::None)
    }

    fn guard_memory_rejected_prefill_prompt_run(
        &self,
        _event: &PrefillPromptRun,
    ) -> Result<bool, ()> {
        Ok(!self.memory_accepted || self.memory_error != PredictorError::None)
    }

    fn guard_memory_rejected_prefill_voice_run(
        &self,
        _event: &PrefillVoiceRun,
    ) -> Result<bool, ()> {
        Ok(!self.memory_accepted || self.memory_error != PredictorError::None)
    }

    fn guard_no_done_callback_begin_prompt_run(&self, event: &BeginPromptRun) -> Result<bool, ()> {
        Ok(event.on_done.is_none())
    }

    fn guard_no_done_callback_init_run(&self, event: &InitRun) -> Result<bool, ()> {
        Ok(event.on_done.is_none())
    }

    fn guard_no_done_callback_load_voice_run(&self, event: &LoadVoiceRun) -> Result<bool, ()> {
        Ok(event.on_done.is_none())
    }

    fn guard_no_done_callback_prefill_prompt_run(
        &self,
        event: &PrefillPromptRun,
    ) -> Result<bool, ()> {
        Ok(event.on_done.is_none())
    }

    fn guard_no_done_callback_prefill_voice_run(
        &self,
        event: &PrefillVoiceRun,
    ) -> Result<bool, ()> {
        Ok(event.on_done.is_none())
    }

    fn guard_no_error_callback_begin_prompt_run(&self, event: &BeginPromptRun) -> Result<bool, ()> {
        Ok(event.on_error.is_none())
    }

    fn guard_no_error_callback_init_run(&self, event: &InitRun) -> Result<bool, ()> {
        Ok(event.on_error.is_none())
    }

    fn guard_no_error_callback_load_voice_run(&self, event: &LoadVoiceRun) -> Result<bool, ()> {
        Ok(event.on_error.is_none())
    }

    fn guard_no_error_callback_prefill_prompt_run(
        &self,
        event: &PrefillPromptRun,
    ) -> Result<bool, ()> {
        Ok(event.on_error.is_none())
    }

    fn guard_no_error_callback_prefill_voice_run(
        &self,
        event: &PrefillVoiceRun,
    ) -> Result<bool, ()> {
        Ok(event.on_error.is_none())
    }

    fn guard_no_error_out_begin_prompt_run(&self, event: &BeginPromptRun) -> Result<bool, ()> {
        Ok(event.error_out.is_none())
    }

    fn guard_no_error_out_execute_run(&self, event: &ExecuteRun) -> Result<bool, ()> {
        Ok(event.error_out.is_none())
    }

    fn guard_no_error_out_init_run(&self, event: &InitRun) -> Result<bool, ()> {
        Ok(event.error_out.is_none())
    }

    fn guard_no_error_out_load_voice_run(&self, event: &LoadVoiceRun) -> Result<bool, ()> {
        Ok(event.error_out.is_none())
    }

    fn guard_no_error_out_predict_run(&self, event: &PredictRun) -> Result<bool, ()> {
        Ok(event.error_out.is_none())
    }

    fn guard_no_error_out_prefill_prompt_run(&self, event: &PrefillPromptRun) -> Result<bool, ()> {
        Ok(event.error_out.is_none())
    }

    fn guard_no_error_out_prefill_voice_run(&self, event: &PrefillVoiceRun) -> Result<bool, ()> {
        Ok(event.error_out.is_none())
    }

    fn guard_no_error_out_sample_run(&self, event: &SampleRun) -> Result<bool, ()> {
        Ok(event.error_out.is_none())
    }

    fn guard_no_graph_error_out_execute_run(&self, event: &ExecuteRun) -> Result<bool, ()> {
        Ok(event.graph_error_out.is_none())
    }

    fn guard_no_graph_error_out_prefill_prompt_run(
        &self,
        event: &PrefillPromptRun,
    ) -> Result<bool, ()> {
        Ok(event.graph_error_out.is_none())
    }

    fn guard_no_graph_error_out_prefill_voice_run(
        &self,
        event: &PrefillVoiceRun,
    ) -> Result<bool, ()> {
        Ok(event.graph_error_out.is_none())
    }

    fn guard_no_graph_error_out_sample_run(&self, event: &SampleRun) -> Result<bool, ()> {
        Ok(event.graph_error_out.is_none())
    }

    fn guard_no_personaplex_prompt_complete_out(
        &self,
        _event: &PrefillPromptRun,
    ) -> Result<bool, ()> {
        Ok(self.prompt_remaining() == 0)
    }

    fn guard_no_personaplex_prompt_remaining_out(
        &self,
        event: &PrefillPromptRun,
    ) -> Result<bool, ()> {
        Ok(event.remaining_frames_out.is_none())
    }

    fn guard_no_voice_complete_out(&self, event: &PrefillVoiceRun) -> Result<bool, ()> {
        Ok(event.complete_out.is_none())
    }

    fn guard_no_voice_remaining_out(&self, event: &PrefillVoiceRun) -> Result<bool, ()> {
        Ok(event.remaining_frames_out.is_none())
    }

    fn guard_personaplex_lmgen(&self, event: &InitRun) -> Result<bool, ()> {
        Ok(event.model.personaplex)
    }
    fn guard_standard_lmgen(&self, event: &InitRun) -> Result<bool, ()> {
        Ok(!event.model.personaplex)
    }

    fn guard_personaplex_prompt_begin_empty_valid(
        &self,
        event: &BeginPromptRun,
    ) -> Result<bool, ()> {
        Ok(self.prompt_begin_valid(event)
            && event.text_token_count == 0
            && event
                .pre_text_silence_frames
                .saturating_add(event.post_text_silence_frames)
                > 0)
    }

    fn guard_personaplex_prompt_begin_invalid(&self, event: &BeginPromptRun) -> Result<bool, ()> {
        Ok(!self.prompt_begin_valid(event))
    }

    fn guard_personaplex_prompt_begin_nonempty_valid(
        &self,
        event: &BeginPromptRun,
    ) -> Result<bool, ()> {
        Ok(self.prompt_begin_valid(event)
            && event
                .text_token_count
                .saturating_add(event.pre_text_silence_frames)
                .saturating_add(event.post_text_silence_frames)
                > 0)
    }

    fn guard_personaplex_prompt_complete(&self, _event: &PrefillPromptRun) -> Result<bool, ()> {
        Ok(self.prompt_remaining() == 0)
    }

    fn guard_personaplex_prompt_pending(&self, _event: &PrefillPromptRun) -> Result<bool, ()> {
        Ok(self.prompt_remaining() > 0)
    }

    fn guard_personaplex_prompt_phase_invalid(&self, event: &PrefillPromptRun) -> Result<bool, ()> {
        Ok(!self.prompt_phase_valid(event))
    }

    fn guard_personaplex_prompt_post_silence_pending(
        &self,
        _event: &PrefillPromptRun,
    ) -> Result<bool, ()> {
        Ok(self.pre_remaining == 0 && self.text_remaining == 0 && self.post_remaining > 0)
    }

    fn guard_personaplex_prompt_pre_silence_pending(
        &self,
        _event: &PrefillPromptRun,
    ) -> Result<bool, ()> {
        Ok(self.pre_remaining > 0)
    }

    fn guard_personaplex_prompt_prefill_request_invalid(
        &self,
        event: &PrefillPromptRun,
    ) -> Result<bool, ()> {
        Ok(!self.prompt_prefill_valid(event))
    }

    fn guard_personaplex_prompt_prefill_request_valid(
        &self,
        event: &PrefillPromptRun,
    ) -> Result<bool, ()> {
        Ok(self.prompt_prefill_valid(event))
    }

    fn guard_personaplex_prompt_text_pending(&self, event: &PrefillPromptRun) -> Result<bool, ()> {
        Ok(self.pre_remaining == 0
            && self.text_remaining > 0
            && (0..self.model.text_card).contains(&event.text_token))
    }

    fn guard_predict_blocked_by_voice_prompt(&self, _event: &PredictRun) -> Result<bool, ()> {
        Ok(self.predict_blocked())
    }

    fn guard_predict_request_shape_invalid(&self, event: &PredictRun) -> Result<bool, ()> {
        Ok(!self.guard_predict_valid(event) && !self.predict_blocked())
    }

    fn guard_predict_request_valid(&self, event: &PredictRun) -> Result<bool, ()> {
        Ok(self.guard_predict_valid(event))
    }

    fn guard_reset_graph_failed(&self, _event: &ResetRun) -> Result<bool, ()> {
        Ok(!self.graph_accepted || self.graph_error != PredictorError::None)
    }

    fn guard_reset_graph_succeeded(&self, _event: &ResetRun) -> Result<bool, ()> {
        Ok(self.graph_accepted && self.graph_error == PredictorError::None)
    }

    fn guard_reset_memory_failed(&self, _event: &ResetRun) -> Result<bool, ()> {
        Ok(!self.memory_accepted || self.memory_error != PredictorError::None)
    }

    fn guard_reset_memory_succeeded(&self, _event: &ResetRun) -> Result<bool, ()> {
        Ok(self.memory_accepted && self.memory_error == PredictorError::None)
    }

    fn guard_sample_request_invalid(&self, event: &SampleRun) -> Result<bool, ()> {
        Ok(!self.sample_valid(event))
    }

    fn guard_sample_request_valid(&self, event: &SampleRun) -> Result<bool, ()> {
        Ok(self.sample_valid(event))
    }

    fn guard_unexpected_error_out_absent(&self) -> Result<bool, ()> {
        Ok(!self.pending_error_out)
    }

    fn guard_unexpected_error_out_present(&self) -> Result<bool, ()> {
        Ok(self.pending_error_out)
    }

    fn guard_voice_contract_invalid(&self, event: &LoadVoiceRun) -> Result<bool, ()> {
        Ok(!self.initialized
            || !event
                .voice
                .valid(&self.model, self.lmgen_rows, self.codebook_count))
    }

    fn guard_voice_contract_valid(&self, event: &LoadVoiceRun) -> Result<bool, ()> {
        Ok(self.initialized
            && event
                .voice
                .valid(&self.model, self.lmgen_rows, self.codebook_count))
    }

    fn guard_voice_embedding_frame_bf16(&self, _event: &PrefillVoiceRun) -> Result<bool, ()> {
        Ok(self.voice.embedding_format == 3)
    }

    fn guard_voice_embedding_frame_f16(&self, _event: &PrefillVoiceRun) -> Result<bool, ()> {
        Ok(self.voice.embedding_format == 2)
    }

    fn guard_voice_embedding_frame_f32(&self, _event: &PrefillVoiceRun) -> Result<bool, ()> {
        Ok(self.voice.embedding_format == 1)
    }

    fn guard_voice_embedding_frame_failed(&self, _event: &PrefillVoiceRun) -> Result<bool, ()> {
        Ok(!self.embedding_frame_ok)
    }

    fn guard_voice_embedding_frame_loaded(&self, _event: &PrefillVoiceRun) -> Result<bool, ()> {
        Ok(self.embedding_frame_ok)
    }

    fn guard_voice_prefill_complete(&self, _event: &PrefillVoiceRun) -> Result<bool, ()> {
        Ok(self.voice_frame_index >= self.voice.prompt_frame_count)
    }

    fn guard_voice_prefill_pending(&self, _event: &PrefillVoiceRun) -> Result<bool, ()> {
        Ok(self.voice_frame_index < self.voice.prompt_frame_count)
    }

    fn guard_voice_prefill_request_invalid(&self, _event: &PrefillVoiceRun) -> Result<bool, ()> {
        Ok(!self.voice_prefill_valid())
    }

    fn guard_voice_prefill_request_valid(&self, _event: &PrefillVoiceRun) -> Result<bool, ()> {
        Ok(self.voice_prefill_valid())
    }
}

/// Public synchronous `Moshi` predictor actor backed by the generated state machine.
pub struct SpeechPredictorMoshiActor {
    machine: SpeechPredictorMoshiStateMachine<SpeechPredictorMoshiContext>,
}

impl fmt::Debug for SpeechPredictorMoshiActor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SpeechPredictorMoshiActor")
            .field("context", self.machine.context())
            .finish()
    }
}

impl Default for SpeechPredictorMoshiActor {
    fn default() -> Self {
        Self::new()
    }
}

impl SpeechPredictorMoshiActor {
    #[must_use]
    pub fn new() -> Self {
        Self {
            machine: SpeechPredictorMoshiStateMachine::new(SpeechPredictorMoshiContext::default()),
        }
    }

    pub fn process_event(
        &mut self,
        event: SpeechPredictorMoshiEvents,
    ) -> Result<(), PredictorError> {
        if self.machine.process_event(event).is_err() {
            return Err(PredictorError::UnexpectedEvent);
        }
        match self.machine.context().last_error {
            PredictorError::None => Ok(()),
            error => Err(error),
        }
    }

    pub fn process_init(&mut self, event: InitRun) -> Result<(), PredictorError> {
        self.process_event(SpeechPredictorMoshiEvents::InitRun(event))
    }

    pub fn process_load_voice(&mut self, event: LoadVoiceRun) -> Result<(), PredictorError> {
        self.process_event(SpeechPredictorMoshiEvents::LoadVoiceRun(event))
    }

    pub fn process_prefill_voice(&mut self, event: PrefillVoiceRun) -> Result<(), PredictorError> {
        self.process_event(SpeechPredictorMoshiEvents::PrefillVoiceRun(event))
    }

    pub fn process_begin_prompt(&mut self, event: BeginPromptRun) -> Result<(), PredictorError> {
        self.process_event(SpeechPredictorMoshiEvents::BeginPromptRun(event))
    }

    pub fn process_prefill_prompt(
        &mut self,
        event: PrefillPromptRun,
    ) -> Result<(), PredictorError> {
        self.process_event(SpeechPredictorMoshiEvents::PrefillPromptRun(event))
    }

    pub fn process_predict(&mut self, event: PredictRun) -> Result<(), PredictorError> {
        self.process_event(SpeechPredictorMoshiEvents::PredictRun(event))
    }

    pub fn process_execute(&mut self, event: ExecuteRun) -> Result<(), PredictorError> {
        self.process_event(SpeechPredictorMoshiEvents::ExecuteRun(event))
    }

    pub fn process_sample(&mut self, event: SampleRun) -> Result<(), PredictorError> {
        self.process_event(SpeechPredictorMoshiEvents::SampleRun(event))
    }

    pub fn process_capture_tokenizer_state(
        &mut self,
        event: EventCaptureTokenizerState,
    ) -> Result<(), PredictorError> {
        self.process_event(SpeechPredictorMoshiEvents::EventCaptureTokenizerState(
            event,
        ))
    }

    pub fn process_reset(&mut self, event: ResetRun) -> Result<(), PredictorError> {
        self.process_event(SpeechPredictorMoshiEvents::ResetRun(event))
    }

    #[must_use]
    pub fn state(&self) -> &SpeechPredictorMoshiStates {
        self.machine.state()
    }

    #[must_use]
    pub fn context(&self) -> &SpeechPredictorMoshiContext {
        self.machine.context()
    }

    #[must_use]
    pub fn is(&self, state: &SpeechPredictorMoshiStates) -> bool {
        self.machine.is(state)
    }
}
