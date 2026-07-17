//! Private, ownership-safe model data schema.
//!
//! This is the non-routing schema from the pinned `emel.cpp/src/emel/model/data.hpp`.
//! Large fixed-capacity regions use exact-length boxed slices so construction is fallible and
//! never materializes the complete model on the stack.

// These layouts intentionally preserve the pinned schema's independent boolean fields and names.
#![allow(clippy::struct_excessive_bools, clippy::struct_field_names)]

use std::collections::TryReserveError;
use std::ops::{Deref, DerefMut};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Fixed<T, const N: usize>([T; N]);

impl<T: Copy + Default, const N: usize> Default for Fixed<T, N> {
    fn default() -> Self {
        Self([T::default(); N])
    }
}

impl<T, const N: usize> Deref for Fixed<T, N> {
    type Target = [T; N];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T, const N: usize> DerefMut for Fixed<T, N> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub const MAX_TENSORS: usize = 65_536;
pub const MAX_NAME_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_ARCHITECTURE_NAME: usize = 64;
pub const MAX_VOCAB_TOKENS: usize = 320_000;
pub const MAX_VOCAB_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_TOKENIZER_MODEL: usize = 64;
pub const MAX_TOKENIZER_PRE: usize = 64;
pub const MAX_MERGES: usize = 600_000;
pub const MAX_MERGE_BYTES: usize = 8 * 1024 * 1024;
pub const MAX_PRECOMPILED_CHARMAP_BYTES: usize = 1024 * 1024;
pub const MAX_SPLIT_FILES: usize = 128;
pub const MAX_METADATA_BLOB_BYTES: usize = 4 * 1024 * 1024;
pub const MAX_METADATA_STRINGS: usize = 4096;
pub const MAX_METADATA_LIST: usize = 256;
pub const MAX_METADATA_ENTITIES: usize = 64;
pub const MAX_METADATA_ARRAYS: usize = 512;
pub const MAX_MATRYOSHKA_DIMS: usize = 16;
pub const MAX_ROPE_DIMENSION_SECTIONS: usize = 32;
pub const MAX_XIELU_VALUES: usize = 256;
pub const MAX_CLIP_IMAGE_STATS: usize = 16;
pub const MAX_CLIP_LAYER_INDEXES: usize = 512;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum GenerationResidualRoute {
    #[default]
    Attention = 0,
    Shortconv = 1,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum GenerationAttentionQkNormRoute {
    #[default]
    None = 0,
    HeadwiseRms = 1,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum GenerationAttentionValueRoute {
    #[default]
    DedicatedValue = 0,
    SharedKeyValue = 1,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum GenerationAttentionVNormRoute {
    #[default]
    None = 0,
    Rms = 1,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum GenerationAttentionWindowRoute {
    #[default]
    FullContext = 0,
    SlidingWindow = 1,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum TokenizerModel {
    None = 0,
    Spm = 1,
    Bpe = 2,
    Wpm = 3,
    Ugm = 4,
    Rwkv = 5,
    Plamo2 = 6,
    #[default]
    Unknown = 7,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u16)]
pub enum TokenizerPre {
    #[default]
    Default = 0,
    Llama3,
    Jais2,
    Dbrx,
    Smaug,
    DeepseekLlm,
    DeepseekCoder,
    Deepseek3Llm,
    Youtu,
    Falcon,
    Mpt,
    Starcoder,
    Gpt2,
    Jais,
    Refact,
    CommandR,
    Qwen2,
    Qwen35,
    Stablelm2,
    Olmo,
    Poro,
    Chatglm4,
    Viking,
    Tekken,
    Smollm,
    Codeshell,
    Bloom,
    Gpt3Finnish,
    Exaone,
    Exaone4,
    ExaoneMoe,
    Chameleon,
    Minerva,
    Megrez,
    Gpt4o,
    TinyAya,
    Superbpe,
    Trillion,
    GraniteDocling,
    Bailingmoe,
    SeedCoder,
    Hunyuan,
    HunyuanDense,
    JoyaiLlm,
    KimiK2,
    Grok2,
    Afmoe,
    MinimaxM2,
    SolarOpen,
    Unknown,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TensorBinding {
    pub(crate) split_index: u16,
    pub(crate) offset: u64,
    pub(crate) length: u64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct TensorRecord {
    pub(crate) name_offset: u32,
    pub(crate) name_length: u32,
    pub(crate) r#type: i32,
    pub(crate) n_dims: i32,
    pub(crate) dims: [i64; 4],
    pub(crate) data_offset: u64,
    pub(crate) file_offset: u64,
    pub(crate) data_size: u64,
    pub(crate) data: Option<TensorBinding>,
    pub(crate) file_index: u16,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HParams {
    pub(crate) n_ctx: i32,
    pub(crate) n_embd: i32,
    pub(crate) n_embd_out: i32,
    pub(crate) n_ff: i32,
    pub(crate) n_head: i32,
    pub(crate) n_head_kv: i32,
    pub(crate) n_rot: i32,
    pub(crate) n_layer: i32,
    pub(crate) n_vocab: i32,
    pub(crate) n_features: i32,
    pub(crate) n_leading_dense_block: i32,
    pub(crate) n_expert_ff: i32,
    pub(crate) n_expert_shared_ff: i32,
    pub(crate) n_expert_chunk_ff: i32,
    pub(crate) n_expert: i32,
    pub(crate) n_expert_used: i32,
    pub(crate) n_expert_shared: i32,
    pub(crate) n_expert_group: i32,
    pub(crate) n_expert_group_used: i32,
    pub(crate) expert_weights_scale: f32,
    pub(crate) expert_weights_norm: bool,
    pub(crate) expert_gating_func: i32,
    pub(crate) expert_group_scale: f32,
    pub(crate) experts_per_group: i32,
    pub(crate) moe_every_n_layers: i32,
    pub(crate) tie_word_embeddings: bool,
    pub(crate) nextn_predict_layers: i32,
    pub(crate) n_deepstack_layers: i32,
    pub(crate) pooling_type: i32,
    pub(crate) logit_scale: f32,
    pub(crate) decoder_start_token_id: i32,
    pub(crate) decoder_block_count: i32,
    pub(crate) attn_logit_softcapping: f32,
    pub(crate) router_logit_softcapping: f32,
    pub(crate) final_logit_softcapping: f32,
    pub(crate) swin_norm: bool,
    pub(crate) rescale_every_n_layers: i32,
    pub(crate) time_mix_extra_dim: i32,
    pub(crate) time_decay_extra_dim: i32,
    pub(crate) residual_scale: f32,
    pub(crate) embedding_scale: f32,
    pub(crate) token_shift_count: i32,
    pub(crate) interleave_moe_layer_step: i32,
    pub(crate) full_attention_interval: i32,
    pub(crate) activation_sparsity_scale: f32,
    pub(crate) altup_active_idx: i32,
    pub(crate) altup_num_inputs: i32,
    pub(crate) embd_length_per_layer_input: i32,
    pub(crate) swiglu_clamp_exp_count: u32,
    pub(crate) swiglu_clamp_exp: Fixed<f32, MAX_XIELU_VALUES>,
    pub(crate) swiglu_clamp_shexp_count: u32,
    pub(crate) swiglu_clamp_shexp: Fixed<f32, MAX_XIELU_VALUES>,
    pub(crate) dense_2_feat_in: i32,
    pub(crate) dense_2_feat_out: i32,
    pub(crate) dense_3_feat_in: i32,
    pub(crate) dense_3_feat_out: i32,
    pub(crate) use_parallel_residual: bool,
    pub(crate) attention_max_alibi_bias: f32,
    pub(crate) attention_clamp_kqv: f32,
    pub(crate) attention_key_length: i32,
    pub(crate) attention_value_length: i32,
    pub(crate) attention_key_length_swa: i32,
    pub(crate) attention_value_length_swa: i32,
    pub(crate) attention_layer_norm_epsilon: f32,
    pub(crate) attention_layer_norm_rms_epsilon: f32,
    pub(crate) attention_group_norm_epsilon: f32,
    pub(crate) attention_group_norm_groups: i32,
    pub(crate) attention_causal: bool,
    pub(crate) attention_q_lora_rank: i32,
    pub(crate) attention_kv_lora_rank: i32,
    pub(crate) attention_decay_lora_rank: i32,
    pub(crate) attention_iclr_lora_rank: i32,
    pub(crate) attention_value_residual_mix_lora_rank: i32,
    pub(crate) attention_gate_lora_rank: i32,
    pub(crate) attention_relative_buckets_count: i32,
    pub(crate) attention_sliding_window: i32,
    pub(crate) attention_sliding_window_pattern: i32,
    pub(crate) attention_sliding_window_pattern_count: u32,
    pub(crate) attention_sliding_window_pattern_flags: Fixed<u8, MAX_METADATA_ARRAYS>,
    pub(crate) attention_layer_pattern_count: u32,
    pub(crate) attention_layer_pattern_flags: Fixed<u8, MAX_METADATA_ARRAYS>,
    pub(crate) attention_scale: f32,
    pub(crate) attention_output_scale: f32,
    pub(crate) attention_temperature_length: i32,
    pub(crate) attention_temperature_scale: f32,
    pub(crate) attention_key_length_mla: i32,
    pub(crate) attention_value_length_mla: i32,
    pub(crate) attention_indexer_head_count: i32,
    pub(crate) attention_indexer_key_length: i32,
    pub(crate) attention_indexer_top_k: i32,
    pub(crate) attention_shared_kv_layers: i32,
    pub(crate) rope_freq_base: f32,
    pub(crate) rope_freq_base_swa: f32,
    pub(crate) n_rot_swa: i32,
    pub(crate) rope_pair_x0_stride: i32,
    pub(crate) rope_pair_x1_stride: i32,
    pub(crate) rope_pair_x1_offset: i32,
    pub(crate) rope_pair_x1_half_rot_offset: i32,
    pub(crate) rope_scale_linear: f32,
    pub(crate) rope_dimension_sections_count: i32,
    pub(crate) rope_dimension_sections: [i32; MAX_ROPE_DIMENSION_SECTIONS],
    pub(crate) rope_scaling_factor: f32,
    pub(crate) rope_scaling_attn_factor: f32,
    pub(crate) rope_scaling_orig_ctx_len: i32,
    pub(crate) rope_scaling_finetuned: bool,
    pub(crate) rope_scaling_yarn_log_multiplier: f32,
    pub(crate) rope_scaling_yarn_ext_factor: f32,
    pub(crate) rope_scaling_yarn_attn_factor: f32,
    pub(crate) rope_scaling_yarn_beta_fast: f32,
    pub(crate) rope_scaling_yarn_beta_slow: f32,
    pub(crate) ssm_conv_kernel: i32,
    pub(crate) ssm_inner_size: i32,
    pub(crate) ssm_state_size: i32,
    pub(crate) ssm_time_step_rank: i32,
    pub(crate) ssm_group_count: i32,
    pub(crate) ssm_dt_b_c_rms: bool,
    pub(crate) kda_head_dim: i32,
    pub(crate) wkv_head_size: i32,
    pub(crate) posnet_embd: i32,
    pub(crate) posnet_block_count: i32,
    pub(crate) convnext_embd: i32,
    pub(crate) convnext_block_count: i32,
    pub(crate) shortconv_l_cache: i32,
    pub(crate) matryoshka_dimension_count: u32,
    pub(crate) matryoshka_dimensions: [i32; MAX_MATRYOSHKA_DIMS],
}

impl HParams {
    fn reset(&mut self) {
        *self = Self::default();
    }
}

impl Default for HParams {
    #[allow(clippy::too_many_lines)]
    fn default() -> Self {
        Self {
            n_ctx: Default::default(),
            n_embd: Default::default(),
            n_embd_out: Default::default(),
            n_ff: Default::default(),
            n_head: Default::default(),
            n_head_kv: Default::default(),
            n_rot: Default::default(),
            n_layer: Default::default(),
            n_vocab: Default::default(),
            n_features: Default::default(),
            n_leading_dense_block: Default::default(),
            n_expert_ff: Default::default(),
            n_expert_shared_ff: Default::default(),
            n_expert_chunk_ff: Default::default(),
            n_expert: Default::default(),
            n_expert_used: Default::default(),
            n_expert_shared: Default::default(),
            n_expert_group: Default::default(),
            n_expert_group_used: Default::default(),
            expert_weights_scale: Default::default(),
            expert_weights_norm: Default::default(),
            expert_gating_func: Default::default(),
            expert_group_scale: Default::default(),
            experts_per_group: Default::default(),
            moe_every_n_layers: Default::default(),
            tie_word_embeddings: Default::default(),
            nextn_predict_layers: Default::default(),
            n_deepstack_layers: Default::default(),
            pooling_type: Default::default(),
            logit_scale: Default::default(),
            decoder_start_token_id: -1,
            decoder_block_count: Default::default(),
            attn_logit_softcapping: Default::default(),
            router_logit_softcapping: Default::default(),
            final_logit_softcapping: Default::default(),
            swin_norm: Default::default(),
            rescale_every_n_layers: Default::default(),
            time_mix_extra_dim: Default::default(),
            time_decay_extra_dim: Default::default(),
            residual_scale: Default::default(),
            embedding_scale: Default::default(),
            token_shift_count: Default::default(),
            interleave_moe_layer_step: Default::default(),
            full_attention_interval: Default::default(),
            activation_sparsity_scale: Default::default(),
            altup_active_idx: -1,
            altup_num_inputs: Default::default(),
            embd_length_per_layer_input: Default::default(),
            swiglu_clamp_exp_count: Default::default(),
            swiglu_clamp_exp: Fixed::default(),
            swiglu_clamp_shexp_count: Default::default(),
            swiglu_clamp_shexp: Fixed::default(),
            dense_2_feat_in: Default::default(),
            dense_2_feat_out: Default::default(),
            dense_3_feat_in: Default::default(),
            dense_3_feat_out: Default::default(),
            use_parallel_residual: Default::default(),
            attention_max_alibi_bias: Default::default(),
            attention_clamp_kqv: Default::default(),
            attention_key_length: Default::default(),
            attention_value_length: Default::default(),
            attention_key_length_swa: Default::default(),
            attention_value_length_swa: Default::default(),
            attention_layer_norm_epsilon: Default::default(),
            attention_layer_norm_rms_epsilon: Default::default(),
            attention_group_norm_epsilon: Default::default(),
            attention_group_norm_groups: Default::default(),
            attention_causal: Default::default(),
            attention_q_lora_rank: Default::default(),
            attention_kv_lora_rank: Default::default(),
            attention_decay_lora_rank: Default::default(),
            attention_iclr_lora_rank: Default::default(),
            attention_value_residual_mix_lora_rank: Default::default(),
            attention_gate_lora_rank: Default::default(),
            attention_relative_buckets_count: Default::default(),
            attention_sliding_window: Default::default(),
            attention_sliding_window_pattern: Default::default(),
            attention_sliding_window_pattern_count: Default::default(),
            attention_sliding_window_pattern_flags: Fixed::default(),
            attention_layer_pattern_count: Default::default(),
            attention_layer_pattern_flags: Fixed::default(),
            attention_scale: Default::default(),
            attention_output_scale: Default::default(),
            attention_temperature_length: Default::default(),
            attention_temperature_scale: Default::default(),
            attention_key_length_mla: Default::default(),
            attention_value_length_mla: Default::default(),
            attention_indexer_head_count: Default::default(),
            attention_indexer_key_length: Default::default(),
            attention_indexer_top_k: Default::default(),
            attention_shared_kv_layers: Default::default(),
            rope_freq_base: Default::default(),
            rope_freq_base_swa: Default::default(),
            n_rot_swa: Default::default(),
            rope_pair_x0_stride: 2,
            rope_pair_x1_stride: 2,
            rope_pair_x1_offset: 1,
            rope_pair_x1_half_rot_offset: Default::default(),
            rope_scale_linear: Default::default(),
            rope_dimension_sections_count: Default::default(),
            rope_dimension_sections: Default::default(),
            rope_scaling_factor: Default::default(),
            rope_scaling_attn_factor: Default::default(),
            rope_scaling_orig_ctx_len: Default::default(),
            rope_scaling_finetuned: Default::default(),
            rope_scaling_yarn_log_multiplier: Default::default(),
            rope_scaling_yarn_ext_factor: Default::default(),
            rope_scaling_yarn_attn_factor: Default::default(),
            rope_scaling_yarn_beta_fast: Default::default(),
            rope_scaling_yarn_beta_slow: Default::default(),
            ssm_conv_kernel: Default::default(),
            ssm_inner_size: Default::default(),
            ssm_state_size: Default::default(),
            ssm_time_step_rank: Default::default(),
            ssm_group_count: Default::default(),
            ssm_dt_b_c_rms: Default::default(),
            kda_head_dim: Default::default(),
            wkv_head_size: Default::default(),
            posnet_embd: Default::default(),
            posnet_block_count: Default::default(),
            convnext_embd: Default::default(),
            convnext_block_count: Default::default(),
            shortconv_l_cache: Default::default(),
            matryoshka_dimension_count: Default::default(),
            matryoshka_dimensions: Default::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(u8)]
pub enum MoshiComponent {
    #[default]
    None = 0,
    Lm = 1,
    Mimi = 2,
    Voice = 3,
}

pub const MAX_MOSHI_DELAYS: usize = 64;
pub const MAX_INFERENCE_PROMPT_TOKENS: usize = MAX_MOSHI_DELAYS;
pub const MAX_DEPFORMER_WEIGHT_SCHEDULE: usize = MAX_MOSHI_DELAYS;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MoshiLmHParams {
    pub(crate) card: i32,
    pub(crate) n_q: i32,
    pub(crate) dep_q: i32,
    pub(crate) inference_dep_q: i32,
    pub(crate) text_card: i32,
    pub(crate) text_padding_id: i32,
    pub(crate) dim: i32,
    pub(crate) num_layers: i32,
    pub(crate) num_heads: i32,
    pub(crate) context: i32,
    pub(crate) max_period: i32,
    pub(crate) dim_feedforward: i32,
    pub(crate) depformer_dim: i32,
    pub(crate) depformer_num_heads: i32,
    pub(crate) depformer_num_layers: i32,
    pub(crate) depformer_dim_feedforward: i32,
    pub(crate) depformer_context: i32,
    pub(crate) depformer_max_period: i32,
    pub(crate) depformer_low_rank_embeddings: i32,
    pub(crate) extra_heads_num_heads: i32,
    pub(crate) inference_pre_text_silence_frames: i32,
    pub(crate) inference_post_text_silence_frames: i32,
    pub(crate) delay_count: u32,
    pub(crate) inference_prompt_token_count: u32,
    pub(crate) depformer_weight_schedule_count: u32,
    pub(crate) delays: Fixed<i32, MAX_MOSHI_DELAYS>,
    pub(crate) inference_prompt_tokens: Fixed<i32, MAX_INFERENCE_PROMPT_TOKENS>,
    pub(crate) depformer_weight_schedule: Fixed<i32, MAX_DEPFORMER_WEIGHT_SCHEDULE>,
    pub(crate) causal: bool,
    pub(crate) cross_attention: bool,
    pub(crate) demux_second_stream: bool,
    pub(crate) depformer_multi_linear: bool,
    pub(crate) depformer_weights_per_step: bool,
}

impl MoshiLmHParams {
    fn reset(&mut self) {
        *self = Self::default();
    }
}

impl Default for MoshiLmHParams {
    fn default() -> Self {
        Self {
            card: Default::default(),
            n_q: Default::default(),
            dep_q: Default::default(),
            inference_dep_q: Default::default(),
            text_card: Default::default(),
            text_padding_id: -1,
            dim: Default::default(),
            num_layers: Default::default(),
            num_heads: Default::default(),
            context: Default::default(),
            max_period: Default::default(),
            dim_feedforward: Default::default(),
            depformer_dim: Default::default(),
            depformer_num_heads: Default::default(),
            depformer_num_layers: Default::default(),
            depformer_dim_feedforward: Default::default(),
            depformer_context: Default::default(),
            depformer_max_period: Default::default(),
            depformer_low_rank_embeddings: Default::default(),
            extra_heads_num_heads: Default::default(),
            inference_pre_text_silence_frames: Default::default(),
            inference_post_text_silence_frames: Default::default(),
            delay_count: Default::default(),
            inference_prompt_token_count: Default::default(),
            depformer_weight_schedule_count: Default::default(),
            delays: Fixed::default(),
            inference_prompt_tokens: Fixed::default(),
            depformer_weight_schedule: Fixed::default(),
            causal: Default::default(),
            cross_attention: Default::default(),
            demux_second_stream: Default::default(),
            depformer_multi_linear: Default::default(),
            depformer_weights_per_step: Default::default(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MimiHParams {
    pub(crate) sample_rate: i32,
    pub(crate) frame_rate: f32,
    pub(crate) n_q: i32,
    pub(crate) card: i32,
    pub(crate) dim: i32,
    pub(crate) semantic_n_q: i32,
    pub(crate) codebook_dim: i32,
    pub(crate) transformer_num_layers: i32,
    pub(crate) transformer_num_heads: i32,
    pub(crate) transformer_context: i32,
    pub(crate) transformer_max_period: i32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct VocabEntry {
    pub(crate) text_offset: u32,
    pub(crate) text_length: u32,
    pub(crate) score: f32,
    pub(crate) r#type: i32,
}

pub const ATTR_FLAG_BYTES: usize = MAX_VOCAB_TOKENS.div_ceil(8);

#[derive(Clone, Debug, PartialEq)]
pub struct Vocab {
    pub(crate) n_tokens: u32,
    pub(crate) n_token_types: u32,
    pub(crate) token_bytes_used: u32,
    pub(crate) n_merges: u32,
    pub(crate) merge_bytes_used: u32,
    pub(crate) precompiled_charsmap_size: u32,
    pub(crate) tokenizer_model_name: [u8; MAX_TOKENIZER_MODEL],
    pub(crate) tokenizer_pre_name: [u8; MAX_TOKENIZER_PRE],
    pub(crate) token_storage: Box<[u8]>,
    pub(crate) merge_storage: Box<[u8]>,
    pub(crate) entries: Box<[VocabEntry]>,
    pub(crate) merge_offsets: Box<[u32]>,
    pub(crate) merge_lengths: Box<[u32]>,
    pub(crate) precompiled_charsmap: Box<[u8]>,
    pub(crate) lstrip_flags: Box<[u8]>,
    pub(crate) rstrip_flags: Box<[u8]>,
    pub(crate) tokenizer_model_id: TokenizerModel,
    pub(crate) tokenizer_pre_id: TokenizerPre,
    pub(crate) bos_id: i32,
    pub(crate) eos_id: i32,
    pub(crate) eot_id: i32,
    pub(crate) eom_id: i32,
    pub(crate) unk_id: i32,
    pub(crate) sep_id: i32,
    pub(crate) pad_id: i32,
    pub(crate) cls_id: i32,
    pub(crate) mask_id: i32,
    pub(crate) prefix_id: i32,
    pub(crate) suffix_id: i32,
    pub(crate) middle_id: i32,
    pub(crate) fim_pre_id: i32,
    pub(crate) fim_suf_id: i32,
    pub(crate) fim_mid_id: i32,
    pub(crate) fim_pad_id: i32,
    pub(crate) fim_rep_id: i32,
    pub(crate) fim_sep_id: i32,
    pub(crate) add_bos: bool,
    pub(crate) add_eos: bool,
    pub(crate) add_sep: bool,
    pub(crate) add_space_prefix: bool,
    pub(crate) remove_extra_whitespaces: bool,
    pub(crate) escape_whitespaces: bool,
    pub(crate) treat_whitespace_as_suffix: bool,
    pub(crate) ignore_merges: bool,
}

impl Vocab {
    fn try_new() -> Result<Self, TryReserveError> {
        Ok(Self {
            n_tokens: 0,
            n_token_types: 0,
            token_bytes_used: 0,
            n_merges: 0,
            merge_bytes_used: 0,
            precompiled_charsmap_size: 0,
            tokenizer_model_name: [0; MAX_TOKENIZER_MODEL],
            tokenizer_pre_name: [0; MAX_TOKENIZER_PRE],
            token_storage: try_boxed_slice(MAX_VOCAB_BYTES)?,
            merge_storage: try_boxed_slice(MAX_MERGE_BYTES)?,
            entries: try_boxed_slice(MAX_VOCAB_TOKENS)?,
            merge_offsets: try_boxed_slice(MAX_MERGES)?,
            merge_lengths: try_boxed_slice(MAX_MERGES)?,
            precompiled_charsmap: try_boxed_slice(MAX_PRECOMPILED_CHARMAP_BYTES)?,
            lstrip_flags: try_boxed_slice(ATTR_FLAG_BYTES)?,
            rstrip_flags: try_boxed_slice(ATTR_FLAG_BYTES)?,
            tokenizer_model_id: TokenizerModel::Unknown,
            tokenizer_pre_id: TokenizerPre::Default,
            bos_id: -1,
            eos_id: -1,
            eot_id: -1,
            eom_id: -1,
            unk_id: -1,
            sep_id: -1,
            pad_id: -1,
            cls_id: -1,
            mask_id: -1,
            prefix_id: -1,
            suffix_id: -1,
            middle_id: -1,
            fim_pre_id: -1,
            fim_suf_id: -1,
            fim_mid_id: -1,
            fim_pad_id: -1,
            fim_rep_id: -1,
            fim_sep_id: -1,
            add_bos: false,
            add_eos: false,
            add_sep: false,
            add_space_prefix: false,
            remove_extra_whitespaces: false,
            escape_whitespaces: true,
            treat_whitespace_as_suffix: false,
            ignore_merges: false,
        })
    }

    fn reset(&mut self) {
        self.n_tokens = 0;
        self.n_token_types = 0;
        self.token_bytes_used = 0;
        self.n_merges = 0;
        self.merge_bytes_used = 0;
        self.precompiled_charsmap_size = 0;
        self.tokenizer_model_name.fill(0);
        self.tokenizer_pre_name.fill(0);
        self.token_storage.fill(0);
        self.merge_storage.fill(0);
        self.entries.fill(VocabEntry::default());
        self.merge_offsets.fill(0);
        self.merge_lengths.fill(0);
        self.precompiled_charsmap.fill(0);
        self.lstrip_flags.fill(0);
        self.rstrip_flags.fill(0);
        self.tokenizer_model_id = TokenizerModel::Unknown;
        self.tokenizer_pre_id = TokenizerPre::Default;
        self.bos_id = -1;
        self.eos_id = -1;
        self.eot_id = -1;
        self.eom_id = -1;
        self.unk_id = -1;
        self.sep_id = -1;
        self.pad_id = -1;
        self.cls_id = -1;
        self.mask_id = -1;
        self.prefix_id = -1;
        self.suffix_id = -1;
        self.middle_id = -1;
        self.fim_pre_id = -1;
        self.fim_suf_id = -1;
        self.fim_mid_id = -1;
        self.fim_pad_id = -1;
        self.fim_rep_id = -1;
        self.fim_sep_id = -1;
        self.add_bos = false;
        self.add_eos = false;
        self.add_sep = false;
        self.add_space_prefix = false;
        self.remove_extra_whitespaces = false;
        self.escape_whitespaces = true;
        self.treat_whitespace_as_suffix = false;
        self.ignore_merges = false;
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MetadataStringView {
    pub(crate) offset: u32,
    pub(crate) length: u32,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct NamedEntity {
    pub(crate) name: MetadataStringView,
    pub(crate) author: MetadataStringView,
    pub(crate) version: MetadataStringView,
    pub(crate) organization: MetadataStringView,
    pub(crate) description: MetadataStringView,
    pub(crate) url: MetadataStringView,
    pub(crate) doi: MetadataStringView,
    pub(crate) uuid: MetadataStringView,
    pub(crate) repo_url: MetadataStringView,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GeneralMetadata {
    pub(crate) r#type: MetadataStringView,
    pub(crate) quantization_version: i32,
    pub(crate) alignment: i32,
    pub(crate) file_type: i32,
    pub(crate) name: MetadataStringView,
    pub(crate) author: MetadataStringView,
    pub(crate) version: MetadataStringView,
    pub(crate) organization: MetadataStringView,
    pub(crate) finetune: MetadataStringView,
    pub(crate) basename: MetadataStringView,
    pub(crate) description: MetadataStringView,
    pub(crate) quantized_by: MetadataStringView,
    pub(crate) size_label: MetadataStringView,
    pub(crate) license: MetadataStringView,
    pub(crate) license_name: MetadataStringView,
    pub(crate) license_link: MetadataStringView,
    pub(crate) url: MetadataStringView,
    pub(crate) doi: MetadataStringView,
    pub(crate) uuid: MetadataStringView,
    pub(crate) repo_url: MetadataStringView,
    pub(crate) source_url: MetadataStringView,
    pub(crate) source_doi: MetadataStringView,
    pub(crate) source_uuid: MetadataStringView,
    pub(crate) source_repo_url: MetadataStringView,
    pub(crate) source_hf_repo: MetadataStringView,
    pub(crate) base_model_count: u32,
    pub(crate) base_models: [NamedEntity; MAX_METADATA_ENTITIES],
    pub(crate) dataset_count: u32,
    pub(crate) datasets: [NamedEntity; MAX_METADATA_ENTITIES],
    pub(crate) tag_count: u32,
    pub(crate) tags: [MetadataStringView; MAX_METADATA_LIST],
    pub(crate) language_count: u32,
    pub(crate) languages: [MetadataStringView; MAX_METADATA_LIST],
}

impl Default for GeneralMetadata {
    fn default() -> Self {
        Self {
            r#type: MetadataStringView::default(),
            quantization_version: -1,
            alignment: 0,
            file_type: -1,
            name: MetadataStringView::default(),
            author: MetadataStringView::default(),
            version: MetadataStringView::default(),
            organization: MetadataStringView::default(),
            finetune: MetadataStringView::default(),
            basename: MetadataStringView::default(),
            description: MetadataStringView::default(),
            quantized_by: MetadataStringView::default(),
            size_label: MetadataStringView::default(),
            license: MetadataStringView::default(),
            license_name: MetadataStringView::default(),
            license_link: MetadataStringView::default(),
            url: MetadataStringView::default(),
            doi: MetadataStringView::default(),
            uuid: MetadataStringView::default(),
            repo_url: MetadataStringView::default(),
            source_url: MetadataStringView::default(),
            source_doi: MetadataStringView::default(),
            source_uuid: MetadataStringView::default(),
            source_repo_url: MetadataStringView::default(),
            source_hf_repo: MetadataStringView::default(),
            base_model_count: 0,
            base_models: [NamedEntity::default(); MAX_METADATA_ENTITIES],
            dataset_count: 0,
            datasets: [NamedEntity::default(); MAX_METADATA_ENTITIES],
            tag_count: 0,
            tags: [MetadataStringView::default(); MAX_METADATA_LIST],
            language_count: 0,
            languages: [MetadataStringView::default(); MAX_METADATA_LIST],
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SamplingMetadata {
    pub(crate) sequence: MetadataStringView,
    pub(crate) top_k: i32,
    pub(crate) top_p: f32,
    pub(crate) min_p: f32,
    pub(crate) xtc_probability: f32,
    pub(crate) xtc_threshold: f32,
    pub(crate) temp: f32,
    pub(crate) penalty_last_n: i32,
    pub(crate) penalty_repeat: f32,
    pub(crate) mirostat: i32,
    pub(crate) mirostat_tau: f32,
    pub(crate) mirostat_eta: f32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TokenizerMetadata {
    pub(crate) hf_json: MetadataStringView,
    pub(crate) rwkv_world: MetadataStringView,
    pub(crate) chat_template: MetadataStringView,
    pub(crate) chat_template_count: u32,
    pub(crate) chat_template_names: [MetadataStringView; MAX_METADATA_LIST],
    pub(crate) chat_template_values: [MetadataStringView; MAX_METADATA_LIST],
}

impl Default for TokenizerMetadata {
    fn default() -> Self {
        Self {
            hf_json: MetadataStringView::default(),
            rwkv_world: MetadataStringView::default(),
            chat_template: MetadataStringView::default(),
            chat_template_count: 0,
            chat_template_names: [MetadataStringView::default(); MAX_METADATA_LIST],
            chat_template_values: [MetadataStringView::default(); MAX_METADATA_LIST],
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ClassifierMetadata {
    pub(crate) label_count: u32,
    pub(crate) labels: [MetadataStringView; MAX_METADATA_LIST],
}

impl Default for ClassifierMetadata {
    fn default() -> Self {
        Self {
            label_count: 0,
            labels: [MetadataStringView::default(); MAX_METADATA_LIST],
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct AdapterMetadata {
    pub(crate) r#type: MetadataStringView,
    pub(crate) lora_alpha: f32,
    pub(crate) lora_task_name: MetadataStringView,
    pub(crate) lora_prompt_prefix: MetadataStringView,
    pub(crate) alora_invocation_count: u32,
    pub(crate) alora_invocation_tokens: Fixed<u32, MAX_METADATA_ARRAYS>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ImatrixMetadata {
    pub(crate) chunk_count: i32,
    pub(crate) chunk_size: i32,
    pub(crate) dataset_count: u32,
    pub(crate) datasets: [MetadataStringView; MAX_METADATA_LIST],
}

impl Default for ImatrixMetadata {
    fn default() -> Self {
        Self {
            chunk_count: 0,
            chunk_size: 0,
            dataset_count: 0,
            datasets: [MetadataStringView::default(); MAX_METADATA_LIST],
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ClipMetadata {
    pub(crate) has_vision_encoder: bool,
    pub(crate) has_audio_encoder: bool,
    pub(crate) has_llava_projector: bool,
    pub(crate) use_gelu: bool,
    pub(crate) use_silu: bool,
    pub(crate) projector_type: MetadataStringView,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ClipVisionMetadata {
    pub(crate) encoder_name: MetadataStringView,
    pub(crate) projector_type: MetadataStringView,
    pub(crate) image_size: i32,
    pub(crate) image_min_pixels: i32,
    pub(crate) image_max_pixels: i32,
    pub(crate) preproc_image_size: i32,
    pub(crate) patch_size: i32,
    pub(crate) embedding_length: i32,
    pub(crate) feed_forward_length: i32,
    pub(crate) projection_dim: i32,
    pub(crate) block_count: i32,
    pub(crate) spatial_merge_size: i32,
    pub(crate) n_wa_pattern: i32,
    pub(crate) window_size: i32,
    pub(crate) attention_head_count: i32,
    pub(crate) attention_layer_norm_epsilon: f32,
    pub(crate) projector_scale_factor: i32,
    pub(crate) image_mean_count: u32,
    pub(crate) image_mean: [f32; MAX_CLIP_IMAGE_STATS],
    pub(crate) image_std_count: u32,
    pub(crate) image_std: [f32; MAX_CLIP_IMAGE_STATS],
    pub(crate) wa_layer_index_count: u32,
    pub(crate) wa_layer_indexes: Fixed<u32, MAX_CLIP_LAYER_INDEXES>,
    pub(crate) deepstack_layer_count: u32,
    pub(crate) deepstack_layers: Fixed<u8, MAX_CLIP_LAYER_INDEXES>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ClipAudioMetadata {
    pub(crate) encoder_name: MetadataStringView,
    pub(crate) projector_type: MetadataStringView,
    pub(crate) sample_rate: i32,
    pub(crate) n_fft: i32,
    pub(crate) win_length: i32,
    pub(crate) hop_size: i32,
    pub(crate) num_mel_bins: i32,
    pub(crate) embedding_length: i32,
    pub(crate) feed_forward_length: i32,
    pub(crate) projection_dim: i32,
    pub(crate) block_count: i32,
    pub(crate) attention_head_count: i32,
    pub(crate) attention_layer_norm_epsilon: f32,
    pub(crate) low_frequency: f32,
    pub(crate) high_frequency: f32,
    pub(crate) preemphasis_coefficient: f32,
    pub(crate) log_offset: f32,
    pub(crate) normalize_bias: f32,
    pub(crate) normalize_scale: f32,
    pub(crate) projector_stack_factor: i32,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RopeMetadata {
    pub(crate) scaling_type: MetadataStringView,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LlmStringsMetadata {
    pub(crate) tensor_data_layout: MetadataStringView,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct XieluMetadata {
    pub(crate) alpha_p_count: u32,
    pub(crate) alpha_p: Fixed<f32, MAX_XIELU_VALUES>,
    pub(crate) alpha_n_count: u32,
    pub(crate) alpha_n: Fixed<f32, MAX_XIELU_VALUES>,
    pub(crate) beta_count: u32,
    pub(crate) beta: Fixed<f32, MAX_XIELU_VALUES>,
    pub(crate) eps_count: u32,
    pub(crate) eps: Fixed<f32, MAX_XIELU_VALUES>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct DiffusionMetadata {
    pub(crate) shift_logits: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Metadata {
    pub(crate) general_data: GeneralMetadata,
    pub(crate) sampling_data: SamplingMetadata,
    pub(crate) tokenizer_data: TokenizerMetadata,
    pub(crate) classifier_data: ClassifierMetadata,
    pub(crate) adapter_data: AdapterMetadata,
    pub(crate) imatrix_data: ImatrixMetadata,
    pub(crate) clip_data: ClipMetadata,
    pub(crate) clip_vision_data: ClipVisionMetadata,
    pub(crate) clip_audio_data: ClipAudioMetadata,
    pub(crate) rope_data: RopeMetadata,
    pub(crate) llm_strings_data: LlmStringsMetadata,
    pub(crate) xielu_data: XieluMetadata,
    pub(crate) diffusion_data: DiffusionMetadata,
    pub(crate) blob_bytes_used: u32,
    pub(crate) blob: Box<[u8]>,
}

impl Metadata {
    fn try_new() -> Result<Self, TryReserveError> {
        Ok(Self {
            general_data: GeneralMetadata::default(),
            sampling_data: SamplingMetadata::default(),
            tokenizer_data: TokenizerMetadata::default(),
            classifier_data: ClassifierMetadata::default(),
            adapter_data: AdapterMetadata::default(),
            imatrix_data: ImatrixMetadata::default(),
            clip_data: ClipMetadata::default(),
            clip_vision_data: ClipVisionMetadata::default(),
            clip_audio_data: ClipAudioMetadata::default(),
            rope_data: RopeMetadata::default(),
            llm_strings_data: LlmStringsMetadata::default(),
            xielu_data: XieluMetadata::default(),
            diffusion_data: DiffusionMetadata::default(),
            blob_bytes_used: 0,
            blob: try_boxed_slice(MAX_METADATA_BLOB_BYTES)?,
        })
    }

    fn reset(&mut self) {
        self.general_data = GeneralMetadata::default();
        self.sampling_data = SamplingMetadata::default();
        self.tokenizer_data = TokenizerMetadata::default();
        self.classifier_data = ClassifierMetadata::default();
        self.adapter_data = AdapterMetadata::default();
        self.imatrix_data = ImatrixMetadata::default();
        self.clip_data = ClipMetadata::default();
        self.clip_vision_data = ClipVisionMetadata::default();
        self.clip_audio_data = ClipAudioMetadata::default();
        self.rope_data = RopeMetadata::default();
        self.llm_strings_data = LlmStringsMetadata::default();
        self.xielu_data = XieluMetadata::default();
        self.diffusion_data = DiffusionMetadata::default();
        self.blob_bytes_used = 0;
        self.blob.fill(0);
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct WeightsBinding;

#[derive(Clone, Debug, PartialEq)]
pub struct Data {
    pub(crate) n_layers: i32,
    pub(crate) n_tensors: u32,
    pub(crate) name_bytes_used: u32,
    pub(crate) architecture_name: [u8; MAX_ARCHITECTURE_NAME],
    pub(crate) name_storage: Box<[u8]>,
    pub(crate) tensors: Box<[TensorRecord]>,
    pub(crate) weights_data: Option<WeightsBinding>,
    pub(crate) weights_size: u64,
    pub(crate) weights_split_count: u16,
    pub(crate) weights_split_sizes: [u64; MAX_SPLIT_FILES],
    pub(crate) weights_split_offsets: [u64; MAX_SPLIT_FILES],
    pub(crate) params: HParams,
    pub(crate) vocab_data: Vocab,
    pub(crate) meta: Metadata,
    pub(crate) moshi_component_id: MoshiComponent,
    pub(crate) moshi_lm: MoshiLmHParams,
    pub(crate) mimi: MimiHParams,
}

impl Data {
    pub(crate) fn try_new() -> Result<Self, TryReserveError> {
        let mut params = HParams::default();
        params.reset();
        let mut moshi_lm = MoshiLmHParams::default();
        moshi_lm.reset();
        Ok(Self {
            n_layers: 0,
            n_tensors: 0,
            name_bytes_used: 0,
            architecture_name: [0; MAX_ARCHITECTURE_NAME],
            name_storage: try_boxed_slice(MAX_NAME_BYTES)?,
            tensors: try_boxed_slice(MAX_TENSORS)?,
            weights_data: None,
            weights_size: 0,
            weights_split_count: 1,
            weights_split_sizes: [0; MAX_SPLIT_FILES],
            weights_split_offsets: [0; MAX_SPLIT_FILES],
            params,
            vocab_data: Vocab::try_new()?,
            meta: Metadata::try_new()?,
            moshi_component_id: MoshiComponent::None,
            moshi_lm,
            mimi: MimiHParams::default(),
        })
    }

    pub(crate) fn reset(&mut self) {
        self.n_layers = 0;
        self.n_tensors = 0;
        self.name_bytes_used = 0;
        self.architecture_name.fill(0);
        self.name_storage.fill(0);
        self.tensors.fill(TensorRecord::default());
        self.weights_data = None;
        self.weights_size = 0;
        self.weights_split_count = 1;
        self.weights_split_sizes.fill(0);
        self.weights_split_offsets.fill(0);
        self.params.reset();
        self.vocab_data.reset();
        self.meta.reset();
        self.moshi_component_id = MoshiComponent::None;
        self.moshi_lm.reset();
        self.mimi = MimiHParams::default();
    }
}

pub fn tensor_name_view<'a>(model_data: &'a Data, tensor: &TensorRecord) -> &'a [u8] {
    checked_view(
        &model_data.name_storage,
        tensor.name_offset,
        tensor.name_length,
    )
}

pub fn metadata_string_view(metadata: &Metadata, value: MetadataStringView) -> &[u8] {
    if value.length == 0 {
        return &[];
    }
    let used = usize::try_from(metadata.blob_bytes_used)
        .unwrap_or(usize::MAX)
        .min(metadata.blob.len());
    checked_view(&metadata.blob[..used], value.offset, value.length)
}

pub fn try_parse_block_index(name: &[u8]) -> Option<i32> {
    let suffix = name.strip_prefix(b"blk.")?;
    let mut digits = suffix.iter().copied();
    let first = digits.next()?;
    if !first.is_ascii_digit() {
        return None;
    }
    let mut value = i32::from(first - b'0');
    for byte in digits {
        if byte == b'.' {
            return Some(value);
        }
        if !byte.is_ascii_digit() {
            return None;
        }
        value = value.checked_mul(10)?.checked_add(i32::from(byte - b'0'))?;
    }
    None
}

pub fn architecture_name_view(model_data: &Data) -> &[u8] {
    let length = model_data
        .architecture_name
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(model_data.architecture_name.len());
    &model_data.architecture_name[..length]
}

fn checked_view(storage: &[u8], offset: u32, length: u32) -> &[u8] {
    let begin = usize::try_from(offset).unwrap_or(usize::MAX);
    let length = usize::try_from(length).unwrap_or(usize::MAX);
    let Some(end) = begin.checked_add(length) else {
        return &[];
    };
    storage.get(begin..end).unwrap_or_default()
}

fn try_boxed_slice<T: Clone + Default>(length: usize) -> Result<Box<[T]>, TryReserveError> {
    let mut values = Vec::new();
    values.try_reserve_exact(length)?;
    values.resize(length, T::default());
    Ok(values.into_boxed_slice())
}

#[cfg(test)]
mod tests {
    use super::*;
    use allocation_counter::measure;
    use std::fmt::Write as _;
    use std::hint::black_box;
    use std::time::Instant;

    #[test]
    fn nested_defaults_match_source_sentinels_and_strides() {
        let params = HParams::default();
        assert_eq!(params.decoder_start_token_id, -1);
        assert_eq!(params.altup_active_idx, -1);
        assert_eq!(params.rope_pair_x0_stride, 2);
        assert_eq!(params.rope_pair_x1_stride, 2);
        assert_eq!(params.rope_pair_x1_offset, 1);
        assert_eq!(params.rope_pair_x1_half_rot_offset, 0);
        assert_eq!(MoshiLmHParams::default().text_padding_id, -1);
    }

    #[test]
    #[allow(clippy::cognitive_complexity)]
    #[allow(clippy::too_many_lines)]
    fn constants_and_enum_discriminants_match_the_pinned_schema() {
        assert_eq!(MAX_TENSORS, 65_536);
        assert_eq!(MAX_NAME_BYTES, 4_194_304);
        assert_eq!(MAX_ARCHITECTURE_NAME, 64);
        assert_eq!(MAX_VOCAB_TOKENS, 320_000);
        assert_eq!(MAX_VOCAB_BYTES, 8_388_608);
        assert_eq!(MAX_TOKENIZER_MODEL, 64);
        assert_eq!(MAX_TOKENIZER_PRE, 64);
        assert_eq!(MAX_MERGES, 600_000);
        assert_eq!(MAX_MERGE_BYTES, 8_388_608);
        assert_eq!(MAX_PRECOMPILED_CHARMAP_BYTES, 1_048_576);
        assert_eq!(MAX_SPLIT_FILES, 128);
        assert_eq!(MAX_METADATA_BLOB_BYTES, 4_194_304);
        assert_eq!(MAX_METADATA_STRINGS, 4096);
        assert_eq!(MAX_METADATA_LIST, 256);
        assert_eq!(MAX_METADATA_ENTITIES, 64);
        assert_eq!(MAX_METADATA_ARRAYS, 512);
        assert_eq!(MAX_MATRYOSHKA_DIMS, 16);
        assert_eq!(MAX_ROPE_DIMENSION_SECTIONS, 32);
        assert_eq!(MAX_XIELU_VALUES, 256);
        assert_eq!(MAX_CLIP_IMAGE_STATS, 16);
        assert_eq!(MAX_CLIP_LAYER_INDEXES, 512);
        assert_eq!(MAX_MOSHI_DELAYS, 64);
        assert_eq!(MAX_INFERENCE_PROMPT_TOKENS, 64);
        assert_eq!(MAX_DEPFORMER_WEIGHT_SCHEDULE, 64);
        assert_eq!(ATTR_FLAG_BYTES, 40_000);
        assert_eq!(
            [
                TokenizerModel::None as u8,
                TokenizerModel::Spm as u8,
                TokenizerModel::Bpe as u8,
                TokenizerModel::Wpm as u8,
                TokenizerModel::Ugm as u8,
                TokenizerModel::Rwkv as u8,
                TokenizerModel::Plamo2 as u8,
                TokenizerModel::Unknown as u8,
            ],
            [0, 1, 2, 3, 4, 5, 6, 7]
        );
        assert_eq!(
            [
                TokenizerPre::Default,
                TokenizerPre::Llama3,
                TokenizerPre::Jais2,
                TokenizerPre::Dbrx,
                TokenizerPre::Smaug,
                TokenizerPre::DeepseekLlm,
                TokenizerPre::DeepseekCoder,
                TokenizerPre::Deepseek3Llm,
                TokenizerPre::Youtu,
                TokenizerPre::Falcon,
                TokenizerPre::Mpt,
                TokenizerPre::Starcoder,
                TokenizerPre::Gpt2,
                TokenizerPre::Jais,
                TokenizerPre::Refact,
                TokenizerPre::CommandR,
                TokenizerPre::Qwen2,
                TokenizerPre::Qwen35,
                TokenizerPre::Stablelm2,
                TokenizerPre::Olmo,
                TokenizerPre::Poro,
                TokenizerPre::Chatglm4,
                TokenizerPre::Viking,
                TokenizerPre::Tekken,
                TokenizerPre::Smollm,
                TokenizerPre::Codeshell,
                TokenizerPre::Bloom,
                TokenizerPre::Gpt3Finnish,
                TokenizerPre::Exaone,
                TokenizerPre::Exaone4,
                TokenizerPre::ExaoneMoe,
                TokenizerPre::Chameleon,
                TokenizerPre::Minerva,
                TokenizerPre::Megrez,
                TokenizerPre::Gpt4o,
                TokenizerPre::TinyAya,
                TokenizerPre::Superbpe,
                TokenizerPre::Trillion,
                TokenizerPre::GraniteDocling,
                TokenizerPre::Bailingmoe,
                TokenizerPre::SeedCoder,
                TokenizerPre::Hunyuan,
                TokenizerPre::HunyuanDense,
                TokenizerPre::JoyaiLlm,
                TokenizerPre::KimiK2,
                TokenizerPre::Grok2,
                TokenizerPre::Afmoe,
                TokenizerPre::MinimaxM2,
                TokenizerPre::SolarOpen,
                TokenizerPre::Unknown,
            ]
            .map(|value| value as u16),
            std::array::from_fn::<_, 50, _>(|index| u16::try_from(index).unwrap())
        );
        assert_eq!(
            [
                MoshiComponent::None as u8,
                MoshiComponent::Lm as u8,
                MoshiComponent::Mimi as u8,
                MoshiComponent::Voice as u8,
            ],
            [0, 1, 2, 3]
        );
        assert_eq!(
            [
                GenerationResidualRoute::Attention as u8,
                GenerationResidualRoute::Shortconv as u8,
            ],
            [0, 1]
        );
        assert_eq!(
            [
                GenerationAttentionQkNormRoute::None as u8,
                GenerationAttentionQkNormRoute::HeadwiseRms as u8,
            ],
            [0, 1]
        );
        assert_eq!(
            [
                GenerationAttentionValueRoute::DedicatedValue as u8,
                GenerationAttentionValueRoute::SharedKeyValue as u8,
            ],
            [0, 1]
        );
        assert_eq!(
            [
                GenerationAttentionVNormRoute::None as u8,
                GenerationAttentionVNormRoute::Rms as u8,
            ],
            [0, 1]
        );
        assert_eq!(
            [
                GenerationAttentionWindowRoute::FullContext as u8,
                GenerationAttentionWindowRoute::SlidingWindow as u8,
            ],
            [0, 1]
        );
    }

    #[test]
    #[allow(clippy::cognitive_complexity)]
    #[allow(clippy::too_many_lines)]
    fn construction_has_exact_capacities_and_source_defaults() {
        let data = Data::try_new().unwrap();
        assert_eq!(data.name_storage.len(), MAX_NAME_BYTES);
        assert_eq!(data.tensors.len(), MAX_TENSORS);
        assert_eq!(data.vocab_data.token_storage.len(), MAX_VOCAB_BYTES);
        assert_eq!(data.vocab_data.merge_storage.len(), MAX_MERGE_BYTES);
        assert_eq!(data.vocab_data.entries.len(), MAX_VOCAB_TOKENS);
        assert_eq!(data.vocab_data.merge_offsets.len(), MAX_MERGES);
        assert_eq!(data.vocab_data.merge_lengths.len(), MAX_MERGES);
        assert_eq!(
            data.vocab_data.precompiled_charsmap.len(),
            MAX_PRECOMPILED_CHARMAP_BYTES
        );
        assert_eq!(data.vocab_data.lstrip_flags.len(), ATTR_FLAG_BYTES);
        assert_eq!(data.vocab_data.rstrip_flags.len(), ATTR_FLAG_BYTES);
        assert_eq!(data.meta.blob.len(), MAX_METADATA_BLOB_BYTES);
        assert_eq!(data.weights_split_count, 1);
        assert_eq!(data.params.decoder_start_token_id, -1);
        assert_eq!(data.params.altup_active_idx, -1);
        assert_eq!(data.params.rope_pair_x0_stride, 2);
        assert_eq!(data.params.rope_pair_x1_stride, 2);
        assert_eq!(data.params.rope_pair_x1_offset, 1);
        assert_eq!(data.vocab_data.tokenizer_model_id, TokenizerModel::Unknown);
        assert_eq!(data.vocab_data.tokenizer_pre_id, TokenizerPre::Default);
        let vocab_sentinel_ids = [
            data.vocab_data.bos_id,
            data.vocab_data.eos_id,
            data.vocab_data.eot_id,
            data.vocab_data.eom_id,
            data.vocab_data.unk_id,
            data.vocab_data.sep_id,
            data.vocab_data.pad_id,
            data.vocab_data.cls_id,
            data.vocab_data.mask_id,
            data.vocab_data.prefix_id,
            data.vocab_data.suffix_id,
            data.vocab_data.middle_id,
            data.vocab_data.fim_pre_id,
            data.vocab_data.fim_suf_id,
            data.vocab_data.fim_mid_id,
            data.vocab_data.fim_pad_id,
            data.vocab_data.fim_rep_id,
            data.vocab_data.fim_sep_id,
        ];
        assert_eq!(vocab_sentinel_ids, [-1; 18]);
        assert!(data.vocab_data.escape_whitespaces);
        assert_eq!(data.meta.general_data.quantization_version, -1);
        assert_eq!(data.meta.general_data.file_type, -1);
        assert_eq!(data.moshi_lm.text_padding_id, -1);
        assert_eq!(data.params.rope_pair_x1_half_rot_offset, 0);
        assert_eq!(data.moshi_component_id, MoshiComponent::None);
        assert_eq!(data.weights_data, None);
        assert!(data.name_storage.iter().all(|value| *value == 0));
        assert!(
            data.tensors
                .iter()
                .all(|value| *value == TensorRecord::default())
        );
        assert!(
            data.vocab_data
                .token_storage
                .iter()
                .all(|value| *value == 0)
        );
        assert!(
            data.vocab_data
                .merge_storage
                .iter()
                .all(|value| *value == 0)
        );
        assert!(
            data.vocab_data
                .entries
                .iter()
                .all(|value| *value == VocabEntry::default())
        );
        assert!(
            data.vocab_data
                .merge_offsets
                .iter()
                .all(|value| *value == 0)
        );
        assert!(
            data.vocab_data
                .merge_lengths
                .iter()
                .all(|value| *value == 0)
        );
        assert!(
            data.vocab_data
                .precompiled_charsmap
                .iter()
                .all(|value| *value == 0)
        );
        assert!(data.vocab_data.lstrip_flags.iter().all(|value| *value == 0));
        assert!(data.vocab_data.rstrip_flags.iter().all(|value| *value == 0));
        assert!(data.meta.blob.iter().all(|value| *value == 0));
        assert!(
            data.params
                .swiglu_clamp_exp
                .iter()
                .all(|value| *value == 0.0)
        );
        assert!(
            data.params
                .attention_sliding_window_pattern_flags
                .iter()
                .all(|value| *value == 0)
        );
        assert!(data.moshi_lm.delays.iter().all(|value| *value == 0));
        assert!(data.meta.xielu_data.eps.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn reset_restores_defaults_without_allocating() {
        let mut data = Data::try_new().unwrap();
        data.n_layers = 9;
        data.architecture_name[..5].copy_from_slice(b"llama");
        data.name_storage[..4].copy_from_slice(b"name");
        data.tensors[0].name_length = 4;
        data.params.decoder_start_token_id = 42;
        data.vocab_data.token_storage[0] = 9;
        data.vocab_data.escape_whitespaces = false;
        data.meta.blob[0] = 7;
        data.meta.general_data.file_type = 3;
        let allocation = measure(|| data.reset());
        assert_eq!(allocation.count_total, 0);
        assert_eq!(data.n_layers, 0);
        assert_eq!(data.architecture_name, [0; MAX_ARCHITECTURE_NAME]);
        assert_eq!(data.name_storage[0], 0);
        assert_eq!(data.tensors[0], TensorRecord::default());
        assert_eq!(data.params.decoder_start_token_id, -1);
        assert_eq!(data.vocab_data.token_storage[0], 0);
        assert!(data.vocab_data.escape_whitespaces);
        assert_eq!(data.meta.blob[0], 0);
        assert_eq!(data.meta.general_data.file_type, -1);
    }

    #[test]
    fn views_are_byte_oriented_checked_and_allocation_free() {
        let mut data = Data::try_new().unwrap();
        data.name_storage[..5].copy_from_slice(&[0xff, b'n', b'a', b'm', b'e']);
        let tensor = TensorRecord {
            name_offset: 0,
            name_length: 5,
            ..TensorRecord::default()
        };
        data.meta.blob[..4].copy_from_slice(&[0xfe, b'm', b'e', b't']);
        data.meta.blob_bytes_used = 4;
        let value = MetadataStringView {
            offset: 0,
            length: 4,
        };
        data.architecture_name[..5].copy_from_slice(b"moshi");
        let allocation = measure(|| {
            assert_eq!(
                tensor_name_view(&data, &tensor),
                &[0xff, b'n', b'a', b'm', b'e']
            );
            assert_eq!(
                metadata_string_view(&data.meta, value),
                &[0xfe, b'm', b'e', b't']
            );
            assert_eq!(architecture_name_view(&data), b"moshi");
            assert_eq!(try_parse_block_index(b"blk.17.attn"), Some(17));
        });
        assert_eq!(allocation.count_total, 0);
    }

    #[test]
    fn views_reject_out_of_bounds_and_empty_metadata_ranges() {
        let mut data = Data::try_new().unwrap();
        data.meta.blob_bytes_used = 4;
        assert!(
            tensor_name_view(
                &data,
                &TensorRecord {
                    name_offset: u32::MAX,
                    name_length: u32::MAX,
                    ..TensorRecord::default()
                }
            )
            .is_empty()
        );
        assert!(
            metadata_string_view(
                &data.meta,
                MetadataStringView {
                    offset: 3,
                    length: 2,
                }
            )
            .is_empty()
        );
        assert!(
            metadata_string_view(
                &data.meta,
                MetadataStringView {
                    offset: u32::MAX,
                    length: u32::MAX,
                }
            )
            .is_empty()
        );
        assert!(
            metadata_string_view(
                &data.meta,
                MetadataStringView {
                    offset: 0,
                    length: 0,
                }
            )
            .is_empty()
        );
        data.architecture_name.fill(b'x');
        assert_eq!(architecture_name_view(&data).len(), MAX_ARCHITECTURE_NAME);
    }

    #[test]
    fn block_index_parser_rejects_malformed_and_overflowing_names() {
        assert_eq!(try_parse_block_index(b"block.1.x"), None);
        assert_eq!(try_parse_block_index(b"blk."), None);
        assert_eq!(try_parse_block_index(b"blk.x.y"), None);
        assert_eq!(try_parse_block_index(b"blk.1"), None);
        assert_eq!(try_parse_block_index(b"blk.1x.y"), None);
        assert_eq!(try_parse_block_index(b"blk.2147483647.x"), Some(i32::MAX));
        assert_eq!(try_parse_block_index(b"blk.2147483648.x"), None);
    }

    #[test]
    fn fallible_storage_rejects_capacity_overflow() {
        assert!(try_boxed_slice::<u64>(usize::MAX).is_err());
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn parity_snapshot_cases_are_derived_from_behavior() {
        let mut data = Data::try_new().unwrap();
        data.name_storage[..5].copy_from_slice(&[0xff, b'n', b'a', b'm', b'e']);
        data.meta.blob[..4].copy_from_slice(&[0xfe, b'm', b'e', b't']);
        data.meta.blob_bytes_used = 4;
        data.architecture_name[..5].copy_from_slice(b"moshi");
        let reset_allocations = measure(|| data.reset()).count_total;
        data.name_storage[..5].copy_from_slice(&[0xff, b'n', b'a', b'm', b'e']);
        data.meta.blob[..4].copy_from_slice(&[0xfe, b'm', b'e', b't']);
        data.meta.blob_bytes_used = 4;
        data.architecture_name[..5].copy_from_slice(b"moshi");
        let observed = [
            format!(
                "case=tensor_name_valid offset=0 length=5 bytes={}",
                hex(tensor_name_view(
                    &data,
                    &TensorRecord {
                        name_length: 5,
                        ..TensorRecord::default()
                    }
                ))
            ),
            format!(
                "case=tensor_name_out_of_bounds offset=4294967295 length=4294967295 bytes={}",
                hex(tensor_name_view(
                    &data,
                    &TensorRecord {
                        name_offset: u32::MAX,
                        name_length: u32::MAX,
                        ..TensorRecord::default()
                    }
                ))
            ),
            format!(
                "case=metadata_valid offset=0 length=4 bytes={}",
                hex(metadata_string_view(
                    &data.meta,
                    MetadataStringView {
                        offset: 0,
                        length: 4,
                    }
                ))
            ),
            format!(
                "case=metadata_zero_length offset=0 length=0 bytes={}",
                hex(metadata_string_view(
                    &data.meta,
                    MetadataStringView {
                        offset: 0,
                        length: 0,
                    }
                ))
            ),
            format!(
                "case=metadata_used_bound offset=3 length=2 used=4 bytes={}",
                hex(metadata_string_view(
                    &data.meta,
                    MetadataStringView {
                        offset: 3,
                        length: 2,
                    }
                ))
            ),
            format!(
                "case=architecture_nul_terminated bytes={}",
                hex(architecture_name_view(&data))
            ),
            {
                data.architecture_name.fill(b'x');
                format!(
                    "case=architecture_full_capacity length={}",
                    architecture_name_view(&data).len()
                )
            },
            format!(
                "case=block_index_valid input=blk.17.attn output={}",
                try_parse_block_index(b"blk.17.attn").unwrap()
            ),
            format!(
                "case=block_index_i32_max input=blk.2147483647.x output={}",
                try_parse_block_index(b"blk.2147483647.x").unwrap()
            ),
            format!(
                "case=block_index_overflow input=blk.2147483648.x output={}",
                try_parse_block_index(b"blk.2147483648.x")
                    .map_or_else(|| "none".to_owned(), |value| value.to_string())
            ),
            format!("case=reset allocation_count={reset_allocations} defaults=restored"),
        ];
        let manifest = format!(
            concat!(
                "model-data-parity-snapshot/v1\n",
                "source_repository=stateforward/emel.cpp\n",
                "source_commit=843a117386ef17dc5a50549bbfc821074c2141d6\n",
                "source_model_tree=278b7b20545b630be33bee8bed0cb4c8db8990c3\n",
                "source_header_blob=78a25b987423d8cbef17965a8ca92596ffc0ecef\n",
                "source_implementation_blob=b33ace170b569d076844a36146c7ca86d4ffa7fc\n",
                "source_header_sha256=4b604d57fef1c22c8a9ebee36779a0c9cd4e7fe811856455d5cafd645498a151\n",
                "source_implementation_sha256=c8231f4feb2bc395a642bf3d66e74f5ee6ad243d706f277245b2b28c620c8816\n",
                "fixture_config=all_constants,enums,fields,defaults,byte_views,block_index_bounds,allocation_free_reset\n",
                "storage_contract=safe_exact_length_boxed_slices;fallible_construction;private_to_emel_model\n",
                "{}\n"
            ),
            observed.join("\n")
        );
        if let Some(output) = std::env::var_os("EMEL_MODEL_DATA_PARITY_OUTPUT") {
            std::fs::write(output, &manifest).unwrap();
        }
        if std::env::var_os("EMEL_MODEL_DATA_PARITY_UPDATE").is_none() {
            assert_eq!(
                manifest,
                include_str!("../../../snapshots/parity/model-data/manifest.txt")
            );
        }
    }

    fn hex(bytes: &[u8]) -> String {
        let mut output = String::with_capacity(bytes.len() * 2);
        for byte in bytes {
            write!(&mut output, "{byte:02x}").unwrap();
        }
        output
    }

    #[test]
    #[ignore = "run through scripts/bench.sh --suite=model-data"]
    fn benchmark_model_data_foundation() {
        const ACCESS_BATCH: usize = 100_000;
        let iterations = benchmark_parameter("EMEL_MODEL_DATA_BENCH_ITERATIONS", 5, false);
        let runs = benchmark_parameter("EMEL_MODEL_DATA_BENCH_RUNS", 5, false);
        let warmup_iterations =
            benchmark_parameter("EMEL_MODEL_DATA_BENCH_WARMUP_ITERATIONS", 1, true);

        let construction = median_ns_per_op(runs, iterations, warmup_iterations, 1, || {
            black_box(Data::try_new().unwrap());
        });
        let mut data = Data::try_new().unwrap();
        let reset = median_ns_per_op(runs, iterations, warmup_iterations, 1, || {
            data.name_storage[0] = 1;
            data.vocab_data.token_storage[0] = 1;
            data.meta.blob[0] = 1;
            data.reset();
            black_box(&data);
        });
        data.architecture_name[..5].copy_from_slice(b"moshi");
        data.name_storage[..8].copy_from_slice(b"blk.name");
        let tensor = TensorRecord {
            name_length: 8,
            ..TensorRecord::default()
        };
        let accessors = median_ns_per_op(runs, iterations, warmup_iterations, ACCESS_BATCH, || {
            black_box(tensor_name_view(&data, &tensor));
            black_box(architecture_name_view(&data));
            black_box(try_parse_block_index(b"blk.2147483647.attn"));
        });

        println!("# bench_host_arch: {}", std::env::consts::ARCH);
        println!("# bench_pointer_width: {}", usize::BITS);
        println!(
            "# benchmark_config: iterations={iterations} runs={runs} sample_policy=median warmup_iterations={warmup_iterations}"
        );
        println!("# source_repository: stateforward/emel.cpp");
        println!("# source_commit: 843a117386ef17dc5a50549bbfc821074c2141d6");
        println!("# source_header_blob: 78a25b987423d8cbef17965a8ca92596ffc0ecef");
        println!("# benchmark_fixture: complete fixed-capacity model data schema");
        println!("# benchmark_validation: safe construction; allocation-free reset and accessors");
        println!(
            "model/data/construction ns_per_op={construction:.3} iter={iterations} runs={runs} batch=1"
        );
        println!("model/data/reset ns_per_op={reset:.3} iter={iterations} runs={runs} batch=1");
        println!(
            "model/data/accessors ns_per_op={accessors:.3} iter={iterations} runs={runs} batch={ACCESS_BATCH}"
        );
    }

    fn benchmark_parameter(name: &str, default: usize, allow_zero: bool) -> usize {
        let value = std::env::var(name).map_or(default, |raw| {
            raw.parse::<usize>()
                .unwrap_or_else(|error| panic!("invalid {name}: {error}"))
        });
        assert!(allow_zero || value != 0, "{name} must be nonzero");
        value
    }

    fn median_ns_per_op(
        runs: usize,
        iterations: usize,
        warmup_iterations: usize,
        batch: usize,
        mut operation: impl FnMut(),
    ) -> f64 {
        for _ in 0..warmup_iterations {
            operation();
        }
        let mut samples = Vec::with_capacity(runs);
        for _ in 0..runs {
            let start = Instant::now();
            for _ in 0..iterations {
                for _ in 0..batch {
                    operation();
                }
            }
            let operation_count = u32::try_from(iterations.checked_mul(batch).unwrap()).unwrap();
            samples
                .push(start.elapsed().as_secs_f64() * 1_000_000_000.0 / f64::from(operation_count));
        }
        samples.sort_by(f64::total_cmp);
        samples[samples.len() / 2]
    }
}
