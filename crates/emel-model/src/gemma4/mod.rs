//! Gemma4 model contract from the pinned `emel.cpp` model domain.

mod actor;
pub mod event;
mod sm;

use crate::loader::hparams::{Accessor, Error};
use emel_gguf::Loader;

pub use actor::Gemma4;
pub use event::Parameters;

pub const ARCHITECTURE_NAME: &[u8] = b"gemma4";
pub const SLIDING_WINDOW_PATTERN_CAPACITY: usize = 35;
pub const GLOBAL_TENSOR_COUNT: u32 = 3;
pub const DEDICATED_KV_BLOCK_TENSOR_COUNT: u32 = 10;
pub const SHARED_KV_BLOCK_TENSOR_COUNT: u32 = 9;
pub const BLOCK_TENSOR_COUNT: u32 = DEDICATED_KV_BLOCK_TENSOR_COUNT;
pub const TOKEN_EMBEDDING_NAME: &[u8] = b"token_embd.weight";
pub const OUTPUT_NORM_NAME: &[u8] = b"output_norm.weight";

#[derive(Clone, Debug, PartialEq)]
pub struct MetadataParameters {
    pub context_length: i32,
    pub embedding_length: i32,
    pub embedding_length_per_layer_input: i32,
    pub feed_forward_length: i32,
    pub attention_head_count: i32,
    pub attention_head_count_kv: i32,
    pub attention_key_length: i32,
    pub attention_key_length_swa: i32,
    pub attention_value_length: i32,
    pub attention_value_length_swa: i32,
    pub block_count: i32,
    pub vocab_size: i32,
    pub attention_sliding_window: i32,
    pub attention_shared_kv_layers: i32,
    pub rope_dimension_count: i32,
    pub rope_dimension_count_swa: i32,
    pub attention_layer_norm_rms_epsilon: f32,
    pub final_logit_softcapping: f32,
    pub rope_freq_base: f32,
    pub rope_freq_base_swa: f32,
    pub sliding_window_pattern: [u8; SLIDING_WINDOW_PATTERN_CAPACITY],
    pub sliding_window_pattern_count: u32,
    pub full_attention_interval: i32,
    pub embedding_length_out: i32,
    pub tie_word_embeddings: bool,
    pub rope_pair_x0_stride: i32,
    pub rope_pair_x1_stride: i32,
    pub rope_pair_x1_offset: i32,
    pub rope_pair_x1_half_rot_offset: i32,
}

impl Default for MetadataParameters {
    fn default() -> Self {
        Self {
            context_length: 0,
            embedding_length: 0,
            embedding_length_per_layer_input: 0,
            feed_forward_length: 0,
            attention_head_count: 0,
            attention_head_count_kv: 0,
            attention_key_length: 0,
            attention_key_length_swa: 0,
            attention_value_length: 0,
            attention_value_length_swa: 0,
            block_count: 0,
            vocab_size: 0,
            attention_sliding_window: 0,
            attention_shared_kv_layers: 0,
            rope_dimension_count: 0,
            rope_dimension_count_swa: 0,
            attention_layer_norm_rms_epsilon: 0.0,
            final_logit_softcapping: 0.0,
            rope_freq_base: 0.0,
            rope_freq_base_swa: 0.0,
            sliding_window_pattern: [0; SLIDING_WINDOW_PATTERN_CAPACITY],
            sliding_window_pattern_count: 0,
            full_attention_interval: 0,
            embedding_length_out: 0,
            tie_word_embeddings: false,
            rope_pair_x0_stride: 0,
            rope_pair_x1_stride: 0,
            rope_pair_x1_offset: 0,
            rope_pair_x1_half_rot_offset: 0,
        }
    }
}

/// Loads the source-defined Gemma4 metadata and derived layout parameters.
///
/// # Errors
///
/// Returns a typed loader error when a required metadata field is missing,
/// malformed, out of range, or exceeds a bounded destination capacity.
pub fn load_hparams(loader: &mut Loader) -> Result<MetadataParameters, Error> {
    let mut accessor = Accessor::new();
    let mut p = MetadataParameters::default();
    accessor.assign_i32(loader, b"gemma4.context_length", &mut p.context_length)?;
    accessor.assign_i32(loader, b"gemma4.embedding_length", &mut p.embedding_length)?;
    accessor.assign_i32(
        loader,
        b"gemma4.embedding_length_per_layer_input",
        &mut p.embedding_length_per_layer_input,
    )?;
    accessor.assign_i32_or_first_array_value(
        loader,
        b"gemma4.feed_forward_length",
        &mut p.feed_forward_length,
    )?;
    accessor.assign_i32(
        loader,
        b"gemma4.attention.head_count",
        &mut p.attention_head_count,
    )?;
    accessor.assign_i32(
        loader,
        b"gemma4.attention.head_count_kv",
        &mut p.attention_head_count_kv,
    )?;
    accessor.assign_i32(
        loader,
        b"gemma4.attention.key_length",
        &mut p.attention_key_length,
    )?;
    accessor.assign_i32(
        loader,
        b"gemma4.attention.key_length_swa",
        &mut p.attention_key_length_swa,
    )?;
    accessor.assign_i32(
        loader,
        b"gemma4.attention.value_length",
        &mut p.attention_value_length,
    )?;
    accessor.assign_i32(
        loader,
        b"gemma4.attention.value_length_swa",
        &mut p.attention_value_length_swa,
    )?;
    accessor.assign_i32(loader, b"gemma4.block_count", &mut p.block_count)?;
    accessor.assign_i32(loader, b"gemma4.vocab_size", &mut p.vocab_size)?;
    accessor.assign_i32(
        loader,
        b"gemma4.attention.sliding_window",
        &mut p.attention_sliding_window,
    )?;
    accessor.assign_i32(
        loader,
        b"gemma4.attention.shared_kv_layers",
        &mut p.attention_shared_kv_layers,
    )?;
    accessor.assign_i32(
        loader,
        b"gemma4.rope.dimension_count",
        &mut p.rope_dimension_count,
    )?;
    accessor.assign_i32(
        loader,
        b"gemma4.rope.dimension_count_swa",
        &mut p.rope_dimension_count_swa,
    )?;
    accessor.assign_f32(
        loader,
        b"gemma4.attention.layer_norm_rms_epsilon",
        &mut p.attention_layer_norm_rms_epsilon,
    )?;
    accessor.assign_f32(
        loader,
        b"gemma4.final_logit_softcapping",
        &mut p.final_logit_softcapping,
    )?;
    accessor.assign_f32(loader, b"gemma4.rope.freq_base", &mut p.rope_freq_base)?;
    accessor.assign_f32(
        loader,
        b"gemma4.rope.freq_base_swa",
        &mut p.rope_freq_base_swa,
    )?;
    p.sliding_window_pattern_count = accessor.copy_flag_array(
        loader,
        b"gemma4.attention.sliding_window_pattern",
        &mut p.sliding_window_pattern,
    )?;
    p.full_attention_interval = 5;
    p.embedding_length_out = p.embedding_length;
    p.tie_word_embeddings = true;
    p.rope_pair_x0_stride = 1;
    p.rope_pair_x1_stride = 1;
    p.rope_pair_x1_half_rot_offset = 1;
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn defaults_preserve_source_shape() {
        assert_eq!(
            MetadataParameters::default().sliding_window_pattern.len(),
            35
        );
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod actor_tests;
