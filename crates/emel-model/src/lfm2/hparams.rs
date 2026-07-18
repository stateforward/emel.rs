//! Source-exact LFM2 hyperparameter decoding through the public GGUF actor.

use emel_gguf::Loader;

use crate::loader::hparams::{Accessor, Error};

use super::Parameters;

/// Loads the ten metadata keys consumed by pinned LFM2 `load_hparams`.
///
/// The KV-head array is decoded twice exactly as the source does: its first
/// nonzero value becomes the shared KV-head count and every element becomes a
/// per-layer attention flag. The source-fixed derived embedding and RoPE-pair
/// facts are represented by the family contract rather than additional keys.
///
/// # Errors
///
/// Returns a typed metadata-access error for missing, malformed, wrong-kind,
/// out-of-range, or over-capacity required array data.
pub fn load_hparams(loader: &mut Loader) -> Result<Parameters, Error> {
    let mut accessor = Accessor::new();
    let mut parameters = Parameters::default();
    accessor.assign_i32(
        loader,
        b"lfm2.context_length",
        &mut parameters.context_length,
    )?;
    accessor.assign_i32(
        loader,
        b"lfm2.embedding_length",
        &mut parameters.embedding_length,
    )?;
    accessor.assign_i32(
        loader,
        b"lfm2.feed_forward_length",
        &mut parameters.feed_forward_length,
    )?;
    accessor.assign_i32(
        loader,
        b"lfm2.attention.head_count",
        &mut parameters.attention_head_count,
    )?;
    accessor.assign_i32(loader, b"lfm2.block_count", &mut parameters.block_count)?;
    accessor.assign_i32(loader, b"lfm2.vocab_size", &mut parameters.vocab_size)?;
    accessor.assign_i32(
        loader,
        b"lfm2.shortconv.l_cache",
        &mut parameters.shortconv_l_cache,
    )?;
    accessor.assign_f32(
        loader,
        b"lfm2.attention.layer_norm_rms_epsilon",
        &mut parameters.attention_layer_norm_rms_epsilon,
    )?;
    accessor.assign_f32(
        loader,
        b"lfm2.rope.freq_base",
        &mut parameters.rope_freq_base,
    )?;
    accessor.assign_first_nonzero_i32_from_array(
        loader,
        b"lfm2.attention.head_count_kv",
        &mut parameters.attention_head_count_kv,
    )?;
    parameters.attention_layer_pattern_count = accessor.copy_flag_array(
        loader,
        b"lfm2.attention.head_count_kv",
        &mut parameters.attention_layer_pattern_flags,
    )?;
    parameters.embedding_length_out = parameters.embedding_length;
    Ok(parameters)
}
