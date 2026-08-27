//! Source-exact Llama hyperparameter decoding through the public GGUF actor.

use emel_gguf::Loader;

use crate::loader::hparams::{Accessor, Error};

use super::Parameters;

/// Loads the eighteen fields consumed by pinned Llama `load_hparams`.
///
/// Missing optional keys preserve their zero defaults, matching the source
/// loader. Wrong kinds, malformed values, and values outside the destination
/// type return the typed common-accessor error.
///
/// # Errors
///
/// Returns a typed metadata-access error for any present field that cannot be
/// decoded into its source destination type.
pub fn load_hparams(loader: &mut Loader) -> Result<Parameters, Error> {
    let mut accessor = Accessor::new();
    let mut parameters = Parameters::default();
    accessor.assign_i32(
        loader,
        b"llama.context_length",
        &mut parameters.context_length,
    )?;
    accessor.assign_i32(
        loader,
        b"llama.embedding_length",
        &mut parameters.embedding_length,
    )?;
    accessor.assign_i32(
        loader,
        b"llama.embedding_length_out",
        &mut parameters.embedding_length_out,
    )?;
    accessor.assign_i32(
        loader,
        b"llama.feed_forward_length",
        &mut parameters.feed_forward_length,
    )?;
    accessor.assign_i32(
        loader,
        b"llama.attention.head_count",
        &mut parameters.attention_head_count,
    )?;
    accessor.assign_i32(
        loader,
        b"llama.attention.head_count_kv",
        &mut parameters.attention_head_count_kv,
    )?;
    accessor.assign_i32(
        loader,
        b"llama.rope.dimension_count",
        &mut parameters.rope_dimension_count,
    )?;
    accessor.assign_i32(loader, b"llama.block_count", &mut parameters.block_count)?;
    accessor.assign_i32(loader, b"llama.vocab_size", &mut parameters.vocab_size)?;
    accessor.assign_f32(
        loader,
        b"llama.attention.layer_norm_epsilon",
        &mut parameters.attention_layer_norm_epsilon,
    )?;
    accessor.assign_f32(
        loader,
        b"llama.attention.layer_norm_rms_epsilon",
        &mut parameters.attention_layer_norm_rms_epsilon,
    )?;
    accessor.assign_f32(
        loader,
        b"llama.attention.clamp_kqv",
        &mut parameters.attention_clamp_kqv,
    )?;
    accessor.assign_f32(
        loader,
        b"llama.attn_logit_softcapping",
        &mut parameters.attn_logit_softcapping,
    )?;
    accessor.assign_f32(
        loader,
        b"llama.final_logit_softcapping",
        &mut parameters.final_logit_softcapping,
    )?;
    accessor.assign_f32(
        loader,
        b"llama.residual_scale",
        &mut parameters.residual_scale,
    )?;
    accessor.assign_f32(
        loader,
        b"llama.embedding_scale",
        &mut parameters.embedding_scale,
    )?;
    accessor.assign_f32(
        loader,
        b"llama.rope.freq_base",
        &mut parameters.rope_freq_base,
    )?;
    accessor.assign_f32(
        loader,
        b"llama.rope.freq_base_swa",
        &mut parameters.rope_freq_base_swa,
    )?;
    Ok(parameters)
}
