//! Source-exact Qwen3 hyperparameter decoding through the public GGUF actor.

use emel_gguf::Loader;

use crate::loader::hparams::{Accessor, Error, ErrorKind, Operation};

use super::Parameters;

/// Inspects the ten metadata keys consumed by pinned Qwen3 `load_hparams`.
///
/// Missing keys preserve their source-compatible defaults. The source-fixed output
/// width, tied-embedding policy, `RoPE` dimension, and pair-layout facts are derived
/// after decoding; key and value attention lengths must both be positive.
///
/// # Errors
///
/// Returns a typed metadata-access error for malformed, wrong-kind, or out-of-range
/// present metadata, or when either attention length is not positive after decoding.
pub fn load_hparams(loader: &mut Loader) -> Result<Parameters, Error> {
    let mut accessor = Accessor::new();
    let mut parameters = Parameters::default();
    accessor.assign_i32(
        loader,
        b"qwen3.context_length",
        &mut parameters.context_length,
    )?;
    accessor.assign_i32(
        loader,
        b"qwen3.embedding_length",
        &mut parameters.embedding_length,
    )?;
    accessor.assign_i32(
        loader,
        b"qwen3.feed_forward_length",
        &mut parameters.feed_forward_length,
    )?;
    accessor.assign_i32(
        loader,
        b"qwen3.attention.head_count",
        &mut parameters.attention_head_count,
    )?;
    accessor.assign_i32(
        loader,
        b"qwen3.attention.head_count_kv",
        &mut parameters.attention_head_count_kv,
    )?;
    accessor.assign_i32(
        loader,
        b"qwen3.attention.key_length",
        &mut parameters.attention_key_length,
    )?;
    accessor.assign_i32(
        loader,
        b"qwen3.attention.value_length",
        &mut parameters.attention_value_length,
    )?;
    accessor.assign_i32(loader, b"qwen3.block_count", &mut parameters.block_count)?;
    accessor.assign_f32(
        loader,
        b"qwen3.attention.layer_norm_rms_epsilon",
        &mut parameters.attention_layer_norm_rms_epsilon,
    )?;
    accessor.assign_f32(
        loader,
        b"qwen3.rope.freq_base",
        &mut parameters.rope_freq_base,
    )?;

    if parameters.attention_key_length <= 0 || parameters.attention_value_length <= 0 {
        return Err(Error {
            operation: Operation::OptionalI32,
            kind: ErrorKind::Range,
        });
    }

    parameters.embedding_length_out = parameters.embedding_length;
    parameters.tie_word_embeddings = true;
    parameters.rope_pair_x0_stride = 1;
    parameters.rope_pair_x1_stride = 1;
    parameters.rope_pair_x1_offset = 0;
    parameters.rope_pair_x1_half_rot_offset = 1;
    if parameters.rope_dimension_count == 0 {
        parameters.rope_dimension_count = parameters.attention_key_length;
    }

    Ok(parameters)
}
