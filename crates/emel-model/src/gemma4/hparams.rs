//! Source-exact Gemma4 hyperparameter decoding through the public GGUF actor.

use emel_gguf::Loader;

use crate::loader::hparams::{Accessor, Error};

use super::Parameters;

/// Loads the 21 metadata keys consumed by pinned Gemma4 `load_hparams`.
///
/// Scalar keys preserve the pinned source defaults when absent. The feed-forward
/// value accepts either a scalar or the first array element. The sliding-window
/// pattern is required and copied into bounded caller-owned storage.
///
/// # Errors
///
/// Returns a typed metadata-access error for malformed, wrong-kind, out-of-range,
/// missing required pattern, or over-capacity metadata.
pub fn load_hparams(loader: &mut Loader) -> Result<Parameters, Error> {
    let mut accessor = Accessor::new();
    let mut parameters = Parameters::default();

    load_shape_hparams(&mut accessor, loader, &mut parameters)?;
    load_attention_hparams(&mut accessor, loader, &mut parameters)?;
    load_rope_hparams(&mut accessor, loader, &mut parameters)?;
    parameters.sliding_window_pattern_count = accessor.copy_flag_array(
        loader,
        b"gemma4.attention.sliding_window_pattern",
        &mut parameters.sliding_window_pattern_flags,
    )?;

    parameters.embedding_length_out = parameters.embedding_length;
    parameters.full_attention_interval = super::FULL_ATTENTION_INTERVAL;
    parameters.tie_word_embeddings = true;
    parameters.rope_pair_x0_stride = 1;
    parameters.rope_pair_x1_stride = 1;
    parameters.rope_pair_x1_offset = 0;
    parameters.rope_pair_x1_half_rot_offset = 1;
    Ok(parameters)
}

fn load_shape_hparams(
    accessor: &mut Accessor,
    loader: &mut Loader,
    parameters: &mut Parameters,
) -> Result<(), Error> {
    accessor.assign_i32(
        loader,
        b"gemma4.context_length",
        &mut parameters.context_length,
    )?;
    accessor.assign_i32(
        loader,
        b"gemma4.embedding_length",
        &mut parameters.embedding_length,
    )?;
    accessor.assign_i32(
        loader,
        b"gemma4.embedding_length_per_layer_input",
        &mut parameters.embedding_length_per_layer_input,
    )?;
    accessor.assign_i32_or_first_array_value(
        loader,
        b"gemma4.feed_forward_length",
        &mut parameters.feed_forward_length,
    )?;
    accessor.assign_i32(loader, b"gemma4.block_count", &mut parameters.block_count)?;
    accessor.assign_i32(loader, b"gemma4.vocab_size", &mut parameters.vocab_size)?;
    Ok(())
}

fn load_attention_hparams(
    accessor: &mut Accessor,
    loader: &mut Loader,
    parameters: &mut Parameters,
) -> Result<(), Error> {
    accessor.assign_i32(
        loader,
        b"gemma4.attention.head_count",
        &mut parameters.attention_head_count,
    )?;
    accessor.assign_i32(
        loader,
        b"gemma4.attention.head_count_kv",
        &mut parameters.attention_head_count_kv,
    )?;
    accessor.assign_i32(
        loader,
        b"gemma4.attention.key_length",
        &mut parameters.attention_key_length,
    )?;
    accessor.assign_i32(
        loader,
        b"gemma4.attention.key_length_swa",
        &mut parameters.attention_key_length_swa,
    )?;
    accessor.assign_i32(
        loader,
        b"gemma4.attention.value_length",
        &mut parameters.attention_value_length,
    )?;
    accessor.assign_i32(
        loader,
        b"gemma4.attention.value_length_swa",
        &mut parameters.attention_value_length_swa,
    )?;
    accessor.assign_i32(
        loader,
        b"gemma4.attention.sliding_window",
        &mut parameters.attention_sliding_window,
    )?;
    accessor.assign_i32(
        loader,
        b"gemma4.attention.shared_kv_layers",
        &mut parameters.attention_shared_kv_layers,
    )?;
    accessor.assign_f32(
        loader,
        b"gemma4.attention.layer_norm_rms_epsilon",
        &mut parameters.attention_layer_norm_rms_epsilon,
    )?;
    accessor.assign_f32(
        loader,
        b"gemma4.final_logit_softcapping",
        &mut parameters.final_logit_softcapping,
    )?;
    Ok(())
}

fn load_rope_hparams(
    accessor: &mut Accessor,
    loader: &mut Loader,
    parameters: &mut Parameters,
) -> Result<(), Error> {
    accessor.assign_i32(
        loader,
        b"gemma4.rope.dimension_count",
        &mut parameters.rope_dimension_count,
    )?;
    accessor.assign_i32(
        loader,
        b"gemma4.rope.dimension_count_swa",
        &mut parameters.rope_dimension_count_swa,
    )?;
    accessor.assign_f32(
        loader,
        b"gemma4.rope.freq_base",
        &mut parameters.rope_freq_base,
    )?;
    accessor.assign_f32(
        loader,
        b"gemma4.rope.freq_base_swa",
        &mut parameters.rope_freq_base_swa,
    )?;
    Ok(())
}
