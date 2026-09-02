//! Moshi family metadata ownership boundary.

use emel_gguf::Loader;

use crate::data::{
    MoshiLmHParams, MoshiLmHParamsError, MoshiLmHParamsInput, MAX_DEPFORMER_WEIGHT_SCHEDULE,
    MAX_INFERENCE_PROMPT_TOKENS, MAX_MOSHI_DELAYS,
};
use crate::loader::hparams::{Accessor, Error as AccessorError};

/// Exact architecture name stored in model-owned Moshi metadata.
pub const ARCHITECTURE_NAME: &[u8] = b"moshi";

/// Typed failure while loading Moshi LM metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    /// The maintained Moshi actor is owned by `emel-speech`.
    OwnedBySpeechCrate,
    /// A required GGUF metadata value could not be decoded.
    Metadata(AccessorError),
    /// Decoded metadata violates the model-owned Moshi LM contract.
    Validation(MoshiLmHParamsError),
}

/// Loads the source-aligned Moshi LM metadata contract.
///
/// This decodes the required `general.architecture=moshi` and
/// `moshi.component=lm` discriminators, every field represented by
/// [`MoshiLmHParamsInput`], and bounded integer arrays before validating
/// through [`MoshiLmHParams::try_new`]. It does not bind tensors or execute a
/// native Moshi graph.
///
/// # Errors
///
/// Returns [`Error::Metadata`] for missing, malformed, wrong-kind, or
/// over-capacity metadata. Returns [`Error::Validation`] for semantic Moshi
/// contract violations.
pub fn load_hparams(loader: &mut Loader) -> Result<MoshiLmHParams, Error> {
    Accessor::require_string(loader, b"general.architecture", ARCHITECTURE_NAME)
        .map_err(Error::Metadata)?;
    Accessor::require_string(loader, b"moshi.component", b"lm").map_err(Error::Metadata)?;

    let mut input = MoshiLmHParamsInput {
        card: 0,
        n_q: 0,
        dep_q: 0,
        inference_dep_q: 0,
        text_card: 0,
        text_padding_id: 0,
        dim: 0,
        num_layers: 0,
        num_heads: 0,
        context: 0,
        max_period: 0,
        dim_feedforward: 0,
        depformer_dim: 0,
        depformer_num_heads: 0,
        depformer_num_layers: 0,
        depformer_dim_feedforward: 0,
        depformer_context: 0,
        depformer_max_period: 0,
        depformer_low_rank_embeddings: 0,
        extra_heads_num_heads: 0,
        inference_pre_text_silence_frames: 0,
        inference_post_text_silence_frames: 0,
        delay_count: 0,
        inference_prompt_token_count: 0,
        depformer_weight_schedule_count: 0,
        delays: [0; MAX_MOSHI_DELAYS],
        inference_prompt_tokens: [0; MAX_INFERENCE_PROMPT_TOKENS],
        depformer_weight_schedule: [0; MAX_DEPFORMER_WEIGHT_SCHEDULE],
        causal: false,
        cross_attention: false,
        demux_second_stream: false,
        depformer_multi_linear: false,
        depformer_weights_per_step: false,
    };

    require_i32(loader, b"moshi.lm.card", &mut input.card)?;
    require_i32(loader, b"moshi.lm.n_q", &mut input.n_q)?;
    require_i32(loader, b"moshi.lm.dep_q", &mut input.dep_q)?;
    require_i32(loader, b"moshi.lm.text_card", &mut input.text_card)?;
    require_i32(loader, b"moshi.lm.existing_text_padding_id", &mut input.text_padding_id)?;
    input.inference_dep_q = input.dep_q;
    require_i32(loader, b"moshi.lm.dim", &mut input.dim)?;
    require_i32(loader, b"moshi.lm.num_layers", &mut input.num_layers)?;
    require_i32(loader, b"moshi.lm.num_heads", &mut input.num_heads)?;
    require_i32(loader, b"moshi.lm.context", &mut input.context)?;
    require_i32(loader, b"moshi.lm.max_period", &mut input.max_period)?;
    require_i32(loader, b"moshi.lm.dim_feedforward", &mut input.dim_feedforward)?;
    require_bool(loader, b"moshi.lm.causal", &mut input.causal)?;
    require_bool(loader, b"moshi.lm.cross_attention", &mut input.cross_attention)?;
    require_bool(loader, b"moshi.lm.demux_second_stream", &mut input.demux_second_stream)?;
    input.delay_count = copy_required_array(loader, b"moshi.lm.delays", &mut input.delays)?;
    require_string(loader, b"moshi.lm.gating", b"silu")?;
    require_string(loader, b"moshi.lm.norm", b"rms_norm_f32")?;
    require_string(loader, b"moshi.lm.positional_embedding", b"rope")?;
    assign_i32(loader, b"moshi.lm.extra_heads.num_heads", &mut input.extra_heads_num_heads)?;

    require_i32(loader, b"moshi.lm.depformer.dim", &mut input.depformer_dim)?;
    require_i32(loader, b"moshi.lm.depformer.num_heads", &mut input.depformer_num_heads)?;
    require_i32(loader, b"moshi.lm.depformer.num_layers", &mut input.depformer_num_layers)?;
    require_i32(loader, b"moshi.lm.depformer.dim_feedforward", &mut input.depformer_dim_feedforward)?;
    require_i32(loader, b"moshi.lm.depformer.context", &mut input.depformer_context)?;
    require_i32(loader, b"moshi.lm.depformer.max_period", &mut input.depformer_max_period)?;
    require_bool(loader, b"moshi.lm.depformer.multi_linear", &mut input.depformer_multi_linear)?;
    require_bool(loader, b"moshi.lm.depformer.weights_per_step", &mut input.depformer_weights_per_step)?;
    require_string(loader, b"moshi.lm.depformer.gating", b"silu")?;
    require_string(loader, b"moshi.lm.depformer.pos_emb", b"none")?;
    assign_i32(loader, b"moshi.lm.depformer.low_rank_embeddings", &mut input.depformer_low_rank_embeddings)?;
    input.depformer_weight_schedule_count = copy_optional_array(
        loader,
        b"moshi.lm.depformer.weights_per_step_schedule",
        &mut input.depformer_weight_schedule,
    )?;

    if input.depformer_weights_per_step {
        require_i32(loader, b"moshi.lm.inference.dep_q", &mut input.inference_dep_q)?;
        require_i32(loader, b"moshi.lm.inference.pre_text_silence_frames", &mut input.inference_pre_text_silence_frames)?;
        require_i32(loader, b"moshi.lm.inference.post_text_silence_frames", &mut input.inference_post_text_silence_frames)?;
        input.inference_prompt_token_count = copy_required_array(
            loader,
            b"moshi.lm.inference.prompt_tokens",
            &mut input.inference_prompt_tokens,
        )?;
    }
    MoshiLmHParams::try_new(&input).map_err(Error::Validation)
}

fn require_i32(loader: &mut Loader, key: &[u8], field: &mut i32) -> Result<(), Error> {
    Accessor::require_i32(loader, key, field).map_err(Error::Metadata)
}

fn assign_i32(loader: &mut Loader, key: &[u8], field: &mut i32) -> Result<(), Error> {
    let mut accessor = Accessor::new();
    accessor.assign_i32(loader, key, field).map_err(Error::Metadata)
}

fn require_bool(loader: &mut Loader, key: &[u8], field: &mut bool) -> Result<(), Error> {
    Accessor::require_bool(loader, key, field).map_err(Error::Metadata)
}

fn require_string(loader: &mut Loader, key: &[u8], expected: &[u8]) -> Result<(), Error> {
    Accessor::require_string(loader, key, expected).map_err(Error::Metadata)
}

fn copy_required_array(loader: &mut Loader, key: &[u8], destination: &mut [i32]) -> Result<u32, Error> {
    Accessor::copy_i32_array(loader, key, destination).map_err(Error::Metadata)
}

fn copy_optional_array(loader: &mut Loader, key: &[u8], destination: &mut [i32]) -> Result<u32, Error> {
    Accessor::copy_optional_i32_array(loader, key, destination).map_err(Error::Metadata)
}

/// Returns whether a model architecture is exactly Moshi.
#[must_use]
pub fn is_execution_architecture(name: &[u8]) -> bool {
    name == ARCHITECTURE_NAME
}

#[derive(Debug, Default)]
pub struct Detail;

impl Detail {
    /// Attempts to bind family layers through the owning speech actor.
    ///
    /// # Errors
    ///
    /// Always returns [`Error::OwnedBySpeechCrate`] at this model-domain
    /// boundary; callers must dispatch the speech owner actor.
    pub const fn bind_layers(&self) -> Result<(), Error> {
        Err(Error::OwnedBySpeechCrate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::test_gguf::{Fixture, BOOL, I32, U32};

    fn array_i32(values: &[i32]) -> Vec<u8> {
        values.iter().flat_map(|value| (*value as u32).to_le_bytes()).collect()
    }

    fn valid_fixture() -> Vec<u8> {
        Fixture::new()
            .string(b"general.architecture", b"moshi")
            .string(b"moshi.component", b"lm")
            .scalar(b"moshi.lm.card", U32, 32u32.to_le_bytes())
            .scalar(b"moshi.lm.n_q", U32, 2u32.to_le_bytes())
            .scalar(b"moshi.lm.dep_q", U32, 2u32.to_le_bytes())
            .scalar(b"moshi.lm.text_card", U32, 16u32.to_le_bytes())
            .scalar(b"moshi.lm.existing_text_padding_id", U32, 0u32.to_le_bytes())
            .scalar(b"moshi.lm.dim", U32, 8u32.to_le_bytes())
            .scalar(b"moshi.lm.num_layers", U32, 2u32.to_le_bytes())
            .scalar(b"moshi.lm.num_heads", U32, 2u32.to_le_bytes())
            .scalar(b"moshi.lm.context", U32, 16u32.to_le_bytes())
            .scalar(b"moshi.lm.max_period", U32, 16u32.to_le_bytes())
            .scalar(b"moshi.lm.dim_feedforward", U32, 16u32.to_le_bytes())
            .scalar(b"moshi.lm.causal", BOOL, [1])
            .scalar(b"moshi.lm.cross_attention", BOOL, [0])
            .scalar(b"moshi.lm.demux_second_stream", BOOL, [0])
            .array(b"moshi.lm.delays", I32, &array_i32(&[0, 1, 2]), 3)
            .string(b"moshi.lm.gating", b"silu")
            .string(b"moshi.lm.norm", b"rms_norm_f32")
            .string(b"moshi.lm.positional_embedding", b"rope")
            .scalar(b"moshi.lm.extra_heads.num_heads", U32, 1u32.to_le_bytes())
            .scalar(b"moshi.lm.depformer.dim", U32, 8u32.to_le_bytes())
            .scalar(b"moshi.lm.depformer.num_heads", U32, 2u32.to_le_bytes())
            .scalar(b"moshi.lm.depformer.num_layers", U32, 2u32.to_le_bytes())
            .scalar(b"moshi.lm.depformer.dim_feedforward", U32, 16u32.to_le_bytes())
            .scalar(b"moshi.lm.depformer.context", U32, 16u32.to_le_bytes())
            .scalar(b"moshi.lm.depformer.max_period", U32, 16u32.to_le_bytes())
            .scalar(b"moshi.lm.depformer.multi_linear", BOOL, [1])
            .scalar(b"moshi.lm.depformer.weights_per_step", BOOL, [0])
            .string(b"moshi.lm.depformer.gating", b"silu")
            .string(b"moshi.lm.depformer.pos_emb", b"none")
            .scalar(b"moshi.lm.depformer.low_rank_embeddings", U32, 0u32.to_le_bytes())
            .build()
    }

    #[test]
    fn valid_metadata_loads_as_typed_hparams() {
        let mut loader = crate::loader::test_gguf::load(valid_fixture());
        let hparams = load_hparams(&mut loader).expect("valid Moshi metadata");
        assert_eq!(hparams.card(), 32);
        assert_eq!(hparams.n_q(), 2);
        assert_eq!(hparams.delay_count(), 3);
        assert_eq!(&hparams.delays()[..3], &[0, 1, 2]);
    }

    #[test]
    fn missing_or_wrong_component_is_rejected() {
        let mut missing = crate::loader::test_gguf::load(
            Fixture::new().string(b"general.architecture", b"moshi").build(),
        );
        assert!(matches!(load_hparams(&mut missing), Err(Error::Metadata(_))));
        let mut wrong = crate::loader::test_gguf::load(
            Fixture::new()
                .string(b"general.architecture", b"moshi")
                .string(b"moshi.component", b"voice")
                .build(),
        );
        assert!(matches!(load_hparams(&mut wrong), Err(Error::Metadata(_))));
    }

    #[test]
    fn wrong_kind_and_capacity_are_typed_metadata_errors() {
        let mut wrong_kind = crate::loader::test_gguf::load(
            Fixture::new()
                .string(b"general.architecture", b"moshi")
                .string(b"moshi.component", b"lm")
                .string(b"moshi.lm.card", b"wrong")
                .build(),
        );
        assert!(matches!(load_hparams(&mut wrong_kind), Err(Error::Metadata(_))));
        let values = vec![0u32; MAX_MOSHI_DELAYS + 1];
        let bytes: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
        let mut oversized = crate::loader::test_gguf::load(
            Fixture::new()
                .string(b"general.architecture", b"moshi")
                .string(b"moshi.component", b"lm")
                .scalar(b"moshi.lm.card", U32, 32u32.to_le_bytes())
                .scalar(b"moshi.lm.n_q", U32, 2u32.to_le_bytes())
                .array(b"moshi.lm.delays", U32, &bytes, (MAX_MOSHI_DELAYS + 1) as u64)
                .build(),
        );
        assert!(matches!(load_hparams(&mut oversized), Err(Error::Metadata(_))));
    }

    #[test]
    fn semantic_array_validation_is_preserved() {
        let mut input = MoshiLmHParamsInput {
            card: 1,
            n_q: 1,
            dep_q: 1,
            text_card: 1,
            text_padding_id: 0,
            inference_dep_q: 1,
            dim: 2,
            num_layers: 1,
            num_heads: 1,
            context: 1,
            max_period: 1,
            dim_feedforward: 1,
            depformer_dim: 2,
            depformer_num_heads: 1,
            depformer_num_layers: 1,
            depformer_dim_feedforward: 1,
            depformer_context: 1,
            depformer_max_period: 1,
            depformer_low_rank_embeddings: 0,
            extra_heads_num_heads: 0,
            inference_pre_text_silence_frames: 0,
            inference_post_text_silence_frames: 0,
            delay_count: 2,
            inference_prompt_token_count: 0,
            depformer_weight_schedule_count: 0,
            delays: [0; MAX_MOSHI_DELAYS],
            inference_prompt_tokens: [0; MAX_INFERENCE_PROMPT_TOKENS],
            depformer_weight_schedule: [0; MAX_DEPFORMER_WEIGHT_SCHEDULE],
            causal: false,
            cross_attention: false,
            demux_second_stream: false,
            depformer_multi_linear: false,
            depformer_weights_per_step: false,
        };
        input.delays[0] = -1;
        assert_eq!(MoshiLmHParams::try_new(&input), Err(MoshiLmHParamsError::InvalidDelays));
    }
}
