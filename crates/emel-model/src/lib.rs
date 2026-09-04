//! Model loading and reusable tensor lifecycle actors.
//!
//! The tensor actor is the production ownership boundary for model bytes.

#![forbid(unsafe_code)]
#![allow(missing_docs, dead_code)]

use emel_kernels as _;

pub mod architecture;
pub mod catalog;
pub(crate) mod data;
pub mod gemma4;
pub mod generation;
pub mod generation_audit;
pub mod lfm2;
pub mod llama;
pub mod loader;
pub mod moshi;
pub mod omniembed;
pub mod qwen3;
pub mod sortformer;
pub mod tensor;
pub mod vocabulary;
pub mod whisper;

/// Safe immutable ownership bridge for speech model preparation.
pub mod bridge {
    pub use crate::data::{
        Data, DataError, MimiBindingInput, MimiDataInput, MimiHParams, MimiHParamsError,
        MimiHParamsInput, MoshiComponent, MoshiLmBindingInput, MoshiLmDataInput, MoshiLmHParams,
        MoshiLmHParamsError, MoshiLmHParamsInput, MoshiVoiceBindingInput, MoshiVoiceDataInput,
        OmniEmbedBindingInput, OmniEmbedDataInput, TensorBinding, TensorInput, TensorMetadata,
        TensorMetadataInput, TensorView,
    };
    pub use crate::omniembed::{
        AudioPreprocessing, Encoder as OmniEmbedEncoder, Error as OmniEmbedError,
        Family as OmniEmbedFamily, HParams as OmniEmbedHParams,
        HParamsInput as OmniEmbedHParamsInput, TensorFamilies as OmniEmbedTensorFamilies,
        VisionPreprocessing,
    };
    pub use crate::whisper::{
        WhisperBindingInput, WhisperDataError, WhisperDataInput, WhisperFamily, WhisperFamilyView,
        WhisperHParams, WhisperHParamsError, WhisperHParamsInput,
    };
}

/// An immutable model identifier supplied by model metadata.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ModelId(pub Box<str>);

#[cfg(test)]
mod port_inventory_tests;
