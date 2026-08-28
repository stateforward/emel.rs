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
pub(crate) mod loader;
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
        MimiHParamsInput, MoshiComponent, TensorBinding, TensorInput, TensorMetadata,
        TensorMetadataInput, TensorView,
    };
}

/// An immutable model identifier supplied by model metadata.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ModelId(pub Box<str>);

#[cfg(test)]
mod port_inventory_tests;
