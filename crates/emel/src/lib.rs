//! Feature-gated public entry point for the emel inference stack.

#![forbid(unsafe_code)]
#![allow(dead_code, unused_crate_dependencies)]

pub use emel_core as core;
pub use emel_grammar as grammar;
pub use emel_io as io;
pub use emel_kernels as kernels;
pub use emel_tensor as tensor;

#[cfg(feature = "batch")]
pub use emel_batch as batch;
#[cfg(feature = "diarization")]
pub use emel_diarization as diarization;
#[cfg(feature = "embeddings")]
pub use emel_embeddings as embeddings;
#[cfg(feature = "gguf")]
pub use emel_gguf as gguf;
#[cfg(feature = "graph")]
pub use emel_graph as graph;
#[cfg(feature = "logits")]
pub use emel_logits as logits;
#[cfg(feature = "memory")]
pub use emel_memory as memory;
#[cfg(feature = "model")]
pub use emel_model as model;
#[cfg(feature = "speech")]
pub use emel_speech as speech;
#[cfg(feature = "text")]
pub use emel_text as text;
#[cfg(feature = "token")]
pub use emel_token as token;

/// Safe state-machine scheduling and RTC infrastructure.
pub mod sm;
