//! Embedding-model components and embedding-generation orchestration.

#![forbid(unsafe_code)]
#![allow(missing_docs, dead_code)]

/// An embedding vector emitted by a model.
#[derive(Clone, Debug, PartialEq)]
pub struct Embedding(pub Box<[f32]>);
pub(crate) mod generator;

/// Public generator actor and typed request contracts.
pub use generator::{
    AudioPrepareFn, AudioRouteKind, BenchmarkNowFn, BenchmarkStageTimings, CopiedMessage,
    EmbedDoneFn, EmbedErrorFn, EmbedPublishFn, EmbeddingsGeneratorActor,
    EmbeddingsGeneratorContext, EmbeddingsGeneratorStates, EmbeddingsGeneratorStatus, ErrorOutFn,
    EventEmbedAudioRun, EventEmbedImageRun, EventEmbedTextRun, EventInitializeRun, ImageEncodeFn,
    ImagePrepareFn, ImageRouteKind, InitDoneFn, InitErrorFn, InitializeBindFn,
    MAX_EMBEDDING_DIMENSION, MAX_MESSAGE_BYTES, MAX_MESSAGES, MAX_RGBA_BYTES, MAX_TOKEN_POSITIONS,
    TextEncodeFn, TextPrepareFn, TextPrepareResult, TextRouteKind,
};

/// Request-event namespace matching the maintained generator facade.
pub mod event {
    pub use crate::generator::event::*;
}

/// Maintained `OmniEmbed` metadata and tensor-family validation actor.
pub use generator::omniembed;
