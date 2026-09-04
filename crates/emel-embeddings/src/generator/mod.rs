//! Module for `generator` state machines.
pub mod sm;

/// OmniEmbed contract actor and bounded generation lifecycle.
///
/// Text preparation is owned by the synchronous conditioner child. Encoder
/// execution remains an explicit typed backend boundary until a maintained
/// model execution actor is available; image and audio therefore report
/// [`sm::EmbeddingsGeneratorStatus::Backend`] rather than succeeding silently.
pub mod omniembed;

/// Public generator request-event facade.
///
/// The event payloads are owned fixed-capacity values, so callers can build
/// requests without exposing the generated state-machine implementation.
pub mod event {
    pub use super::sm::{
        EventEmbedAudioRun, EventEmbedImageRun, EventEmbedTextRun, EventInitializeRun,
    };
}

/// Public actor and request contract for the maintained generator machine.
pub use sm::{
    AudioPrepareFn, AudioRouteKind, BenchmarkNowFn, BenchmarkStageTimings, CopiedMessage,
    EmbedDoneFn, EmbedErrorFn, EmbedPublishFn, EmbeddingsGeneratorActor,
    EmbeddingsGeneratorContext, EmbeddingsGeneratorStates, EmbeddingsGeneratorStatus, ErrorOutFn,
    EventEmbedAudioRun, EventEmbedImageRun, EventEmbedTextRun, EventInitializeRun, ImageEncodeFn,
    ImagePrepareFn, ImageRouteKind, InitDoneFn, InitErrorFn, InitializeBindFn,
    MAX_EMBEDDING_DIMENSION, MAX_MESSAGE_BYTES, MAX_MESSAGES, MAX_RGBA_BYTES, MAX_TOKEN_POSITIONS,
    TextEncodeFn, TextPrepareFn, TextPrepareResult, TextRouteKind,
};
