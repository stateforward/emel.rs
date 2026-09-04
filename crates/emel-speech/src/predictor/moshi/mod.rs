//! Module for `predictor/moshi` state machines.
mod binding;
pub(crate) mod executor;
pub use binding::{
    MoshiTextEmbeddingBinding, MoshiTextEmbeddingBindingError, MoshiTextEmbeddingRow,
    TEXT_EMBEDDING_TENSOR,
};
pub mod sm;
pub use sm::{SpeechPredictorMoshiActor, SpeechPredictorMoshiContext, SpeechPredictorMoshiStates};
