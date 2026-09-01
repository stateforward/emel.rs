//! Module for `generator` state machines.
pub mod sm;

/// OmniEmbed contract actor and bounded generation lifecycle.
///
/// Text preparation is owned by the synchronous conditioner child. Encoder
/// execution remains an explicit typed backend boundary until a maintained
/// model execution actor is available; image and audio therefore report
/// [`sm::EmbeddingsGeneratorStatus::Backend`] rather than succeeding silently.
pub mod omniembed;
