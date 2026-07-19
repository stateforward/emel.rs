//! `OmniEmbed` multimodal model-contract actor.
//!
//! The actor consumes and retains one parsed public GGUF loader, then performs
//! an exact private RTC scan to own the six tensor-family summaries required by
//! the pinned model contract. No loader, catalog, scan event, or mutable state
//! crosses this boundary.

mod actor;
pub mod event;
mod hparams;
mod query;
mod sm;

pub use actor::OmniEmbed;
pub use event::{ContractDescriptor, Family, FamilyDescriptor, Parameters, Storage};
pub use hparams::{
    Error as HparamError, ErrorKind as HparamErrorKind, Field as HparamField, load_hparams,
};

pub const ARCHITECTURE_NAME: &[u8] = b"omniembed";
pub const IMAGE_ENCODER_NAME: &[u8] = b"mobilenetv4_conv_medium.e180_r384_in12k";
pub const AUDIO_ENCODER_NAME: &[u8] = b"efficientat_mn20_as";
pub const IMAGE_SIZE: i32 = 384;
pub const IMAGE_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGE_STD: [f32; 3] = [0.229, 0.224, 0.225];
pub const AUDIO_SAMPLE_RATE: i32 = 32_000;
pub const AUDIO_N_FFT: i32 = 1_024;
pub const AUDIO_WIN_LENGTH: i32 = 800;
pub const AUDIO_HOP_SIZE: i32 = 320;
pub const AUDIO_NUM_MEL_BINS: i32 = 128;
pub const AUDIO_LOW_FREQUENCY: f32 = 0.0;
pub const AUDIO_HIGH_FREQUENCY: f32 = 15_000.0;
pub const AUDIO_PREEMPHASIS: f32 = 0.97;
pub const AUDIO_LOG_OFFSET: f32 = 1.0e-5;
pub const AUDIO_NORMALIZE_BIAS: f32 = 4.5;
pub const AUDIO_NORMALIZE_SCALE: f32 = 5.0;
pub const MAX_MATRYOSHKA_DIMENSIONS: usize = 16;

#[cfg(test)]
mod tests;
