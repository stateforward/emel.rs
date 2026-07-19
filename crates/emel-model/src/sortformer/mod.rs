//! Sortformer streaming-diarization model contract actor.
//!
//! The actor consumes and retains one parsed public GGUF loader, then performs
//! an exact private RTC scan to own the four family summaries required by the
//! pinned model contract. No loader, catalog, scan event, or mutable state
//! crosses this boundary.

mod actor;
pub mod event;
mod hparams;
mod query;
mod sm;

pub use actor::Sortformer;
pub use event::{ContractDescriptor, Family, FamilyDescriptor, Parameters, Storage};
pub use hparams::{
    Error as HparamError, ErrorKind as HparamErrorKind, Field as HparamField, load_hparams,
};

pub const ARCHITECTURE_NAME: &[u8] = b"sortformer";
pub const SOURCE_FORMAT: &[u8] = b"nemo";
pub const TENSOR_NAME_SCHEME: &[u8] = b"compact_v1";
pub const OUTTYPE: &[u8] = b"f32";
pub const SAMPLE_RATE: i32 = 16_000;
pub const SPEAKER_COUNT: i32 = 4;
pub const FRAME_SHIFT_MS: i32 = 80;
pub const CHUNK_LEN: i32 = 188;
pub const CHUNK_RIGHT_CONTEXT: i32 = 1;
pub const FIFO_LEN: i32 = 0;
pub const SPKCACHE_UPDATE_PERIOD: i32 = 188;
pub const SPKCACHE_LEN: i32 = 188;

#[cfg(test)]
mod tests;
