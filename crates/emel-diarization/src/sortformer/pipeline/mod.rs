//! Module for `sortformer/pipeline` state machines.
pub mod sm;
pub use sm::{CHUNK_LEN, SAMPLE_RATE, SPEAKER_COUNT};
pub use sm::{
    DiarizationSortformerPipeline, EventRunFlow, Pipeline, PipelineError, RunDone, RunError,
};
