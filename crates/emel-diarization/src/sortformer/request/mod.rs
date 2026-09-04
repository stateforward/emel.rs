//! Module for `sortformer/request` state machines.
pub mod sm;
pub use sm::{CHUNK_LEN, CHUNK_RIGHT_CONTEXT, SAMPLE_RATE, SPEAKER_COUNT};
pub use sm::{
    DiarizationSortformerRequestContext, Error, EventPrepareRun, ExecutionContract, PrepareDone,
    PrepareError, Request, RequestError,
};
