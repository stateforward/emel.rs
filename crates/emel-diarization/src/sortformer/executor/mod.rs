//! Module for `sortformer/executor` state machines and native kernels.
pub mod native_projection;
pub mod native_transformer;
pub mod sm;
pub use native_projection::{
    Binding as NativeProjectionBinding, Error as NativeProjectionError,
    Route as NativeProjectionRoute, Workspace as NativeProjectionWorkspace,
};
pub use native_transformer::{
    Binding as NativeTransformerBinding, Error as NativeTransformerError,
    Route as NativeTransformerRoute, Workspace as NativeTransformerWorkspace,
};
pub use sm::{Error, EventExecuteRun, ExecuteDone, ExecuteError, Executor};
pub use sm::{FRAME_COUNT, SPEAKER_COUNT};
