//! Allocation-free, actor-owned public model catalog.
//!
//! The public boundary is [`Catalog`], [`event`], and opaque semantic values.
//! Generated machine state, raw records, name offsets, and index storage remain
//! private.
//!
//! ```compile_fail
//! use emel_model::catalog::storage::Record;
//! ```
//!
//! ```compile_fail
//! let catalog = emel_model::catalog::Catalog::try_new().unwrap();
//! let _ = catalog.state();
//! ```

mod actor;
pub mod event;
mod ingest;
mod name_query;
mod sm;
mod storage;

pub use actor::Catalog;

pub(crate) const MAX_TENSORS: usize = 65_536;
pub(crate) const MAX_NAME_BYTES: usize = 4 * 1024 * 1024;

use actor::{
    BindRuntime, DescribeModelRuntime, DescribeTensorRuntime, FindRuntime, ReleaseRuntime,
    ResetRuntime, SealRuntime, UnexpectedRuntime,
};

#[cfg(test)]
mod tests;
