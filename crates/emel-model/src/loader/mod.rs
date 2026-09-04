//! Model loader components.

#![allow(clippy::redundant_pub_crate)]
#![allow(clippy::enum_variant_names)]

pub mod actor;
pub mod event;
pub mod hparams;
pub(crate) mod sm;

pub use actor::{ModelLoader, NoTensorLoader, OwnedTensorLoader, TensorLoader};
pub use event::{
    DoneCallback, Error, ErrorCallback, LoadError, LoadRequest, LoadStats, ModelCheck, ParseModel,
    Source,
};

#[cfg(test)]
pub mod test_gguf;
