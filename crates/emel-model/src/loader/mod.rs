//! Model loader components.

#![allow(clippy::redundant_pub_crate)]
#![allow(clippy::enum_variant_names)]

pub(crate) mod actor;
pub(crate) mod event;
pub mod hparams;
pub mod sm;

#[allow(unused_imports)]
pub(crate) use actor::{ModelLoader, NoTensorLoader, TensorLoader};

#[cfg(test)]
pub mod test_gguf;
