//! `x86_64` kernel.

#![cfg(target_arch = "x86_64")]
#![allow(clippy::redundant_pub_crate)]

mod actor;
pub mod event;
mod sm;

mod binary;
mod broadcast;
mod conv_transpose_1d;
mod dup;
mod f16_matmul;
mod fma;
mod gemv;
mod matmul;
mod power;
mod unary;

pub use actor::{Kernel, X86Kernel};
pub use binary::*;
pub use broadcast::*;
pub use conv_transpose_1d::*;
pub use dup::*;
pub use fma::*;
pub use gemv::*;
pub use power::*;
pub use sm::*;
pub use unary::*;
