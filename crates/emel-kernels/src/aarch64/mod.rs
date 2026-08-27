//! `AArch64` kernel.

#![cfg(target_arch = "aarch64")]

mod actor;
pub mod event;
mod sm;

mod binary;
mod broadcast;
mod conv_transpose_1d;
mod dup;
mod f16_matmul;
mod gemv;
mod mul_mat_f32;
mod power;
mod q2_k;
mod q3_k;
mod q4_0;
mod q4_1;
mod q4_k;
mod q4_packed;
mod q5_0;
mod q6;
mod q6_packed;
mod q6_prepared;
mod q8_0;
mod q8_0_packed;
mod unary;

pub use actor::Kernel;
pub use binary::*;
pub use broadcast::*;
pub use conv_transpose_1d::*;
pub use dup::*;
pub use gemv::*;
pub use mul_mat_f32::*;
pub use power::*;
pub use q2_k::*;
pub use q3_k::*;
pub use q4_0::*;
pub use q4_1::*;
pub use q4_k::*;
pub use q4_packed::*;
pub use q5_0::*;
pub use q6::*;
pub use q6_packed::*;
pub use q6_prepared::*;
pub use q8_0::*;
pub use q8_0_packed::*;
pub use sm::*;
pub use unary::*;
