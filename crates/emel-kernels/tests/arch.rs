#![allow(missing_docs)]

#[path = "arch/aarch64_sm.rs"]
mod aarch64_sm;
#[path = "arch/target_aarch64_binary_f32.rs"]
mod target_aarch64_binary_f32;
#[path = "arch/target_aarch64_broadcast_f32.rs"]
mod target_aarch64_broadcast_f32;
#[path = "arch/target_aarch64_conv_transpose_1d_f32.rs"]
mod target_aarch64_conv_transpose_1d_f32;
#[path = "arch/target_aarch64_dup_f32.rs"]
mod target_aarch64_dup_f32;
#[path = "arch/target_aarch64_kernel.rs"]
mod target_aarch64_kernel;
#[path = "arch/target_aarch64_power_f32.rs"]
mod target_aarch64_power_f32;
#[path = "arch/target_f16_matmul.rs"]
mod target_f16_matmul;
#[path = "arch/target_generic.rs"]
mod target_generic;
#[path = "arch/target_get_rows.rs"]
mod target_get_rows;
#[path = "arch/target_im2col.rs"]
mod target_im2col;
#[path = "arch/target_normalization.rs"]
mod target_normalization;
#[path = "arch/target_portable_bridge.rs"]
mod target_portable_bridge;
#[path = "arch/target_rope.rs"]
mod target_rope;
#[path = "arch/target_scalar_trig.rs"]
mod target_scalar_trig;
#[path = "arch/target_softmax.rs"]
mod target_softmax;
#[path = "arch/target_x86_broadcast_f32.rs"]
mod target_x86_broadcast_f32;
#[path = "arch/target_x86_conv_transpose.rs"]
mod target_x86_conv_transpose;
#[path = "arch/target_x86_kernel.rs"]
mod target_x86_kernel;
#[path = "arch/target_x86_matmul.rs"]
mod target_x86_matmul;
#[path = "arch/x86_sm_surface.rs"]
mod x86_sm_surface;
