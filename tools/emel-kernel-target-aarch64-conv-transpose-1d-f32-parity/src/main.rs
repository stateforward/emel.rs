//! Public target-router observer for dense AArch64 F32 conv-transpose-1d.

#![cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]

use emel_kernels::aarch64::{ConvTranspose1d, Kernel};
use emel_kernels::aarch64::{ConvTranspose1dF32Error, ConvTranspose1dF32Shape};

const COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const GUARDS_BLOB: &str = "c25714566ec9a02679daef85089544575123408e";
const ACTIONS_BLOB: &str = "267d4f74e6e7498155c8535920322ffef2c02fb6";
const SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";
const SENTINEL: f32 = f32::from_bits(0x7fc0_0001);

#[cfg(not(target_arch = "aarch64"))]
fn main() {
    eprintln!("AArch64 conv-transpose observer requires an AArch64 build");
    std::process::exit(2);
}

#[cfg(target_arch = "aarch64")]
fn bits(values: &[f32]) -> String {
    values
        .iter()
        .map(|value| format!("{:08x}", value.to_bits()))
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(target_arch = "aarch64")]
const fn shape(input_length: usize) -> ConvTranspose1dF32Shape {
    ConvTranspose1dF32Shape {
        kernel: 5,
        out_channels: 2,
        in_channels: 1,
        input_length,
        stride: 2,
        padding: 0,
        dilation: 1,
    }
}

#[cfg(target_arch = "aarch64")]
fn main() {
    let weights = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 0.5, 1.5, 2.5, 3.5, 4.5];
    let input = [1.0_f32, 2.0];
    let mut output = [0.0_f32; 14];
    println!("kernel-target-aarch64-conv-transpose-1d-f32-live/v1");
    println!("source_repository=stateforward/emel.cpp");
    println!("source_commit={COMMIT}");
    println!("source_kernel_aarch64_guards_blob={GUARDS_BLOB}");
    println!("source_kernel_aarch64_actions_blob={ACTIONS_BLOB}");
    println!("source_kernel_aarch64_sm_blob={SM_BLOB}");
    println!("target_arch=aarch64");
    println!(
        "event=op_conv_transpose_1d shape=kernel5_out2_in1_length2_stride2_padding0_dilation1"
    );
    println!("scope=dense_f32_weights_input_output");
    println!("execution=split_pinned_aarch64_sm_and_public_target_router");

    let mut actor = Kernel::try_new().expect("AArch64 target router");
    assert_eq!(
        actor.process_event(
            ConvTranspose1d::new(&weights, &input, shape(2)),
            &mut output,
        ),
        Ok(())
    );
    println!(
        "case=valid_dense_f32_neon status=ok output_bits={}",
        bits(&output)
    );

    let mut invalid_output = [SENTINEL; 14];
    assert_eq!(
        actor.process_event(
            ConvTranspose1d::new(&weights, &input, shape(3)),
            &mut invalid_output,
        ),
        Err(ConvTranspose1dF32Error::InvalidShape)
    );
    println!(
        "case=invalid_input_length status=reject error=InvalidShape output_bits={}",
        bits(&invalid_output)
    );

    let mut invalid_params_output = [SENTINEL; 14];
    let mut invalid_shape = shape(2);
    invalid_shape.padding = 1;
    assert_eq!(
        actor.process_event(
            ConvTranspose1d::new(&weights, &input, invalid_shape),
            &mut invalid_params_output,
        ),
        Err(ConvTranspose1dF32Error::InvalidShape)
    );
    println!(
        "case=invalid_padding status=reject error=InvalidShape output_bits={}",
        bits(&invalid_params_output)
    );
}
