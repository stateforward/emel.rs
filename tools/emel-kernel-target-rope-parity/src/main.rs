#![cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]

use emel_kernels::any::rope::{
    I32View, OpRope, ROPE_MODE_NEOX, ROPE_MODE_NORM, ROPE_MODE_TIMESTEP, RopeError, RopeParams,
};
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};
use emel_kernels::aarch64::Kernel;

const COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const EVENTS_BLOB: &str = "4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9";
const DETAIL_BLOB: &str = "c8a82643eabfe8f2d7883e655955f455794511b0";
const GUARDS_BLOB: &str = "c25714566ec9a02679daef85089544575123408e";
const ACTIONS_BLOB: &str = "267d4f74e6e7498155c8535920322ffef2c02fb6";
const SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";
const SENTINEL: f32 = f32::from_bits(0x7fc0_0001);

#[cfg(not(target_arch = "aarch64"))]
fn main() {
    eprintln!("AArch64 target RoPE observer requires an AArch64 build");
    std::process::exit(2);
}

#[cfg(target_arch = "aarch64")]
const fn layout(dtype: DType, ne: [u64; 4]) -> Layout {
    Layout::new(
        dtype,
        ne,
        [4, 4 * ne[0], 4 * ne[0] * ne[1], 4 * ne[0] * ne[1] * ne[2]],
    )
}

#[cfg(target_arch = "aarch64")]
const fn params(mode: i32) -> RopeParams {
    RopeParams {
        n_dims: 4,
        mode,
        freq_base: 10_000.0,
        freq_scale: 1.0,
        ext_factor: 0.0,
        attn_factor: 1.0,
    }
}

#[cfg(target_arch = "aarch64")]
fn bits(values: &[f32]) -> String {
    values
        .iter()
        .map(|v| format!("{:08x}", v.to_bits()))
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(target_arch = "aarch64")]
fn run(
    actor: &mut Kernel,
    source: &[f32],
    positions: &[i32],
    ne: [u64; 4],
    mode: i32,
    output: &mut [f32],
) -> Result<(), RopeError> {
    actor.process_rope(OpRope::new(
        TensorView::new(source, layout(DType::F32, ne)),
        I32View::new(positions, layout(DType::I32, [ne[2], 1, 1, 1])),
        TensorViewMut::new(output, layout(DType::F32, ne)),
        params(mode),
    ))
}

#[cfg(target_arch = "aarch64")]
fn main() {
    println!("kernel-target-rope-live/v1");
    println!("source_repository=stateforward/emel.cpp");
    println!("source_commit={COMMIT}");
    println!("source_kernel_events_blob={EVENTS_BLOB}");
    println!("source_kernel_detail_blob={DETAIL_BLOB}");
    println!("source_kernel_aarch64_guards_blob={GUARDS_BLOB}");
    println!("source_kernel_aarch64_actions_blob={ACTIONS_BLOB}");
    println!("source_kernel_aarch64_sm_blob={SM_BLOB}");
    println!("target_arch=aarch64");
    println!(
        "scope=dense_contiguous_f32_target_router_rope_norm_neox_timestep_and_typed_rejection"
    );
    println!("execution=split_pinned_aarch64_sm_and_public_target_aarch64_kernel");

    let mut actor = Kernel::try_new().expect("AArch64 target router");
    let source = [
        1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let positions = [1_i32, 2_i32];
    let mut output = [0.0_f32; 12];
    assert_eq!(
        run(
            &mut actor,
            &source,
            &positions,
            [6, 1, 2, 1],
            ROPE_MODE_NORM,
            &mut output
        ),
        Ok(())
    );
    println!("case=norm status=ok output_bits={}", bits(&output));

    let short = [1.0_f32, 2.0, 3.0, 4.0];
    let one_position = [1_i32];
    let mut output = [0.0_f32; 4];
    assert_eq!(
        run(
            &mut actor,
            &short,
            &one_position,
            [4, 1, 1, 1],
            ROPE_MODE_NEOX,
            &mut output
        ),
        Ok(())
    );
    println!("case=neox status=ok output_bits={}", bits(&output));
    let mut output = [0.0_f32; 4];
    assert_eq!(
        run(
            &mut actor,
            &short,
            &one_position,
            [4, 1, 1, 1],
            ROPE_MODE_TIMESTEP,
            &mut output
        ),
        Ok(())
    );
    println!("case=timestep status=ok output_bits={}", bits(&output));

    let mut invalid_output = [SENTINEL; 4];
    assert_eq!(
        run(
            &mut actor,
            &short,
            &one_position,
            [4, 1, 1, 1],
            99,
            &mut invalid_output
        ),
        Err(RopeError::InvalidMode)
    );
    assert!(
        invalid_output
            .iter()
            .all(|v| v.to_bits() == SENTINEL.to_bits())
    );
    println!(
        "case=invalid_mode status=reject error=InvalidMode output_bits={}",
        bits(&invalid_output)
    );
}
