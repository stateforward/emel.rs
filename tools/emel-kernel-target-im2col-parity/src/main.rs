//! Split Rust observer for the public AArch64 target `im2col` router.

use emel_kernels::any::im2col::{Im2ColError, Im2ColParams};
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};
use emel_kernels::aarch64::{OpIm2Col, Kernel};

const COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const EVENTS_BLOB: &str = "4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9";
const DETAIL_BLOB: &str = "c8a82643eabfe8f2d7883e655955f455794511b0";
const GUARDS_BLOB: &str = "c25714566ec9a02679daef85089544575123408e";
const ACTIONS_BLOB: &str = "267d4f74e6e7498155c8535920322ffef2c02fb6";
const SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";
const SENTINEL: f32 = f32::from_bits(0x7fc0_0001);

#[cfg(not(target_arch = "aarch64"))]
fn main() {
    eprintln!("AArch64 im2col observer requires an AArch64 build");
    std::process::exit(2);
}

#[cfg(target_arch = "aarch64")]
fn layout(ne: [u64; 4]) -> Layout {
    Layout::contiguous(DType::F32, ne).expect("fixture layout fits")
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
fn main() {
    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let kernel = layout([3, 1, 1, 1]);
    let input_layout = layout([4, 1, 1, 1]);
    let output_layout = layout([3, 4, 1, 1]);
    let params = Im2ColParams {
        stride: 1,
        padding: 1,
        dilation: 1,
        is_2d: 0,
    };

    println!("kernel-target-im2col-live/v1");
    println!("source_repository=stateforward/emel.cpp");
    println!("source_commit={COMMIT}");
    println!("source_kernel_events_blob={EVENTS_BLOB}");
    println!("source_kernel_detail_blob={DETAIL_BLOB}");
    println!("source_kernel_aarch64_guards_blob={GUARDS_BLOB}");
    println!("source_kernel_aarch64_actions_blob={ACTIONS_BLOB}");
    println!("source_kernel_aarch64_sm_blob={SM_BLOB}");
    println!("target_arch=aarch64");
    println!("scope=target_router_im2col_f32_1d_dense");
    println!("execution=split_pinned_aarch64_sm_and_public_target_router");

    let mut actor = Kernel::try_new().expect("AArch64 target router");
    let mut output = [0.0_f32; 12];
    actor
        .process_im2col(OpIm2Col::new(
            kernel,
            TensorView::new(&input, input_layout),
            TensorViewMut::new(&mut output, output_layout),
            params,
        ))
        .expect("im2col");
    println!("case=zero_padding status=ok output_bits={}", bits(&output));

    let mut invalid = [SENTINEL; 12];
    assert_eq!(
        actor.process_im2col(OpIm2Col::new(
            kernel,
            TensorView::new(&input, input_layout),
            TensorViewMut::new(&mut invalid, output_layout),
            Im2ColParams { is_2d: 1, ..params },
        )),
        Err(Im2ColError::InvalidParameters)
    );
    println!(
        "case=invalid_parameters status=reject error=InvalidParameters output_bits={}",
        bits(&invalid)
    );

    let mismatch_layout = layout([3, 3, 1, 1]);
    let mut mismatch = [SENTINEL; 9];
    assert_eq!(
        actor.process_im2col(OpIm2Col::new(
            kernel,
            TensorView::new(&input, input_layout),
            TensorViewMut::new(&mut mismatch, mismatch_layout),
            params,
        )),
        Err(Im2ColError::ShapeMismatch)
    );
    println!(
        "case=shape_mismatch status=reject error=ShapeMismatch output_bits={}",
        bits(&mismatch)
    );
}
