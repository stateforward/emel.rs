//! Split Rust observer for the public AArch64 target normalization router.

use emel_kernels::aarch64::Kernel;
use emel_kernels::any::normalization::{NormalizationError, OpNorm, OpRmsNorm};
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};

const COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const EVENTS_BLOB: &str = "4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9";
const DETAIL_BLOB: &str = "c8a82643eabfe8f2d7883e655955f455794511b0";
const GUARDS_BLOB: &str = "c25714566ec9a02679daef85089544575123408e";
const ACTIONS_BLOB: &str = "267d4f74e6e7498155c8535920322ffef2c02fb6";
const SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";
const SENTINEL: f32 = f32::from_bits(0x7fc0_0001);

#[cfg(not(target_arch = "aarch64"))]
fn main() {
    eprintln!("AArch64 normalization observer requires an AArch64 build");
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
fn main() {
    let input = [1.0_f32, 2.0, 4.0, 2.0, 5.0, 8.0];
    let layout = Layout::contiguous(DType::F32, [3, 2, 1, 1]).expect("layout");
    println!("kernel-target-normalization-live/v1");
    println!("source_repository=stateforward/emel.cpp");
    println!("source_commit={COMMIT}");
    println!("source_kernel_events_blob={EVENTS_BLOB}");
    println!("source_kernel_detail_blob={DETAIL_BLOB}");
    println!("source_kernel_aarch64_guards_blob={GUARDS_BLOB}");
    println!("source_kernel_aarch64_actions_blob={ACTIONS_BLOB}");
    println!("source_kernel_aarch64_sm_blob={SM_BLOB}");
    println!("target_arch=aarch64");
    println!("scope=target_router_norm_rms_norm_f32_dense");
    println!("execution=split_pinned_aarch64_sm_and_public_target_router");

    let mut actor = Kernel::try_new().expect("AArch64 target router");
    let mut norm_output = [0.0_f32; 6];
    actor
        .process_normalization(OpNorm::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut norm_output, layout),
            1.0,
        ))
        .expect("norm");
    println!("case=norm status=ok output_bits={}", bits(&norm_output));

    let mut rms_output = [0.0_f32; 6];
    actor
        .process_normalization(OpRmsNorm::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut rms_output, layout),
            1.0,
        ))
        .expect("rms norm");
    println!("case=rms_norm status=ok output_bits={}", bits(&rms_output));

    let mut invalid_parameters = [SENTINEL; 6];
    assert_eq!(
        actor.process_normalization(OpNorm::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut invalid_parameters, layout),
            f32::NAN,
        )),
        Err(NormalizationError::InvalidParameters)
    );
    println!(
        "case=invalid_epsilon status=reject error=InvalidParameters output_bits={}",
        bits(&invalid_parameters)
    );

    let empty_layout = Layout::new(DType::F32, [0, 2, 1, 1], [4, 0, 0, 0]);
    let mut empty_output = [SENTINEL; 1];
    assert_eq!(
        actor.process_normalization(OpRmsNorm::new(
            TensorView::new(&[SENTINEL], empty_layout),
            TensorViewMut::new(&mut empty_output, empty_layout),
            1.0,
        )),
        Err(NormalizationError::InvalidView)
    );
    println!(
        "case=zero_columns status=reject error=InvalidView output_bits={}",
        bits(&empty_output)
    );

    let mismatch_layout = Layout::contiguous(DType::F32, [2, 2, 1, 1]).expect("layout");
    let mut mismatch_output = [SENTINEL; 4];
    assert_eq!(
        actor.process_normalization(OpNorm::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut mismatch_output, mismatch_layout),
            1.0,
        )),
        Err(NormalizationError::ShapeMismatch)
    );
    println!(
        "case=shape_mismatch status=reject error=ShapeMismatch output_bits={}",
        bits(&mismatch_output)
    );
}
