//! Split Rust observer for the public AArch64 target scalar-trigonometry router.

use emel_kernels::aarch64::{Kernel, OpScalarCos, OpScalarLog, OpScalarSin, UnexpectedScalarTrig};
use emel_kernels::any::reductions::ReductionError;
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
    eprintln!("AArch64 scalar-trig observer requires an AArch64 build");
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
    let input = [-3.5_f32, -0.5, 0.0, 0.5, 2.25, 7.0];
    let layout = Layout::contiguous(DType::F32, [6, 1, 1, 1]).expect("layout");
    println!("kernel-target-scalar-trig-live/v1");
    println!("source_repository=stateforward/emel.cpp");
    println!("source_commit={COMMIT}");
    println!("source_kernel_events_blob={EVENTS_BLOB}");
    println!("source_kernel_detail_blob={DETAIL_BLOB}");
    println!("source_kernel_aarch64_guards_blob={GUARDS_BLOB}");
    println!("source_kernel_aarch64_actions_blob={ACTIONS_BLOB}");
    println!("source_kernel_aarch64_sm_blob={SM_BLOB}");
    println!("target_arch=aarch64");
    println!("scope=target_router_scalar_trig_log_sin_cos_f32_dense_and_rejection");
    println!("execution=split_pinned_aarch64_sm_and_public_target_router");

    let mut actor = Kernel::try_new().expect("AArch64 target router");
    let mut output = [0.0_f32; 6];
    actor
        .process_scalar_trig(OpScalarLog::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout),
        ))
        .expect("log");
    println!("case=log status=ok output_bits={}", bits(&output));
    actor
        .process_scalar_trig(OpScalarSin::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout),
        ))
        .expect("sin");
    println!("case=sin status=ok output_bits={}", bits(&output));
    actor
        .process_scalar_trig(OpScalarCos::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout),
        ))
        .expect("cos");
    println!("case=cos status=ok output_bits={}", bits(&output));

    let empty_layout = Layout::new(DType::F32, [0, 1, 1, 1], [4, 0, 0, 0]);
    let mut empty_output = [SENTINEL; 1];
    assert_eq!(
        actor.process_scalar_trig(OpScalarLog::new(
            TensorView::new(&[SENTINEL], empty_layout),
            TensorViewMut::new(&mut empty_output, empty_layout),
        )),
        Err(ReductionError::InvalidView)
    );
    println!(
        "case=zero_count status=reject error=InvalidView output_bits={}",
        bits(&empty_output)
    );

    let mismatch_layout = Layout::contiguous(DType::F32, [5, 1, 1, 1]).expect("layout");
    let mut mismatch_output = [SENTINEL; 5];
    assert_eq!(
        actor.process_scalar_trig(OpScalarSin::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut mismatch_output, mismatch_layout),
        )),
        Err(ReductionError::ShapeMismatch)
    );
    println!(
        "case=shape_mismatch status=reject error=ShapeMismatch output_bits={}",
        bits(&mismatch_output)
    );

    assert_eq!(
        actor.process_scalar_trig(UnexpectedScalarTrig),
        Err(ReductionError::UnexpectedEvent)
    );
    println!("case=unexpected status=reject error=UnexpectedEvent");
}
