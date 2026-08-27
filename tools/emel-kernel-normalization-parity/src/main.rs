//! Public-actor observer for the bounded F32 normalization parity slice.

use emel_kernels::any::normalization::{NormalizationKernel, OpNorm, OpRmsNorm};
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};

const SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const EVENTS_BLOB: &str = "4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9";
const DETAIL_BLOB: &str = "c8a82643eabfe8f2d7883e655955f455794511b0";
const X86_SM_BLOB: &str = "0b4d635ebbd0fbd52dbca8a2345547fb571205c8";
const AARCH64_SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";

const fn layout(ne: [u64; 4], nb: [u64; 4]) -> Layout {
    Layout::new(DType::F32, ne, nb)
}

fn bits(values: &[f32]) -> String {
    values
        .iter()
        .map(|value| format!("{:08x}", value.to_bits()))
        .collect::<Vec<_>>()
        .join(",")
}

fn main() {
    println!("kernel-normalization-parity/v1");
    println!("source_repository=stateforward/emel.cpp");
    println!("source_commit={SOURCE_COMMIT}");
    println!("source_kernel_events_blob={EVENTS_BLOB}");
    println!("source_kernel_detail_blob={DETAIL_BLOB}");
    println!("source_kernel_x86_sm_blob={X86_SM_BLOB}");
    println!("source_kernel_aarch64_sm_blob={AARCH64_SM_BLOB}");
    println!("source_detail_guard_span=4546-4564");
    println!("source_detail_rms_run_span=4568-4603");
    println!("source_detail_norm_run_span=4605-4648");
    println!("scope=f32_norm_and_rms_norm_aligned_explicit_or_implicit_contiguous");

    let input = [1.0_f32, 2.0, 4.0, 2.0, 5.0, 8.0];
    let implicit = layout([3, 2, 1, 1], [0, 0, 0, 0]);
    let mut rms_output = [0.0_f32; 6];
    let mut kernel = NormalizationKernel::new();
    let rms_status = kernel
        .process_event(OpRmsNorm::new(
            TensorView::new(&input, implicit),
            TensorViewMut::new(&mut rms_output, implicit),
            1.0,
        ))
        .is_ok();
    println!(
        "case=implicit_rms_norm status={} output_bits={}",
        if rms_status { "ok" } else { "rejected" },
        bits(&rms_output)
    );

    let strided = layout([3, 2, 1, 1], [4, 16, 32, 32]);
    let strided_input = [1.0_f32, 2.0, 99.0, 99.0, 2.0, 5.0, 8.0];
    let mut norm_output = [9.0_f32; 7];
    let norm_status = kernel
        .process_event(OpNorm::new(
            TensorView::new(&strided_input, strided),
            TensorViewMut::new(&mut norm_output, strided),
            1.0,
        ))
        .is_ok();
    println!(
        "case=strided_norm status={} output_bits={}",
        if norm_status { "ok" } else { "rejected" },
        bits(&norm_output)
    );

    let empty = layout([3, 0, 1, 1], [4, 12, 0, 0]);
    let mut empty_output = [9.0_f32; 1];
    let empty_status = kernel
        .process_event(OpNorm::new(
            TensorView::new(&[1.0_f32], empty),
            TensorViewMut::new(&mut empty_output, empty),
            1.0,
        ))
        .is_ok();
    println!(
        "case=empty_outer_norm status={} output_bits={}",
        if empty_status { "ok" } else { "rejected" },
        bits(&empty_output)
    );

    let mut invalid_output = [9.0_f32; 6];
    let invalid_status = kernel
        .process_event(OpNorm::new(
            TensorView::new(&input, implicit),
            TensorViewMut::new(&mut invalid_output, implicit),
            f32::NAN,
        ))
        .is_ok();
    println!(
        "case=invalid_epsilon status={} output_bits={}",
        if invalid_status { "ok" } else { "rejected" },
        bits(&invalid_output)
    );
}
