//! Public-actor observer for the bounded F32 conv-transpose-1d parity slice.

use emel_kernels::Kernel;
use emel_kernels::any::event::{ConvTranspose1dParams, OpConvTranspose1d};
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};

const SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";

fn bits(values: &[f32]) -> String {
    values
        .iter()
        .map(|value| format!("{:08x}", value.to_bits()))
        .collect::<Vec<_>>()
        .join(",")
}

const fn layout(ne: [u64; 4], nb: [u64; 4]) -> Layout {
    Layout::new(DType::F32, ne, nb)
}

fn main() {
    println!("kernel-conv-transpose-1d-parity/v1");
    println!("source_repository=stateforward/emel.cpp");
    println!("source_commit={SOURCE_COMMIT}");
    println!("source_kernel_events_blob=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9");
    println!("source_kernel_detail_blob=c8a82643eabfe8f2d7883e655955f455794511b0");
    println!("source_kernel_x86_sm_blob=0b4d635ebbd0fbd52dbca8a2345547fb571205c8");
    println!("source_kernel_aarch64_sm_blob=865a9cc6ba6115382ed043c464f3d62bcd851357");
    println!("source_detail_guard_span=4955-5009");
    println!("source_detail_run_span=5011-5064");
    println!("scope=f32_weights_aligned_nonzero_strides_explicit_dense_output");

    let weights = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input = [1.0_f32, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0, 0.0, 5.0, 0.0, 6.0];
    let mut output = [0.0_f32; 15];
    let mut kernel = Kernel::new();
    let result = kernel.process_event(OpConvTranspose1d::new(
        TensorView::new(&weights, layout([3, 2, 1, 1], [4, 12, 24, 24])),
        TensorView::new(&input, layout([3, 1, 1, 1], [8, 24, 24, 24])),
        TensorViewMut::new(&mut output, layout([7, 2, 1, 1], [4, 28, 56, 56])),
        ConvTranspose1dParams {
            stride: 2,
            padding: 0,
            dilation: 1,
        },
    ));
    println!(
        "case=valid_strided_f32 status={} output_bits={}",
        if result.is_ok() { "ok" } else { "rejected" },
        bits(&output[..14])
    );

    let mut invalid_output = [9.0_f32; 15];
    let result = kernel.process_event(OpConvTranspose1d::new(
        TensorView::new(&weights, layout([3, 2, 1, 1], [4, 12, 24, 24])),
        TensorView::new(&input, layout([3, 1, 1, 1], [8, 24, 24, 24])),
        TensorViewMut::new(&mut invalid_output, layout([6, 2, 1, 1], [4, 24, 48, 48])),
        ConvTranspose1dParams {
            stride: 2,
            padding: 0,
            dilation: 1,
        },
    ));
    println!(
        "case=invalid_shape status={} output_bits={}",
        if result.is_ok() { "ok" } else { "rejected" },
        bits(&invalid_output[..8])
    );
}
