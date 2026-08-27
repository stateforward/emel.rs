//! Public observer for the maintained F32 1-D `im2col` slice.

use emel_kernels::Kernel;
use emel_kernels::any::event::{Im2ColParams, OpIm2Col};
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};
use std::time::Instant;

const SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";

fn layout(ne: [u64; 4]) -> Layout {
    Layout::contiguous(DType::F32, ne).expect("fixture layout fits")
}

const fn implicit_layout(ne: [u64; 4]) -> Layout {
    Layout::new(DType::F32, ne, [0, 0, 0, 0])
}

const fn params(stride: i32, padding: i32, dilation: i32) -> Im2ColParams {
    Im2ColParams {
        stride,
        padding,
        dilation,
        is_2d: 0,
    }
}

fn bits(values: &[f32]) -> String {
    values
        .iter()
        .map(|value| format!("{value:08x}", value = value.to_bits()))
        .collect::<Vec<_>>()
        .join(",")
}

fn run_case(
    name: &str,
    kernel: [u64; 4],
    input_shape: [u64; 4],
    output_shape: [u64; 4],
    input: &[f32],
    params: Im2ColParams,
    implicit_layouts: bool,
) {
    let output_len = output_shape.iter().fold(1_usize, |count, extent| {
        count
            .checked_mul(usize::try_from(*extent).expect("fixture extent fits usize"))
            .expect("fixture output size fits usize")
    });
    let mut output = vec![0.0_f32; output_len];
    let mut actor = Kernel::new();
    let make_layout = if implicit_layouts {
        implicit_layout
    } else {
        layout
    };
    let result = actor.process_event(OpIm2Col::new(
        layout(kernel),
        TensorView::new(input, make_layout(input_shape)),
        TensorViewMut::new(&mut output, make_layout(output_shape)),
        params,
    ));
    match result {
        Ok(()) => println!("case={name} status=ok output_bits={}", bits(&output)),
        Err(_) => println!("case={name} status=error"),
    }
}

fn main() {
    if std::env::args().nth(1).as_deref() == Some("--benchmark") {
        benchmark();
        return;
    }
    println!("kernel-im2col-parity/v2");
    println!("source_repository=stateforward/emel.cpp");
    println!("source_commit={SOURCE_COMMIT}");
    println!("source_kernel_events_blob=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9");
    println!("source_kernel_detail_blob=c8a82643eabfe8f2d7883e655955f455794511b0");
    println!("source_kernel_x86_sm_blob=0b4d635ebbd0fbd52dbca8a2345547fb571205c8");
    println!("scope=portable_scalar_f32_1d_dense_output");

    run_case(
        "zero_padding",
        [3, 1, 1, 1],
        [4, 1, 1, 1],
        [3, 4, 1, 1],
        &[1.0, 2.0, 3.0, 4.0],
        params(1, 1, 1),
        false,
    );
    run_case(
        "channels_batches_stride_dilation",
        [2, 2, 1, 1],
        [5, 2, 2, 1],
        [4, 2, 2, 1],
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 101.0, 102.0, 103.0, 104.0,
            105.0, 110.0, 120.0, 130.0, 140.0, 150.0,
        ],
        params(2, 0, 2),
        false,
    );
    run_case(
        "implicit_contiguous",
        [2, 1, 1, 1],
        [3, 1, 1, 1],
        [2, 2, 1, 1],
        &[7.0, 8.0, 9.0],
        params(1, 0, 1),
        true,
    );
    run_case(
        "zero_batch_implicit_noop",
        [2, 1, 1, 1],
        [3, 1, 0, 1],
        [2, 2, 0, 1],
        &[0.0],
        params(1, 0, 1),
        true,
    );
}

fn benchmark() {
    const ITERATIONS: u32 = 10_000;
    const WARMUP: usize = 1_000;
    const INPUT_LENGTH: usize = 64;
    const KERNEL_LENGTH: usize = 3;
    const OUTPUT_LENGTH: usize = INPUT_LENGTH;
    const OUTPUT_ELEMENTS: usize = KERNEL_LENGTH * OUTPUT_LENGTH;
    const INPUT: [f32; INPUT_LENGTH] = [1.0; INPUT_LENGTH];
    let kernel_layout = layout([KERNEL_LENGTH as u64, 1, 1, 1]);
    let input_layout = layout([INPUT_LENGTH as u64, 1, 1, 1]);
    let output_layout = layout([KERNEL_LENGTH as u64, OUTPUT_LENGTH as u64, 1, 1]);
    let params = params(1, 1, 1);
    let mut output = [0.0_f32; OUTPUT_ELEMENTS];
    let mut actor = Kernel::new();

    for _ in 0..WARMUP {
        let _ = actor.process_event(OpIm2Col::new(
            kernel_layout,
            TensorView::new(&INPUT, input_layout),
            TensorViewMut::new(&mut output, output_layout),
            params,
        ));
    }
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = actor.process_event(OpIm2Col::new(
            kernel_layout,
            TensorView::new(&INPUT, input_layout),
            TensorViewMut::new(&mut output, output_layout),
            params,
        ));
    }
    let ns_per_dispatch = start.elapsed().as_secs_f64() * 1e9 / f64::from(ITERATIONS);
    assert_eq!(output[0].to_bits(), 0.0_f32.to_bits());
    assert_eq!(output[1].to_bits(), 1.0_f32.to_bits());
    assert_eq!(output[2].to_bits(), 1.0_f32.to_bits());
    println!(
        "case=op_im2col rust_ns_per_dispatch={ns_per_dispatch:.3} output={OUTPUT_ELEMENTS} probe_bits=00000000,3f800000,3f800000"
    );
}
