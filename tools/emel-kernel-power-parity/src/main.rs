//! Public observer for the maintained portable F32 power actor.

use emel_kernels::Kernel;
use emel_kernels::any::event::{OpSqr as DenseOpSqr, OpSqrt as DenseOpSqrt};
use emel_kernels::any::power::{OpSqr, PowerKernel};
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};

const SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const EVENTS_BLOB: &str = "4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9";
const DETAIL_BLOB: &str = "c8a82643eabfe8f2d7883e655955f455794511b0";
const X86_SM_BLOB: &str = "0b4d635ebbd0fbd52dbca8a2345547fb571205c8";
const X86_ACTIONS_BLOB: &str = "d45558f5eb96950f43c16a09d768cb4f382d6d61";
const X86_GUARDS_BLOB: &str = "cb3dac8253f8417c9b44acff1de414f6d0a3a3cf";

fn bits(values: &[f32]) -> String {
    values
        .iter()
        .map(|value| format!("{value:08x}", value = value.to_bits()))
        .collect::<Vec<_>>()
        .join(",")
}

fn layout(ne: [u64; 4], nb: [u64; 4]) -> Layout {
    Layout::new(DType::F32, ne, nb)
}

fn logical_values(data: &[f32], ne: [u64; 4], nb: [u64; 4]) -> Vec<f32> {
    let mut strides = nb;
    if strides[0] == 0 {
        strides[0] = 4;
        for dimension in 1..4 {
            strides[dimension] = strides[dimension - 1]
                .checked_mul(ne[dimension - 1])
                .expect("fixture stride fits");
        }
    }
    let count = ne
        .iter()
        .try_fold(1_usize, |count, extent| {
            count.checked_mul(usize::try_from(*extent).expect("fixture extent fits"))
        })
        .expect("fixture count fits");
    (0..count)
        .map(|ordinal| {
            let mut remaining = ordinal as u64;
            let mut byte_offset = 0_u64;
            for dimension in 0..4 {
                byte_offset += (remaining % ne[dimension]) * strides[dimension];
                remaining /= ne[dimension];
            }
            data[usize::try_from(byte_offset / 4).expect("fixture offset fits")]
        })
        .collect()
}

fn run_sqr_case(
    name: &str,
    input: &[f32],
    input_ne: [u64; 4],
    input_nb: [u64; 4],
    output: &mut [f32],
    output_ne: [u64; 4],
    output_nb: [u64; 4],
) {
    let mut actor = PowerKernel::new();
    let result = actor.process_event(OpSqr::new(
        TensorView::new(input, layout(input_ne, input_nb)),
        TensorViewMut::new(output, layout(output_ne, output_nb)),
    ));
    match result {
        Ok(()) => println!(
            "case={name} status=ok output_bits={}",
            bits(&logical_values(output, output_ne, output_nb))
        ),
        Err(_) => println!(
            "case={name} status=error output_bits={}",
            bits(&logical_values(output, output_ne, output_nb))
        ),
    }
}

fn run_root_sqr_case(name: &str, input: &[f32], output: &mut [f32]) {
    let mut actor = Kernel::new();
    let result = actor.process_event(DenseOpSqr::new(input, output));
    match result {
        Ok(()) => println!("case={name} status=ok output_bits={}", bits(output)),
        Err(_) => println!("case={name} status=error output_bits={}", bits(output)),
    }
}

fn run_root_sqrt_case(name: &str, input: &[f32], output: &mut [f32]) {
    let mut actor = Kernel::new();
    let result = actor.process_event(DenseOpSqrt::new(input, output));
    match result {
        Ok(()) => println!("case={name} status=ok output_bits={}", bits(output)),
        Err(error) => println!("case={name} status=error error={error:?}"),
    }
}

fn main() {
    println!("kernel-power-parity/v1");
    println!("source_repository=stateforward/emel.cpp");
    println!("source_commit={SOURCE_COMMIT}");
    println!("source_sml_commit=49207123cd3f39767764bae774932cb48623f92f");
    println!("source_kernel_events_blob={EVENTS_BLOB}");
    println!("source_kernel_detail_blob={DETAIL_BLOB}");
    println!("source_kernel_x86_sm_blob={X86_SM_BLOB}");
    println!("source_kernel_x86_actions_blob={X86_ACTIONS_BLOB}");
    println!("source_kernel_x86_guards_blob={X86_GUARDS_BLOB}");
    println!("source_detail_guard_span=3838-3841");
    println!("source_detail_run_span=5353-5356");
    println!("scope=portable_f32_dense_and_validated_strided_square_and_square_root");
    println!(
        "composition_root_cases=dense_square,count_equal_different_dimensions,dense_square_root,shape_rejection"
    );
    println!("composition_direct_child_cases=strided_square,implicit_contiguous_square");

    let mut dense_square = [0.0_f32; 4];
    run_root_sqr_case("dense_square", &[-2.0, -0.5, 0.5, 2.0], &mut dense_square);

    let mut strided_square = [0.0_f32; 5];
    run_sqr_case(
        "strided_square",
        &[3.0, 99.0, -4.0, 99.0, 0.5],
        [3, 1, 1, 1],
        [8, 24, 72, 72],
        &mut strided_square,
        [3, 1, 1, 1],
        [8, 24, 72, 72],
    );

    let mut reshaped_square = [0.0_f32; 4];
    run_root_sqr_case(
        "count_equal_different_dimensions",
        &[1.0, 2.0, 3.0, 4.0],
        &mut reshaped_square,
    );

    let mut implicit_square = [0.0_f32; 4];
    run_sqr_case(
        "implicit_contiguous_square",
        &[1.0, 2.0, 3.0, 4.0],
        [4, 1, 1, 1],
        [0, 0, 0, 0],
        &mut implicit_square,
        [4, 1, 1, 1],
        [0, 0, 0, 0],
    );

    let mut square_root = [0.0_f32; 4];
    run_root_sqrt_case("dense_square_root", &[0.0, 1.0, 4.0, 9.0], &mut square_root);

    let mut rejected = [9.0_f32; 3];
    run_root_sqr_case("shape_rejection", &[1.0, 2.0, 3.0, 4.0], &mut rejected);
}
