//! Public observer for the maintained portable F32 activation actor.

use emel_kernels::any::activation::{ActivationKernel, OpClamp, OpLeakyRelu, OpScale, OpSiluBack};
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};

const SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const SML_COMMIT: &str = "49207123cd3f39767764bae774932cb48623f92f";
const EVENTS_BLOB: &str = "4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9";
const DETAIL_BLOB: &str = "c8a82643eabfe8f2d7883e655955f455794511b0";
const X86_SM_BLOB: &str = "0b4d635ebbd0fbd52dbca8a2345547fb571205c8";
const X86_ACTIONS_BLOB: &str = "d45558f5eb96950f43c16a09d768cb4f382d6d61";
const X86_GUARDS_BLOB: &str = "cb3dac8253f8417c9b44acff1de414f6d0a3a3cf";

fn bits(values: &[f32]) -> String {
    values
        .iter()
        .map(|value| format!("{:08x}", value.to_bits()))
        .collect::<Vec<_>>()
        .join(",")
}

fn layout(count: usize) -> Layout {
    Layout::contiguous(DType::F32, [count as u64, 1, 1, 1]).expect("fixture layout fits")
}

fn print_header() {
    println!("kernel-activation-parity/v1");
    println!("source_repository=stateforward/emel.cpp");
    println!("source_commit={SOURCE_COMMIT}");
    println!("source_sml_commit={SML_COMMIT}");
    println!("source_kernel_events_blob={EVENTS_BLOB}");
    println!("source_kernel_detail_blob={DETAIL_BLOB}");
    println!("source_kernel_x86_sm_blob={X86_SM_BLOB}");
    println!("source_kernel_x86_actions_blob={X86_ACTIONS_BLOB}");
    println!("source_kernel_x86_guards_blob={X86_GUARDS_BLOB}");
    println!("source_detail_activation_declarations=45,55,73,93");
    println!("source_x86_activation_rows=296-303,456-463,671-678,881-888");
    println!("scope=portable_f32_dense_activation_scale_clamp_silu_back_leaky_relu");
}

fn main() {
    print_header();
    let input = [-2.0_f32, -0.5, 0.0, 2.0, f32::from_bits(0x7fc0_1234)];
    let view = layout(input.len());
    let mut output = [0.0_f32; 5];
    let mut kernel = ActivationKernel::new();

    let status = kernel.process_event(OpScale::new(
        TensorView::new(&input, view),
        TensorViewMut::new(&mut output, view),
        1.5,
    ));
    println!(
        "case=op_scale status={status:?} output_bits={}",
        bits(&output)
    );

    let status = kernel.process_event(OpClamp::new(
        TensorView::new(&input, view),
        TensorViewMut::new(&mut output, view),
        -1.0,
        1.0,
    ));
    println!(
        "case=op_clamp status={status:?} output_bits={}",
        bits(&output)
    );

    let silu_input = [-2.0_f32, -0.5, 0.0, 2.0, 0.25];
    let gradient = [0.5_f32, -2.0, 1.0, 3.0, -0.25];
    let status = kernel.process_event(OpSiluBack::new(
        TensorView::new(&silu_input, view),
        TensorView::new(&gradient, view),
        TensorViewMut::new(&mut output, view),
    ));
    println!(
        "case=op_silu_back status={status:?} output_bits={}",
        bits(&output)
    );

    let status = kernel.process_event(OpLeakyRelu::new(
        TensorView::new(&input, view),
        TensorViewMut::new(&mut output, view),
        0.125,
    ));
    println!(
        "case=op_leaky_relu status={status:?} output_bits={}",
        bits(&output)
    );
}
