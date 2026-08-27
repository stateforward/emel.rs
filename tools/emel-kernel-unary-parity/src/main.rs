//! Public observer for the maintained unary kernel routes.

use emel_kernels::Kernel;
use emel_kernels::any::event::{OpUnary, UnarySubOp};
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};
use std::time::Instant;

const SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const SML_COMMIT: &str = "49207123cd3f39767764bae774932cb48623f92f";
const EVENTS_BLOB: &str = "4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9";
const DETAIL_BLOB: &str = "c8a82643eabfe8f2d7883e655955f455794511b0";
const X86_SM_BLOB: &str = "0b4d635ebbd0fbd52dbca8a2345547fb571205c8";
const X86_GUARDS_BLOB: &str = "cb3dac8253f8417c9b44acff1de414f6d0a3a3cf";
const X86_ACTIONS_BLOB: &str = "d45558f5eb96950f43c16a09d768cb4f382d6d61";

fn bits(values: &[f32]) -> String {
    values
        .iter()
        .map(|value| format!("{:08x}", value.to_bits()))
        .collect::<Vec<_>>()
        .join(",")
}

fn main() {
    const INPUT: [f32; 8] = [-10.1, -10.0, -1.2345, -0.125, 0.0, 0.12345, 10.0, 10.1];

    if std::env::args().nth(1).as_deref() == Some("--benchmark") {
        benchmark();
        return;
    }
    let layout = Layout::contiguous(DType::F32, [8, 1, 1, 1]).expect("fixture layout fits");
    println!("kernel-unary-parity/v1");
    println!("source_repository=stateforward/emel.cpp");
    println!("source_commit={SOURCE_COMMIT}");
    println!("source_sml_commit={SML_COMMIT}");
    println!("source_kernel_events_blob={EVENTS_BLOB}");
    println!("source_kernel_detail_blob={DETAIL_BLOB}");
    println!("source_kernel_x86_sm_blob={X86_SM_BLOB}");
    println!("source_kernel_x86_guards_blob={X86_GUARDS_BLOB}");
    println!("source_kernel_x86_actions_blob={X86_ACTIONS_BLOB}");
    println!("source_supported_scalar_subops=abs,neg,relu,exp,tanh,elu,gelu,silu");
    println!("scope=portable_f32_generic_op_unary_equal_count_dense");

    for (name, subop) in [
        ("abs", UnarySubOp::Abs),
        ("neg", UnarySubOp::Neg),
        ("relu", UnarySubOp::Relu),
        ("exp", UnarySubOp::Exp),
        ("tanh", UnarySubOp::Tanh),
        ("elu", UnarySubOp::Elu),
        ("gelu", UnarySubOp::Gelu),
        ("silu", UnarySubOp::Silu),
    ] {
        let mut output = [0.0_f32; 8];
        let mut actor = Kernel::new();
        let result = actor.process_event(OpUnary::new(
            subop,
            TensorView::new(&INPUT, layout),
            TensorViewMut::new(&mut output, layout),
        ));
        println!(
            "case={name} status={} output_bits={}",
            if result.is_ok() { "ok" } else { "error" },
            bits(&output)
        );
    }
}

fn benchmark() {
    const ITERATIONS: u32 = 10_000;
    const WARMUP: usize = 1_000;
    const INPUT: [f32; 1024] = [1.25; 1024];
    let layout = Layout::contiguous(DType::F32, [1024, 1, 1, 1]).expect("fixture layout fits");
    let mut output = [0.0_f32; 1024];
    let mut actor = Kernel::new();
    for _ in 0..WARMUP {
        let _ = actor.process_event(OpUnary::new(
            UnarySubOp::Abs,
            TensorView::new(&INPUT, layout),
            TensorViewMut::new(&mut output, layout),
        ));
    }
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = actor.process_event(OpUnary::new(
            UnarySubOp::Abs,
            TensorView::new(&INPUT, layout),
            TensorViewMut::new(&mut output, layout),
        ));
    }
    let ns_per_dispatch = start.elapsed().as_secs_f64() * 1e9 / f64::from(ITERATIONS);
    println!(
        "case=op_unary_abs rust_ns_per_dispatch={ns_per_dispatch:.3} output={}",
        output[0]
    );
}
