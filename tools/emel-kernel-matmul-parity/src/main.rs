//! Public observer for the maintained root matmul routes.

use emel_kernels::Kernel;
use emel_kernels::any::event::{OpMulMat, OpMulMatArgmax};
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};
use std::time::Instant;

const SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const SML_COMMIT: &str = "49207123cd3f39767764bae774932cb48623f92f";
const EVENTS_BLOB: &str = "4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9";
const DETAIL_BLOB: &str = "c8a82643eabfe8f2d7883e655955f455794511b0";
const X86_SM_BLOB: &str = "0b4d635ebbd0fbd52dbca8a2345547fb571205c8";
const X86_GUARDS_BLOB: &str = "cb3dac8253f8417c9b44acff1de414f6d0a3a3cf";
const X86_ACTIONS_BLOB: &str = "d45558f5eb96950f43c16a09d768cb4f382d6d61";
const AARCH64_SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";
const AARCH64_GUARDS_BLOB: &str = "c25714566ec9a02679daef85089544575123408e";
const AARCH64_ACTIONS_BLOB: &str = "267d4f74e6e7498155c8535920322ffef2c02fb6";

fn f32_layout(ne: [u64; 4]) -> Layout {
    Layout::contiguous(DType::F32, ne).expect("fixed parity layout fits")
}

fn bits(values: &[f32]) -> String {
    values
        .iter()
        .map(|value| format!("{:08x}", value.to_bits()))
        .collect::<Vec<_>>()
        .join(",")
}

fn main() {
    if std::env::args().nth(1).as_deref() == Some("--benchmark") {
        benchmark();
        return;
    }
    println!("kernel-matmul-parity/v1");
    println!("source_repository=stateforward/emel.cpp");
    println!("source_commit={SOURCE_COMMIT}");
    println!("source_sml_commit={SML_COMMIT}");
    println!("source_kernel_events_blob={EVENTS_BLOB}");
    println!("source_kernel_detail_blob={DETAIL_BLOB}");
    println!("source_kernel_x86_sm_blob={X86_SM_BLOB}");
    println!("source_kernel_x86_guards_blob={X86_GUARDS_BLOB}");
    println!("source_kernel_x86_actions_blob={X86_ACTIONS_BLOB}");
    println!("source_kernel_aarch64_sm_blob={AARCH64_SM_BLOB}");
    println!("source_kernel_aarch64_guards_blob={AARCH64_GUARDS_BLOB}");
    println!("source_kernel_aarch64_actions_blob={AARCH64_ACTIONS_BLOB}");
    println!("scope=f32_public_root_matmul_and_argmax_dense_rank4");

    let lhs = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs = [1.0_f32, 0.0, 0.0, 1.0, 1.0, 1.0];
    let mut output = [0.0_f32; 4];
    let result = Kernel::new().process_event(OpMulMat::new(
        TensorView::new(&lhs, f32_layout([3, 2, 1, 1])),
        TensorView::new(&rhs, f32_layout([2, 3, 1, 1])),
        TensorViewMut::new(&mut output, f32_layout([2, 2, 1, 1])),
    ));
    println!(
        "case=matrix status={} output_bits={}",
        if result.is_ok() { "ok" } else { "error" },
        bits(&output)
    );

    let mut best_value = [0.0_f32; 1];
    let mut actor = Kernel::new();
    let argmax = actor.process_event(OpMulMatArgmax::new(
        TensorView::new(&lhs, f32_layout([3, 2, 1, 1])),
        TensorView::new(&[1.0_f32, 0.0, 0.0], f32_layout([1, 3, 1, 1])),
        TensorViewMut::new(&mut best_value, f32_layout([1, 1, 1, 1])),
    ));
    println!(
        "case=argmax status={} index={} output_bits={}",
        if argmax.is_ok() { "ok" } else { "error" },
        argmax.unwrap_or(-1),
        bits(&best_value)
    );
}

fn benchmark() {
    const K: usize = 16;
    const ITERATIONS: u32 = 10_000;
    const WARMUP: usize = 1_000;
    const LHS: [f32; K] = [1.0; K];
    const RHS: [f32; K] = [1.0; K];
    let lhs_layout = f32_layout([K as u64, 1, 1, 1]);
    let rhs_layout = f32_layout([1, K as u64, 1, 1]);
    let destination_layout = f32_layout([1, 1, 1, 1]);
    let mut output = [0.0_f32; 1];
    let mut actor = Kernel::new();
    for _ in 0..WARMUP {
        let _ = actor.process_event(OpMulMat::new(
            TensorView::new(&LHS, lhs_layout),
            TensorView::new(&RHS, rhs_layout),
            TensorViewMut::new(&mut output, destination_layout),
        ));
    }
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = actor.process_event(OpMulMat::new(
            TensorView::new(&LHS, lhs_layout),
            TensorView::new(&RHS, rhs_layout),
            TensorViewMut::new(&mut output, destination_layout),
        ));
    }
    let ns_per_dispatch = start.elapsed().as_secs_f64() * 1e9 / f64::from(ITERATIONS);
    println!(
        "case=op_mul_mat_f32 rust_ns_per_dispatch={ns_per_dispatch:.3} output={}",
        output[0]
    );
}
