//! Public Rust observer for the maintained portable root kernel F16 matmul route.

use emel_kernels::Kernel;
use emel_kernels::any::event::{F16View, OpMulMatF16};
use emel_kernels::any::tensor_view::{DType, Layout, TensorViewMut};
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

const fn f16_layout(ne: [u64; 4]) -> Layout {
    let mut nb = [2_u64; 4];
    nb[1] = 2 * ne[0];
    nb[2] = nb[1] * ne[1];
    nb[3] = nb[2] * ne[2];
    Layout::new(DType::F16, ne, nb)
}

fn f32_layout(ne: [u64; 4]) -> Layout {
    Layout::contiguous(DType::F32, ne).expect("fixture layout fits")
}

fn bits(values: &[f32]) -> String {
    values
        .iter()
        .map(|value| format!("{:08x}", value.to_bits()))
        .collect::<Vec<_>>()
        .join(",")
}

fn main() {
    const LHS: [u16; 6] = [0x3c00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600];
    const RHS: [u16; 6] = [0x3c00, 0x0000, 0x0000, 0x3c00, 0x3c00, 0x3c00];
    const ACC_LHS: [u16; 16] = [
        22892, 382, 28000, 56722, 59028, 37094, 30984, 61306, 6332, 3406, 63920, 15970, 36836,
        5814, 36696, 27210,
    ];
    const ACC_RHS: [u16; 16] = [
        34267, 15045, 49471, 24521, 41187, 62221, 18631, 47249, 7403, 46165, 49487, 10841, 6643,
        40605, 19159, 54561,
    ];

    if std::env::args().nth(1).as_deref() == Some("--benchmark") {
        benchmark();
        return;
    }

    println!("kernel-f16-matmul-parity/v1");
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
    println!("scope=dense_explicit_nonzero_stride_f16_by_f16_to_f32_scalar_double_accumulation");

    let mut output = [0.0_f32; 4];
    let result = Kernel::new().process_event(OpMulMatF16::new(
        F16View::new(&LHS, f16_layout([3, 2, 1, 1])),
        F16View::new(&RHS, f16_layout([3, 2, 1, 1])),
        TensorViewMut::new(&mut output, f32_layout([2, 2, 1, 1])),
    ));
    println!(
        "case=matrix status={} output_bits={}",
        if result.is_ok() { "ok" } else { "error" },
        bits(&output)
    );

    let mut accumulation = [0.0_f32; 1];
    let result = Kernel::new().process_event(OpMulMatF16::new(
        F16View::new(&ACC_LHS, f16_layout([16, 1, 1, 1])),
        F16View::new(&ACC_RHS, f16_layout([16, 1, 1, 1])),
        TensorViewMut::new(&mut accumulation, f32_layout([1, 1, 1, 1])),
    ));
    println!(
        "case=double_accumulation status={} output_bits={}",
        if result.is_ok() { "ok" } else { "error" },
        bits(&accumulation)
    );
}

fn benchmark() {
    const K: usize = 16;
    const ITERATIONS: u32 = 10_000;
    const WARMUP: usize = 1_000;
    const LHS: [u16; K] = [0x3c00; K];
    const RHS: [u16; K] = [0x3c00; K];
    let lhs_layout = f16_layout([K as u64, 1, 1, 1]);
    let destination_layout = f32_layout([1, 1, 1, 1]);
    let mut output = [0.0_f32; 1];
    let mut actor = Kernel::new();
    for _ in 0..WARMUP {
        let _ = actor.process_event(OpMulMatF16::new(
            F16View::new(&LHS, lhs_layout),
            F16View::new(&RHS, lhs_layout),
            TensorViewMut::new(&mut output, destination_layout),
        ));
    }
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _ = actor.process_event(OpMulMatF16::new(
            F16View::new(&LHS, lhs_layout),
            F16View::new(&RHS, lhs_layout),
            TensorViewMut::new(&mut output, destination_layout),
        ));
    }
    let ns_per_dispatch = start.elapsed().as_secs_f64() * 1e9 / f64::from(ITERATIONS);
    println!(
        "case=op_mul_mat_f16 rust_ns_per_dispatch={ns_per_dispatch:.3} output={}",
        output[0]
    );
}
