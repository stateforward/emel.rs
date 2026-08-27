//! Rust-side observer for the pinned scalar `op_mul_mat_argmax` routes.
//!
//! This binary is deliberately a split lane: it owns only Rust fixtures and
//! public actor dispatch. The companion C++ observer owns an independent copy
//! of every fixture; the shell harness compares text outputs after both lanes
//! finish, so no reference object or state crosses the boundary.

use emel_kernels::Kernel;
use emel_kernels::any::event::{
    MatmulArgmaxResult, OpMulMatArgmax, OpMulMatArgmaxQ2K, OpMulMatArgmaxQ3K, OpMulMatArgmaxQ4K,
    OpMulMatArgmaxQ5_0, OpMulMatArgmaxQ6K, OpMulMatArgmaxQ8_0, QuantizedView,
};
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};

const SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const DETAIL_BLOB: &str = "c8a82643eabfe8f2d7883e655955f455794511b0";
const EVENTS_BLOB: &str = "4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9";

fn f32_layout(ne: [u64; 4]) -> Layout {
    Layout::contiguous(DType::F32, ne).expect("fixed fixture layout fits")
}

const fn packed_layout(code: u8, k: u64, rows: u64, row_bytes: u64) -> Layout {
    Layout::new(
        DType::from_code(code),
        [k, rows, 1, 1],
        [1, row_bytes, row_bytes * rows, row_bytes * rows],
    )
}

fn bits(value: f32) -> String {
    format!("{:08x}", value.to_bits())
}

fn print_result(name: &str, result: MatmulArgmaxResult, output: f32) {
    match result {
        Ok(index) => println!(
            "case={name} status=ok index={index} output_bits={}",
            bits(output)
        ),
        Err(error) => println!(
            "case={name} status=error error={error:?} index=-77 output_bits={}",
            bits(output)
        ),
    }
}

fn q5_fixture() -> [u8; 352] {
    let mut bytes = [0_u8; 352];
    for block in 0..16 {
        let offset = block * 22;
        bytes[offset..offset + 2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        if block >= 8 {
            bytes[offset + 2..offset + 6].fill(0xff);
            bytes[offset + 6..offset + 22].fill(0x0f);
        }
    }
    bytes
}

fn q8_fixture() -> [u8; 544] {
    let mut bytes = [0_u8; 544];
    for block in 0..16 {
        let offset = block * 34;
        bytes[offset..offset + 2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        if block >= 8 {
            bytes[offset + 2..offset + 34].fill(1);
        }
    }
    bytes
}

fn q2_fixture() -> [u8; 168] {
    let mut bytes = [0_u8; 168];
    bytes[84 + 80..84 + 82].copy_from_slice(&0x3c00_u16.to_le_bytes());
    bytes[84..84 + 16].fill(0xff);
    bytes[84 + 16..84 + 80].fill(0xff);
    bytes
}

fn q3_fixture() -> [u8; 220] {
    let mut bytes = [0_u8; 220];
    bytes[110..110 + 32].fill(0xff);
    bytes[110 + 32..110 + 96].fill(0xff);
    bytes[110 + 96..110 + 108].fill(0x21);
    bytes[110 + 108..110 + 110].copy_from_slice(&0x3c00_u16.to_le_bytes());
    bytes
}

fn q4_k_fixture() -> [u8; 288] {
    let mut bytes = [0_u8; 288];
    bytes[144..144 + 2].copy_from_slice(&0x3c00_u16.to_le_bytes());
    bytes[144 + 4..144 + 16].fill(1);
    bytes[144 + 16..].fill(0xff);
    bytes
}

fn q6_k_fixture() -> [u8; 420] {
    let mut bytes = [0_u8; 420];
    bytes[210..210 + 128].fill(0xff);
    bytes[210 + 128..210 + 192].fill(0xff);
    bytes[210 + 192..210 + 208].fill(1);
    bytes[210 + 208..210 + 210].copy_from_slice(&0x3c00_u16.to_le_bytes());
    bytes
}

fn run_quantized(
    name: &str,
    lhs: &[u8],
    code: u8,
    row_bytes: u64,
    event: impl FnOnce(QuantizedView<'_>, TensorView<'_>, TensorViewMut<'_>) -> MatmulArgmaxResult,
) {
    let rhs = [1.0_f32; 256];
    let mut output = [f32::from_bits(0x7fc0_1234)];
    let result = event(
        QuantizedView::new(lhs, packed_layout(code, 256, 2, row_bytes)),
        TensorView::new(&rhs, f32_layout([1, 256, 1, 1])),
        TensorViewMut::new(&mut output, f32_layout([1, 1, 1, 1])),
    );
    print_result(name, result, output[0]);
}

fn main() {
    println!("kernel-argmax-parity/v1");
    println!("source_repository=stateforward/emel.cpp");
    println!("source_commit={SOURCE_COMMIT}");
    println!("source_detail_blob={DETAIL_BLOB}");
    println!("source_events_blob={EVENTS_BLOB}");
    println!("scope=f32_plus_aligned_native_quantized_argmax");
    println!("families=q5_0,q8_0,q2_k,q3_k,q4_k,q6_k");
    println!("excluded=q4_0,q4_1_reference_run_has_no_branch");

    let lhs = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs = [1.0_f32, 1.0, 1.0];
    let mut output = [f32::from_bits(0x7fc0_1234)];
    let mut actor = Kernel::new();
    let result = actor.process_event(OpMulMatArgmax::new(
        TensorView::new(&lhs, f32_layout([3, 2, 1, 1])),
        TensorView::new(&rhs, f32_layout([1, 3, 1, 1])),
        TensorViewMut::new(&mut output, f32_layout([1, 1, 1, 1])),
    ));
    print_result("f32_success", result, output[0]);

    let mut invalid_output = [f32::from_bits(0x7fc0_5678)];
    let invalid_result = actor.process_event(OpMulMatArgmax::new(
        TensorView::new(&lhs, f32_layout([3, 2, 1, 1])),
        TensorView::new(&rhs[..2], f32_layout([1, 2, 1, 1])),
        TensorViewMut::new(&mut invalid_output, f32_layout([1, 1, 1, 1])),
    ));
    print_result("f32_shape_error", invalid_result, invalid_output[0]);

    let mut invalid_view_output = [f32::from_bits(0x7fc0_9abc)];
    let invalid_view_result = actor.process_event(OpMulMatArgmax::new(
        TensorView::new(&lhs, f32_layout([3, 2, 1, 1])),
        TensorView::new(&rhs, Layout::new(DType::F16, [1, 3, 1, 1], [2, 2, 6, 6])),
        TensorViewMut::new(&mut invalid_view_output, f32_layout([1, 1, 1, 1])),
    ));
    print_result(
        "f32_invalid_view",
        invalid_view_result,
        invalid_view_output[0],
    );

    let q5 = q5_fixture();
    run_quantized("q5_0_success", &q5, 6, 176, |lhs, rhs, dst| {
        actor.process_event(OpMulMatArgmaxQ5_0::new(lhs, rhs, dst))
    });
    let q8 = q8_fixture();
    run_quantized("q8_0_success", &q8, 8, 272, |lhs, rhs, dst| {
        actor.process_event(OpMulMatArgmaxQ8_0::new(lhs, rhs, dst))
    });
    let q2 = q2_fixture();
    run_quantized("q2_k_success", &q2, 10, 84, |lhs, rhs, dst| {
        actor.process_event(OpMulMatArgmaxQ2K::new(lhs, rhs, dst))
    });
    let q3 = q3_fixture();
    run_quantized("q3_k_success", &q3, 11, 110, |lhs, rhs, dst| {
        actor.process_event(OpMulMatArgmaxQ3K::new(lhs, rhs, dst))
    });
    let q4 = q4_k_fixture();
    run_quantized("q4_k_success", &q4, 12, 144, |lhs, rhs, dst| {
        actor.process_event(OpMulMatArgmaxQ4K::new(lhs, rhs, dst))
    });
    let q6 = q6_k_fixture();
    run_quantized("q6_k_success", &q6, 14, 210, |lhs, rhs, dst| {
        actor.process_event(OpMulMatArgmaxQ6K::new(lhs, rhs, dst))
    });
}
