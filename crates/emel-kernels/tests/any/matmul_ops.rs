#![allow(clippy::float_cmp, missing_docs)]

use allocation_counter::measure;
use emel_kernels::any::matmul::{
    MatmulArgmaxKernel, MatmulError, MatmulKernel, OpMulMat, OpMulMatArgmax, OpMulMatArgmaxQ2K,
    OpMulMatArgmaxQ3K, OpMulMatArgmaxQ4_0, OpMulMatArgmaxQ4_1, OpMulMatArgmaxQ4K,
    OpMulMatArgmaxQ5_0, OpMulMatArgmaxQ6K, OpMulMatArgmaxQ8_0, OpMulMatQ2K, OpMulMatQ3K,
    OpMulMatQ4_0, OpMulMatQ4_1, OpMulMatQ4K, OpMulMatQ5_0, OpMulMatQ6K, OpMulMatQ8_0,
    QuantizedView, UnexpectedMatmul,
};
use emel_kernels::any::quant_more::{
    Q3KRow, Q4KRow, Q6KRow, Q8_K_BLOCK_BYTES, Q8KRow, dot_q3_k_q8_k, dot_q4_k_q8_k, dot_q6_k_q8_k,
};
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};
use emel_tensor as _;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use pulp as _;
use sml as _;

const PINNED_EMEL_CPP_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const PINNED_DETAIL_BLOB: &str = "c8a82643eabfe8f2d7883e655955f455794511b0";
const PINNED_X86_SM_BLOB: &str = "0b4d635ebbd0fbd52dbca8a2345547fb571205c8";
const PARITY_MANIFEST: &str =
    include_str!("../../../../snapshots/parity/kernel-matmul/manifest.txt");
const MATMUL_SOURCE: &str = include_str!("../../src/any/matmul.rs");

fn manifest_field(key: &str) -> &str {
    PARITY_MANIFEST
        .lines()
        .find_map(|line| {
            line.strip_prefix(key)
                .and_then(|value| value.strip_prefix('='))
        })
        .unwrap_or_else(|| panic!("manifest field `{key}` is missing"))
}

fn layout(ne: [u64; 4]) -> Layout {
    Layout::contiguous(DType::F32, ne).expect("F32 layout fits")
}

const fn packed_layout(code: u8, k: u64, rows: u64, row_bytes: u64) -> Layout {
    Layout::new(
        DType::Other(code),
        [k, rows, 1, 1],
        [1, row_bytes, row_bytes * rows, row_bytes * rows],
    )
}

fn q5_0_constant_block(encoded: u8) -> [u8; 22] {
    let mut block = [0_u8; 22];
    block[0] = 0x00;
    block[1] = 0x3c;
    block[2..6].fill(0xff);
    block[6..22].fill(encoded | (encoded << 4));
    block
}

fn q4_0_constant_block(encoded: u8) -> [u8; 18] {
    let mut block = [0_u8; 18];
    block[0] = 0x00;
    block[1] = 0x3c;
    block[2..18].fill(encoded | (encoded << 4));
    block
}

fn q4_1_constant_block(encoded: u8) -> [u8; 20] {
    let mut block = [0_u8; 20];
    block[0] = 0x00;
    block[1] = 0x3c;
    block[2] = 0x00;
    block[3] = 0x00;
    block[4..20].fill(encoded | (encoded << 4));
    block
}

fn q8_0_constant_block(encoded: i8) -> [u8; 34] {
    let mut block = [0_u8; 34];
    block[0] = 0x00;
    block[1] = 0x3c;
    block[2..34].fill(encoded.to_ne_bytes()[0]);
    block
}

const fn q2_k_zero_block() -> [u8; 84] {
    [0_u8; 84]
}

const fn q3_k_zero_block() -> [u8; 110] {
    [0_u8; 110]
}

fn varied_q3_k_block() -> [u8; 110] {
    let mut block = [0_u8; 110];
    for (index, byte) in block[..32].iter_mut().enumerate() {
        *byte = u8::try_from((index * 37 + 0x53) & 0xff).expect("mask fits");
    }
    for (index, byte) in block[32..96].iter_mut().enumerate() {
        *byte = u8::try_from((index * 29 + 0xa7) & 0xff).expect("mask fits");
    }
    block[96..108].copy_from_slice(&[
        0x21, 0x43, 0x65, 0x87, 0xa9, 0xcb, 0xed, 0x0f, 0x12, 0x34, 0x56, 0x78,
    ]);
    block[108..].copy_from_slice(&0x7bff_u16.to_le_bytes());
    block
}

fn pinned_nearest_int(value: f32) -> i32 {
    let biased = value + 12_582_912.0_f32;
    i32::try_from(biased.to_bits() & 0x007f_ffff).expect("mantissa fits") - 0x0040_0000
}

fn q8_k_fixture_from_rhs(rhs: &[f32; 256]) -> [u8; Q8_K_BLOCK_BYTES] {
    let mut maximum = 0.0_f32;
    let mut absolute = 0.0_f32;
    for value in rhs {
        if value.abs() > absolute {
            absolute = value.abs();
            maximum = *value;
        }
    }
    let inverse = -127.0_f32 / maximum;
    let scale = 1.0_f32 / inverse;
    let mut block = [0_u8; Q8_K_BLOCK_BYTES];
    block[..4].copy_from_slice(&scale.to_le_bytes());
    for (index, value) in rhs.iter().enumerate() {
        let quantized = pinned_nearest_int(inverse * value).clamp(-127, 127);
        block[4 + index] = i8::try_from(quantized)
            .expect("fixture quantized value fits")
            .to_ne_bytes()[0];
    }
    for group in 0..16 {
        let mut sum = 0_i16;
        for index in 0..16 {
            sum += i16::from(i8::from_ne_bytes([block[4 + group * 16 + index]]));
        }
        block[260 + group * 2..262 + group * 2].copy_from_slice(&sum.to_le_bytes());
    }
    block
}

const fn q4_k_zero_block() -> [u8; 144] {
    [0_u8; 144]
}

const fn q6_k_zero_block() -> [u8; 210] {
    [0_u8; 210]
}

fn varied_q6_k_block() -> [u8; 210] {
    let mut block = [0_u8; 210];
    for (index, byte) in block[..128].iter_mut().enumerate() {
        *byte = u8::try_from((index * 37 + 0x53) & 0xff).expect("mask fits");
    }
    for (index, byte) in block[128..192].iter_mut().enumerate() {
        *byte = u8::try_from((index * 29 + 0xa7) & 0xff).expect("mask fits");
    }
    for (index, byte) in block[192..208].iter_mut().enumerate() {
        *byte = u8::try_from((index * 11 + 0x31) & 0xff).expect("mask fits");
    }
    block[208..].copy_from_slice(&0x3c00_u16.to_le_bytes());
    block
}

fn varied_q4_k_block() -> [u8; 144] {
    let mut block = [0_u8; 144];
    block[..2].copy_from_slice(&0x7bff_u16.to_le_bytes());
    block[2..4].copy_from_slice(&0x3555_u16.to_le_bytes());
    block[4..16].copy_from_slice(&[
        0x81, 0x42, 0xc3, 0x24, 0x85, 0x46, 0xc7, 0x28, 0x39, 0x7a, 0x5b, 0x6c,
    ]);
    for (index, byte) in block[16..].iter_mut().enumerate() {
        let low = u8::try_from((5 * index + 1) % 16).expect("nibble fits");
        let high = u8::try_from((7 * index + 3) % 16).expect("nibble fits");
        *byte = low | (high << 4);
    }
    block
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
    }
}

#[test]
#[allow(clippy::too_many_lines)]
fn source_identity_and_orientation_match_pinned_contract() {
    assert_eq!(manifest_field("schema"), "emel.kernel.parity.v1");
    assert_eq!(manifest_field("component"), "emel-kernels::matmul_ops");
    assert_eq!(manifest_field("operation"), "op_mul_mat,op_mul_mat_argmax");
    assert_eq!(manifest_field("source_repository"), "../emel.cpp");
    assert_eq!(manifest_field("source_commit"), PINNED_EMEL_CPP_COMMIT);
    assert_eq!(manifest_field("source_detail_blob"), PINNED_DETAIL_BLOB);
    assert_eq!(manifest_field("source_x86_sm_blob"), PINNED_X86_SM_BLOB);
    assert_eq!(manifest_field("source_detail_span"), "3357-3570");
    assert_eq!(manifest_field("source_shape_guard_span"), "3844-3871");
    assert_eq!(manifest_field("source_argmax_guard_span"), "3574-3607");
    assert_eq!(manifest_field("source_argmax_execution_span"), "3609-3731");
    assert_eq!(manifest_field("source_rhs_quantization_span"), "804-848");
    assert_eq!(manifest_field("source_nearest_int_span"), "692-697");
    assert_eq!(
        manifest_field("source_packed_dot_spans"),
        "2459-2565,2567-2754,2756-3060,3062-3155,3183-3211"
    );
    assert_eq!(manifest_field("source_matmul_transition_span"), "416-423");
    assert_eq!(manifest_field("source_argmax_transition_span"), "425-433");
    assert_eq!(
        manifest_field("source_orientation"),
        "lhs[k,m],rhs[n,k],destination[n,m]"
    );
    assert_eq!(
        manifest_field("source_argmax_orientation"),
        "lhs[k,m],rhs[1,k],destination[1,1]"
    );
    assert_eq!(
        manifest_field("scope_quantized"),
        "q4_0,q4_1,q5_0,q8_0,q2_k,q3_k,q4_k,q6_k_with_actor_owned_q8_scratch"
    );
    assert_eq!(
        manifest_field("scope_regular_quantized"),
        "q4_0,q4_1,q5_0,q8_0,q2_k,q3_k,q4_k,q6_k_lhs_packed_rhs_f32_destination"
    );
    assert_eq!(
        manifest_field("source_regular_q4_0_execution_span"),
        "3376-3403"
    );
    assert_eq!(
        manifest_field("source_regular_q4_0_transition_span"),
        "416-423"
    );
    assert_eq!(
        manifest_field("source_regular_q4_1_execution_span"),
        "3405-3432"
    );
    assert_eq!(
        manifest_field("source_regular_q4_1_transition_span"),
        "416-423"
    );
    assert_eq!(
        manifest_field("source_regular_q5_0_execution_span"),
        "3436-3464"
    );
    assert_eq!(
        manifest_field("source_regular_q5_0_transition_span"),
        "416-423"
    );
    assert_eq!(
        manifest_field("source_regular_q8_0_execution_span"),
        "3466-3494"
    );
    assert_eq!(
        manifest_field("source_regular_q8_0_transition_span"),
        "416-423"
    );
    assert_eq!(
        manifest_field("source_regular_q2_k_execution_span"),
        "3509-3520"
    );
    assert_eq!(
        manifest_field("source_regular_q2_k_transition_span"),
        "416-423"
    );
    assert_eq!(
        manifest_field("source_regular_q3_k_execution_span"),
        "3521-3525"
    );
    assert_eq!(
        manifest_field("source_regular_q3_k_transition_span"),
        "416-423"
    );
    assert_eq!(
        manifest_field("source_regular_q4_k_execution_span"),
        "3517-3528"
    );
    assert_eq!(
        manifest_field("source_regular_q4_k_transition_span"),
        "416-423"
    );
    assert_eq!(
        manifest_field("source_regular_q6_k_execution_span"),
        "3496-3537"
    );
    assert_eq!(
        manifest_field("source_regular_q6_k_transition_span"),
        "416-423"
    );
    assert_eq!(
        manifest_field("scope_dispatch"),
        "explicit_sml_rtc_guards_and_bounded_action"
    );
    assert_eq!(
        manifest_field("scope_allocation"),
        "zero_allocations_in_128_dispatches"
    );
    assert_eq!(
        manifest_field("scope_unexpected"),
        "typed_unexpected_event_and_non_silent_generic_rejection"
    );
    assert_eq!(manifest_field("observer"), "live_cpp_observer");
    assert_eq!(manifest_field("result"), "live_match");
}

#[test]
fn source_layout_scope_matches_pinned_contract() {
    assert_eq!(
        manifest_field("scope_layouts"),
        "dense_contiguous_f32_and_packed_rows"
    );
}

#[test]
fn source_q4_legacy_dot_span_is_pinned() {
    assert_eq!(manifest_field("source_q4_legacy_dot_spans"), "3131-3178");
}

#[test]
fn source_fp32_to_fp16_span_is_pinned() {
    assert_eq!(manifest_field("source_fp32_to_fp16_span"), "372-392");
}

#[test]
fn public_wrappers_inspect_ready_states() {
    assert!(MATMUL_SOURCE.contains("let output = event.dispatch(self);"));
    assert!(MATMUL_SOURCE.contains("self.machine.is(&MatmulMachineStates::Ready)"));
    assert!(MATMUL_SOURCE.contains("self.machine.is(&MatmulArgmaxMachineStates::Ready)"));
}

#[test]
fn dense_f32_matmul_matches_pinned_orientation() {
    let lhs_layout = layout([3, 2, 1, 1]);
    let rhs_layout = layout([2, 3, 1, 1]);
    let destination_layout = layout([2, 2, 1, 1]);
    let lhs = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs = [7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut destination = [0.0_f32; 4];
    let mut actor = MatmulKernel::new();

    assert_eq!(
        actor.process_event(OpMulMat::new(
            TensorView::new(&lhs, lhs_layout),
            TensorView::new(&rhs, rhs_layout),
            TensorViewMut::new(&mut destination, destination_layout),
        )),
        Ok(())
    );
    assert_close(&destination, &[58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn strided_f32_operands_follow_logical_coordinates() {
    let lhs_layout = Layout::new(DType::F32, [3, 2, 1, 1], [8, 32, 64, 64]);
    let rhs_layout = Layout::new(DType::F32, [2, 3, 1, 1], [4, 24, 48, 48]);
    let destination_layout = Layout::new(DType::F32, [2, 2, 1, 1], [8, 16, 32, 32]);
    let lhs = [
        1.0_f32, 99.0, 2.0, 88.0, 3.0, 77.0, 66.0, 55.0, 4.0, 98.0, 5.0, 97.0, 6.0,
    ];
    let rhs = [
        7.0_f32, 8.0, 90.0, 91.0, 92.0, 93.0, 9.0, 10.0, 94.0, 95.0, 96.0, 97.0, 11.0, 12.0,
    ];
    let mut destination = [0.0_f32; 8];
    let mut actor = MatmulKernel::new();

    assert_eq!(
        actor.process_event(OpMulMat::new(
            TensorView::new(&lhs, lhs_layout),
            TensorView::new(&rhs, rhs_layout),
            TensorViewMut::new(&mut destination, destination_layout),
        )),
        Ok(())
    );
    assert_close(
        &[
            destination[0],
            destination[2],
            destination[4],
            destination[6],
        ],
        &[58.0, 64.0, 139.0, 154.0],
    );
}

#[test]
fn q4_0_regular_matmul_uses_packed_lhs_and_strided_rhs_columns() {
    let lhs_layout = packed_layout(2, 32, 2, 18);
    let rhs_layout = layout([2, 32, 1, 1]);
    let destination_layout = layout([2, 2, 1, 1]);
    let mut lhs = [0_u8; 36];
    lhs[..18].copy_from_slice(&q4_0_constant_block(9));
    lhs[18..].copy_from_slice(&q4_0_constant_block(10));
    let mut rhs = [0.0_f32; 64];
    let mut index = 0;
    while index < 32 {
        rhs[index * 2] = 1.0;
        rhs[index * 2 + 1] = 2.0;
        index += 1;
    }
    let mut destination = [0.0_f32; 4];
    let mut actor = MatmulKernel::new();

    assert_eq!(
        actor.process_event(OpMulMatQ4_0::new(
            QuantizedView::new(&lhs, lhs_layout),
            TensorView::new(&rhs, rhs_layout),
            TensorViewMut::new(&mut destination, destination_layout),
        )),
        Ok(())
    );
    assert_close(
        &destination,
        &[31.998_047, 63.996_094, 63.996_094, 127.992_19],
    );
}

#[test]
fn q4_0_regular_matmul_rejects_shape_without_writing() {
    let lhs_layout = packed_layout(2, 32, 2, 18);
    let rhs_layout = layout([2, 31, 1, 1]);
    let destination_layout = layout([2, 2, 1, 1]);
    let lhs = [0_u8; 36];
    let rhs = [1.0_f32; 62];
    let mut destination = [9.0_f32; 4];
    let mut actor = MatmulKernel::new();

    assert_eq!(
        actor.process_event(OpMulMatQ4_0::new(
            QuantizedView::new(&lhs, lhs_layout),
            TensorView::new(&rhs, rhs_layout),
            TensorViewMut::new(&mut destination, destination_layout),
        )),
        Err(MatmulError::ShapeMismatch)
    );
    assert_eq!(destination, [9.0; 4]);
}

#[test]
fn q4_0_regular_matmul_dispatch_is_allocation_free() {
    let lhs_layout = packed_layout(2, 32, 1, 18);
    let rhs_layout = layout([1, 32, 1, 1]);
    let destination_layout = layout([1, 1, 1, 1]);
    let lhs = q4_0_constant_block(9);
    let rhs = [1.0_f32; 32];
    let mut destination = [0.0_f32; 1];
    let mut actor = MatmulKernel::new();

    let allocations = measure(|| {
        assert_eq!(
            actor.process_event(OpMulMatQ4_0::new(
                QuantizedView::new(&lhs, lhs_layout),
                TensorView::new(&rhs, rhs_layout),
                TensorViewMut::new(&mut destination, destination_layout),
            )),
            Ok(())
        );
    });
    assert_eq!(allocations.count_total, 0);
    assert!((destination[0] - 31.998_047).abs() < 1e-6);
}

#[test]
fn q4_1_regular_matmul_uses_packed_lhs_and_strided_rhs_columns() {
    let lhs_layout = packed_layout(3, 32, 2, 20);
    let rhs_layout = layout([2, 32, 1, 1]);
    let destination_layout = layout([2, 2, 1, 1]);
    let mut lhs = [0_u8; 40];
    lhs[..20].copy_from_slice(&q4_1_constant_block(1));
    lhs[20..].copy_from_slice(&q4_1_constant_block(2));
    let rhs = [1.0_f32; 64];
    let mut destination = [0.0_f32; 4];
    let mut actor = MatmulKernel::new();

    assert_eq!(
        actor.process_event(OpMulMatQ4_1::new(
            QuantizedView::new(&lhs, lhs_layout),
            TensorView::new(&rhs, rhs_layout),
            TensorViewMut::new(&mut destination, destination_layout),
        )),
        Ok(())
    );
    assert_close(
        &destination,
        &[31.998_047, 31.998_047, 63.996_094, 63.996_094],
    );
}

#[test]
fn q4_1_regular_matmul_rejects_shape_and_view_without_writing() {
    let lhs_layout = packed_layout(3, 32, 2, 20);
    let lhs = [0_u8; 40];
    let rhs = [1.0_f32; 62];
    let mut destination = [9.0_f32; 4];
    let mut actor = MatmulKernel::new();

    assert_eq!(
        actor.process_event(OpMulMatQ4_1::new(
            QuantizedView::new(&lhs, lhs_layout),
            TensorView::new(&rhs, layout([2, 31, 1, 1])),
            TensorViewMut::new(&mut destination, layout([2, 2, 1, 1])),
        )),
        Err(MatmulError::ShapeMismatch)
    );
    assert_eq!(destination, [9.0; 4]);

    assert_eq!(
        actor.process_event(OpMulMatQ4_1::new(
            QuantizedView::new(&lhs, lhs_layout),
            TensorView::new(&[1.0_f32; 64], layout([2, 32, 1, 1])),
            TensorViewMut::new(
                &mut destination,
                Layout::new(DType::F16, [2, 2, 1, 1], [2, 4, 8, 8])
            ),
        )),
        Err(MatmulError::InvalidView)
    );
    assert_eq!(destination, [9.0; 4]);

    assert_eq!(
        actor.process_event(UnexpectedMatmul),
        Err(MatmulError::UnexpectedEvent)
    );
}

#[test]
fn q4_1_regular_matmul_dispatch_is_allocation_free() {
    let lhs_layout = packed_layout(3, 32, 1, 20);
    let rhs_layout = layout([1, 32, 1, 1]);
    let destination_layout = layout([1, 1, 1, 1]);
    let lhs = q4_1_constant_block(1);
    let rhs = [1.0_f32; 32];
    let mut destination = [0.0_f32; 1];
    let mut actor = MatmulKernel::new();

    let allocations = measure(|| {
        assert_eq!(
            actor.process_event(OpMulMatQ4_1::new(
                QuantizedView::new(&lhs, lhs_layout),
                TensorView::new(&rhs, rhs_layout),
                TensorViewMut::new(&mut destination, destination_layout),
            )),
            Ok(())
        );
    });
    assert_eq!(allocations.count_total, 0);
    assert!((destination[0] - 31.998_047).abs() < 1e-6);
}

#[test]
fn q5_0_regular_matmul_uses_packed_lhs_and_strided_rhs_columns() {
    let lhs_layout = packed_layout(6, 32, 2, 22);
    let rhs_layout = layout([2, 32, 1, 1]);
    let destination_layout = layout([2, 2, 1, 1]);
    let mut lhs = [0_u8; 44];
    lhs[..22].copy_from_slice(&q5_0_constant_block(0x11));
    lhs[22..].copy_from_slice(&q5_0_constant_block(0x12));
    let mut rhs = [0.0_f32; 64];
    let mut index = 0;
    while index < 32 {
        rhs[index * 2] = 1.0;
        rhs[index * 2 + 1] = 2.0;
        index += 1;
    }
    let mut destination = [0.0_f32; 4];
    let mut actor = MatmulKernel::new();

    assert_eq!(
        actor.process_event(OpMulMatQ5_0::new(
            QuantizedView::new(&lhs, lhs_layout),
            TensorView::new(&rhs, rhs_layout),
            TensorViewMut::new(&mut destination, destination_layout),
        )),
        Ok(())
    );
    assert_close(
        &destination,
        &[31.998_047, 63.996_094, 79.995_12, 159.990_23],
    );
}

#[test]
fn q5_0_regular_matmul_rejects_shape_and_view_without_writing() {
    let lhs_layout = packed_layout(6, 32, 2, 22);
    let lhs = [0_u8; 44];
    let rhs = [1.0_f32; 64];
    let mut destination = [9.0_f32; 4];
    let mut actor = MatmulKernel::new();

    assert_eq!(
        actor.process_event(OpMulMatQ5_0::new(
            QuantizedView::new(&lhs, lhs_layout),
            TensorView::new(&rhs[..62], layout([2, 31, 1, 1])),
            TensorViewMut::new(&mut destination, layout([2, 2, 1, 1])),
        )),
        Err(MatmulError::ShapeMismatch)
    );
    assert_eq!(destination, [9.0; 4]);

    assert_eq!(
        actor.process_event(OpMulMatQ5_0::new(
            QuantizedView::new(&lhs, packed_layout(6, 32, 2, 22)),
            TensorView::new(&rhs, layout([2, 32, 1, 1])),
            TensorViewMut::new(
                &mut destination,
                Layout::new(DType::F16, [2, 2, 1, 1], [2, 4, 8, 8]),
            ),
        )),
        Err(MatmulError::InvalidView)
    );
    assert_eq!(destination, [9.0; 4]);
}

#[test]
fn q5_0_regular_matmul_dispatch_is_allocation_free() {
    let lhs_layout = packed_layout(6, 32, 1, 22);
    let rhs_layout = layout([1, 32, 1, 1]);
    let destination_layout = layout([1, 1, 1, 1]);
    let lhs = q5_0_constant_block(0x11);
    let rhs = [1.0_f32; 32];
    let mut destination = [0.0_f32; 1];
    let mut actor = MatmulKernel::new();

    let allocations = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                actor.process_event(OpMulMatQ5_0::new(
                    QuantizedView::new(&lhs, lhs_layout),
                    TensorView::new(&rhs, rhs_layout),
                    TensorViewMut::new(&mut destination, destination_layout),
                )),
                Ok(())
            );
        }
    });
    assert_eq!(allocations.count_total, 0);
    assert_eq!(allocations.bytes_total, 0);
    assert!((destination[0] - 31.998_047).abs() < 1e-6);
}

#[test]
fn q8_0_regular_matmul_uses_packed_lhs_and_strided_rhs_columns() {
    let lhs_layout = packed_layout(8, 32, 2, 34);
    let rhs_layout = layout([2, 32, 1, 1]);
    let destination_layout = layout([2, 2, 1, 1]);
    let mut lhs = [0_u8; 68];
    lhs[..34].copy_from_slice(&q8_0_constant_block(1));
    lhs[34..].copy_from_slice(&q8_0_constant_block(2));
    let mut rhs = [0.0_f32; 64];
    let mut index = 0;
    while index < 32 {
        rhs[index * 2] = 1.0;
        rhs[index * 2 + 1] = 2.0;
        index += 1;
    }
    let mut destination = [0.0_f32; 4];
    let mut actor = MatmulKernel::new();

    assert_eq!(
        actor.process_event(OpMulMatQ8_0::new(
            QuantizedView::new(&lhs, lhs_layout),
            TensorView::new(&rhs, rhs_layout),
            TensorViewMut::new(&mut destination, destination_layout),
        )),
        Ok(())
    );
    assert_close(
        &destination,
        &[31.998_047, 63.996_094, 63.996_094, 127.992_19],
    );
}

#[test]
fn q8_0_regular_matmul_rejects_shape_and_view_without_writing() {
    let lhs = [0_u8; 34];
    let rhs = [1.0_f32; 64];
    let mut destination = [9.0_f32; 1];
    let mut actor = MatmulKernel::new();

    assert_eq!(
        actor.process_event(OpMulMatQ8_0::new(
            QuantizedView::new(&lhs, packed_layout(8, 32, 1, 34)),
            TensorView::new(&rhs[..62], layout([2, 31, 1, 1])),
            TensorViewMut::new(&mut destination, layout([1, 1, 1, 1])),
        )),
        Err(MatmulError::ShapeMismatch)
    );
    assert_eq!(destination, [9.0]);
}

#[test]
fn q8_0_regular_matmul_dispatch_is_allocation_free() {
    let lhs = q8_0_constant_block(1);
    let rhs = [1.0_f32; 32];
    let mut destination = [0.0_f32; 1];
    let mut actor = MatmulKernel::new();
    let allocations = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                actor.process_event(OpMulMatQ8_0::new(
                    QuantizedView::new(&lhs, packed_layout(8, 32, 1, 34)),
                    TensorView::new(&rhs, layout([1, 32, 1, 1])),
                    TensorViewMut::new(&mut destination, layout([1, 1, 1, 1])),
                )),
                Ok(())
            );
        }
    });
    assert_eq!(allocations.count_total, 0);
    assert_eq!(allocations.bytes_total, 0);
}

#[test]
fn q2_k_regular_matmul_uses_packed_lhs_and_strided_rhs_columns() {
    let lhs_layout = packed_layout(10, 256, 1, 84);
    let rhs_layout = layout([1, 256, 1, 1]);
    let destination_layout = layout([1, 1, 1, 1]);
    let lhs = q2_k_zero_block();
    let rhs = [1.0_f32; 256];
    let mut destination = [9.0_f32; 1];
    let mut actor = MatmulKernel::new();

    assert_eq!(
        actor.process_event(OpMulMatQ2K::new(
            QuantizedView::new(&lhs, lhs_layout),
            TensorView::new(&rhs, rhs_layout),
            TensorViewMut::new(&mut destination, destination_layout),
        )),
        Ok(())
    );
    assert_eq!(destination, [0.0]);
}

#[test]
fn q2_k_regular_matmul_dispatch_is_allocation_free() {
    let lhs = q2_k_zero_block();
    let rhs = [1.0_f32; 256];
    let mut destination = [0.0_f32; 1];
    let mut actor = MatmulKernel::new();
    let allocations = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                actor.process_event(OpMulMatQ2K::new(
                    QuantizedView::new(&lhs, packed_layout(10, 256, 1, 84)),
                    TensorView::new(&rhs, layout([1, 256, 1, 1])),
                    TensorViewMut::new(&mut destination, layout([1, 1, 1, 1])),
                )),
                Ok(())
            );
        }
    });
    assert_eq!(allocations.count_total, 0);
    assert_eq!(allocations.bytes_total, 0);
}

#[test]
fn q3_k_regular_matmul_uses_packed_lhs_and_strided_rhs_columns() {
    let lhs_layout = packed_layout(11, 256, 1, 110);
    let rhs_layout = layout([2, 256, 1, 1]);
    let destination_layout = layout([2, 1, 1, 1]);
    let lhs = q3_k_zero_block();
    let rhs = [1.0_f32; 512];
    let mut destination = [9.0_f32; 2];
    let mut actor = MatmulKernel::new();

    assert_eq!(
        actor.process_event(OpMulMatQ3K::new(
            QuantizedView::new(&lhs, lhs_layout),
            TensorView::new(&rhs, rhs_layout),
            TensorViewMut::new(&mut destination, destination_layout),
        )),
        Ok(())
    );
    assert_eq!(destination, [0.0, 0.0]);
}

#[test]
fn q3_k_regular_matmul_preserves_pinned_lane_accumulation_bits() {
    let lhs = varied_q3_k_block();
    let mut rhs = [0.0_f32; 256];
    for (index, value) in rhs.iter_mut().enumerate() {
        let quantized = -f32::from(u8::try_from(index % 127 + 1).expect("bounded"));
        *value = quantized * 0.73;
    }
    let q8_fixture = q8_k_fixture_from_rhs(&rhs);
    let expected = dot_q3_k_q8_k(
        Q3KRow::from_bytes(&lhs).expect("valid q3 row"),
        Q8KRow::from_bytes(&q8_fixture).expect("valid q8 row"),
    )
    .expect("matching block counts");
    assert_ne!(expected, 0.0);

    let mut destination = [0.0_f32; 1];
    let mut actor = MatmulKernel::new();
    assert_eq!(
        actor.process_event(OpMulMatQ3K::new(
            QuantizedView::new(&lhs, packed_layout(11, 256, 1, 110)),
            TensorView::new(&rhs, layout([1, 256, 1, 1])),
            TensorViewMut::new(&mut destination, layout([1, 1, 1, 1])),
        )),
        Ok(())
    );
    assert_eq!(destination[0].to_bits(), expected.to_bits());
}

#[test]
fn q3_k_regular_matmul_rejects_shape_and_view_without_writing() {
    let lhs = q3_k_zero_block();
    let rhs = [1.0_f32; 256];
    let mut destination = [9.0_f32; 1];
    let mut actor = MatmulKernel::new();

    assert_eq!(
        actor.process_event(OpMulMatQ3K::new(
            QuantizedView::new(&lhs, packed_layout(11, 256, 1, 110)),
            TensorView::new(&rhs[..255], layout([1, 255, 1, 1])),
            TensorViewMut::new(&mut destination, layout([1, 1, 1, 1])),
        )),
        Err(MatmulError::ShapeMismatch)
    );
    assert_eq!(destination, [9.0]);

    assert_eq!(
        actor.process_event(OpMulMatQ3K::new(
            QuantizedView::new(&lhs, packed_layout(11, 256, 1, 110)),
            TensorView::new(&rhs, layout([1, 256, 1, 1])),
            TensorViewMut::new(
                &mut destination,
                Layout::new(DType::F16, [1, 1, 1, 1], [2, 2, 2, 2]),
            ),
        )),
        Err(MatmulError::InvalidView)
    );
    assert_eq!(destination, [9.0]);
}

#[test]
fn q3_k_regular_matmul_dispatch_is_allocation_free() {
    let lhs = q3_k_zero_block();
    let rhs = [1.0_f32; 256];
    let mut destination = [0.0_f32; 1];
    let mut actor = MatmulKernel::new();
    let allocations = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                actor.process_event(OpMulMatQ3K::new(
                    QuantizedView::new(&lhs, packed_layout(11, 256, 1, 110)),
                    TensorView::new(&rhs, layout([1, 256, 1, 1])),
                    TensorViewMut::new(&mut destination, layout([1, 1, 1, 1])),
                )),
                Ok(())
            );
        }
    });
    assert_eq!(allocations.count_total, 0);
    assert_eq!(allocations.bytes_total, 0);
}

#[test]
fn q4_k_regular_matmul_uses_packed_lhs_and_strided_rhs_columns() {
    let lhs_layout = packed_layout(12, 256, 1, 144);
    let rhs_layout = layout([2, 256, 1, 1]);
    let destination_layout = layout([2, 1, 1, 1]);
    let lhs = q4_k_zero_block();
    let rhs = [1.0_f32; 512];
    let mut destination = [9.0_f32; 2];
    let mut actor = MatmulKernel::new();

    assert_eq!(
        actor.process_event(OpMulMatQ4K::new(
            QuantizedView::new(&lhs, lhs_layout),
            TensorView::new(&rhs, rhs_layout),
            TensorViewMut::new(&mut destination, destination_layout),
        )),
        Ok(())
    );
    assert_eq!(destination, [0.0, 0.0]);
}

#[test]
fn q4_k_regular_matmul_preserves_pinned_lane_accumulation_bits() {
    let lhs = varied_q4_k_block();
    let mut rhs = [0.0_f32; 256];
    for (index, value) in rhs.iter_mut().enumerate() {
        let quantized = -f32::from(u8::try_from(index % 127 + 1).expect("bounded"));
        *value = quantized * 0.73;
    }
    let q8_fixture = q8_k_fixture_from_rhs(&rhs);
    let expected = dot_q4_k_q8_k(
        Q4KRow::from_bytes(&lhs).expect("valid q4 row"),
        Q8KRow::from_bytes(&q8_fixture).expect("valid q8 row"),
    )
    .expect("matching block counts");
    assert_ne!(expected, 0.0);

    let mut destination = [0.0_f32; 1];
    let mut actor = MatmulKernel::new();
    assert_eq!(
        actor.process_event(OpMulMatQ4K::new(
            QuantizedView::new(&lhs, packed_layout(12, 256, 1, 144)),
            TensorView::new(&rhs, layout([1, 256, 1, 1])),
            TensorViewMut::new(&mut destination, layout([1, 1, 1, 1])),
        )),
        Ok(())
    );
    assert_eq!(destination[0].to_bits(), expected.to_bits());
}

#[test]
fn q4_k_regular_matmul_rejects_shape_and_view_without_writing() {
    let lhs = q4_k_zero_block();
    let rhs = [1.0_f32; 256];
    let mut destination = [9.0_f32; 1];
    let mut actor = MatmulKernel::new();

    assert_eq!(
        actor.process_event(OpMulMatQ4K::new(
            QuantizedView::new(&lhs, packed_layout(12, 256, 1, 144)),
            TensorView::new(&rhs[..255], layout([1, 255, 1, 1])),
            TensorViewMut::new(&mut destination, layout([1, 1, 1, 1])),
        )),
        Err(MatmulError::ShapeMismatch)
    );
    assert_eq!(destination, [9.0]);

    assert_eq!(
        actor.process_event(OpMulMatQ4K::new(
            QuantizedView::new(&lhs, packed_layout(12, 256, 1, 144)),
            TensorView::new(&rhs, layout([1, 256, 1, 1])),
            TensorViewMut::new(
                &mut destination,
                Layout::new(DType::F16, [1, 1, 1, 1], [2, 2, 2, 2]),
            ),
        )),
        Err(MatmulError::InvalidView)
    );
    assert_eq!(destination, [9.0]);
}

#[test]
fn q4_k_regular_matmul_dispatch_is_allocation_free() {
    let lhs = q4_k_zero_block();
    let rhs = [1.0_f32; 256];
    let mut destination = [0.0_f32; 1];
    let mut actor = MatmulKernel::new();
    let allocations = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                actor.process_event(OpMulMatQ4K::new(
                    QuantizedView::new(&lhs, packed_layout(12, 256, 1, 144)),
                    TensorView::new(&rhs, layout([1, 256, 1, 1])),
                    TensorViewMut::new(&mut destination, layout([1, 1, 1, 1])),
                )),
                Ok(())
            );
        }
    });
    assert_eq!(allocations.count_total, 0);
    assert_eq!(allocations.bytes_total, 0);
}

#[test]
fn q6_k_regular_matmul_uses_packed_lhs_and_strided_rhs_columns() {
    let lhs = q6_k_zero_block();
    let rhs = [1.0_f32; 512];
    let mut destination = [9.0_f32; 2];
    let mut actor = MatmulKernel::new();

    assert_eq!(
        actor.process_event(OpMulMatQ6K::new(
            QuantizedView::new(&lhs, packed_layout(14, 256, 1, 210)),
            TensorView::new(&rhs, layout([2, 256, 1, 1])),
            TensorViewMut::new(&mut destination, layout([2, 1, 1, 1])),
        )),
        Ok(())
    );
    assert_eq!(destination, [0.0, 0.0]);
}

#[test]
fn q6_k_regular_matmul_preserves_pinned_lane_accumulation_bits() {
    let lhs = varied_q6_k_block();
    let mut rhs = [0.0_f32; 256];
    for (index, value) in rhs.iter_mut().enumerate() {
        let quantized = -f32::from(u8::try_from(index % 127 + 1).expect("bounded"));
        *value = quantized * 0.73;
    }
    let q8_fixture = q8_k_fixture_from_rhs(&rhs);
    let expected = dot_q6_k_q8_k(
        Q6KRow::from_bytes(&lhs).expect("valid q6 row"),
        Q8KRow::from_bytes(&q8_fixture).expect("valid q8 row"),
    )
    .expect("matching block counts");
    assert_ne!(expected, 0.0);

    let mut destination = [0.0_f32; 1];
    let mut actor = MatmulKernel::new();
    assert_eq!(
        actor.process_event(OpMulMatQ6K::new(
            QuantizedView::new(&lhs, packed_layout(14, 256, 1, 210)),
            TensorView::new(&rhs, layout([1, 256, 1, 1])),
            TensorViewMut::new(&mut destination, layout([1, 1, 1, 1])),
        )),
        Ok(())
    );
    assert_eq!(destination[0].to_bits(), expected.to_bits());
}

#[test]
fn q6_k_regular_matmul_rejects_shape_and_view_without_writing() {
    let lhs = q6_k_zero_block();
    let rhs = [1.0_f32; 256];
    let mut destination = [9.0_f32; 1];
    let mut actor = MatmulKernel::new();

    assert_eq!(
        actor.process_event(OpMulMatQ6K::new(
            QuantizedView::new(&lhs, packed_layout(14, 256, 1, 210)),
            TensorView::new(&rhs[..255], layout([1, 255, 1, 1])),
            TensorViewMut::new(&mut destination, layout([1, 1, 1, 1])),
        )),
        Err(MatmulError::ShapeMismatch)
    );
    assert_eq!(destination, [9.0]);

    assert_eq!(
        actor.process_event(OpMulMatQ6K::new(
            QuantizedView::new(&lhs, packed_layout(14, 256, 1, 210)),
            TensorView::new(&rhs, layout([1, 256, 1, 1])),
            TensorViewMut::new(
                &mut destination,
                Layout::new(DType::F16, [1, 1, 1, 1], [2, 2, 2, 2]),
            ),
        )),
        Err(MatmulError::InvalidView)
    );
    assert_eq!(destination, [9.0]);
}

#[test]
fn q6_k_regular_matmul_dispatch_is_allocation_free() {
    let lhs = q6_k_zero_block();
    let rhs = [1.0_f32; 256];
    let mut destination = [0.0_f32; 1];
    let mut actor = MatmulKernel::new();
    let allocations = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                actor.process_event(OpMulMatQ6K::new(
                    QuantizedView::new(&lhs, packed_layout(14, 256, 1, 210)),
                    TensorView::new(&rhs, layout([1, 256, 1, 1])),
                    TensorViewMut::new(&mut destination, layout([1, 1, 1, 1])),
                )),
                Ok(())
            );
        }
    });
    assert_eq!(allocations.count_total, 0);
    assert_eq!(allocations.bytes_total, 0);
}

#[test]
fn guards_reject_shape_and_view_without_writing() {
    let lhs_layout = layout([3, 2, 1, 1]);
    let rhs_layout = layout([2, 3, 1, 1]);
    let destination_layout = layout([2, 2, 1, 1]);
    let lhs = [1.0_f32; 6];
    let rhs = [1.0_f32; 6];
    let mut destination = [9.0_f32; 6];
    let mut actor = MatmulKernel::new();

    let mismatch = layout([3, 2, 1, 1]);
    assert_eq!(
        actor.process_event(OpMulMat::new(
            TensorView::new(&lhs, lhs_layout),
            TensorView::new(&rhs, rhs_layout),
            TensorViewMut::new(&mut destination, mismatch),
        )),
        Err(MatmulError::ShapeMismatch)
    );
    assert_eq!(destination, [9.0; 6]);

    let invalid_rhs = Layout::new(DType::F16, [2, 3, 1, 1], [4, 8, 24, 24]);
    assert_eq!(
        actor.process_event(OpMulMat::new(
            TensorView::new(&lhs, lhs_layout),
            TensorView::new(&rhs, invalid_rhs),
            TensorViewMut::new(&mut destination, destination_layout),
        )),
        Err(MatmulError::InvalidView)
    );
    assert_eq!(destination, [9.0; 6]);
}

#[test]
fn f32_matmul_argmax_returns_index_and_writes_best_score() {
    let lhs_layout = layout([3, 2, 1, 1]);
    let rhs_layout = layout([1, 3, 1, 1]);
    let destination_layout = layout([1, 1, 1, 1]);
    let lhs = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs = [1.0_f32, 1.0, 1.0];
    let mut destination = [0.0_f32];
    let mut actor = MatmulArgmaxKernel::new();

    assert_eq!(
        actor.process_event(OpMulMatArgmax::new(
            TensorView::new(&lhs, lhs_layout),
            TensorView::new(&rhs, rhs_layout),
            TensorViewMut::new(&mut destination, destination_layout),
        )),
        Ok(1)
    );
    assert_eq!(destination, [15.0]);
}

#[test]
fn matmul_argmax_rejects_invalid_shape_without_writing() {
    let lhs_layout = layout([3, 2, 1, 1]);
    let rhs_layout = layout([1, 2, 1, 1]);
    let destination_layout = layout([1, 1, 1, 1]);
    let lhs = [1.0_f32; 6];
    let rhs = [1.0_f32; 2];
    let mut destination = [9.0_f32];
    let mut actor = MatmulArgmaxKernel::new();

    assert_eq!(
        actor.process_event(OpMulMatArgmax::new(
            TensorView::new(&lhs, lhs_layout),
            TensorView::new(&rhs, rhs_layout),
            TensorViewMut::new(&mut destination, destination_layout),
        )),
        Err(MatmulError::ShapeMismatch)
    );
    assert_eq!(destination, [9.0]);

    let legacy_rhs_layout = layout([3, 1, 1, 1]);
    assert_eq!(
        actor.process_event(OpMulMatArgmax::new(
            TensorView::new(&lhs, lhs_layout),
            TensorView::new(&[1.0_f32; 3], legacy_rhs_layout),
            TensorViewMut::new(&mut destination, destination_layout),
        )),
        Err(MatmulError::ShapeMismatch)
    );
    assert_eq!(destination, [9.0]);
}

#[test]
fn matmul_argmax_rejects_invalid_view_and_unexpected_event() {
    let lhs_layout = layout([3, 2, 1, 1]);
    let rhs_layout = Layout::new(DType::F16, [1, 3, 1, 1], [4, 4, 12, 12]);
    let destination_layout = layout([1, 1, 1, 1]);
    let lhs = [1.0_f32; 6];
    let rhs = [1.0_f32; 3];
    let mut destination = [9.0_f32];
    let mut actor = MatmulArgmaxKernel::new();

    assert_eq!(
        actor.process_event(OpMulMatArgmax::new(
            TensorView::new(&lhs, lhs_layout),
            TensorView::new(&rhs, rhs_layout),
            TensorViewMut::new(&mut destination, destination_layout),
        )),
        Err(MatmulError::InvalidView)
    );
    assert_eq!(destination, [9.0]);

    let strided_rhs_layout = Layout::new(DType::F32, [1, 3, 1, 1], [4, 8, 24, 24]);
    assert_eq!(
        actor.process_event(OpMulMatArgmax::new(
            TensorView::new(&lhs, lhs_layout),
            TensorView::new(&[1.0_f32; 5], strided_rhs_layout),
            TensorViewMut::new(&mut destination, destination_layout),
        )),
        Err(MatmulError::InvalidView)
    );
    assert_eq!(destination, [9.0]);

    assert_eq!(
        actor.process_event(UnexpectedMatmul),
        Err(MatmulError::UnexpectedEvent)
    );
}

#[test]
fn unexpected_event_is_typed_and_actor_recovers() {
    let lhs_layout = layout([1, 1, 1, 1]);
    let rhs_layout = layout([1, 1, 1, 1]);
    let mut destination = [0.0_f32; 1];
    let lhs = [2.0_f32];
    let rhs = [3.0_f32];
    let mut actor = MatmulKernel::new();

    assert_eq!(
        actor.process_event(UnexpectedMatmul),
        Err(MatmulError::UnexpectedEvent)
    );
    assert_eq!(
        actor.process_event(OpMulMat::new(
            TensorView::new(&lhs, lhs_layout),
            TensorView::new(&rhs, rhs_layout),
            TensorViewMut::new(&mut destination, lhs_layout),
        )),
        Ok(())
    );
    assert_eq!(destination, [6.0]);
}

#[test]
fn dispatch_is_allocation_free() {
    let lhs_layout = layout([3, 2, 1, 1]);
    let rhs_layout = layout([2, 3, 1, 1]);
    let destination_layout = layout([2, 2, 1, 1]);
    let lhs = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs = [7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut destination = [0.0_f32; 4];
    let mut actor = MatmulKernel::new();

    let allocation = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                actor.process_event(OpMulMat::new(
                    TensorView::new(&lhs, lhs_layout),
                    TensorView::new(&rhs, rhs_layout),
                    TensorViewMut::new(&mut destination, destination_layout),
                )),
                Ok(())
            );
        }
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(allocation.bytes_total, 0);
}

#[test]
fn matmul_argmax_dispatch_is_allocation_free() {
    let lhs_layout = layout([3, 2, 1, 1]);
    let rhs_layout = layout([1, 3, 1, 1]);
    let destination_layout = layout([1, 1, 1, 1]);
    let lhs = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs = [1.0_f32, 1.0, 1.0];
    let mut destination = [0.0_f32];
    let mut actor = MatmulArgmaxKernel::new();
    let allocation = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                actor.process_event(OpMulMatArgmax::new(
                    TensorView::new(&lhs, lhs_layout),
                    TensorView::new(&rhs, rhs_layout),
                    TensorViewMut::new(&mut destination, destination_layout),
                )),
                Ok(1)
            );
        }
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(allocation.bytes_total, 0);
}

#[test]
fn public_surfaces_are_typed_and_debuggable() {
    assert!(format!("{}", MatmulError::InvalidView).contains("view"));
    assert!(format!("{}", MatmulError::ShapeMismatch).contains("shapes"));
    assert!(format!("{}", MatmulError::UnexpectedEvent).contains("unexpected"));
    assert!(format!("{:?}", MatmulKernel::default()).contains("MatmulKernel"));
}

#[test]
fn q5_0_and_q8_0_argmax_use_packed_rows_and_first_tie() {
    let rhs_layout = layout([1, 32, 1, 1]);
    let destination_layout = layout([1, 1, 1, 1]);
    let rhs = [1.0_f32; 32];
    let mut destination = [0.0_f32];
    let lhs_q5 = [q5_0_constant_block(0x11), q5_0_constant_block(0x12)].concat();
    let lhs_q5_layout = packed_layout(6, 32, 2, 22);
    let mut actor = MatmulArgmaxKernel::new();
    assert_eq!(
        actor.process_event(OpMulMatArgmaxQ5_0::new(
            QuantizedView::new(&lhs_q5, lhs_q5_layout),
            TensorView::new(&rhs, rhs_layout),
            TensorViewMut::new(&mut destination, destination_layout),
        )),
        Ok(1)
    );
    assert!(destination[0] > 0.0);

    let lhs_q8 = [q8_0_constant_block(1), q8_0_constant_block(2)].concat();
    let lhs_q8_layout = packed_layout(8, 32, 2, 34);
    assert_eq!(
        actor.process_event(OpMulMatArgmaxQ8_0::new(
            QuantizedView::new(&lhs_q8, lhs_q8_layout),
            TensorView::new(&rhs, rhs_layout),
            TensorViewMut::new(&mut destination, destination_layout),
        )),
        Ok(1)
    );
    let allocation = measure(|| {
        for _ in 0..32 {
            assert_eq!(
                actor.process_event(OpMulMatArgmaxQ8_0::new(
                    QuantizedView::new(&lhs_q8, lhs_q8_layout),
                    TensorView::new(&rhs, rhs_layout),
                    TensorViewMut::new(&mut destination, destination_layout),
                )),
                Ok(1)
            );
        }
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(allocation.bytes_total, 0);
}

#[test]
fn q4_0_and_q4_1_argmax_use_packed_rows_and_first_tie() {
    let rhs_layout = layout([1, 32, 1, 1]);
    let destination_layout = layout([1, 1, 1, 1]);
    let rhs = [1.0_f32; 32];
    let mut destination = [0.0_f32];
    let mut actor = MatmulArgmaxKernel::new();

    let lhs_q4_0 = [q4_0_constant_block(0x09), q4_0_constant_block(0x0a)].concat();
    assert_eq!(
        actor.process_event(OpMulMatArgmaxQ4_0::new(
            QuantizedView::new(&lhs_q4_0, packed_layout(2, 32, 2, 18)),
            TensorView::new(&rhs, rhs_layout),
            TensorViewMut::new(&mut destination, destination_layout),
        )),
        Ok(1)
    );
    assert!(destination[0] > 0.0);

    let lhs_q4_1 = [q4_1_constant_block(1), q4_1_constant_block(2)].concat();
    assert_eq!(
        actor.process_event(OpMulMatArgmaxQ4_1::new(
            QuantizedView::new(&lhs_q4_1, packed_layout(3, 32, 2, 20)),
            TensorView::new(&rhs, rhs_layout),
            TensorViewMut::new(&mut destination, destination_layout),
        )),
        Ok(1)
    );
    assert!(destination[0] > 0.0);

    let allocation = measure(|| {
        for _ in 0..32 {
            assert_eq!(
                actor.process_event(OpMulMatArgmaxQ4_1::new(
                    QuantizedView::new(&lhs_q4_1, packed_layout(3, 32, 2, 20)),
                    TensorView::new(&rhs, rhs_layout),
                    TensorViewMut::new(&mut destination, destination_layout),
                )),
                Ok(1)
            );
        }
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(allocation.bytes_total, 0);
}

#[test]
fn q4_argmax_rejects_metadata_without_mutation() {
    let rhs = [1.0_f32; 32];
    let rhs_layout = layout([1, 32, 1, 1]);
    let mut destination = [9.0_f32];
    let destination_layout = layout([1, 1, 1, 1]);
    let lhs = [0_u8; 20];
    let mut actor = MatmulArgmaxKernel::new();

    assert_eq!(
        actor.process_event(OpMulMatArgmaxQ4_1::new(
            QuantizedView::new(&lhs, packed_layout(3, 32, 1, 21)),
            TensorView::new(&rhs, rhs_layout),
            TensorViewMut::new(&mut destination, destination_layout),
        )),
        Err(MatmulError::InvalidView)
    );
    assert_eq!(destination, [9.0]);
}

#[test]
fn q_k_argmax_routes_all_native_families_without_allocation() {
    let rhs_layout = layout([1, 256, 1, 1]);
    let destination_layout = layout([1, 1, 1, 1]);
    let rhs = [0.0_f32; 256];
    let mut destination = [7.0_f32];
    let mut actor = MatmulArgmaxKernel::new();

    let q2 = [0_u8; 84 * 2];
    assert_eq!(
        actor.process_event(OpMulMatArgmaxQ2K::new(
            QuantizedView::new(&q2, packed_layout(10, 256, 2, 84)),
            TensorView::new(&rhs, rhs_layout),
            TensorViewMut::new(&mut destination, destination_layout),
        )),
        Ok(0)
    );
    let q3 = [0_u8; 110 * 2];
    assert_eq!(
        actor.process_event(OpMulMatArgmaxQ3K::new(
            QuantizedView::new(&q3, packed_layout(11, 256, 2, 110)),
            TensorView::new(&rhs, rhs_layout),
            TensorViewMut::new(&mut destination, destination_layout),
        )),
        Ok(0)
    );
    let q4 = [0_u8; 144 * 2];
    assert_eq!(
        actor.process_event(OpMulMatArgmaxQ4K::new(
            QuantizedView::new(&q4, packed_layout(12, 256, 2, 144)),
            TensorView::new(&rhs, rhs_layout),
            TensorViewMut::new(&mut destination, destination_layout),
        )),
        Ok(0)
    );
    let q6 = [0_u8; 210 * 2];
    assert_eq!(
        actor.process_event(OpMulMatArgmaxQ6K::new(
            QuantizedView::new(&q6, packed_layout(14, 256, 2, 210)),
            TensorView::new(&rhs, rhs_layout),
            TensorViewMut::new(&mut destination, destination_layout),
        )),
        Ok(0)
    );
}

#[test]
fn packed_argmax_rejects_metadata_and_recovers_without_mutation() {
    let rhs = [1.0_f32; 32];
    let rhs_layout = layout([1, 32, 1, 1]);
    let mut destination = [9.0_f32];
    let destination_layout = layout([1, 1, 1, 1]);
    let lhs = [0_u8; 22];
    let invalid_layout = packed_layout(6, 32, 1, 23);
    let mut actor = MatmulArgmaxKernel::new();
    assert_eq!(
        actor.process_event(OpMulMatArgmaxQ5_0::new(
            QuantizedView::new(&lhs, invalid_layout),
            TensorView::new(&rhs, rhs_layout),
            TensorViewMut::new(&mut destination, destination_layout),
        )),
        Err(MatmulError::InvalidView)
    );
    assert_eq!(destination, [9.0]);
    assert_eq!(
        actor.process_event(UnexpectedMatmul),
        Err(MatmulError::UnexpectedEvent)
    );
}
