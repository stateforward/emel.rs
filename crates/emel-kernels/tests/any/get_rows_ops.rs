#![allow(clippy::float_cmp, missing_docs)]

use allocation_counter::measure;
use emel_kernels::any::f16_matmul::F16View;
use emel_kernels::any::get_rows::{
    Bf16View, GetRowsError, GetRowsKernel, IndexView, OpGetRows, OpGetRowsBf16, OpGetRowsF16,
    OpGetRowsF32Bytes, OpGetRowsQ4_0, OpGetRowsQ4K, OpGetRowsQ8_0, UnexpectedGetRows,
};
use emel_kernels::any::matmul::QuantizedView;
use emel_kernels::any::quant::{Q4_0_BLOCK_BYTES, Q8_0_BLOCK_BYTES};
use emel_kernels::any::quant_more::{Q4_K_BLOCK_BYTES, QK_K_VALUES};
use emel_kernels::any::tensor_view::{ByteTensorView, DType, Layout, TensorView, TensorViewMut};
use emel_tensor as _;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use pulp as _;
use sml as _;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

const PINNED_EMEL_CPP_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const DETAIL_BLOB: &str = "c8a82643eabfe8f2d7883e655955f455794511b0";
const X86_SM_BLOB: &str = "0b4d635ebbd0fbd52dbca8a2345547fb571205c8";
const X86_GUARDS_BLOB: &str = "cb3dac8253f8417c9b44acff1de414f6d0a3a3cf";
const PINNED_SOURCE_DIR: &str =
    "emel.cpp-worktrees/pinned-843a117386ef17dc5a50549bbfc821074c2141d6";
const SNAPSHOT_MANIFEST: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../snapshots/parity/kernel-get-rows/manifest.txt"
));

fn contiguous(ne: [u64; 4]) -> Layout {
    Layout::contiguous(DType::F32, ne).expect("test layout fits")
}

fn manifest_field(name: &str) -> &'static str {
    SNAPSHOT_MANIFEST
        .lines()
        .find_map(|line| line.strip_prefix(name)?.strip_prefix('='))
        .unwrap_or_else(|| panic!("manifest is missing {name}"))
}

fn pinned_source_file(relative: &str) -> std::path::PathBuf {
    let root = std::env::var_os("EMEL_CPP_PINNED_ROOT").map_or_else(
        || {
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../..")
                .join(PINNED_SOURCE_DIR)
        },
        std::path::PathBuf::from,
    );
    root.join(relative)
}

fn sha256_file(relative: &str) -> String {
    let bytes = fs::read(pinned_source_file(relative)).expect("pinned source file must exist");
    let digest = sha256(&bytes);
    let mut result = String::with_capacity(64);
    for byte in digest {
        write!(&mut result, "{byte:02x}").expect("writing to a String cannot fail");
    }
    result
}

#[allow(clippy::chunks_exact_to_as_chunks, clippy::too_many_lines)]
fn sha256(bytes: &[u8]) -> [u8; 32] {
    const ROUND_CONSTANTS: [u32; 64] = [
        0x428a_2f98,
        0x7137_4491,
        0xb5c0_fbcf,
        0xe9b5_dba5,
        0x3956_c25b,
        0x59f1_11f1,
        0x923f_82a4,
        0xab1c_5ed5,
        0xd807_aa98,
        0x1283_5b01,
        0x2431_85be,
        0x550c_7dc3,
        0x72be_5d74,
        0x80de_b1fe,
        0x9bdc_06a7,
        0xc19b_f174,
        0xe49b_69c1,
        0xefbe_4786,
        0x0fc1_9dc6,
        0x240c_a1cc,
        0x2de9_2c6f,
        0x4a74_84aa,
        0x5cb0_a9dc,
        0x76f9_88da,
        0x983e_5152,
        0xa831_c66d,
        0xb003_27c8,
        0xbf59_7fc7,
        0xc6e0_0bf3,
        0xd5a7_9147,
        0x06ca_6351,
        0x1429_2967,
        0x27b7_0a85,
        0x2e1b_2138,
        0x4d2c_6dfc,
        0x5338_0d13,
        0x650a_7354,
        0x766a_0abb,
        0x81c2_c92e,
        0x9272_2c85,
        0xa2bf_e8a1,
        0xa81a_664b,
        0xc24b_8b70,
        0xc76c_51a3,
        0xd192_e819,
        0xd699_0624,
        0xf40e_3585,
        0x106a_a070,
        0x19a4_c116,
        0x1e37_6c08,
        0x2748_774c,
        0x34b0_bcb5,
        0x391c_0cb3,
        0x4ed8_aa4a,
        0x5b9c_ca4f,
        0x682e_6ff3,
        0x748f_82ee,
        0x78a5_636f,
        0x84c8_7814,
        0x8cc7_0208,
        0x90be_fffa,
        0xa450_6ceb,
        0xbef9_a3f7,
        0xc671_78f2,
    ];
    let mut padded = Vec::with_capacity((bytes.len() + 9).div_ceil(64) * 64);
    padded.extend_from_slice(bytes);
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    let bit_length = u64::try_from(bytes.len())
        .expect("test source length fits u64")
        .saturating_mul(8);
    padded.extend_from_slice(&bit_length.to_be_bytes());

    let mut state: [u32; 8] = [
        0x6a09_e667,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];
    for chunk in padded.chunks_exact(64) {
        let mut schedule = [0_u32; 64];
        for (index, word) in schedule[..16].iter_mut().enumerate() {
            let offset = index * 4;
            *word = u32::from_be_bytes([
                chunk[offset],
                chunk[offset + 1],
                chunk[offset + 2],
                chunk[offset + 3],
            ]);
        }
        for index in 16..64 {
            let first = schedule[index - 15];
            let second = schedule[index - 2];
            let sigma0 = first.rotate_right(7) ^ first.rotate_right(18) ^ (first >> 3);
            let sigma1 = second.rotate_right(17) ^ second.rotate_right(19) ^ (second >> 10);
            schedule[index] = schedule[index - 16]
                .wrapping_add(sigma0)
                .wrapping_add(schedule[index - 7])
                .wrapping_add(sigma1);
        }
        let [
            mut state_a,
            mut state_b,
            mut state_c,
            mut state_d,
            mut state_e,
            mut state_f,
            mut state_g,
            mut state_h,
        ] = state;
        for index in 0..64 {
            let sigma1 =
                state_e.rotate_right(6) ^ state_e.rotate_right(11) ^ state_e.rotate_right(25);
            let choice = (state_e & state_f) ^ ((!state_e) & state_g);
            let temporary1 = state_h
                .wrapping_add(sigma1)
                .wrapping_add(choice)
                .wrapping_add(ROUND_CONSTANTS[index])
                .wrapping_add(schedule[index]);
            let sigma0 =
                state_a.rotate_right(2) ^ state_a.rotate_right(13) ^ state_a.rotate_right(22);
            let majority = (state_a & state_b) ^ (state_a & state_c) ^ (state_b & state_c);
            let temporary2 = sigma0.wrapping_add(majority);
            state_h = state_g;
            state_g = state_f;
            state_f = state_e;
            state_e = state_d.wrapping_add(temporary1);
            state_d = state_c;
            state_c = state_b;
            state_b = state_a;
            state_a = temporary1.wrapping_add(temporary2);
        }
        state[0] = state[0].wrapping_add(state_a);
        state[1] = state[1].wrapping_add(state_b);
        state[2] = state[2].wrapping_add(state_c);
        state[3] = state[3].wrapping_add(state_d);
        state[4] = state[4].wrapping_add(state_e);
        state[5] = state[5].wrapping_add(state_f);
        state[6] = state[6].wrapping_add(state_g);
        state[7] = state[7].wrapping_add(state_h);
    }
    let mut digest = [0_u8; 32];
    for (index, word) in state.iter().enumerate() {
        digest[index * 4..index * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    digest
}

const fn packed_layout(
    code: u8,
    cols: u64,
    rows: u64,
    block_values: u64,
    block_bytes: u64,
) -> Layout {
    packed_layout_with_outer(code, cols, rows, 1, 1, block_values, block_bytes)
}

const fn packed_layout_with_outer(
    code: u8,
    cols: u64,
    rows: u64,
    outer_2: u64,
    outer_3: u64,
    block_values: u64,
    block_bytes: u64,
) -> Layout {
    let row_bytes = (cols / block_values) * block_bytes;
    let row_stride_2 = row_bytes * rows;
    Layout::new(
        DType::Other(code),
        [cols, rows, outer_2, outer_3],
        [1, row_bytes, row_stride_2, row_stride_2 * outer_2],
    )
}

fn q4_0_block(scale: u16, packed: u8) -> [u8; Q4_0_BLOCK_BYTES] {
    let mut block = [0_u8; Q4_0_BLOCK_BYTES];
    block[..2].copy_from_slice(&scale.to_le_bytes());
    block[2..].fill(packed);
    block
}

fn q8_0_block(scale: u16, value: i8) -> [u8; Q8_0_BLOCK_BYTES] {
    let mut block = [0_u8; Q8_0_BLOCK_BYTES];
    block[..2].copy_from_slice(&scale.to_le_bytes());
    block[2..].fill(value.to_ne_bytes()[0]);
    block
}

fn q4_k_block(scale: u16, minimum: u16, packed: u8) -> [u8; Q4_K_BLOCK_BYTES] {
    q4_k_block_with_scales(
        scale,
        minimum,
        [1, 1, 1, 1, 1, 1, 1, 1, 0x11, 0x11, 0x11, 0x11],
        packed,
    )
}

fn q4_k_block_with_scales(
    scale: u16,
    minimum: u16,
    scales: [u8; 12],
    packed: u8,
) -> [u8; Q4_K_BLOCK_BYTES] {
    let mut block = [0_u8; Q4_K_BLOCK_BYTES];
    block[..2].copy_from_slice(&scale.to_le_bytes());
    block[2..4].copy_from_slice(&minimum.to_le_bytes());
    block[4..16].copy_from_slice(&scales);
    block[16..].fill(packed);
    block
}

fn q4_0_multiblock_source(rows: usize, outer_2: usize, outer_3: usize) -> Vec<u8> {
    let mut source = Vec::new();
    for row_id in 0..rows * outer_2 * outer_3 {
        for block in 0..2 {
            let low = (8 + row_id + block) & 0x0f;
            let high = (9 + row_id + block) & 0x0f;
            let packed = u8::try_from(low | (high << 4)).expect("test nibble pair fits u8");
            source.extend_from_slice(&q4_0_block(0x3c00, packed));
        }
    }
    source
}

fn q8_0_multiblock_source(rows: usize, outer_2: usize, outer_3: usize) -> Vec<u8> {
    let mut source = Vec::new();
    for row_id in 0..rows * outer_2 * outer_3 {
        for block in 0..2 {
            let value = i8::try_from(row_id * 3 + block).expect("test q8 value fits i8");
            source.extend_from_slice(&q8_0_block(0x3c00, value));
        }
    }
    source
}

fn q4_k_multiblock_source(rows: usize, outer_2: usize, outer_3: usize) -> Vec<u8> {
    let mut source = Vec::new();
    for row_id in 0..rows * outer_2 * outer_3 {
        for block in 0..2 {
            let low = u8::try_from(1 + row_id + block).expect("test q4_k low fits u8");
            let high = u8::try_from(2 + row_id + block).expect("test q4_k high fits u8");
            let packed = low | (high << 4);
            source.extend_from_slice(&q4_k_block(0x3c00, 0x3800, packed));
        }
    }
    source
}

fn assert_sentinel(output: &[f32], sentinel: f32) {
    assert!(output.iter().all(|value| *value == sentinel));
}

#[test]
fn source_identity_matches_pinned_get_rows_contract() {
    for (field, expected) in [
        ("source_repository", "stateforward/emel.cpp"),
        ("source_commit", PINNED_EMEL_CPP_COMMIT),
        ("source_kernel_detail_blob", DETAIL_BLOB),
        ("source_kernel_x86_sm_blob", X86_SM_BLOB),
        ("source_kernel_x86_guards_blob", X86_GUARDS_BLOB),
        ("rust_module", "crates/emel-kernels/src/get_rows_ops.rs"),
        ("rust_tests", "crates/emel-kernels/tests/get_rows_ops.rs"),
        (
            "source_detail_spans",
            "convert_row_to_f32_as:4423-4447,can_run_get_rows:4460-4501,run_get_rows_as:4503-4525",
        ),
        (
            "source_x86_spans",
            "dispatch_op_get_rows_rows:535-568,action_aliases:2608-2623,variant_guards:269-348",
        ),
    ] {
        assert_eq!(manifest_field(field), expected, "manifest field {field}");
    }
}

#[test]
fn snapshot_records_bf16_stride_and_empty_index_scope() {
    let scope = manifest_field("scope");
    assert!(scope.contains("f32_nb0=0_implicit"));
    assert!(scope.contains("byte_backed_unaligned_outer_strides"));
    assert!(scope.contains("bf16_nb0=0_implicit"));
    assert!(manifest_field("fixture_unaligned_outer_f32_source").contains("nb=[4,9,18,18]"));
    assert!(manifest_field("fixture_implicit_bf16_source").contains("effective_nb=[2,4,8,8]"));
    assert!(manifest_field("fixture_singleton_outer_zero_bf16_source").contains("nb=[2,0,0,0]"));
    assert_eq!(
        manifest_field("source_q4_k_decode_spans"),
        "get_scale_min_k4:475-485,dequantize_row_q4_k:523-552"
    );
    assert_eq!(
        manifest_field("q4_k_decoder_proof"),
        "upper_scale_min_bits_and_low_high_nibbles_tested_in_get_rows_ops"
    );
    assert!(manifest_field("invalid_routes").contains("bf16_non_singleton_zero_stride"));
    let residuals = manifest_field("residuals");
    assert!(!residuals.contains("unaligned_outer_byte_strides"));
    assert!(!residuals.contains("bf16_singleton_zero_stride_forms"));
    assert!(!residuals.contains("zero_index_noop_bf16_forms"));
}

#[test]
#[ignore = "requires EMEL_CPP_PINNED_ROOT or the local pinned emel.cpp sibling"]
fn live_pinned_source_hashes_match_manifest() {
    for (field, relative) in [
        ("source_detail_sha256", "src/emel/kernel/detail.hpp"),
        ("source_x86_sm_sha256", "src/emel/kernel/x86_64/sm.hpp"),
        (
            "source_x86_guards_sha256",
            "src/emel/kernel/x86_64/guards.hpp",
        ),
    ] {
        assert_eq!(
            sha256_file(relative),
            manifest_field(field),
            "live pinned source hash for {relative}",
        );
    }
}

#[test]
fn f32_get_rows_matches_pinned_row_and_batch_order() {
    let source_layout = contiguous([3, 2, 2, 1]);
    let source = [
        1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
    ];
    let indices = [1_i32, 0, 0, 1];
    let mut output = [0.0_f32; 12];
    let mut kernel = GetRowsKernel::new();

    assert_eq!(
        kernel.process_event(OpGetRows::new(
            TensorView::new(&source, source_layout),
            IndexView::new(&indices, [2, 2, 1]),
            TensorViewMut::new(&mut output, contiguous([3, 2, 2, 1])),
        )),
        Ok(())
    );
    assert_eq!(
        output,
        [
            4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0
        ]
    );
}

#[test]
fn f32_get_rows_reads_strided_source_rows() {
    let source_layout = Layout::new(DType::F32, [2, 2, 1, 1], [4, 12, 24, 24]);
    let source = [1.0_f32, 2.0, 99.0, 3.0, 4.0];
    let indices = [1_i32, 0];
    let mut output = [0.0_f32; 4];
    let mut kernel = GetRowsKernel::default();

    assert_eq!(
        kernel.process_event(OpGetRows::new(
            TensorView::new(&source, source_layout),
            IndexView::new(&indices, [2, 1, 1]),
            TensorViewMut::new(&mut output, contiguous([2, 2, 1, 1])),
        )),
        Ok(())
    );
    assert_eq!(output, [3.0, 4.0, 1.0, 2.0]);
}

#[test]
fn f32_bytes_get_rows_reads_unaligned_outer_source_rows() {
    let source_layout = Layout::new(DType::F32, [2, 2, 1, 1], [4, 9, 18, 18]);
    let mut source = [0_u8; 17];
    source[0..4].copy_from_slice(&1.0_f32.to_ne_bytes());
    source[4..8].copy_from_slice(&2.0_f32.to_ne_bytes());
    source[9..13].copy_from_slice(&3.0_f32.to_ne_bytes());
    source[13..17].copy_from_slice(&4.0_f32.to_ne_bytes());
    let indices = [1_i32, 0];
    let mut output = [0.0_f32; 4];
    let mut kernel = GetRowsKernel::new();

    assert_eq!(
        kernel.process_event(OpGetRowsF32Bytes::new(
            ByteTensorView::new(&source, source_layout),
            IndexView::new(&indices, [2, 1, 1]),
            TensorViewMut::new(&mut output, contiguous([2, 2, 1, 1])),
        )),
        Ok(())
    );
    assert_eq!(output, [3.0, 4.0, 1.0, 2.0]);
}

#[test]
fn f32_bytes_get_rows_rejects_short_unaligned_outer_storage() {
    let source_layout = Layout::new(DType::F32, [2, 2, 1, 1], [4, 9, 18, 18]);
    let source = [0_u8; 16];
    let indices = [1_i32, 0];
    let mut output = [9.0_f32; 4];
    let mut kernel = GetRowsKernel::new();

    assert_eq!(
        kernel.process_event(OpGetRowsF32Bytes::new(
            ByteTensorView::new(&source, source_layout),
            IndexView::new(&indices, [2, 1, 1]),
            TensorViewMut::new(&mut output, contiguous([2, 2, 1, 1])),
        )),
        Err(GetRowsError::InvalidView)
    );
    assert_eq!(output, [9.0; 4]);
}

#[test]
fn f32_get_rows_accepts_singleton_outer_zero_strides() {
    let source_layout = Layout::new(DType::F32, [2, 1, 1, 1], [4, 0, 0, 0]);
    let source = [1.0_f32, 2.0];
    let indices = [0_i32];
    let mut output = [9.0_f32; 2];
    let mut kernel = GetRowsKernel::new();

    assert_eq!(
        kernel.process_event(OpGetRows::new(
            TensorView::new(&source, source_layout),
            IndexView::new(&indices, [1, 1, 1]),
            TensorViewMut::new(&mut output, contiguous([2, 1, 1, 1])),
        )),
        Ok(())
    );
    assert_eq!(output, [1.0, 2.0]);
}

#[test]
fn f32_get_rows_rejects_non_singleton_zero_stride_without_mutation() {
    let source_layout = Layout::new(DType::F32, [2, 2, 1, 1], [4, 0, 8, 8]);
    let source = [1.0_f32, 2.0];
    let indices = [0_i32];
    let mut output = [9.0_f32; 2];
    let mut kernel = GetRowsKernel::new();

    assert_eq!(
        kernel.process_event(OpGetRows::new(
            TensorView::new(&source, source_layout),
            IndexView::new(&indices, [1, 1, 1]),
            TensorViewMut::new(&mut output, contiguous([2, 1, 1, 1])),
        )),
        Err(GetRowsError::InvalidView)
    );
    assert_eq!(output, [9.0; 2]);
}

#[test]
fn f32_get_rows_reads_non_contiguous_i32_indices() {
    let source_layout = contiguous([3, 2, 2, 1]);
    let source = [
        1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
    ];
    let indices = [1_i32, 0, 99, 99, 0, 1];
    let mut output = [0.0_f32; 12];
    let mut kernel = GetRowsKernel::new();

    assert_eq!(
        kernel.process_event(OpGetRows::new(
            TensorView::new(&source, source_layout),
            IndexView::with_strides(&indices, [2, 2, 1], [4, 16, 32]),
            TensorViewMut::new(&mut output, contiguous([3, 2, 2, 1])),
        )),
        Ok(())
    );
    assert_eq!(
        output,
        [
            4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0
        ]
    );
}

#[test]
fn implicit_i32_stride_sentinel_normalizes_non_singleton_indices() {
    let source_layout = contiguous([3, 2, 2, 1]);
    let source = [
        1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0,
    ];
    let indices = [1_i32, 0, 0, 1];
    let view = IndexView::with_strides(&indices, [2, 2, 1], [0, u64::MAX, 3]);
    assert!(view.metadata_valid());
    let mut output = [0.0_f32; 12];
    let mut kernel = GetRowsKernel::new();

    assert_eq!(
        kernel.process_event(OpGetRows::new(
            TensorView::new(&source, source_layout),
            view,
            TensorViewMut::new(&mut output, contiguous([3, 2, 2, 1])),
        )),
        Ok(())
    );
    assert_eq!(
        output,
        [
            4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0
        ]
    );
}

#[test]
fn implicit_i32_stride_sentinel_accepts_singleton_and_zero_extent_forms() {
    let singleton = IndexView::with_strides(&[0_i32], [1, 1, 1], [0, 99, 99]);
    assert!(singleton.metadata_valid());

    let source_layout = contiguous([3, 2, 2, 1]);
    let source = [0.0_f32; 12];
    let mut output: [f32; 0] = [];
    let zero_extent = IndexView::with_strides(&[], [0, 2, 1], [0, 0, 0]);
    assert!(zero_extent.metadata_valid());
    let mut kernel = GetRowsKernel::new();

    assert_eq!(
        kernel.process_event(OpGetRows::new(
            TensorView::new(&source, source_layout),
            zero_extent,
            TensorViewMut::new(
                &mut output,
                Layout::new(DType::F32, [3, 0, 2, 1], [4, 12, 0, 0])
            ),
        )),
        Ok(())
    );
}

#[test]
fn strided_i32_index_metadata_and_span_errors_are_invalid_views() {
    let source_layout = contiguous([3, 2, 2, 1]);
    let source = [0.0_f32; 12];
    let mut output = [9.0_f32; 12];
    let mut kernel = GetRowsKernel::new();

    for indices in [
        IndexView::with_strides(&[0_i32, 1, 0, 1], [2, 2, 1], [2, 8, 8]),
        IndexView::with_strides(&[0_i32, 1, 0, 1], [2, 2, 1], [4, 0, 8]),
        IndexView::with_strides(
            &[0_i32, 1, 0, 1],
            [2, 2, 1],
            [u64::MAX - 3, u64::MAX - 3, 4],
        ),
        IndexView::with_strides(&[0_i32, 1], [2, 2, 1], [4, 16, 32]),
    ] {
        assert_eq!(
            kernel.process_event(OpGetRows::new(
                TensorView::new(&source, source_layout),
                indices,
                TensorViewMut::new(&mut output, contiguous([3, 2, 2, 1])),
            )),
            Err(GetRowsError::InvalidView)
        );
        assert_eq!(output, [9.0; 12]);
    }
}

#[test]
fn q4_0_get_rows_matches_pinned_dequantized_row_order() {
    let source_layout = packed_layout(2, 32, 2, 32, Q4_0_BLOCK_BYTES as u64);
    let source = [q4_0_block(0x3c00, 0x98), q4_0_block(0x3c00, 0xa7)].concat();
    let indices = [1_i32, 0];
    let mut output = [0.0_f32; 64];
    let mut kernel = GetRowsKernel::new();

    assert_eq!(
        kernel.process_event(OpGetRowsQ4_0::new(
            QuantizedView::new(&source, source_layout),
            IndexView::new(&indices, [2, 1, 1]),
            TensorViewMut::new(&mut output, contiguous([32, 2, 1, 1])),
        )),
        Ok(())
    );
    assert_eq!(&output[..16], &[-1.0; 16]);
    assert_eq!(&output[16..32], &[2.0; 16]);
    assert_eq!(&output[32..48], &[0.0; 16]);
    assert_eq!(&output[48..], &[1.0; 16]);
}

#[test]
fn q8_0_get_rows_matches_pinned_dequantized_row_order() {
    let source_layout = packed_layout(8, 32, 2, 32, Q8_0_BLOCK_BYTES as u64);
    let source = [q8_0_block(0x3c00, -2), q8_0_block(0x3c00, 1)].concat();
    let indices = [1_i32, 0];
    let mut output = [0.0_f32; 64];
    let mut kernel = GetRowsKernel::new();

    assert_eq!(
        kernel.process_event(OpGetRowsQ8_0::new(
            QuantizedView::new(&source, source_layout),
            IndexView::new(&indices, [2, 1, 1]),
            TensorViewMut::new(&mut output, contiguous([32, 2, 1, 1])),
        )),
        Ok(())
    );
    assert_eq!(&output[..32], &[1.0; 32]);
    assert_eq!(&output[32..], &[-2.0; 32]);
}

#[test]
fn q4_k_get_rows_matches_pinned_dequantized_row_order() {
    let source_layout = packed_layout(
        12,
        QK_K_VALUES as u64,
        2,
        QK_K_VALUES as u64,
        Q4_K_BLOCK_BYTES as u64,
    );
    let source = [
        q4_k_block(0x3c00, 0x3800, 0x43),
        q4_k_block(0x3c00, 0x3800, 0x21),
    ]
    .concat();
    let indices = [1_i32, 0];
    let mut output = [0.0_f32; QK_K_VALUES * 2];
    let mut kernel = GetRowsKernel::new();

    assert_eq!(
        kernel.process_event(OpGetRowsQ4K::new(
            QuantizedView::new(&source, source_layout),
            IndexView::new(&indices, [2, 1, 1]),
            TensorViewMut::new(&mut output, contiguous([QK_K_VALUES as u64, 2, 1, 1])),
        )),
        Ok(())
    );
    for chunk in output[..QK_K_VALUES].as_chunks::<64>().0 {
        assert_eq!(&chunk[..32], &[0.5; 32]);
        assert_eq!(&chunk[32..], &[1.5; 32]);
    }
    for chunk in output[QK_K_VALUES..].as_chunks::<64>().0 {
        assert_eq!(&chunk[..32], &[2.5; 32]);
        assert_eq!(&chunk[32..], &[3.5; 32]);
    }
}

#[test]
fn q4_k_get_rows_decodes_upper_scale_min_groups() {
    let source_layout = packed_layout(
        12,
        QK_K_VALUES as u64,
        1,
        QK_K_VALUES as u64,
        Q4_K_BLOCK_BYTES as u64,
    );
    let source = q4_k_block_with_scales(
        0x3c00,
        0x3800,
        [
            0x41, 0x82, 0xc3, 0x04, 0x45, 0x86, 0xc7, 0x08, 0x89, 0x9a, 0xab, 0xbc,
        ],
        0x21,
    );
    let indices = [0_i32];
    let mut output = [0.0_f32; QK_K_VALUES];
    let mut kernel = GetRowsKernel::new();

    assert_eq!(
        kernel.process_event(OpGetRowsQ4K::new(
            QuantizedView::new(&source, source_layout),
            IndexView::new(&indices, [1, 1, 1]),
            TensorViewMut::new(&mut output, contiguous([QK_K_VALUES as u64, 1, 1, 1])),
        )),
        Ok(())
    );

    let expected = [(-1.5, 1.0), (-0.5, 4.0), (13.0, 63.5), (30.0, 18.5)];
    for (group, (low, high)) in expected.into_iter().enumerate() {
        let chunk = &output[group * 64..(group + 1) * 64];
        assert_eq!(&chunk[..32], &[low; 32]);
        assert_eq!(&chunk[32..], &[high; 32]);
    }
}

fn q4_0_multiblock_outer_case(indices: &[i32], source_row_ids: &[usize]) {
    let source = q4_0_multiblock_source(2, 2, 2);
    let layout = packed_layout_with_outer(2, 64, 2, 2, 2, 32, Q4_0_BLOCK_BYTES as u64);
    let mut output = [0.0_f32; 64 * 8];
    let mut kernel = GetRowsKernel::new();
    assert_eq!(
        kernel.process_event(OpGetRowsQ4_0::new(
            QuantizedView::new(&source, layout),
            IndexView::new(indices, [2, 2, 2]),
            TensorViewMut::new(&mut output, contiguous([64, 2, 2, 2])),
        )),
        Ok(())
    );
    for (destination_row, source_row) in source_row_ids.iter().copied().enumerate() {
        let row = &output[destination_row * 64..(destination_row + 1) * 64];
        for block in 0..2 {
            let low =
                f32::from(u8::try_from((8 + source_row + block) & 0x0f).expect("test nibble fits"))
                    - 8.0;
            let high =
                f32::from(u8::try_from((9 + source_row + block) & 0x0f).expect("test nibble fits"))
                    - 8.0;
            assert_eq!(&row[block * 32..block * 32 + 16], &[low; 16]);
            assert_eq!(&row[block * 32 + 16..block * 32 + 32], &[high; 16]);
        }
    }
}

fn q8_0_multiblock_outer_case(indices: &[i32], source_row_ids: &[usize]) {
    let source = q8_0_multiblock_source(2, 2, 2);
    let layout = packed_layout_with_outer(8, 64, 2, 2, 2, 32, Q8_0_BLOCK_BYTES as u64);
    let mut output = [0.0_f32; 64 * 8];
    let mut kernel = GetRowsKernel::new();
    assert_eq!(
        kernel.process_event(OpGetRowsQ8_0::new(
            QuantizedView::new(&source, layout),
            IndexView::new(indices, [2, 2, 2]),
            TensorViewMut::new(&mut output, contiguous([64, 2, 2, 2])),
        )),
        Ok(())
    );
    for (destination_row, source_row) in source_row_ids.iter().copied().enumerate() {
        let row = &output[destination_row * 64..(destination_row + 1) * 64];
        for block in 0..2 {
            let expected =
                f32::from(i8::try_from(source_row * 3 + block).expect("test q8 value fits"));
            assert_eq!(&row[block * 32..block * 32 + 32], &[expected; 32]);
        }
    }
}

fn q4_k_multiblock_outer_case(indices: &[i32], source_row_ids: &[usize]) {
    let source = q4_k_multiblock_source(2, 2, 2);
    let layout = packed_layout_with_outer(
        12,
        (QK_K_VALUES * 2) as u64,
        2,
        2,
        2,
        QK_K_VALUES as u64,
        Q4_K_BLOCK_BYTES as u64,
    );
    let mut output = [0.0_f32; QK_K_VALUES * 2 * 8];
    let mut kernel = GetRowsKernel::new();
    assert_eq!(
        kernel.process_event(OpGetRowsQ4K::new(
            QuantizedView::new(&source, layout),
            IndexView::new(indices, [2, 2, 2]),
            TensorViewMut::new(&mut output, contiguous([(QK_K_VALUES * 2) as u64, 2, 2, 2]),),
        )),
        Ok(())
    );
    for (destination_row, source_row) in source_row_ids.iter().copied().enumerate() {
        let row =
            &output[destination_row * QK_K_VALUES * 2..(destination_row + 1) * QK_K_VALUES * 2];
        for block in 0..2 {
            let low =
                f32::from(u8::try_from(1 + source_row + block).expect("test q4_k low fits")) - 0.5;
            let high =
                f32::from(u8::try_from(2 + source_row + block).expect("test q4_k high fits")) - 0.5;
            for group in 0..4 {
                let offset = block * QK_K_VALUES + group * 64;
                assert_eq!(&row[offset..offset + 32], &[low; 32]);
                assert_eq!(&row[offset + 32..offset + 64], &[high; 32]);
            }
        }
    }
}

#[test]
fn quantized_get_rows_covers_multiple_blocks_and_outer_batches() {
    let indices = [0_i32, 1, 1, 0, 0, 1, 1, 0];
    let source_row_ids = [0_usize, 1, 3, 2, 4, 5, 7, 6];
    q4_0_multiblock_outer_case(&indices, &source_row_ids);
    q8_0_multiblock_outer_case(&indices, &source_row_ids);
    q4_k_multiblock_outer_case(&indices, &source_row_ids);
}

#[test]
fn quantized_get_rows_empty_indices_are_metadata_only_noops() {
    let indices = IndexView::new(&[], [0, 1, 1]);
    let mut kernel = GetRowsKernel::new();

    let q4_source_layout = packed_layout(2, 32, 1, 32, Q4_0_BLOCK_BYTES as u64);
    let q4_source = q4_0_block(0x3c00, 0x98);
    let mut q4_output: [f32; 0] = [];
    assert_eq!(
        kernel.process_event(OpGetRowsQ4_0::new(
            QuantizedView::new(&q4_source, q4_source_layout),
            indices,
            TensorViewMut::new(&mut q4_output, contiguous([32, 0, 1, 1])),
        )),
        Ok(())
    );

    let q8_source_layout = packed_layout(8, 32, 1, 32, Q8_0_BLOCK_BYTES as u64);
    let q8_source = q8_0_block(0x3c00, 1);
    let mut q8_output: [f32; 0] = [];
    assert_eq!(
        kernel.process_event(OpGetRowsQ8_0::new(
            QuantizedView::new(&q8_source, q8_source_layout),
            indices,
            TensorViewMut::new(&mut q8_output, contiguous([32, 0, 1, 1])),
        )),
        Ok(())
    );

    let q4_k_source_layout = packed_layout(
        12,
        QK_K_VALUES as u64,
        1,
        QK_K_VALUES as u64,
        Q4_K_BLOCK_BYTES as u64,
    );
    let q4_k_source = q4_k_block(0x3c00, 0x3800, 0x21);
    let mut q4_k_output: [f32; 0] = [];
    assert_eq!(
        kernel.process_event(OpGetRowsQ4K::new(
            QuantizedView::new(&q4_k_source, q4_k_source_layout),
            indices,
            TensorViewMut::new(&mut q4_k_output, contiguous([QK_K_VALUES as u64, 0, 1, 1])),
        )),
        Ok(())
    );
}

#[test]
fn q4_0_get_rows_rejects_malformed_views_and_negative_index_without_mutation() {
    let source_layout = packed_layout(2, 32, 1, 32, Q4_0_BLOCK_BYTES as u64);
    let source = q4_0_block(0x3c00, 0x98);
    let mut output = [7.0_f32; 32];
    let mut kernel = GetRowsKernel::new();
    let malformed_layout = Layout::new(DType::Other(2), [32, 1, 1, 1], [2, 18, 18, 18]);

    assert_eq!(
        kernel.process_event(OpGetRowsQ4_0::new(
            QuantizedView::new(&source, malformed_layout),
            IndexView::new(&[0_i32], [1, 1, 1]),
            TensorViewMut::new(&mut output, contiguous([32, 1, 1, 1])),
        )),
        Err(GetRowsError::InvalidView)
    );
    assert_sentinel(&output, 7.0);

    let short_source = [0_u8; Q4_0_BLOCK_BYTES - 1];
    assert_eq!(
        kernel.process_event(OpGetRowsQ4_0::new(
            QuantizedView::new(&short_source, source_layout),
            IndexView::new(&[0_i32], [1, 1, 1]),
            TensorViewMut::new(&mut output, contiguous([32, 1, 1, 1])),
        )),
        Err(GetRowsError::InvalidView)
    );
    assert_sentinel(&output, 7.0);

    let non_dense_destination = Layout::new(DType::F32, [32, 1, 1, 1], [8, 256, 256, 256]);
    assert_eq!(
        kernel.process_event(OpGetRowsQ4_0::new(
            QuantizedView::new(&source, source_layout),
            IndexView::new(&[0_i32], [1, 1, 1]),
            TensorViewMut::new(&mut output, non_dense_destination),
        )),
        Err(GetRowsError::InvalidView)
    );
    assert_sentinel(&output, 7.0);

    assert_eq!(
        kernel.process_event(OpGetRowsQ4_0::new(
            QuantizedView::new(&source, source_layout),
            IndexView::new(&[-1_i32], [1, 1, 1]),
            TensorViewMut::new(&mut output, contiguous([32, 1, 1, 1])),
        )),
        Err(GetRowsError::IndexOutOfBounds)
    );
    assert_sentinel(&output, 7.0);
}

#[test]
fn q8_0_get_rows_rejects_malformed_views_and_negative_index_without_mutation() {
    let source_layout = packed_layout(8, 32, 1, 32, Q8_0_BLOCK_BYTES as u64);
    let source = q8_0_block(0x3c00, 1);
    let mut output = [7.0_f32; 32];
    let mut kernel = GetRowsKernel::new();
    let malformed_layout = Layout::new(DType::Other(8), [32, 1, 1, 1], [2, 34, 34, 34]);

    assert_eq!(
        kernel.process_event(OpGetRowsQ8_0::new(
            QuantizedView::new(&source, malformed_layout),
            IndexView::new(&[0_i32], [1, 1, 1]),
            TensorViewMut::new(&mut output, contiguous([32, 1, 1, 1])),
        )),
        Err(GetRowsError::InvalidView)
    );
    assert_sentinel(&output, 7.0);

    let short_source = [0_u8; Q8_0_BLOCK_BYTES - 1];
    assert_eq!(
        kernel.process_event(OpGetRowsQ8_0::new(
            QuantizedView::new(&short_source, source_layout),
            IndexView::new(&[0_i32], [1, 1, 1]),
            TensorViewMut::new(&mut output, contiguous([32, 1, 1, 1])),
        )),
        Err(GetRowsError::InvalidView)
    );
    assert_sentinel(&output, 7.0);

    let non_dense_destination = Layout::new(DType::F32, [32, 1, 1, 1], [8, 256, 256, 256]);
    assert_eq!(
        kernel.process_event(OpGetRowsQ8_0::new(
            QuantizedView::new(&source, source_layout),
            IndexView::new(&[0_i32], [1, 1, 1]),
            TensorViewMut::new(&mut output, non_dense_destination),
        )),
        Err(GetRowsError::InvalidView)
    );
    assert_sentinel(&output, 7.0);

    assert_eq!(
        kernel.process_event(OpGetRowsQ8_0::new(
            QuantizedView::new(&source, source_layout),
            IndexView::new(&[-1_i32], [1, 1, 1]),
            TensorViewMut::new(&mut output, contiguous([32, 1, 1, 1])),
        )),
        Err(GetRowsError::IndexOutOfBounds)
    );
    assert_sentinel(&output, 7.0);
}

#[test]
fn q4_k_get_rows_rejects_malformed_views_and_negative_index_without_mutation() {
    let source_layout = packed_layout(
        12,
        QK_K_VALUES as u64,
        1,
        QK_K_VALUES as u64,
        Q4_K_BLOCK_BYTES as u64,
    );
    let source = q4_k_block(0x3c00, 0x3800, 0x21);
    let mut output = [7.0_f32; QK_K_VALUES];
    let mut kernel = GetRowsKernel::new();
    let malformed_layout = Layout::new(
        DType::Other(12),
        [QK_K_VALUES as u64, 1, 1, 1],
        [
            2,
            Q4_K_BLOCK_BYTES as u64,
            Q4_K_BLOCK_BYTES as u64,
            Q4_K_BLOCK_BYTES as u64,
        ],
    );

    assert_eq!(
        kernel.process_event(OpGetRowsQ4K::new(
            QuantizedView::new(&source, malformed_layout),
            IndexView::new(&[0_i32], [1, 1, 1]),
            TensorViewMut::new(&mut output, contiguous([QK_K_VALUES as u64, 1, 1, 1]),),
        )),
        Err(GetRowsError::InvalidView)
    );
    assert_sentinel(&output, 7.0);

    let short_source = [0_u8; Q4_K_BLOCK_BYTES - 1];
    assert_eq!(
        kernel.process_event(OpGetRowsQ4K::new(
            QuantizedView::new(&short_source, source_layout),
            IndexView::new(&[0_i32], [1, 1, 1]),
            TensorViewMut::new(&mut output, contiguous([QK_K_VALUES as u64, 1, 1, 1]),),
        )),
        Err(GetRowsError::InvalidView)
    );
    assert_sentinel(&output, 7.0);

    let non_dense_destination = Layout::new(
        DType::F32,
        [QK_K_VALUES as u64, 1, 1, 1],
        [
            8,
            (QK_K_VALUES * 8) as u64,
            (QK_K_VALUES * 8) as u64,
            (QK_K_VALUES * 8) as u64,
        ],
    );
    assert_eq!(
        kernel.process_event(OpGetRowsQ4K::new(
            QuantizedView::new(&source, source_layout),
            IndexView::new(&[0_i32], [1, 1, 1]),
            TensorViewMut::new(&mut output, non_dense_destination),
        )),
        Err(GetRowsError::InvalidView)
    );
    assert_sentinel(&output, 7.0);

    assert_eq!(
        kernel.process_event(OpGetRowsQ4K::new(
            QuantizedView::new(&source, source_layout),
            IndexView::new(&[-1_i32], [1, 1, 1]),
            TensorViewMut::new(&mut output, contiguous([QK_K_VALUES as u64, 1, 1, 1]),),
        )),
        Err(GetRowsError::IndexOutOfBounds)
    );
    assert_sentinel(&output, 7.0);
}

#[test]
fn quantized_get_rows_dispatch_is_allocation_free() {
    let indices = [0_i32];
    let mut kernel = GetRowsKernel::new();

    let q4_source_layout = packed_layout(2, 32, 1, 32, Q4_0_BLOCK_BYTES as u64);
    let q4_source = q4_0_block(0x3c00, 0x98);
    let mut q4_output = [0.0_f32; 32];
    let q4_info = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                kernel.process_event(OpGetRowsQ4_0::new(
                    QuantizedView::new(&q4_source, q4_source_layout),
                    IndexView::new(&indices, [1, 1, 1]),
                    TensorViewMut::new(&mut q4_output, contiguous([32, 1, 1, 1])),
                )),
                Ok(())
            );
        }
    });
    assert_eq!(q4_info.count_total, 0);
    assert_eq!(q4_info.bytes_total, 0);

    let q8_source_layout = packed_layout(8, 32, 1, 32, Q8_0_BLOCK_BYTES as u64);
    let q8_source = q8_0_block(0x3c00, 1);
    let mut q8_output = [0.0_f32; 32];
    let q8_info = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                kernel.process_event(OpGetRowsQ8_0::new(
                    QuantizedView::new(&q8_source, q8_source_layout),
                    IndexView::new(&indices, [1, 1, 1]),
                    TensorViewMut::new(&mut q8_output, contiguous([32, 1, 1, 1])),
                )),
                Ok(())
            );
        }
    });
    assert_eq!(q8_info.count_total, 0);
    assert_eq!(q8_info.bytes_total, 0);

    let q4_k_source_layout = packed_layout(
        12,
        QK_K_VALUES as u64,
        1,
        QK_K_VALUES as u64,
        Q4_K_BLOCK_BYTES as u64,
    );
    let q4_k_source = q4_k_block(0x3c00, 0x3800, 0x21);
    let mut q4_k_output = [0.0_f32; QK_K_VALUES];
    let q4_k_info = measure(|| {
        for _ in 0..32 {
            assert_eq!(
                kernel.process_event(OpGetRowsQ4K::new(
                    QuantizedView::new(&q4_k_source, q4_k_source_layout),
                    IndexView::new(&indices, [1, 1, 1]),
                    TensorViewMut::new(
                        &mut q4_k_output,
                        contiguous([QK_K_VALUES as u64, 1, 1, 1]),
                    ),
                )),
                Ok(())
            );
        }
    });
    assert_eq!(q4_k_info.count_total, 0);
    assert_eq!(q4_k_info.bytes_total, 0);
}

#[test]
fn f16_get_rows_matches_pinned_conversion_and_batch_order() {
    let source_layout = Layout::new(DType::F16, [3, 2, 2, 1], [2, 6, 12, 24]);
    let source = [
        0x3c00_u16, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600, 0x4900, 0x4a00, 0x4b00, 0x4c00, 0x4d00,
        0x4e00,
    ];
    let indices = [1_i32, 0, 0, 1];
    let mut output = [0.0_f32; 12];
    let mut kernel = GetRowsKernel::new();

    assert_eq!(
        kernel.process_event(OpGetRowsF16::new(
            F16View::new(&source, source_layout),
            IndexView::new(&indices, [2, 2, 1]),
            TensorViewMut::new(&mut output, contiguous([3, 2, 2, 1])),
        )),
        Ok(())
    );
    assert_eq!(
        output,
        [
            4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0
        ]
    );
}

#[test]
fn f16_get_rows_reads_pinned_strided_source_rows() {
    let source_layout = Layout::new(DType::F16, [2, 2, 1, 1], [4, 12, 24, 24]);
    let source = [
        0x3c00_u16, 0x7e00, 0x4000, 0x7e00, 0x7e00, 0x7e00, 0x4200, 0x7e00, 0x4400,
    ];
    let indices = [1_i32, 0];
    let mut output = [0.0_f32; 4];
    let mut kernel = GetRowsKernel::new();

    assert_eq!(
        kernel.process_event(OpGetRowsF16::new(
            F16View::new(&source, source_layout),
            IndexView::new(&indices, [2, 1, 1]),
            TensorViewMut::new(&mut output, contiguous([2, 2, 1, 1])),
        )),
        Ok(())
    );
    assert_eq!(output, [3.0, 4.0, 1.0, 2.0]);
}

#[test]
fn bf16_get_rows_matches_pinned_conversion_and_batch_order() {
    let source_layout = Layout::new(DType::Bf16, [3, 2, 2, 1], [2, 6, 12, 24]);
    let source = [
        0x3f80_u16, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0, 0x4120, 0x41a0, 0x41f0, 0x4220, 0x4248,
        0x4270,
    ];
    let indices = [1_i32, 0, 0, 1];
    let mut output = [0.0_f32; 12];
    let mut kernel = GetRowsKernel::new();

    assert_eq!(
        kernel.process_event(OpGetRowsBf16::new(
            Bf16View::new(&source, source_layout),
            IndexView::new(&indices, [2, 2, 1]),
            TensorViewMut::new(&mut output, contiguous([3, 2, 2, 1])),
        )),
        Ok(())
    );
    assert_eq!(
        output,
        [
            4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0
        ]
    );
}

#[test]
fn bf16_get_rows_reads_pinned_strided_source_rows() {
    let source_layout = Layout::new(DType::Bf16, [2, 2, 1, 1], [2, 12, 24, 24]);
    let source = [
        0x3f80_u16, 0x4000, 0x7fc0, 0x7fc0, 0x7fc0, 0x7fc0, 0x4040, 0x4080,
    ];
    let indices = [1_i32, 0];
    let mut output = [0.0_f32; 4];
    let mut kernel = GetRowsKernel::new();

    assert_eq!(
        kernel.process_event(OpGetRowsBf16::new(
            Bf16View::new(&source, source_layout),
            IndexView::new(&indices, [2, 1, 1]),
            TensorViewMut::new(&mut output, contiguous([2, 2, 1, 1])),
        )),
        Ok(())
    );
    assert_eq!(output, [3.0, 4.0, 1.0, 2.0]);
}

#[test]
fn bf16_get_rows_accepts_implicit_source_stride() {
    let source_layout = Layout::new(DType::Bf16, [2, 2, 1, 1], [0, 4, 8, 8]);
    let source = [0x3f80_u16, 0x4000, 0x4040, 0x4080];
    let indices = [1_i32, 0];
    let mut output = [0.0_f32; 4];
    let mut kernel = GetRowsKernel::new();

    assert_eq!(
        kernel.process_event(OpGetRowsBf16::new(
            Bf16View::new(&source, source_layout),
            IndexView::new(&indices, [2, 1, 1]),
            TensorViewMut::new(&mut output, contiguous([2, 2, 1, 1])),
        )),
        Ok(())
    );
    assert_eq!(output, [3.0, 4.0, 1.0, 2.0]);
}

#[test]
fn bf16_get_rows_accepts_singleton_outer_zero_strides() {
    let source_layout = Layout::new(DType::Bf16, [2, 1, 1, 1], [2, 0, 0, 0]);
    let source = [0x3f80_u16, 0x4000];
    let mut output = [9.0_f32; 2];
    let mut kernel = GetRowsKernel::new();

    assert_eq!(
        kernel.process_event(OpGetRowsBf16::new(
            Bf16View::new(&source, source_layout),
            IndexView::new(&[0_i32], [1, 1, 1]),
            TensorViewMut::new(&mut output, contiguous([2, 1, 1, 1])),
        )),
        Ok(())
    );
    assert_eq!(output, [1.0, 2.0]);
}

#[test]
fn bf16_get_rows_rejects_non_singleton_zero_stride_without_mutation() {
    let source_layout = Layout::new(DType::Bf16, [2, 2, 1, 1], [2, 0, 4, 4]);
    let source = [0x3f80_u16, 0x4000];
    let mut output = [9.0_f32; 2];
    let mut kernel = GetRowsKernel::new();

    assert_eq!(
        kernel.process_event(OpGetRowsBf16::new(
            Bf16View::new(&source, source_layout),
            IndexView::new(&[0_i32], [1, 1, 1]),
            TensorViewMut::new(&mut output, contiguous([2, 1, 1, 1])),
        )),
        Err(GetRowsError::InvalidView)
    );
    assert_eq!(output, [9.0; 2]);
}

#[test]
fn bf16_zero_index_implicit_source_metadata_is_a_noop() {
    let source_layout = Layout::new(DType::Bf16, [3, 2, 0, 0], [0, 6, 0, 0]);
    let mut output: [f32; 0] = [];
    let mut kernel = GetRowsKernel::new();

    assert_eq!(
        kernel.process_event(OpGetRowsBf16::new(
            Bf16View::new(&[], source_layout),
            IndexView::new(&[], [0, 0, 0]),
            TensorViewMut::new(&mut output, contiguous([3, 0, 0, 0])),
        )),
        Ok(())
    );
    assert!(output.is_empty());
}

#[test]
fn bf16_get_rows_rejects_shape_and_indices_without_mutation() {
    let source_layout = Layout::new(DType::Bf16, [3, 2, 1, 1], [2, 6, 12, 12]);
    let source = [0x3f80_u16; 6];
    let mut output = [9.0_f32; 6];
    let mut kernel = GetRowsKernel::new();

    assert_eq!(
        kernel.process_event(OpGetRowsBf16::new(
            Bf16View::new(&source, source_layout),
            IndexView::new(&[0_i32, 1], [2, 1, 1]),
            TensorViewMut::new(&mut output, contiguous([2, 2, 1, 1])),
        )),
        Err(GetRowsError::ShapeMismatch)
    );
    assert_eq!(output, [9.0; 6]);

    assert_eq!(
        kernel.process_event(OpGetRowsBf16::new(
            Bf16View::new(&source, source_layout),
            IndexView::new(&[2_i32], [1, 1, 1]),
            TensorViewMut::new(&mut output, contiguous([3, 1, 1, 1])),
        )),
        Err(GetRowsError::IndexOutOfBounds)
    );
    assert_eq!(output, [9.0; 6]);
}

#[test]
fn bf16_get_rows_dispatch_is_allocation_free() {
    let source_layout = Layout::new(DType::Bf16, [2, 1, 1, 1], [2, 4, 4, 4]);
    let source = [0x3f80_u16, 0x4000];
    let indices = [0_i32];
    let mut output = [0.0_f32; 2];
    let mut kernel = GetRowsKernel::new();
    let info = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                kernel.process_event(OpGetRowsBf16::new(
                    Bf16View::new(&source, source_layout),
                    IndexView::new(&indices, [1, 1, 1]),
                    TensorViewMut::new(&mut output, contiguous([2, 1, 1, 1])),
                )),
                Ok(())
            );
        }
    });
    assert_eq!(info.count_total, 0);
    assert_eq!(info.bytes_total, 0);
    assert_eq!(output, [1.0, 2.0]);
}

#[test]
fn f16_get_rows_rejects_shape_and_indices_without_mutation() {
    let source_layout = Layout::new(DType::F16, [3, 2, 1, 1], [2, 6, 12, 12]);
    let source = [0x3c00_u16; 6];
    let mut output = [9.0_f32; 6];
    let mut kernel = GetRowsKernel::new();

    assert_eq!(
        kernel.process_event(OpGetRowsF16::new(
            F16View::new(&source, source_layout),
            IndexView::new(&[0_i32, 1], [2, 1, 1]),
            TensorViewMut::new(&mut output, contiguous([2, 2, 1, 1])),
        )),
        Err(GetRowsError::ShapeMismatch)
    );
    assert_eq!(output, [9.0; 6]);

    assert_eq!(
        kernel.process_event(OpGetRowsF16::new(
            F16View::new(&source, source_layout),
            IndexView::new(&[2_i32], [1, 1, 1]),
            TensorViewMut::new(&mut output, contiguous([3, 1, 1, 1])),
        )),
        Err(GetRowsError::IndexOutOfBounds)
    );
    assert_eq!(output, [9.0; 6]);
}

#[test]
fn f16_get_rows_dispatch_is_allocation_free() {
    let source_layout = Layout::new(DType::F16, [2, 1, 1, 1], [2, 4, 4, 4]);
    let source = [0x3c00_u16, 0x4000];
    let indices = [0_i32];
    let mut output = [0.0_f32; 2];
    let mut kernel = GetRowsKernel::new();
    let info = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                kernel.process_event(OpGetRowsF16::new(
                    F16View::new(&source, source_layout),
                    IndexView::new(&indices, [1, 1, 1]),
                    TensorViewMut::new(&mut output, contiguous([2, 1, 1, 1])),
                )),
                Ok(())
            );
        }
    });
    assert_eq!(info.count_total, 0);
    assert_eq!(info.bytes_total, 0);
    assert_eq!(output, [1.0, 2.0]);
}

#[test]
fn f32_get_rows_matches_pinned_contiguous_copy_for_non_unit_source_stride() {
    let source_layout = Layout::new(DType::F32, [2, 2, 1, 1], [8, 16, 32, 32]);
    let source = [1.0_f32, 99.0, 2.0, 99.0, 3.0, 99.0, 4.0];
    let indices = [1_i32, 0];
    let mut output = [9.0_f32; 4];
    let mut kernel = GetRowsKernel::new();

    assert_eq!(
        kernel.process_event(OpGetRows::new(
            TensorView::new(&source, source_layout),
            IndexView::new(&indices, [2, 1, 1]),
            TensorViewMut::new(&mut output, contiguous([2, 2, 1, 1])),
        )),
        Ok(())
    );
    // The pinned F32 conversion uses memcpy(cols * 4) from each row start;
    // it does not gather logical elements through nb[0].
    assert_eq!(output, [3.0, 99.0, 1.0, 99.0]);
}

#[test]
fn f32_get_rows_accepts_implicit_source_stride() {
    let source_layout = Layout::new(DType::F32, [2, 2, 1, 1], [0, 8, 16, 16]);
    let source = [1.0_f32, 2.0, 3.0, 4.0];
    let indices = [1_i32, 0];
    let mut output = [9.0_f32; 4];
    let mut kernel = GetRowsKernel::new();

    assert_eq!(
        kernel.process_event(OpGetRows::new(
            TensorView::new(&source, source_layout),
            IndexView::new(&indices, [2, 1, 1]),
            TensorViewMut::new(&mut output, contiguous([2, 2, 1, 1])),
        )),
        Ok(())
    );
    assert_eq!(output, [3.0, 4.0, 1.0, 2.0]);
}

#[test]
fn zero_index_shape_is_a_valid_allocation_free_noop() {
    let source_layout = contiguous([3, 2, 1, 1]);
    let source = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut output: [f32; 0] = [];
    let mut kernel = GetRowsKernel::new();

    assert_eq!(
        kernel.process_event(OpGetRows::new(
            TensorView::new(&source, source_layout),
            IndexView::new(&[], [0, 1, 1]),
            TensorViewMut::new(&mut output, contiguous([3, 0, 1, 1])),
        )),
        Ok(())
    );
    assert!(output.is_empty());
}

#[test]
fn zero_index_outer_extents_follow_pinned_noop_guard() {
    let source_layout = Layout::new(DType::F32, [3, 2, 0, 0], [4, 12, 0, 0]);
    let destination_layout = Layout::new(DType::F32, [3, 0, 0, 0], [4, 12, 0, 0]);
    let mut output: [f32; 0] = [];
    let mut kernel = GetRowsKernel::new();

    assert_eq!(
        kernel.process_event(OpGetRows::new(
            TensorView::new(&[], source_layout),
            IndexView::new(&[], [0, 0, 0]),
            TensorViewMut::new(&mut output, destination_layout),
        )),
        Ok(())
    );
    assert!(output.is_empty());
}

#[test]
fn zero_index_implicit_f32_source_metadata_is_a_noop() {
    let source_layout = Layout::new(DType::F32, [3, 2, 0, 0], [0, 12, 0, 0]);
    let destination_layout = contiguous([3, 0, 0, 0]);
    let mut output: [f32; 0] = [];
    let mut kernel = GetRowsKernel::new();

    assert_eq!(
        kernel.process_event(OpGetRows::new(
            TensorView::new(&[], source_layout),
            IndexView::new(&[], [0, 0, 0]),
            TensorViewMut::new(&mut output, destination_layout),
        )),
        Ok(())
    );
    assert!(output.is_empty());
}

#[test]
fn index_metadata_accepts_valid_zero_extent_layouts_and_rejects_gaps() {
    assert!(IndexView::new(&[], [0, 1, 1]).metadata_valid());
    assert!(IndexView::new(&[], [0, 0, 0]).metadata_valid());
    assert!(IndexView::new(&[], [1, 2, 0]).metadata_valid());
    assert!(!IndexView::new(&[], [u64::MAX, 1, 1]).metadata_valid());
    assert!(!IndexView::new(&[], [0, 2, 1]).metadata_valid());
    assert!(IndexView::with_strides(&[], [0, 2, 1], [0, 0, 0]).metadata_valid());
}

#[test]
fn index_metadata_zero_stride_gap_rejects_noop_without_mutation() {
    let source_layout = contiguous([3, 2, 2, 1]);
    let source = [0.0_f32; 12];
    let mut output: [f32; 0] = [];
    let mut kernel = GetRowsKernel::new();
    let indices = IndexView::new(&[], [0, 2, 1]);
    assert!(!indices.metadata_valid());

    assert_eq!(
        kernel.process_event(OpGetRows::new(
            TensorView::new(&source, source_layout),
            indices,
            TensorViewMut::new(&mut output, contiguous([3, 0, 2, 1])),
        )),
        Err(GetRowsError::InvalidView)
    );
    assert!(output.is_empty());
}

#[test]
fn zero_index_rejects_invalid_outer_stride_metadata_without_mutation() {
    let source_layout = Layout::new(DType::F32, [3, 2, 0, 0], [4, 0, 0, 0]);
    let destination_layout = Layout::new(DType::F32, [3, 0, 0, 0], [4, 12, 0, 0]);
    let mut output = [9.0_f32; 3];
    let mut kernel = GetRowsKernel::new();

    assert_eq!(
        kernel.process_event(OpGetRows::new(
            TensorView::new(&[], source_layout),
            IndexView::new(&[], [0, 0, 0]),
            TensorViewMut::new(&mut output, destination_layout),
        )),
        Err(GetRowsError::InvalidView)
    );
    assert_eq!(output, [9.0; 3]);
}

#[test]
fn guards_reject_shape_and_index_errors_without_mutation() {
    let source_layout = contiguous([3, 2, 1, 1]);
    let source = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut output = [9.0_f32; 6];
    let mut kernel = GetRowsKernel::new();

    assert_eq!(
        kernel.process_event(OpGetRows::new(
            TensorView::new(&source, source_layout),
            IndexView::new(&[0_i32, 1], [2, 1, 1]),
            TensorViewMut::new(&mut output, contiguous([2, 2, 1, 1])),
        )),
        Err(GetRowsError::ShapeMismatch)
    );
    assert_eq!(output, [9.0; 6]);

    assert_eq!(
        kernel.process_event(OpGetRows::new(
            TensorView::new(&source, source_layout),
            IndexView::new(&[-1_i32, 0], [2, 1, 1]),
            TensorViewMut::new(&mut output, contiguous([3, 2, 1, 1])),
        )),
        Err(GetRowsError::IndexOutOfBounds)
    );
    assert_eq!(output, [9.0; 6]);

    assert_eq!(
        kernel.process_event(OpGetRows::new(
            TensorView::new(&source, source_layout),
            IndexView::new(&[2_i32, 0], [2, 1, 1]),
            TensorViewMut::new(&mut output, contiguous([3, 2, 1, 1])),
        )),
        Err(GetRowsError::IndexOutOfBounds)
    );
    assert_eq!(output, [9.0; 6]);
}

#[test]
fn guards_reject_unsupported_view_and_short_indices() {
    let source = [1.0_f32; 6];
    let mut output = [9.0_f32; 6];
    let valid_source_layout = contiguous([3, 2, 1, 1]);
    let invalid_source_layout = Layout::new(DType::F16, [3, 2, 1, 1], [4, 12, 24, 24]);
    let mut kernel = GetRowsKernel::default();

    assert_eq!(
        kernel.process_event(OpGetRows::new(
            TensorView::new(&source, invalid_source_layout),
            IndexView::new(&[0_i32, 1], [2, 1, 1]),
            TensorViewMut::new(&mut output, valid_source_layout),
        )),
        Err(GetRowsError::InvalidView)
    );
    assert_eq!(output, [9.0; 6]);

    assert_eq!(
        kernel.process_event(OpGetRows::new(
            TensorView::new(&source, valid_source_layout),
            IndexView::new(&[0_i32], [2, 1, 1]),
            TensorViewMut::new(&mut output, valid_source_layout),
        )),
        Err(GetRowsError::InvalidView)
    );
    assert_eq!(output, [9.0; 6]);
}

#[test]
fn get_rows_error_surfaces_are_explicit() {
    assert_eq!(
        format!("{}", GetRowsError::InvalidView),
        "invalid get_rows tensor view"
    );
    assert_eq!(
        format!("{}", GetRowsError::ShapeMismatch),
        "get_rows tensor shapes differ"
    );
    assert_eq!(
        format!("{}", GetRowsError::IndexOutOfBounds),
        "get_rows index is out of bounds"
    );
    assert_eq!(
        format!("{}", GetRowsError::UnsupportedSourceStride),
        "get_rows source element stride is unsupported"
    );
    assert_eq!(
        format!("{}", GetRowsError::UnexpectedEvent),
        "unexpected get_rows event"
    );
    assert_eq!(
        format!("{}", GetRowsError::Internal),
        "internal get_rows dispatch error"
    );
    assert!(!format!("{:?}", GetRowsError::Internal).is_empty());
}

#[test]
fn unexpected_public_event_is_typed_and_actor_recovers() {
    let mut kernel = GetRowsKernel::default();
    assert_eq!(
        kernel.process_event(UnexpectedGetRows),
        Err(GetRowsError::UnexpectedEvent)
    );

    let source_layout = contiguous([2, 1, 1, 1]);
    let source = [1.0_f32, 2.0];
    let indices = [0_i32];
    let mut output = [0.0_f32; 2];
    assert_eq!(
        kernel.process_event(OpGetRows::new(
            TensorView::new(&source, source_layout),
            IndexView::new(&indices, [1, 1, 1]),
            TensorViewMut::new(&mut output, source_layout),
        )),
        Ok(())
    );
    assert_eq!(output, [1.0, 2.0]);
}

#[test]
fn get_rows_dispatch_is_allocation_free() {
    let source_layout = contiguous([4, 2, 1, 1]);
    let source = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let indices = [1_i32, 0];
    let mut output = [0.0_f32; 8];
    let mut kernel = GetRowsKernel::new();
    let info = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                kernel.process_event(OpGetRows::new(
                    TensorView::new(&source, source_layout),
                    IndexView::new(&indices, [2, 1, 1]),
                    TensorViewMut::new(&mut output, source_layout),
                )),
                Ok(())
            );
        }
    });
    assert_eq!(info.count_total, 0);
    assert_eq!(info.bytes_total, 0);
    assert_eq!(output, [5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn f32_bytes_get_rows_dispatch_is_allocation_free() {
    let source_layout = Layout::new(DType::F32, [2, 2, 1, 1], [4, 9, 18, 18]);
    let mut source = [0_u8; 17];
    source[0..4].copy_from_slice(&1.0_f32.to_ne_bytes());
    source[4..8].copy_from_slice(&2.0_f32.to_ne_bytes());
    source[9..13].copy_from_slice(&3.0_f32.to_ne_bytes());
    source[13..17].copy_from_slice(&4.0_f32.to_ne_bytes());
    let indices = [1_i32, 0];
    let mut output = [0.0_f32; 4];
    let mut kernel = GetRowsKernel::new();
    let info = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                kernel.process_event(OpGetRowsF32Bytes::new(
                    ByteTensorView::new(&source, source_layout),
                    IndexView::new(&indices, [2, 1, 1]),
                    TensorViewMut::new(&mut output, contiguous([2, 2, 1, 1])),
                )),
                Ok(())
            );
        }
    });
    assert_eq!(info.count_total, 0);
    assert_eq!(info.bytes_total, 0);
    assert_eq!(output, [3.0, 4.0, 1.0, 2.0]);
}

#[test]
fn strided_i32_index_dispatch_is_allocation_free() {
    let source_layout = contiguous([2, 2, 2, 1]);
    let source = [1.0_f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
    let indices = [1_i32, 0, 99, 99, 0, 1];
    let indices_view = IndexView::with_strides(&indices, [2, 2, 1], [4, 16, 32]);
    let mut output = [0.0_f32; 8];
    let mut kernel = GetRowsKernel::new();
    let info = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                kernel.process_event(OpGetRows::new(
                    TensorView::new(&source, source_layout),
                    indices_view,
                    TensorViewMut::new(&mut output, source_layout),
                )),
                Ok(())
            );
        }
    });
    assert_eq!(info.count_total, 0);
    assert_eq!(info.bytes_total, 0);
    assert_eq!(output, [3.0, 4.0, 1.0, 2.0, 10.0, 20.0, 30.0, 40.0]);
}
