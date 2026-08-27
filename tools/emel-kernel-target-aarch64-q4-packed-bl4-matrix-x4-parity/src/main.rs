//! Split Rust observer for the public AArch64 q4 packed bl4 matrix_x4 router.

#![cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::needless_range_loop
)]

use emel_kernels::aarch64::{Kernel, MulMatQ4PackedBl4MatrixX4, Q4PackedBl4Error};

const COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const GUARDS_BLOB: &str = "c25714566ec9a02679daef85089544575123408e";
const ACTIONS_BLOB: &str = "267d4f74e6e7498155c8535920322ffef2c02fb6";
const SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";
const SENTINEL: f32 = f32::from_bits(0x7fc0_0001);
const K: usize = 256;
const M: usize = 8;
const RHS_ROWS: usize = 4;
const QK_K: usize = 256;
const Q4_K_BLOCK_BYTES: usize = 144;
const Q4_K_X8_ROWS: usize = 8;
const Q4_K_X8_BLOCK_BYTES: usize = 1152;
const Q8_K_BLOCK_BYTES: usize = 292;
const INTERLEAVE: usize = 4;

#[cfg(not(target_arch = "aarch64"))]
fn main() {
    eprintln!("AArch64 q4 packed bl4 matrix_x4 observer requires an AArch64 build");
    std::process::exit(2);
}

#[cfg(target_arch = "aarch64")]
fn bits(values: &[f32]) -> String {
    values
        .iter()
        .map(|value| format!("{:08x}", value.to_bits()))
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(target_arch = "aarch64")]
fn native_q4_k_row(row: usize) -> [u8; Q4_K_BLOCK_BYTES] {
    let mut out = [0_u8; Q4_K_BLOCK_BYTES];
    let d = (0x3c00 + row * 17) as u16;
    let dmin = (0x3400 + row * 11) as u16;
    out[0..2].copy_from_slice(&d.to_le_bytes());
    out[2..4].copy_from_slice(&dmin.to_le_bytes());
    for index in 0..12 {
        out[4 + index] = ((row * 37 + index * 13 + 9) & 0xff) as u8;
    }
    for index in 0..128 {
        out[16 + index] = ((row * 29 + index * 5 + 3) & 0xff) as u8;
    }
    out
}

#[cfg(target_arch = "aarch64")]
fn write_packed_scale12(
    out: &mut [u8],
    base: usize,
    scales: [u8; Q4_K_X8_ROWS],
    mins: [u8; Q4_K_X8_ROWS],
) {
    out[32 + base] = (scales[0] & 63).wrapping_add((scales[4] & 48) << 2);
    out[32 + base + 1] = (scales[1] & 63).wrapping_add((scales[5] & 48) << 2);
    out[32 + base + 2] = (scales[2] & 63).wrapping_add((scales[6] & 48) << 2);
    out[32 + base + 3] = (scales[3] & 63).wrapping_add((scales[7] & 48) << 2);
    out[32 + base + 4] = (mins[0] & 63).wrapping_add((mins[4] & 48) << 2);
    out[32 + base + 5] = (mins[1] & 63).wrapping_add((mins[5] & 48) << 2);
    out[32 + base + 6] = (mins[2] & 63).wrapping_add((mins[6] & 48) << 2);
    out[32 + base + 7] = (mins[3] & 63).wrapping_add((mins[7] & 48) << 2);
    out[32 + base + 8] = (scales[4] & 15).wrapping_add((mins[4] & 15) << 4);
    out[32 + base + 9] = (scales[5] & 15).wrapping_add((mins[5] & 15) << 4);
    out[32 + base + 10] = (scales[6] & 15).wrapping_add((mins[6] & 15) << 4);
    out[32 + base + 11] = (scales[7] & 15).wrapping_add((mins[7] & 15) << 4);
}

#[cfg(target_arch = "aarch64")]
fn make_block_q4_k_x8_bl4(
    rows: &[[u8; Q4_K_BLOCK_BYTES]; Q4_K_X8_ROWS],
) -> [u8; Q4_K_X8_BLOCK_BYTES] {
    let mut out = [0_u8; Q4_K_X8_BLOCK_BYTES];
    for row in 0..Q4_K_X8_ROWS {
        out[row * 2..row * 2 + 2].copy_from_slice(&rows[row][0..2]);
        out[16 + row * 2..18 + row * 2].copy_from_slice(&rows[row][2..4]);
    }
    let end = (QK_K * 4) / INTERLEAVE;
    for index in 0..end {
        let src_row = index % Q4_K_X8_ROWS;
        let src_offset = (index / Q4_K_X8_ROWS) * INTERLEAVE;
        let dst_offset = index * INTERLEAVE;
        out[128 + dst_offset..128 + dst_offset + INTERLEAVE]
            .copy_from_slice(&rows[src_row][16 + src_offset..16 + src_offset + INTERLEAVE]);
    }
    let mut scales = [0_u8; Q4_K_X8_ROWS];
    let mut mins = [0_u8; Q4_K_X8_ROWS];
    for group in 0..4 {
        for row in 0..Q4_K_X8_ROWS {
            scales[row] = rows[row][4 + group] & 63;
            mins[row] = rows[row][8 + group] & 63;
        }
        write_packed_scale12(&mut out, group * 12, scales, mins);
    }
    for group in 0..4 {
        for row in 0..Q4_K_X8_ROWS {
            scales[row] = ((rows[row][4 + group] & 192) >> 2) | (rows[row][12 + group] & 15);
            mins[row] = ((rows[row][8 + group] & 192) >> 2) | ((rows[row][12 + group] & 240) >> 4);
        }
        write_packed_scale12(&mut out, group * 12 + 48, scales, mins);
    }
    out
}

#[cfg(target_arch = "aarch64")]
fn pack_lhs() -> [u8; Q4_K_X8_BLOCK_BYTES] {
    let mut group_rows = [[0_u8; Q4_K_BLOCK_BYTES]; Q4_K_X8_ROWS];
    for row in 0..Q4_K_X8_ROWS {
        group_rows[row] = native_q4_k_row(row);
    }
    make_block_q4_k_x8_bl4(&group_rows)
}

#[cfg(target_arch = "aarch64")]
fn pack_q8_k_row(row: usize) -> [u8; Q8_K_BLOCK_BYTES] {
    let mut out = [0_u8; Q8_K_BLOCK_BYTES];
    let scale = 0.0625_f32 * (row as f32 + 1.0);
    out[0..4].copy_from_slice(&scale.to_le_bytes());
    let mut qs = [0_i8; QK_K];
    for index in 0..QK_K {
        qs[index] = ((index as i32 * 7 + row as i32 * 11) % 31 - 15) as i8;
        out[4 + index] = qs[index].to_ne_bytes()[0];
    }
    for group in 0..(QK_K / 16) {
        let mut sum = 0_i32;
        for lane in 0..16 {
            sum += i32::from(qs[group * 16 + lane]);
        }
        let offset = 260 + group * 2;
        out[offset..offset + 2].copy_from_slice(&(sum as i16).to_le_bytes());
    }
    out
}

#[cfg(target_arch = "aarch64")]
fn pack_rhs() -> [u8; RHS_ROWS * Q8_K_BLOCK_BYTES] {
    let mut out = [0_u8; RHS_ROWS * Q8_K_BLOCK_BYTES];
    for row in 0..RHS_ROWS {
        let packed = pack_q8_k_row(row);
        let offset = row * Q8_K_BLOCK_BYTES;
        out[offset..offset + Q8_K_BLOCK_BYTES].copy_from_slice(&packed);
    }
    out
}

#[cfg(target_arch = "aarch64")]
fn main() {
    println!("kernel-target-aarch64-q4-packed-bl4-matrix-x4-live/v1");
    println!("source_repository=stateforward/emel.cpp");
    println!("source_commit={COMMIT}");
    println!("source_kernel_aarch64_guards_blob={GUARDS_BLOB}");
    println!("source_kernel_aarch64_actions_blob={ACTIONS_BLOB}");
    println!("source_kernel_aarch64_sm_blob={SM_BLOB}");
    println!("target_arch=aarch64");
    println!("scope=target_router_q4_packed_bl4_matrix_x4_m8_k256_and_typed_rejection");
    println!("execution=split_pinned_aarch64_sm_and_public_target_router");

    let lhs = pack_lhs();
    let rhs = pack_rhs();
    let mut actor = Kernel::try_new().expect("AArch64 target router");
    let mut output = [0.0_f32; RHS_ROWS * M];
    assert_eq!(
        actor.process_event(
            MulMatQ4PackedBl4MatrixX4::new(&lhs, &rhs, M, K),
            &mut output
        ),
        Ok(())
    );
    println!(
        "case=q4_packed_bl4_matrix_x4_m8_k256 status=ok output_bits={}",
        bits(&output)
    );

    let mut zero_output = [SENTINEL; 4];
    assert_eq!(
        actor.process_event(
            MulMatQ4PackedBl4MatrixX4::new(&[], &[], 0, 0),
            &mut zero_output
        ),
        Err(Q4PackedBl4Error::InvalidShape)
    );
    println!(
        "case=zero_count status=reject error=InvalidShape output_bits={}",
        bits(&zero_output)
    );

    let mut k_zero_output = [SENTINEL; RHS_ROWS * M];
    assert_eq!(
        actor.process_event(
            MulMatQ4PackedBl4MatrixX4::new(&[], &[], M, 0),
            &mut k_zero_output
        ),
        Err(Q4PackedBl4Error::InvalidShape)
    );
    println!(
        "case=k_zero status=reject error=InvalidShape output_bits={}",
        bits(&k_zero_output)
    );

    let short_rhs = pack_q8_k_row(0);
    let mut invalid_output = [SENTINEL; RHS_ROWS * M];
    assert_eq!(
        actor.process_event(
            MulMatQ4PackedBl4MatrixX4::new(&lhs, &short_rhs, M, K),
            &mut invalid_output
        ),
        Err(Q4PackedBl4Error::InvalidShape)
    );
    println!(
        "case=invalid_rhs_rows status=reject error=InvalidShape output_bits={}",
        bits(&invalid_output)
    );
}
