//! Split Rust observer for the public AArch64 q4 packed x8/bl8 F32-rhs router.

#![cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::needless_range_loop
)]

use emel_kernels::aarch64::{Kernel, MulMatQ4PackedF32Bl8, Q4PackedBl4Error};

const COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const GUARDS_BLOB: &str = "c25714566ec9a02679daef85089544575123408e";
const ACTIONS_BLOB: &str = "267d4f74e6e7498155c8535920322ffef2c02fb6";
const SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";
const SENTINEL: f32 = f32::from_bits(0x7fc0_0001);
const K: usize = 256;
const M: usize = 5;
const QK_K: usize = 256;
const Q4_K_BLOCK_BYTES: usize = 144;
const Q4_K_X8_ROWS: usize = 8;
const Q4_K_X8_BLOCK_BYTES: usize = 1152;
const INTERLEAVE: usize = 8;

#[cfg(not(target_arch = "aarch64"))]
fn main() {
    eprintln!("AArch64 q4 packed f32 bl8 observer requires an AArch64 build");
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
fn make_block_q4_k_x8_bl8(
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
    let mut rows = [[0_u8; Q4_K_BLOCK_BYTES]; M];
    for row in 0..M {
        rows[row] = native_q4_k_row(row);
    }
    let mut group_rows = [[0_u8; Q4_K_BLOCK_BYTES]; Q4_K_X8_ROWS];
    for row in 0..Q4_K_X8_ROWS {
        if row < M {
            group_rows[row] = rows[row];
        }
    }
    make_block_q4_k_x8_bl8(&group_rows)
}

#[cfg(target_arch = "aarch64")]
fn dense_f32_rhs() -> [f32; K] {
    let mut out = [0.0_f32; K];
    for index in 0..K {
        out[index] = ((index as i32 * 7) % 31 - 15) as f32 * 0.0625;
    }
    out
}

#[cfg(target_arch = "aarch64")]
fn main() {
    println!("kernel-target-aarch64-q4-packed-f32-bl8-live/v1");
    println!("source_repository=stateforward/emel.cpp");
    println!("source_commit={COMMIT}");
    println!("source_kernel_aarch64_guards_blob={GUARDS_BLOB}");
    println!("source_kernel_aarch64_actions_blob={ACTIONS_BLOB}");
    println!("source_kernel_aarch64_sm_blob={SM_BLOB}");
    println!("target_arch=aarch64");
    println!("scope=target_router_q4_packed_f32_bl8_m5_k256_and_typed_rejection");
    println!("execution=split_pinned_aarch64_sm_and_public_target_router");

    let lhs = pack_lhs();
    let rhs = dense_f32_rhs();
    let mut actor = Kernel::try_new().expect("AArch64 target router");
    let mut output = [0.0_f32; M];
    assert_eq!(
        actor.process_event(MulMatQ4PackedF32Bl8::new(&lhs, &rhs, M, K), &mut output),
        Ok(())
    );
    println!(
        "case=q4_packed_f32_bl8_m5_k256 status=ok output_bits={}",
        bits(&output)
    );

    let mut zero_output = [SENTINEL];
    assert_eq!(
        actor.process_event(MulMatQ4PackedF32Bl8::new(&[], &[], 0, 0), &mut zero_output),
        Err(Q4PackedBl4Error::InvalidShape)
    );
    println!(
        "case=zero_count status=reject error=InvalidShape output_bits={}",
        bits(&zero_output)
    );

    let mut invalid_output = [SENTINEL; M];
    assert_eq!(
        actor.process_event(
            MulMatQ4PackedF32Bl8::new(&lhs, &rhs, M, 0),
            &mut invalid_output
        ),
        Err(Q4PackedBl4Error::InvalidShape)
    );
    println!(
        "case=invalid_shape status=reject error=InvalidShape output_bits={}",
        bits(&invalid_output)
    );
}
