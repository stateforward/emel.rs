//! Split Rust observer for the public AArch64 q8_0 packed x4/bl8 router.

#![cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]

use emel_kernels::aarch64::{Kernel, MulMatQ8_0PackedBl8, Q8_0PackedBl4Error};
use emel_kernels::any::quant::Q8_0_BLOCK_BYTES;

const COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const GUARDS_BLOB: &str = "c25714566ec9a02679daef85089544575123408e";
const ACTIONS_BLOB: &str = "267d4f74e6e7498155c8535920322ffef2c02fb6";
const SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";
const SENTINEL: f32 = f32::from_bits(0x7fc0_0001);
const K: usize = 32;
const M: usize = 5;
const Q8_0_X4_ROWS: usize = 4;
const Q8_0_X4_BLOCK_BYTES: usize = 136;
const INTERLEAVE: usize = 8;

#[cfg(not(target_arch = "aarch64"))]
fn main() {
    eprintln!("AArch64 q8_0 packed bl8 observer requires an AArch64 build");
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
fn pack_row(scale: u16, row: usize) -> [u8; Q8_0_BLOCK_BYTES] {
    let mut block = [0_u8; Q8_0_BLOCK_BYTES];
    block[..2].copy_from_slice(&scale.to_le_bytes());
    for index in 0..32 {
        let value = i8::try_from(index as i32 - 16 + i32::try_from(row).unwrap()).unwrap();
        block[2 + index] = value.to_ne_bytes()[0];
    }
    block
}

#[cfg(target_arch = "aarch64")]
fn make_block_q8_0_x4_bl4(
    rows: &[[u8; Q8_0_BLOCK_BYTES]; Q8_0_X4_ROWS],
) -> [u8; Q8_0_X4_BLOCK_BYTES] {
    let mut out = [0_u8; Q8_0_X4_BLOCK_BYTES];
    for row in 0..Q8_0_X4_ROWS {
        out[row * 2..row * 2 + 2].copy_from_slice(&rows[row][..2]);
    }
    let end = (32 * Q8_0_X4_ROWS) / INTERLEAVE;
    for index in 0..end {
        let src_row = index % Q8_0_X4_ROWS;
        let src_offset = (index / Q8_0_X4_ROWS) * INTERLEAVE;
        let dst_offset = index * INTERLEAVE;
        out[8 + dst_offset..8 + dst_offset + INTERLEAVE]
            .copy_from_slice(&rows[src_row][2 + src_offset..2 + src_offset + INTERLEAVE]);
    }
    out
}

#[cfg(target_arch = "aarch64")]
fn pack_lhs() -> [u8; 2 * Q8_0_X4_BLOCK_BYTES] {
    let mut rows = [[0_u8; Q8_0_BLOCK_BYTES]; M];
    for row in 0..M {
        rows[row] = pack_row(0x3c00, row);
    }
    let mut packed = [0_u8; 2 * Q8_0_X4_BLOCK_BYTES];
    for group in 0..2 {
        let row_base = group * Q8_0_X4_ROWS;
        let mut group_rows = [[0_u8; Q8_0_BLOCK_BYTES]; Q8_0_X4_ROWS];
        for row in 0..Q8_0_X4_ROWS {
            let logical = row_base + row;
            if logical < M {
                group_rows[row] = rows[logical];
            }
        }
        let offset = group * Q8_0_X4_BLOCK_BYTES;
        packed[offset..offset + Q8_0_X4_BLOCK_BYTES]
            .copy_from_slice(&make_block_q8_0_x4_bl4(&group_rows));
    }
    packed
}

#[cfg(target_arch = "aarch64")]
fn main() {
    println!("kernel-target-aarch64-q8-0-x4-bl8-live/v1");
    println!("source_repository=stateforward/emel.cpp");
    println!("source_commit={COMMIT}");
    println!("source_kernel_aarch64_guards_blob={GUARDS_BLOB}");
    println!("source_kernel_aarch64_actions_blob={ACTIONS_BLOB}");
    println!("source_kernel_aarch64_sm_blob={SM_BLOB}");
    println!("target_arch=aarch64");
    println!("scope=target_router_q8_0_packed_bl8_m5_k32_and_typed_rejection");
    println!("execution=split_pinned_aarch64_sm_and_public_target_router");

    let lhs = pack_lhs();
    let rhs = pack_row(0x3800, 0);
    let mut actor = Kernel::try_new().expect("AArch64 target router");
    let mut output = [0.0_f32; M];
    assert_eq!(
        actor.process_event(MulMatQ8_0PackedBl8::new(&lhs, &rhs, M, K), &mut output),
        Ok(())
    );
    println!(
        "case=q8_0_packed_bl8_m5_k32 status=ok output_bits={}",
        bits(&output)
    );

    let mut zero_output = [SENTINEL];
    assert_eq!(
        actor.process_event(MulMatQ8_0PackedBl8::new(&[], &[], 0, 0), &mut zero_output),
        Err(Q8_0PackedBl4Error::InvalidShape)
    );
    println!(
        "case=zero_count status=reject error=InvalidShape output_bits={}",
        bits(&zero_output)
    );

    let mut invalid_output = [SENTINEL; M];
    assert_eq!(
        actor.process_event(
            MulMatQ8_0PackedBl8::new(&lhs, &rhs, M, 31),
            &mut invalid_output
        ),
        Err(Q8_0PackedBl4Error::InvalidShape)
    );
    println!(
        "case=invalid_shape status=reject error=InvalidShape output_bits={}",
        bits(&invalid_output)
    );
}
