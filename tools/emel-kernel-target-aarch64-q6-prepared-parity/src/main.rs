//! Split Rust observer for the public AArch64 q6 prepared x8/q8-rhs router.

#![cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::needless_range_loop
)]

use emel_kernels::aarch64::{Kernel, MulMatQ6Prepared, Q6PreparedError};

const COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const GUARDS_BLOB: &str = "c25714566ec9a02679daef85089544575123408e";
const ACTIONS_BLOB: &str = "267d4f74e6e7498155c8535920322ffef2c02fb6";
const SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";
const SENTINEL: f32 = f32::from_bits(0x7fc0_0001);
const K: usize = 256;
const M: usize = 5;
const QK_K: usize = 256;
const Q6_K_BLOCK_BYTES: usize = 210;
const Q6_K_X8_ROWS: usize = 8;
const Q6_K_X8_PREPARED_BLOCK_BYTES: usize = 2192;
const Q8_K_BLOCK_BYTES: usize = 292;
const SCALE_OFFSET: usize = 16;
const QS_OFFSET: usize = 144;

#[cfg(not(target_arch = "aarch64"))]
fn main() {
    eprintln!("AArch64 q6 prepared observer requires an AArch64 build");
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
fn native_q6_k_row(row: usize) -> [u8; Q6_K_BLOCK_BYTES] {
    let mut out = [0_u8; Q6_K_BLOCK_BYTES];
    let d = (0x3c00 + row * 17) as u16;
    for index in 0..128 {
        out[index] = ((row * 29 + index * 5 + 3) & 0xff) as u8;
    }
    for index in 0..64 {
        out[128 + index] = ((row * 19 + index * 7 + 11) & 0xff) as u8;
    }
    for index in 0..16 {
        let scale = (row as i32 * 3 + index as i32).rem_euclid(15) - 7;
        out[192 + index] = (scale as i8).to_ne_bytes()[0];
    }
    out[208..210].copy_from_slice(&d.to_le_bytes());
    out
}

#[cfg(target_arch = "aarch64")]
fn decode_q6_nibble(ql_nibble: u8, qh_bits: u8) -> i8 {
    (i32::from((ql_nibble & 0x0f) | ((qh_bits & 0x03) << 4)) - 32) as i8
}

#[cfg(target_arch = "aarch64")]
fn decode_q6_k_row(row: &[u8; Q6_K_BLOCK_BYTES]) -> [i8; QK_K] {
    let mut decoded = [0_i8; QK_K];
    for half in 0..2 {
        let half_value_base = half * 128;
        let ql_half_base = half * 64;
        let qh_half_base = half * 32;
        for lane in 0..32 {
            let qh_byte = row[128 + qh_half_base + lane];
            let ql_low = row[ql_half_base + lane];
            let ql_high = row[ql_half_base + 32 + lane];
            decoded[half_value_base + lane] = decode_q6_nibble(ql_low, qh_byte);
            decoded[half_value_base + lane + 32] = decode_q6_nibble(ql_high, qh_byte >> 2);
            decoded[half_value_base + lane + 64] = decode_q6_nibble(ql_low >> 4, qh_byte >> 4);
            decoded[half_value_base + lane + 96] = decode_q6_nibble(ql_high >> 4, qh_byte >> 6);
        }
    }
    decoded
}

#[cfg(target_arch = "aarch64")]
fn make_block_q6_k_x8_q8_prepared(
    rows: &[[u8; Q6_K_BLOCK_BYTES]; Q6_K_X8_ROWS],
) -> [u8; Q6_K_X8_PREPARED_BLOCK_BYTES] {
    let mut out = [0_u8; Q6_K_X8_PREPARED_BLOCK_BYTES];
    for row in 0..Q6_K_X8_ROWS {
        out[row * 2..row * 2 + 2].copy_from_slice(&rows[row][208..210]);
        let decoded = decode_q6_k_row(&rows[row]);
        for scale in 0..(QK_K / 16) {
            out[SCALE_OFFSET + scale * Q6_K_X8_ROWS + row] = rows[row][192 + scale];
            let pair_index = row / 2;
            let pair_base = QS_OFFSET + scale * 128 + pair_index * 32;
            let half_base = pair_base + (row % 2) * 8;
            for lane in 0..8 {
                out[half_base + lane] = decoded[scale * 16 + lane].to_ne_bytes()[0];
                out[half_base + 16 + lane] = decoded[scale * 16 + 8 + lane].to_ne_bytes()[0];
            }
        }
    }
    out
}

#[cfg(target_arch = "aarch64")]
fn pack_lhs() -> [u8; Q6_K_X8_PREPARED_BLOCK_BYTES] {
    let mut group_rows = [[0_u8; Q6_K_BLOCK_BYTES]; Q6_K_X8_ROWS];
    for row in 0..M {
        group_rows[row] = native_q6_k_row(row);
    }
    make_block_q6_k_x8_q8_prepared(&group_rows)
}

#[cfg(target_arch = "aarch64")]
fn pack_q8_k_row() -> [u8; Q8_K_BLOCK_BYTES] {
    let mut out = [0_u8; Q8_K_BLOCK_BYTES];
    out[0..4].copy_from_slice(&0.0625_f32.to_le_bytes());
    let mut qs = [0_i8; QK_K];
    for index in 0..QK_K {
        qs[index] = ((index as i32 * 7) % 31 - 15) as i8;
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
fn main() {
    println!("kernel-target-aarch64-q6-prepared-live/v1");
    println!("source_repository=stateforward/emel.cpp");
    println!("source_commit={COMMIT}");
    println!("source_kernel_aarch64_guards_blob={GUARDS_BLOB}");
    println!("source_kernel_aarch64_actions_blob={ACTIONS_BLOB}");
    println!("source_kernel_aarch64_sm_blob={SM_BLOB}");
    println!("target_arch=aarch64");
    println!("scope=target_router_q6_prepared_m5_k256_and_typed_rejection");
    println!("execution=split_pinned_aarch64_sm_and_public_target_router");
    println!("live_path=execute_neon_mul_mat_q6_vector_prepared_q8_rhs_i8mm");

    let lhs = pack_lhs();
    let rhs = pack_q8_k_row();
    let mut actor = Kernel::try_new().expect("AArch64 target router");
    let mut output = [0.0_f32; M];
    assert_eq!(
        actor.process_event(MulMatQ6Prepared::new(&lhs, &rhs, M, K), &mut output),
        Ok(())
    );
    println!(
        "case=q6_prepared_m5_k256 status=ok output_bits={}",
        bits(&output)
    );

    let mut zero_output = [SENTINEL];
    assert_eq!(
        actor.process_event(MulMatQ6Prepared::new(&[], &[], 0, 0), &mut zero_output),
        Err(Q6PreparedError::InvalidShape)
    );
    println!(
        "case=zero_count status=reject error=InvalidShape output_bits={}",
        bits(&zero_output)
    );

    let mut invalid_output = [SENTINEL; M];
    assert_eq!(
        actor.process_event(
            MulMatQ6Prepared::new(&lhs, &rhs, M, 255),
            &mut invalid_output
        ),
        Err(Q6PreparedError::InvalidShape)
    );
    println!(
        "case=invalid_shape status=reject error=InvalidShape output_bits={}",
        bits(&invalid_output)
    );
}
