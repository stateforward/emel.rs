//! Split Rust observer for the public AArch64 q6_k GEMM router.

#![cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]

use emel_kernels::aarch64::{Kernel, MulMatQ6, Q6VectorError};

const COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const GUARDS_BLOB: &str = "c25714566ec9a02679daef85089544575123408e";
const ACTIONS_BLOB: &str = "267d4f74e6e7498155c8535920322ffef2c02fb6";
const SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";
const SENTINEL: f32 = f32::from_bits(0x7fc0_0001);
const QK_K: usize = 256;
const Q6_K_BLOCK_BYTES: usize = 210;
const M: usize = 2;
const N: usize = 2;

#[cfg(not(target_arch = "aarch64"))]
fn main() {
    eprintln!("AArch64 q6 mul_mat observer requires an AArch64 build");
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
fn pack_q6_row(scale: u16, ql: u8, qh: u8, group_scale: i8) -> [u8; Q6_K_BLOCK_BYTES] {
    let mut block = [0_u8; Q6_K_BLOCK_BYTES];
    block[..128].fill(ql);
    block[128..192].fill(qh);
    for index in 0..16 {
        block[192 + index] = group_scale.to_ne_bytes()[0];
    }
    block[208..210].copy_from_slice(&scale.to_le_bytes());
    block
}

#[cfg(target_arch = "aarch64")]
fn main() {
    println!("kernel-target-aarch64-q6-mul-mat-live/v1");
    println!("source_repository=stateforward/emel.cpp");
    println!("source_commit={COMMIT}");
    println!("source_kernel_aarch64_guards_blob={GUARDS_BLOB}");
    println!("source_kernel_aarch64_actions_blob={ACTIONS_BLOB}");
    println!("source_kernel_aarch64_sm_blob={SM_BLOB}");
    println!("target_arch=aarch64");
    println!("scope=target_router_q6_mul_mat_m2_k256_n2_and_typed_rejection");
    println!("execution=split_pinned_aarch64_sm_and_public_target_router");

    let row0 = pack_q6_row(0x3c00, 0x12, 0x34, 2);
    let row1 = pack_q6_row(0x4000, 0x56, 0x78, -1);
    let mut lhs = [0_u8; 2 * Q6_K_BLOCK_BYTES];
    lhs[..Q6_K_BLOCK_BYTES].copy_from_slice(&row0);
    lhs[Q6_K_BLOCK_BYTES..].copy_from_slice(&row1);
    let mut rhs = [0.0_f32; QK_K * N];
    for index in 0..QK_K {
        rhs[index * N] = (index as f32 - 128.0) * 0.03125;
        rhs[index * N + 1] = (64.0 - index as f32) * 0.015_625;
    }
    let mut actor = Kernel::try_new().expect("AArch64 target router");
    let mut output = [0.0_f32; M * N];
    assert_eq!(
        actor.process_event(MulMatQ6::new(&lhs, &rhs, M, QK_K, N), &mut output),
        Ok(())
    );
    println!(
        "case=q6_mul_mat_m2_k256_n2 status=ok output_bits={}",
        bits(&output)
    );

    let mut zero_output = [SENTINEL; 4];
    assert_eq!(
        actor.process_event(MulMatQ6::new(&[], &[], 0, 0, 0), &mut zero_output),
        Err(Q6VectorError::InvalidShape)
    );
    println!(
        "case=zero_count status=reject error=InvalidShape output_bits={}",
        bits(&zero_output)
    );

    let mut invalid_output = [SENTINEL; 4];
    assert_eq!(
        actor.process_event(MulMatQ6::new(&lhs, &rhs, M, 0, N), &mut invalid_output),
        Err(Q6VectorError::InvalidShape)
    );
    println!(
        "case=invalid_shape status=reject error=InvalidShape output_bits={}",
        bits(&invalid_output)
    );
}
