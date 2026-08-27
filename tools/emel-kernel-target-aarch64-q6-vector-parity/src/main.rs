//! Split Rust observer for the public AArch64 q6_k vector router.

#![cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]

use emel_kernels::aarch64::{Kernel, MulMatQ6Vector, Q6VectorError};

const COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const GUARDS_BLOB: &str = "c25714566ec9a02679daef85089544575123408e";
const ACTIONS_BLOB: &str = "267d4f74e6e7498155c8535920322ffef2c02fb6";
const SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";
const SENTINEL: f32 = f32::from_bits(0x7fc0_0001);
const K: usize = 256;
const M: usize = 5;
const Q6_K_BLOCK_BYTES: usize = 210;

#[cfg(not(target_arch = "aarch64"))]
fn main() {
    eprintln!("AArch64 q6_k vector observer requires an AArch64 build");
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
fn pack_lhs() -> [u8; M * Q6_K_BLOCK_BYTES] {
    let mut lhs = [0_u8; M * Q6_K_BLOCK_BYTES];
    for row in 0..M {
        let offset = row * Q6_K_BLOCK_BYTES;
        let ql = u8::try_from(0x11 * (row + 1)).unwrap();
        for index in 0..128 {
            lhs[offset + index] = ql;
        }
        for index in 0..64 {
            lhs[offset + 128 + index] = 0xa5;
        }
        for index in 0..16 {
            lhs[offset + 192 + index] = i8::try_from(row as i32 + 1).unwrap().to_ne_bytes()[0];
        }
        lhs[offset + 208..offset + 210].copy_from_slice(&0x3c00_u16.to_le_bytes());
    }
    lhs
}

#[cfg(target_arch = "aarch64")]
fn rhs_values() -> [f32; K] {
    let mut rhs = [0.0_f32; K];
    for index in 0..K {
        rhs[index] = (i32::try_from(index % 17).unwrap() - 8) as f32 * 0.125;
    }
    rhs
}

#[cfg(target_arch = "aarch64")]
fn main() {
    println!("kernel-target-aarch64-q6-vector-live/v1");
    println!("source_repository=stateforward/emel.cpp");
    println!("source_commit={COMMIT}");
    println!("source_kernel_aarch64_guards_blob={GUARDS_BLOB}");
    println!("source_kernel_aarch64_actions_blob={ACTIONS_BLOB}");
    println!("source_kernel_aarch64_sm_blob={SM_BLOB}");
    println!("target_arch=aarch64");
    println!("scope=target_router_q6_vector_m5_k256_and_typed_rejection");
    println!("execution=split_pinned_aarch64_sm_and_public_target_router");

    let lhs = pack_lhs();
    let rhs = rhs_values();
    let mut actor = Kernel::try_new().expect("AArch64 target router");
    let mut output = [0.0_f32; M];
    assert_eq!(
        actor.process_event(MulMatQ6Vector::new(&lhs, &rhs, M, K), &mut output),
        Ok(())
    );
    println!(
        "case=q6_vector_m5_k256 status=ok output_bits={}",
        bits(&output)
    );

    let mut zero_output = [SENTINEL];
    assert_eq!(
        actor.process_event(MulMatQ6Vector::new(&[], &[], 0, 0), &mut zero_output),
        Err(Q6VectorError::InvalidShape)
    );
    println!(
        "case=zero_count status=reject error=InvalidShape output_bits={}",
        bits(&zero_output)
    );

    let mut invalid_output = [SENTINEL; M];
    assert_eq!(
        actor.process_event(MulMatQ6Vector::new(&lhs, &rhs, M, 32), &mut invalid_output),
        Err(Q6VectorError::InvalidShape)
    );
    println!(
        "case=invalid_shape status=reject error=InvalidShape output_bits={}",
        bits(&invalid_output)
    );
}
