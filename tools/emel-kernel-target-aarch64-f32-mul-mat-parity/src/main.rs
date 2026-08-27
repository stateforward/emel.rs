//! Split Rust observer for the public AArch64 dense F32 GEMM router.

#![cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]

use emel_kernels::aarch64::{F32MulMatError, Kernel, MulMatF32};

const COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const GUARDS_BLOB: &str = "c25714566ec9a02679daef85089544575123408e";
const ACTIONS_BLOB: &str = "267d4f74e6e7498155c8535920322ffef2c02fb6";
const SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";
const SENTINEL: f32 = f32::from_bits(0x7fc0_0001);
const M: usize = 5;
const K: usize = 8;
const N: usize = 4;

#[cfg(not(target_arch = "aarch64"))]
fn main() {
    eprintln!("AArch64 F32 mul_mat observer requires an AArch64 build");
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
fn fill_lhs() -> [f32; M * K] {
    let mut values = [0.0_f32; M * K];
    for row in 0..M {
        for depth in 0..K {
            values[row * K + depth] = (row as f32 + 1.0) * 0.125 + depth as f32 * 0.25;
        }
    }
    values
}

#[cfg(target_arch = "aarch64")]
fn fill_rhs() -> [f32; K * N] {
    let mut values = [0.0_f32; K * N];
    for depth in 0..K {
        for col in 0..N {
            values[depth * N + col] = (col as f32 + 1.0) * 0.5 - depth as f32 * 0.0625;
        }
    }
    values
}

#[cfg(target_arch = "aarch64")]
fn main() {
    println!("kernel-target-aarch64-f32-mul-mat-live/v1");
    println!("source_repository=stateforward/emel.cpp");
    println!("source_commit={COMMIT}");
    println!("source_kernel_aarch64_guards_blob={GUARDS_BLOB}");
    println!("source_kernel_aarch64_actions_blob={ACTIONS_BLOB}");
    println!("source_kernel_aarch64_sm_blob={SM_BLOB}");
    println!("target_arch=aarch64");
    println!("scope=target_router_f32_mul_mat_m5_k8_n4_and_typed_rejection");
    println!("execution=split_pinned_aarch64_sm_and_public_target_router");

    let lhs = fill_lhs();
    let rhs = fill_rhs();
    let mut actor = Kernel::try_new().expect("AArch64 target router");
    let mut output = [0.0_f32; M * N];
    assert_eq!(
        actor.process_event(MulMatF32::new(&lhs, &rhs, M, K, N), &mut output),
        Ok(())
    );
    println!(
        "case=f32_mul_mat_m5_k8_n4 status=ok output_bits={}",
        bits(&output)
    );

    let mut zero_output = [SENTINEL; 4];
    assert_eq!(
        actor.process_event(MulMatF32::new(&[], &[], 0, 0, 0), &mut zero_output),
        Err(F32MulMatError::InvalidShape)
    );
    println!(
        "case=zero_count status=reject error=InvalidShape output_bits={}",
        bits(&zero_output)
    );

    let mut invalid_output = [SENTINEL; M * N];
    assert_eq!(
        actor.process_event(MulMatF32::new(&[], &[], M, 0, N), &mut invalid_output),
        Err(F32MulMatError::InvalidShape)
    );
    println!(
        "case=invalid_shape status=reject error=InvalidShape output_bits={}",
        bits(&invalid_output)
    );
}
