//! Split Rust observer for the public AArch64 F32 GEMV router.

#![cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]

use emel_kernels::aarch64::{
    Gemv, Kernel, KernelError,
};

const COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const GUARDS_BLOB: &str = "c25714566ec9a02679daef85089544575123408e";
const ACTIONS_BLOB: &str = "267d4f74e6e7498155c8535920322ffef2c02fb6";
const SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";
const SENTINEL: f32 = f32::from_bits(0x7fc0_0001);

#[cfg(not(target_arch = "aarch64"))]
fn main() {
    eprintln!("AArch64 F32 GEMV observer requires an AArch64 build");
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
fn main() {
    let lhs = [
        0.25_f32, -1.5, 2.0, 3.25, -4.5, 5.0, 6.75, -7.0, 8.5, 9.0, -10.25, 11.5, 12.0, -13.75,
        14.25, 15.0, -16.5, 1.0, 2.25, -3.5, 4.0, 5.75, -6.0, 7.5, 8.0, -9.25, 10.5, 11.0, -12.75,
        13.25, 14.0, -15.5, 16.0, 17.75, -1.0, 2.0, 3.5, -4.0, 5.25, 6.0, -7.5, 8.75, 9.0, -10.0,
        11.25, 12.5, -13.0, 14.75, 15.0, -16.25, 17.0, 2.0, -2.5, 3.0, 4.25, -5.0, 6.5, 7.0, -8.75,
        9.5, 10.0, -11.5, 12.0, 13.25, -14.0, 15.5, 16.0, -17.75, -0.5, 1.5, 2.75, -3.0, 4.5, 5.25,
        -6.0, 7.75, 8.0, -9.5, 10.25, 11.0, -12.5, 13.0, 14.25, -15.0, 16.5,
    ];
    let rhs = [
        0.5_f32, -1.0, 1.5, 2.0, -2.5, 3.0, 3.5, -4.0, 4.5, 5.0, -5.5, 6.0, 6.5, -7.0, 7.5, 8.0,
        -8.5,
    ];
    println!("kernel-target-f32-gemv-aarch64-live/v1");
    println!("source_repository=stateforward/emel.cpp");
    println!("source_commit={COMMIT}");
    println!("source_kernel_aarch64_guards_blob={GUARDS_BLOB}");
    println!("source_kernel_aarch64_actions_blob={ACTIONS_BLOB}");
    println!("source_kernel_aarch64_sm_blob={SM_BLOB}");
    println!("target_arch=aarch64");
    println!("scope=target_router_f32_gemv_dense_contiguous_body_and_tail_rejection");
    println!("execution=split_pinned_aarch64_sm_and_public_target_router");

    let mut actor = Kernel::try_new().expect("AArch64 target router");
    let mut output = [0.0_f32; 5];
    assert_eq!(
        actor.process_event(Gemv::new(&lhs, &rhs, 5, 17), &mut output),
        Ok(())
    );
    println!("case=gemv_m5_k17 status=ok output_bits={}", bits(&output));

    let mut zero_output = [SENTINEL];
    assert_eq!(
        actor.process_event(Gemv::new(&[], &[], 0, 0), &mut zero_output),
        Err(emel_kernels::aarch64::F32GemvError::InvalidShape)
    );
    println!(
        "case=zero_count status=reject error=InvalidShape output_bits={}",
        bits(&zero_output)
    );

    let mut invalid_output = [SENTINEL; 5];
    assert_eq!(
        actor.process_event(
            Gemv::new(&lhs[..16], &rhs, 5, 17),
            &mut invalid_output
        ),
        Err(emel_kernels::aarch64::F32GemvError::InvalidShape)
    );
    assert!(
        invalid_output
            .iter()
            .all(|value| value.to_bits() == SENTINEL.to_bits())
    );
    println!(
        "case=invalid_shape status=reject error=InvalidShape output_bits={}",
        bits(&invalid_output)
    );
    assert!(actor.is_ready());
    let _ = KernelError::UnexpectedEvent;
}
