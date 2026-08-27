//! Split Rust observer for the public AArch64 target row-broadcast router.

#![cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]

use emel_kernels::aarch64::{BroadcastAdd, BroadcastF32Error, BroadcastMul, Kernel};

const COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const EVENTS_BLOB: &str = "4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9";
const DETAIL_BLOB: &str = "c8a82643eabfe8f2d7883e655955f455794511b0";
const GUARDS_BLOB: &str = "c25714566ec9a02679daef85089544575123408e";
const ACTIONS_BLOB: &str = "267d4f74e6e7498155c8535920322ffef2c02fb6";
const SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";
const SENTINEL: f32 = f32::from_bits(0x7fc0_0001);

#[cfg(not(target_arch = "aarch64"))]
fn main() {
    eprintln!("AArch64 broadcast observer requires an AArch64 build");
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
    let source = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let row = [10.0_f32, 20.0, 30.0];
    println!("kernel-target-aarch64-broadcast-f32-live/v1");
    println!("source_repository=stateforward/emel.cpp");
    println!("source_commit={COMMIT}");
    println!("source_kernel_events_blob={EVENTS_BLOB}");
    println!("source_kernel_detail_blob={DETAIL_BLOB}");
    println!("source_kernel_aarch64_guards_blob={GUARDS_BLOB}");
    println!("source_kernel_aarch64_actions_blob={ACTIONS_BLOB}");
    println!("source_kernel_aarch64_sm_blob={SM_BLOB}");
    println!("target_arch=aarch64");
    println!("scope=target_router_binary_f32_row_broadcast_add_mul_dense");
    println!("execution=split_pinned_aarch64_sm_and_public_target_router");

    let mut actor = Kernel::try_new().expect("AArch64 target router");
    let mut output = [0.0_f32; 6];
    assert_eq!(
        actor.process_event(BroadcastAdd::new(&source, &row), &mut output),
        Ok(())
    );
    println!("case=row_add status=ok output_bits={}", bits(&output));
    assert_eq!(
        actor.process_event(BroadcastMul::new(&source, &row), &mut output),
        Ok(())
    );
    println!("case=row_mul status=ok output_bits={}", bits(&output));

    let mut invalid_output = [SENTINEL; 6];
    assert_eq!(
        actor.process_event(BroadcastAdd::new(&source, &[]), &mut invalid_output,),
        Err(BroadcastF32Error::InvalidShape)
    );
    assert!(
        invalid_output
            .iter()
            .all(|value| value.to_bits() == SENTINEL.to_bits())
    );
    println!(
        "case=invalid_shape status=reject error={:?} output_bits={}",
        BroadcastF32Error::InvalidShape,
        bits(&invalid_output)
    );
}
