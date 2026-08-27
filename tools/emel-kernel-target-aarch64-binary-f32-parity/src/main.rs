//! Split Rust observer for the public AArch64 target binary router.

#![cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]

use emel_kernels::aarch64::{BinaryAdd, BinaryDiv, BinaryF32Error, BinaryMul, BinarySub, Kernel};

const SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const EVENTS_BLOB: &str = "4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9";
const DETAIL_BLOB: &str = "c8a82643eabfe8f2d7883e655955f455794511b0";
const GUARDS_BLOB: &str = "c25714566ec9a02679daef85089544575123408e";
const ACTIONS_BLOB: &str = "267d4f74e6e7498155c8535920322ffef2c02fb6";
const SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";
const SENTINEL: f32 = f32::from_bits(0x7fc0_0001);

#[cfg(not(target_arch = "aarch64"))]
fn main() {
    eprintln!("AArch64 binary observer requires an AArch64 build");
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
    let lhs = [8.0_f32, -9.0, 6.0, 4.0, -2.0, 3.0, 7.0, 11.0, 5.0];
    let rhs = [2.0_f32, 3.0, -2.0, 4.0, 2.0, -3.0, 7.0, 11.0, 5.0];
    println!("kernel-target-aarch64-binary-f32-live/v1");
    println!("source_repository=stateforward/emel.cpp");
    println!("source_commit={SOURCE_COMMIT}");
    println!("source_kernel_events_blob={EVENTS_BLOB}");
    println!("source_kernel_detail_blob={DETAIL_BLOB}");
    println!("source_kernel_aarch64_guards_blob={GUARDS_BLOB}");
    println!("source_kernel_aarch64_actions_blob={ACTIONS_BLOB}");
    println!("source_kernel_aarch64_sm_blob={SM_BLOB}");
    println!("target_arch=aarch64");
    println!("scope=target_router_binary_f32_add_sub_mul_div_dense_equal_length_and_rejection");
    println!("execution=split_pinned_aarch64_sm_and_public_target_router");

    let mut actor = Kernel::try_new().expect("AArch64 target router");
    let mut output = [0.0_f32; 9];

    assert_eq!(
        actor.process_event(BinaryAdd::new(&lhs, &rhs), &mut output),
        Ok(())
    );
    println!("case=add status=ok output_bits={}", bits(&output));
    assert_eq!(
        actor.process_event(BinarySub::new(&lhs, &rhs), &mut output),
        Ok(())
    );
    println!("case=sub status=ok output_bits={}", bits(&output));
    assert_eq!(
        actor.process_event(BinaryMul::new(&lhs, &rhs), &mut output),
        Ok(())
    );
    println!("case=mul status=ok output_bits={}", bits(&output));
    assert_eq!(
        actor.process_event(BinaryDiv::new(&lhs, &rhs), &mut output),
        Ok(())
    );
    println!("case=div status=ok output_bits={}", bits(&output));

    let mut invalid_output = [SENTINEL; 9];
    assert_eq!(
        actor.process_event(BinaryAdd::new(&[1.0_f32], &[2.0, 3.0]), &mut invalid_output,),
        Err(emel_kernels::aarch64::BinaryF32Error::InvalidShape)
    );
    assert!(
        invalid_output
            .iter()
            .all(|value| value.to_bits() == SENTINEL.to_bits())
    );
    println!(
        "case=invalid_shape status=reject error={:?} output_bits={}",
        BinaryF32Error::InvalidShape,
        bits(&invalid_output)
    );
}
