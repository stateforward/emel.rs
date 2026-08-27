//! Live observer for four public AArch64 target-router events.

#![cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]

use emel_kernels::aarch64::{
    BinaryAdd, Dup, Gemv, Kernel,
    KernelError, UnaryAbs, UnexpectedAarch64Kernel,
};

const COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const EVENTS_BLOB: &str = "4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9";
const DETAIL_BLOB: &str = "c8a82643eabfe8f2d7883e655955f455794511b0";
const GUARDS_BLOB: &str = "c25714566ec9a02679daef85089544575123408e";
const ACTIONS_BLOB: &str = "267d4f74e6e7498155c8535920322ffef2c02fb6";
const SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";
const SENTINEL: f32 = f32::from_bits(0x7fc0_0001);

#[cfg(not(target_arch = "aarch64"))]
fn main() {
    eprintln!("AArch64 router observer requires an AArch64 build");
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
    let unary_input = [-3.5_f32, -1.0, 0.0, 0.5, 2.25, -4.0, 5.5, -8.0, 9.75];
    let binary_lhs = [8.0_f32, -9.0, 6.0, 4.0, -2.0, 3.0, 7.0, 11.0, 5.0];
    let binary_rhs = [2.0_f32, 3.0, -2.0, 4.0, 2.0, -3.0, 7.0, 11.0, 5.0];
    let dup_input = [0.0_f32, -1.5, 2.25, 4.0, -8.0, 16.0, 32.0, -64.0, 128.0];
    let gemv_lhs = [
        0.25_f32, -1.5, 2.0, 3.25, -4.5, 5.0, 6.75, -7.0, 8.5, 9.0, -10.25, 11.5, 12.0, -13.75,
        14.25, 15.0, -16.5, 1.0, 2.25, -3.5, 4.0, 5.75, -6.0, 7.5, 8.0, -9.25, 10.5, 11.0, -12.75,
        13.25, 14.0, -15.5, 16.0, 17.75, -1.0, 2.0, 3.5, -4.0, 5.25, 6.0, -7.5, 8.75, 9.0, -10.0,
        11.25, 12.5, -13.0, 14.75, 15.0, -16.25, 17.0, 2.0, -2.5, 3.0, 4.25, -5.0, 6.5, 7.0, -8.75,
        9.5, 10.0, -11.5, 12.0, 13.25, -14.0, 15.5, 16.0, -17.75, -0.5, 1.5, 2.75, -3.0, 4.5, 5.25,
        -6.0, 7.75, 8.0, -9.5, 10.25, 11.0, -12.5, 13.0, 14.25, -15.0, 16.5,
    ];
    let gemv_rhs = [
        0.5_f32, -1.0, 1.5, 2.0, -2.5, 3.0, 3.5, -4.0, 4.5, 5.0, -5.5, 6.0, 6.5, -7.0, 7.5, 8.0,
        -8.5,
    ];

    println!("kernel-target-aarch64-router-live/v1");
    println!("source_repository=stateforward/emel.cpp");
    println!("source_commit={COMMIT}");
    println!("source_kernel_events_blob={EVENTS_BLOB}");
    println!("source_kernel_detail_blob={DETAIL_BLOB}");
    println!("source_kernel_aarch64_guards_blob={GUARDS_BLOB}");
    println!("source_kernel_aarch64_actions_blob={ACTIONS_BLOB}");
    println!("source_kernel_aarch64_sm_blob={SM_BLOB}");
    println!("target_arch=aarch64");
    println!("scope=target_router_unary_abs_binary_add_dup_gemv_and_unexpected_rejection");
    println!("execution=one_public_target_router_dispatching_four_live_events");

    let mut actor = Kernel::try_new().expect("AArch64 target router");
    let mut output = [0.0_f32; 9];
    assert_eq!(
        actor.process_event(UnaryAbs::new(&unary_input), &mut output),
        Ok(())
    );
    println!("case=unary_abs status=ok output_bits={}", bits(&output));
    assert_eq!(
        actor.process_event(
            BinaryAdd::new(&binary_lhs, &binary_rhs),
            &mut output
        ),
        Ok(())
    );
    println!("case=binary_add status=ok output_bits={}", bits(&output));
    assert_eq!(
        actor.process_event(Dup::new(&dup_input), &mut output),
        Ok(())
    );
    println!("case=dup status=ok output_bits={}", bits(&output));
    let mut gemv_output = [0.0_f32; 5];
    assert_eq!(
        actor.process_event(
            Gemv::new(&gemv_lhs, &gemv_rhs, 5, 17),
            &mut gemv_output
        ),
        Ok(())
    );
    println!(
        "case=gemv_m5_k17 status=ok output_bits={}",
        bits(&gemv_output)
    );

    let mut rejected = [SENTINEL; 9];
    assert_eq!(
        actor.process_event(UnexpectedAarch64Kernel, &mut rejected),
        Err(KernelError::UnexpectedEvent)
    );
    assert!(
        rejected
            .iter()
            .all(|value| value.to_bits() == SENTINEL.to_bits())
    );
    println!(
        "case=unexpected status=reject error=UnexpectedEvent output_bits={}",
        bits(&rejected)
    );
}
