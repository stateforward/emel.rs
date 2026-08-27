#![allow(clippy::float_cmp, missing_docs)]
#![cfg(target_arch = "aarch64")]

use allocation_counter::measure;
use emel_kernels::aarch64::{
    OpAarch64PowerSqr, OpAarch64PowerSqrt, PowerF32Error, PowerF32Kernel, UnexpectedAarch64PowerF32,
};
use emel_tensor as _;
use pulp as _;
use sml as _;

fn kernel() -> PowerF32Kernel {
    PowerF32Kernel::try_new().expect("NEON is required on AArch64")
}

#[test]
fn power_routes_match_pinned_neon_formulas() {
    let input = [0.0_f32, 0.25, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0];
    let mut output = [0.0_f32; 9];
    let mut actor = kernel();
    assert_eq!(
        actor.process_event(OpAarch64PowerSqr::new(&input), &mut output),
        Ok(())
    );
    assert_eq!(
        output,
        [0.0, 0.0625, 1.0, 16.0, 81.0, 256.0, 625.0, 1296.0, 2401.0]
    );
    let squared = output;
    assert_eq!(
        actor.process_event(OpAarch64PowerSqrt::new(&squared), &mut output),
        Ok(())
    );
    assert_eq!(output, input);
    assert!(actor.is_ready());
}

#[test]
fn invalid_shape_is_explicit_and_does_not_mutate_output() {
    let mut actor = kernel();
    let mut output = [7.0_f32; 2];
    assert_eq!(
        actor.process_event(OpAarch64PowerSqr::new(&[1.0, 4.0, 9.0]), &mut output),
        Err(PowerF32Error::InvalidShape)
    );
    assert_eq!(output, [7.0; 2]);
    assert_eq!(
        actor.process_event(OpAarch64PowerSqrt::new(&[]), &mut output),
        Err(PowerF32Error::InvalidShape)
    );
    assert_eq!(output, [7.0; 2]);
}

#[test]
fn unexpected_event_is_typed_and_actor_recovers() {
    let mut actor = kernel();
    assert_eq!(
        actor.process_event(UnexpectedAarch64PowerF32, &mut []),
        Err(PowerF32Error::UnexpectedEvent)
    );
    assert!(actor.is_ready());
}

#[test]
fn dispatch_is_allocation_free_after_construction() {
    let mut actor = kernel();
    let input = [1.0_f32, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0];
    let mut output = [0.0_f32; 9];
    let allocation = measure(|| {
        for _ in 0..64 {
            assert_eq!(
                actor.process_event(OpAarch64PowerSqr::new(&input), &mut output),
                Ok(())
            );
            assert_eq!(
                actor.process_event(OpAarch64PowerSqrt::new(&input), &mut output),
                Ok(())
            );
        }
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(allocation.bytes_total, 0);
}

#[test]
fn errors_are_typed_and_debuggable() {
    assert_eq!(
        format!("{:?}", PowerF32Kernel::try_new().map(|_| ())),
        "Some(())"
    );
    assert_eq!(
        PowerF32Error::InvalidShape.to_string(),
        "invalid AArch64 power F32 shape"
    );
}
