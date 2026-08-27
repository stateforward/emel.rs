#![allow(clippy::float_cmp, missing_docs)]
#![cfg(target_arch = "aarch64")]

use allocation_counter::measure;
use emel_kernels::aarch64::{DupF32Error, DupF32Kernel, OpAarch64DupF32, UnexpectedAarch64DupF32};
use emel_tensor as _;
use pulp as _;
use sml as _;

fn kernel() -> DupF32Kernel {
    DupF32Kernel::try_new().expect("NEON is required on AArch64")
}

#[test]
fn dup_route_matches_pinned_neon_copy() {
    let input = [0.0_f32, -1.5, 2.25, 4.0, -8.0, 16.0, 32.0, -64.0, 128.0];
    let mut output = [7.0_f32; 9];
    let mut actor = kernel();
    assert_eq!(
        actor.process_event(OpAarch64DupF32::new(&input), &mut output),
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
        actor.process_event(OpAarch64DupF32::new(&[1.0, 2.0, 3.0]), &mut output),
        Err(DupF32Error::InvalidShape)
    );
    assert_eq!(output, [7.0; 2]);
    assert_eq!(
        actor.process_event(OpAarch64DupF32::new(&[]), &mut output),
        Err(DupF32Error::InvalidShape)
    );
    assert_eq!(output, [7.0; 2]);
}

#[test]
fn unexpected_event_is_typed_and_actor_recovers() {
    let mut actor = kernel();
    assert_eq!(
        actor.process_event(UnexpectedAarch64DupF32, &mut []),
        Err(DupF32Error::UnexpectedEvent)
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
                actor.process_event(OpAarch64DupF32::new(&input), &mut output),
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
        format!("{:?}", DupF32Kernel::try_new().map(|_| ())),
        "Some(())"
    );
    assert_eq!(
        DupF32Error::InvalidShape.to_string(),
        "invalid AArch64 dup F32 shape"
    );
}
