#![allow(clippy::float_cmp, missing_docs)]
#![cfg(target_arch = "aarch64")]

use allocation_counter::measure;
use emel_kernels::aarch64::{
    BroadcastF32Error, BroadcastF32Kernel, OpAarch64BroadcastAdd, OpAarch64BroadcastMul,
    UnexpectedAarch64BroadcastF32,
};
use emel_tensor as _;
use pulp as _;
use sml as _;

fn kernel() -> BroadcastF32Kernel {
    BroadcastF32Kernel::try_new().expect("NEON is required on AArch64")
}

#[test]
fn row_broadcast_routes_match_pinned_formulas() {
    let input = [8.0_f32, -9.0, 6.0, 4.0, -2.0, 3.0, 7.0, 11.0];
    let row = [2.0_f32, 3.0, -2.0, 4.0];
    let mut output = [0.0_f32; 8];
    let mut actor = kernel();
    assert_eq!(
        actor.process_event(OpAarch64BroadcastAdd::new(&input, &row), &mut output),
        Ok(())
    );
    assert_eq!(output, [10.0, -6.0, 4.0, 8.0, 0.0, 6.0, 5.0, 15.0]);
    assert_eq!(
        actor.process_event(OpAarch64BroadcastMul::new(&input, &row), &mut output),
        Ok(())
    );
    assert_eq!(output, [16.0, -27.0, -12.0, 16.0, -4.0, 9.0, -14.0, 44.0]);
    assert!(actor.is_ready());
}

#[test]
fn invalid_shape_is_explicit_and_does_not_mutate_output() {
    let mut actor = kernel();
    let mut output = [9.0_f32; 8];
    assert_eq!(
        actor.process_event(
            OpAarch64BroadcastAdd::new(&[1.0, 2.0, 3.0], &[1.0, 2.0]),
            &mut output
        ),
        Err(BroadcastF32Error::InvalidShape)
    );
    assert_eq!(output, [9.0; 8]);
    assert_eq!(
        actor.process_event(OpAarch64BroadcastMul::new(&[1.0, 2.0], &[]), &mut output),
        Err(BroadcastF32Error::InvalidShape)
    );
    assert_eq!(output, [9.0; 8]);
}

#[test]
fn unexpected_event_is_typed_and_machine_stays_ready() {
    let mut actor = kernel();
    assert_eq!(
        actor.process_event(UnexpectedAarch64BroadcastF32, &mut []),
        Err(BroadcastF32Error::UnexpectedEvent)
    );
    assert!(actor.is_ready());
}

#[test]
fn dispatch_is_allocation_free_after_construction() {
    let mut actor = kernel();
    let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let row = [2.0_f32, 3.0, 4.0];
    let mut output = [0.0_f32; 6];
    let allocations = measure(|| {
        assert_eq!(
            actor.process_event(OpAarch64BroadcastAdd::new(&input, &row), &mut output),
            Ok(())
        );
    });
    assert_eq!(allocations.count_total, 0);
}
