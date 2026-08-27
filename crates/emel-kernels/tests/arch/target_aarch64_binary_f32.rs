#![allow(clippy::float_cmp, missing_docs)]
#![cfg(target_arch = "aarch64")]

use allocation_counter::measure;
use emel_kernels::aarch64::{
    BinaryF32Error, BinaryF32Kernel, OpAarch64BinaryAdd, OpAarch64BinaryDiv, OpAarch64BinaryMul,
    OpAarch64BinarySub, UnexpectedAarch64BinaryF32,
};
use emel_tensor as _;
use pulp as _;
use sml as _;

fn kernel() -> BinaryF32Kernel {
    BinaryF32Kernel::try_new().expect("NEON is required on AArch64")
}

#[test]
fn binary_routes_match_pinned_neon_formulas() {
    let lhs = [8.0_f32, -9.0, 6.0, 4.0, -2.0, 3.0, 7.0, 11.0, 5.0];
    let rhs = [2.0_f32, 3.0, -2.0, 4.0, 2.0, -3.0, 7.0, 11.0, 5.0];
    let mut output = [0.0_f32; 9];
    let mut actor = kernel();
    assert_eq!(
        actor.process_event(OpAarch64BinaryAdd::new(&lhs, &rhs), &mut output),
        Ok(())
    );
    assert_eq!(output, [10.0, -6.0, 4.0, 8.0, 0.0, 0.0, 14.0, 22.0, 10.0]);
    assert_eq!(
        actor.process_event(OpAarch64BinarySub::new(&lhs, &rhs), &mut output),
        Ok(())
    );
    assert_eq!(output, [6.0, -12.0, 8.0, 0.0, -4.0, 6.0, 0.0, 0.0, 0.0]);
    assert_eq!(
        actor.process_event(OpAarch64BinaryMul::new(&lhs, &rhs), &mut output),
        Ok(())
    );
    assert_eq!(
        output,
        [16.0, -27.0, -12.0, 16.0, -4.0, -9.0, 49.0, 121.0, 25.0]
    );
    assert_eq!(
        actor.process_event(OpAarch64BinaryDiv::new(&lhs, &rhs), &mut output),
        Ok(())
    );
    assert_eq!(output, [4.0, -3.0, -3.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0]);
    assert!(actor.is_ready());
}

#[test]
fn invalid_shape_is_explicit_and_does_not_mutate_output() {
    let mut actor = kernel();
    let mut output = [9.0_f32; 3];
    assert_eq!(
        actor.process_event(OpAarch64BinaryAdd::new(&[1.0], &[2.0, 3.0]), &mut output),
        Err(BinaryF32Error::InvalidShape)
    );
    assert_eq!(output, [9.0; 3]);
    assert_eq!(
        actor.process_event(OpAarch64BinaryMul::new(&[], &[]), &mut output),
        Err(BinaryF32Error::InvalidShape)
    );
    assert_eq!(output, [9.0; 3]);
}

#[test]
fn unexpected_event_is_typed_and_machine_stays_ready() {
    let mut actor = kernel();
    assert_eq!(
        actor.process_event(UnexpectedAarch64BinaryF32, &mut []),
        Err(BinaryF32Error::UnexpectedEvent)
    );
    assert!(actor.is_ready());
}

#[test]
fn dispatch_is_allocation_free_after_construction() {
    let mut actor = kernel();
    let lhs = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let rhs = [2.0_f32, 3.0, 4.0, 5.0, 6.0];
    let mut output = [0.0_f32; 5];
    let allocations = measure(|| {
        assert_eq!(
            actor.process_event(OpAarch64BinaryAdd::new(&lhs, &rhs), &mut output),
            Ok(())
        );
    });
    assert_eq!(allocations.count_total, 0);
}
