#![allow(clippy::float_cmp)]
#![allow(missing_docs)]
#![cfg(target_arch = "x86_64")]

use allocation_counter as _;
use emel_kernels as _;
use emel_tensor as _;
use pulp as _;
use sml as _;

#[cfg(target_arch = "x86_64")]
use allocation_counter::measure;
#[cfg(target_arch = "x86_64")]
use emel_kernels::x86_64::{
    OpX86BroadcastAdd, OpX86BroadcastMul, UnexpectedX86BroadcastF32, X86BroadcastF32Error,
    X86BroadcastF32Kernel,
};

#[test]
fn row_broadcast_routes_match_pinned_formulas() {
    let Some(mut actor) = X86BroadcastF32Kernel::try_new() else {
        return;
    };
    let input = [8.0_f32, -9.0, 6.0, 4.0, -2.0, 3.0, 7.0, 11.0];
    let row = [2.0_f32, 3.0, -2.0, 4.0];
    let mut output = [0.0_f32; 8];
    assert_eq!(
        actor.process_event(OpX86BroadcastAdd::new(&input, &row), &mut output),
        Ok(())
    );
    assert_eq!(output, [10.0, -6.0, 4.0, 8.0, 0.0, 6.0, 5.0, 15.0]);
    assert_eq!(
        actor.process_event(OpX86BroadcastMul::new(&input, &row), &mut output),
        Ok(())
    );
    assert_eq!(output, [16.0, -27.0, -12.0, 16.0, -4.0, 9.0, -14.0, 44.0]);
    assert!(actor.is_ready());
}

#[test]
fn tails_and_multiple_rows_are_processed_without_scalar_backend() {
    let Some(mut actor) = X86BroadcastF32Kernel::try_new() else {
        return;
    };
    let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let row = [10.0_f32, 20.0, 30.0];
    let mut output = [0.0_f32; 9];
    assert_eq!(
        actor.process_event(OpX86BroadcastMul::new(&input, &row), &mut output),
        Ok(())
    );
    assert_eq!(
        output,
        [10.0, 40.0, 90.0, 40.0, 100.0, 180.0, 70.0, 160.0, 270.0]
    );
}

#[test]
fn invalid_shape_is_explicit_and_does_not_mutate_output() {
    let Some(mut actor) = X86BroadcastF32Kernel::try_new() else {
        return;
    };
    let mut output = [9.0_f32; 8];
    assert_eq!(
        actor.process_event(
            OpX86BroadcastAdd::new(&[1.0, 2.0, 3.0], &[1.0, 2.0]),
            &mut output,
        ),
        Err(X86BroadcastF32Error::InvalidShape)
    );
    assert_eq!(output, [9.0; 8]);
    assert_eq!(
        actor.process_event(OpX86BroadcastMul::new(&[1.0, 2.0], &[]), &mut output,),
        Err(X86BroadcastF32Error::InvalidShape)
    );
    assert_eq!(output, [9.0; 8]);
}

#[test]
fn unexpected_event_is_typed_and_machine_stays_ready() {
    let Some(mut actor) = X86BroadcastF32Kernel::try_new() else {
        return;
    };
    assert_eq!(
        actor.process_event(UnexpectedX86BroadcastF32, &mut []),
        Err(X86BroadcastF32Error::UnexpectedEvent)
    );
    assert!(actor.is_ready());
}

#[test]
fn dispatch_is_allocation_free_after_construction() {
    let Some(mut actor) = X86BroadcastF32Kernel::try_new() else {
        return;
    };
    let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let row = [2.0_f32, 3.0, 4.0];
    let mut output = [0.0_f32; 6];
    let allocations = measure(|| {
        assert_eq!(
            actor.process_event(OpX86BroadcastAdd::new(&input, &row), &mut output),
            Ok(())
        );
    });
    assert_eq!(allocations.count_total, 0);
}
