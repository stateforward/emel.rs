#![allow(missing_docs)]

use allocation_counter::measure;
use emel_kernels::Kernel;
use emel_kernels::any::event::{CountEqualError, OpCountEqual};

#[test]
fn count_equal_writes_matching_i32_count() {
    let lhs = [1_i32, 2, 3, 2, 8];
    let rhs = [1_i32, 4, 3, 2, 9];
    let mut output = [-1_i32];
    let mut kernel = Kernel::new();

    assert_eq!(
        kernel.process_event(OpCountEqual::new(&lhs, &rhs, &mut output)),
        Ok(())
    );
    assert_eq!(output, [3]);
    assert!(kernel.is_ready());
}

#[test]
fn count_equal_rejects_invalid_shapes_without_mutation() {
    let lhs = [1_i32, 2];
    let rhs = [1_i32];
    let mut output = [7_i32];
    let mut kernel = Kernel::new();

    assert_eq!(
        kernel.process_event(OpCountEqual::new(&lhs, &rhs, &mut output)),
        Err(CountEqualError::InvalidShape)
    );
    assert_eq!(output, [7]);
    assert_eq!(
        kernel.process_event(OpCountEqual::new(&[], &[], &mut output)),
        Err(CountEqualError::InvalidShape)
    );
    assert_eq!(output, [7]);
    assert!(kernel.is_ready());
}

#[test]
fn count_equal_dispatch_is_allocation_free() {
    let lhs = [1_i32, 2, 3, 4];
    let rhs = [1_i32, 0, 3, 9];
    let mut output = [0_i32];
    let mut kernel = Kernel::new();
    let info = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                kernel.process_event(OpCountEqual::new(&lhs, &rhs, &mut output)),
                Ok(())
            );
        }
    });
    assert_eq!(info.count_total, 0);
    assert_eq!(info.bytes_total, 0);
    assert_eq!(output, [2]);
}

#[test]
fn count_equal_source_route_is_explicit() {
    let source = include_str!("../../src/any/count_equal.rs");
    assert!(source.contains("guard_count_equal_valid"));
    assert!(source.contains("guard_count_equal_invalid"));
    assert!(source.contains("unexpected_event<_>"));
    assert!(source.contains("CountEqualMachineStates::Ready"));
}
