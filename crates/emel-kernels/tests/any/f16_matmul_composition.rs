#![allow(clippy::float_cmp, missing_docs)]

use allocation_counter::measure;
use emel_kernels::Kernel;
use emel_kernels::any::event::{Event, F16MatmulError, F16View, OpMulMatF16, UnexpectedF16Matmul};
use emel_kernels::any::tensor_view::{DType, Layout, TensorViewMut};
use emel_tensor as _;
use pulp as _;
use sml as _;

const fn f16_layout(ne: [u64; 4]) -> Layout {
    let mut nb = [2_u64; 4];
    nb[1] = 2 * ne[0];
    nb[2] = nb[1] * ne[1];
    nb[3] = nb[2] * ne[2];
    Layout::new(DType::F16, ne, nb)
}

fn f32_layout(ne: [u64; 4]) -> Layout {
    Layout::contiguous(DType::F32, ne).expect("fixture layout fits")
}

#[test]
fn root_kernel_composes_f16_matmul_child_and_preserves_orientation() {
    let lhs_layout = f16_layout([3, 2, 1, 1]);
    let rhs_layout = f16_layout([3, 2, 1, 1]);
    let destination_layout = f32_layout([2, 2, 1, 1]);
    let lhs = [0x3c00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600];
    let rhs = [0x3c00, 0x0000, 0x0000, 0x3c00, 0x3c00, 0x3c00];
    let mut destination = [0.0_f32; 4];
    let mut actor = Kernel::new();

    assert_eq!(
        actor.process_event(OpMulMatF16::new(
            F16View::new(&lhs, lhs_layout),
            F16View::new(&rhs, rhs_layout),
            TensorViewMut::new(&mut destination, destination_layout),
        )),
        Ok(())
    );
    assert_eq!(destination, [1.0, 4.0, 6.0, 15.0]);
    assert!(actor.is_ready());
}

#[test]
fn root_kernel_composes_typed_rejection_without_mutating_output() {
    let lhs_layout = f16_layout([3, 2, 1, 1]);
    let destination_layout = f32_layout([2, 2, 1, 1]);
    let lhs = [0x3c00_u16; 6];
    let mut destination = [9.0_f32; 4];
    let mut actor = Kernel::new();

    assert_eq!(
        actor.process_event(OpMulMatF16::new(
            F16View::new(&lhs, lhs_layout),
            F16View::new(&lhs, f16_layout([2, 2, 1, 1])),
            TensorViewMut::new(&mut destination, destination_layout),
        )),
        Err(F16MatmulError::ShapeMismatch)
    );
    assert_eq!(destination, [9.0; 4]);

    assert_eq!(
        actor.process_event(OpMulMatF16::new(
            F16View::new(&lhs, lhs_layout),
            F16View::new(
                &[0_u16; 6],
                Layout::new(DType::F16, [3, 2, 1, 1], [4, 12, 24, 48]),
            ),
            TensorViewMut::new(&mut destination, destination_layout),
        )),
        Err(F16MatmulError::InvalidView)
    );
    assert!(actor.is_ready());
}

#[test]
fn root_kernel_exposes_explicit_unexpected_event_and_allocation_free_dispatch() {
    let mut actor = Kernel::new();
    assert_eq!(
        actor.process_event(UnexpectedF16Matmul),
        Err(F16MatmulError::UnexpectedEvent)
    );
    assert!(actor.is_ready());

    let lhs_layout = f16_layout([1, 1, 1, 1]);
    let destination_layout = f32_layout([1, 1, 1, 1]);
    let lhs = [0x3c00_u16];
    let mut destination = [0.0_f32];
    let allocation = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                actor.process_event(OpMulMatF16::new(
                    F16View::new(&lhs, lhs_layout),
                    F16View::new(&lhs, lhs_layout),
                    TensorViewMut::new(&mut destination, destination_layout),
                )),
                Ok(())
            );
        }
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(allocation.bytes_total, 0);
}

#[test]
fn root_kernel_event_trait_dispatches_public_f16_event() {
    let lhs_layout = f16_layout([1, 1, 1, 1]);
    let destination_layout = f32_layout([1, 1, 1, 1]);
    let lhs = [0x3c00_u16];
    let mut destination = [0.0_f32];
    let mut actor = Kernel::new();
    let result = <OpMulMatF16<'_> as Event>::dispatch(
        OpMulMatF16::new(
            F16View::new(&lhs, lhs_layout),
            F16View::new(&lhs, lhs_layout),
            TensorViewMut::new(&mut destination, destination_layout),
        ),
        &mut actor,
    );
    assert_eq!(result, Ok(()));
    assert_eq!(destination, [1.0]);
}
