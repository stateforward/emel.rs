#![allow(clippy::float_cmp, missing_docs)]

use allocation_counter::measure;
use emel_kernels::Kernel;
use emel_kernels::any::event::{
    Event, MatmulError, OpMulMat, OpMulMatArgmax, UnexpectedMatmul, UnexpectedMatmulArgmax,
};
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};
use emel_tensor as _;
use pulp as _;
use sml as _;

fn layout(ne: [u64; 4]) -> Layout {
    Layout::contiguous(DType::F32, ne).expect("F32 layout fits")
}

#[test]
fn root_forwards_dense_matmul_to_the_public_child_actor() {
    let lhs_layout = layout([3, 2, 1, 1]);
    let rhs_layout = layout([2, 3, 1, 1]);
    let destination_layout = layout([2, 2, 1, 1]);
    let lhs = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs = [7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut destination = [0.0_f32; 4];
    let mut kernel = Kernel::new();

    assert_eq!(
        kernel.process_event(OpMulMat::new(
            TensorView::new(&lhs, lhs_layout),
            TensorView::new(&rhs, rhs_layout),
            TensorViewMut::new(&mut destination, destination_layout),
        )),
        Ok(())
    );
    assert_eq!(destination, [58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn root_forwards_argmax_to_the_separate_public_child_actor() {
    let lhs_layout = layout([3, 2, 1, 1]);
    let rhs_layout = layout([1, 3, 1, 1]);
    let destination_layout = layout([1, 1, 1, 1]);
    let lhs = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs = [1.0_f32, 1.0, 1.0];
    let mut destination = [0.0_f32];
    let mut kernel = Kernel::new();

    assert_eq!(
        kernel.process_event(OpMulMatArgmax::new(
            TensorView::new(&lhs, lhs_layout),
            TensorView::new(&rhs, rhs_layout),
            TensorViewMut::new(&mut destination, destination_layout),
        )),
        Ok(1)
    );
    assert_eq!(destination, [15.0]);
}

#[test]
fn root_rejects_invalid_matmul_without_mutating_output() {
    let lhs_layout = layout([3, 2, 1, 1]);
    let rhs_layout = layout([2, 2, 1, 1]);
    let destination_layout = layout([2, 2, 1, 1]);
    let lhs = [1.0_f32; 6];
    let rhs = [1.0_f32; 4];
    let mut destination = [9.0_f32; 4];
    let mut kernel = Kernel::new();

    assert_eq!(
        kernel.process_event(OpMulMat::new(
            TensorView::new(&lhs, lhs_layout),
            TensorView::new(&rhs, rhs_layout),
            TensorViewMut::new(&mut destination, destination_layout),
        )),
        Err(MatmulError::ShapeMismatch)
    );
    assert_eq!(destination, [9.0; 4]);
}

#[test]
fn root_unexpected_matmul_is_typed_and_recovers() {
    let lhs_layout = layout([1, 1, 1, 1]);
    let rhs_layout = layout([1, 1, 1, 1]);
    let mut destination = [0.0_f32; 1];
    let lhs = [2.0_f32];
    let rhs = [3.0_f32];
    let mut kernel = Kernel::new();

    assert_eq!(
        kernel.process_event(UnexpectedMatmul),
        Err(MatmulError::UnexpectedEvent)
    );
    assert_eq!(
        kernel.process_event(OpMulMat::new(
            TensorView::new(&lhs, lhs_layout),
            TensorView::new(&rhs, rhs_layout),
            TensorViewMut::new(&mut destination, lhs_layout),
        )),
        Ok(())
    );
    assert_eq!(destination, [6.0]);
}

#[test]
fn root_unexpected_argmax_is_typed_and_recovers() {
    let lhs_layout = layout([1, 1, 1, 1]);
    let rhs_layout = layout([1, 1, 1, 1]);
    let mut destination = [0.0_f32; 1];
    let lhs = [2.0_f32];
    let rhs = [3.0_f32];
    let mut kernel = Kernel::new();

    assert_eq!(
        kernel.process_event(UnexpectedMatmulArgmax),
        Err(MatmulError::UnexpectedEvent)
    );
    assert_eq!(
        kernel.process_event(OpMulMatArgmax::new(
            TensorView::new(&lhs, lhs_layout),
            TensorView::new(&rhs, rhs_layout),
            TensorViewMut::new(&mut destination, lhs_layout),
        )),
        Ok(0)
    );
    assert_eq!(destination, [6.0]);
}

#[test]
fn root_matmul_dispatch_is_allocation_free_after_construction() {
    let lhs_layout = layout([3, 2, 1, 1]);
    let rhs_layout = layout([2, 3, 1, 1]);
    let destination_layout = layout([2, 2, 1, 1]);
    let lhs = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs = [7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let mut destination = [0.0_f32; 4];
    let mut kernel = Kernel::new();

    let allocation = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                kernel.process_event(OpMulMat::new(
                    TensorView::new(&lhs, lhs_layout),
                    TensorView::new(&rhs, rhs_layout),
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
fn all_root_matmul_event_implementations_are_publicly_dispatchable() {
    fn dispatches<E: Event>(event: E, kernel: &mut Kernel) -> E::Output {
        event.dispatch(kernel)
    }

    let lhs_layout = layout([1, 1, 1, 1]);
    let rhs_layout = layout([1, 1, 1, 1]);
    let lhs = [1.0_f32];
    let rhs = [1.0_f32];
    let mut destination = [0.0_f32; 1];
    let mut kernel = Kernel::new();
    assert_eq!(
        dispatches(
            OpMulMat::new(
                TensorView::new(&lhs, lhs_layout),
                TensorView::new(&rhs, rhs_layout),
                TensorViewMut::new(&mut destination, lhs_layout),
            ),
            &mut kernel,
        ),
        Ok(())
    );
}
