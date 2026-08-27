#![allow(clippy::float_cmp, missing_docs)]

use allocation_counter::measure;
use emel_kernels::Kernel;
use emel_kernels::any::binary::BinaryKernel;
use emel_kernels::any::event::{
    BinaryError, OpAddTensorView, OpDivTensorView, OpMulTensorView, OpSubTensorView,
    UnexpectedTensorBinary,
};
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};
use emel_tensor as _;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use pulp as _;
use sml as _;

fn layout(count: u64) -> Layout {
    Layout::contiguous(DType::F32, [count, 1, 1, 1]).expect("layout fits")
}

#[test]
fn root_composes_all_four_tensor_view_binary_operations() {
    let layout = layout(4);
    let lhs = [8.0_f32, -6.0, 4.0, -2.0];
    let rhs = [2.0_f32, 3.0, -2.0, -4.0];
    let mut output = [0.0_f32; 4];
    let mut kernel = Kernel::new();
    assert!(kernel.is_ready());

    assert_eq!(
        kernel.process_event(OpAddTensorView::new(
            TensorView::new(&lhs, layout),
            TensorView::new(&rhs, layout),
            TensorViewMut::new(&mut output, layout),
        )),
        Ok(())
    );
    assert_eq!(output, [10.0, -3.0, 2.0, -6.0]);

    assert_eq!(
        kernel.process_event(OpSubTensorView::new(
            TensorView::new(&lhs, layout),
            TensorView::new(&rhs, layout),
            TensorViewMut::new(&mut output, layout),
        )),
        Ok(())
    );
    assert_eq!(output, [6.0, -9.0, 6.0, 2.0]);

    assert_eq!(
        kernel.process_event(OpMulTensorView::new(
            TensorView::new(&lhs, layout),
            TensorView::new(&rhs, layout),
            TensorViewMut::new(&mut output, layout),
        )),
        Ok(())
    );
    assert_eq!(output, [16.0, -18.0, -8.0, 8.0]);

    assert_eq!(
        kernel.process_event(OpDivTensorView::new(
            TensorView::new(&lhs, layout),
            TensorView::new(&rhs, layout),
            TensorViewMut::new(&mut output, layout),
        )),
        Ok(())
    );
    assert_eq!(output, [4.0, -2.0, -2.0, 0.5]);
    assert!(kernel.is_ready());
}

#[test]
fn root_binary_guards_reject_shape_and_view_without_mutation() {
    let valid = layout(4);
    let mismatch = layout(3);
    let invalid = Layout::new(DType::F16, [4, 1, 1, 1], [2, 8, 32, 32]);
    let lhs = [1.0_f32; 4];
    let rhs = [2.0_f32; 4];
    let mut output = [9.0_f32; 4];
    let mut kernel = Kernel::new();

    assert_eq!(
        kernel.process_event(OpMulTensorView::new(
            TensorView::new(&lhs, valid),
            TensorView::new(&rhs, mismatch),
            TensorViewMut::new(&mut output, valid),
        )),
        Err(BinaryError::ShapeMismatch)
    );
    assert_eq!(output, [9.0; 4]);

    assert_eq!(
        kernel.process_event(OpDivTensorView::new(
            TensorView::new(&lhs, invalid),
            TensorView::new(&rhs, valid),
            TensorViewMut::new(&mut output, valid),
        )),
        Err(BinaryError::InvalidView)
    );
    assert_eq!(output, [9.0; 4]);
    assert!(kernel.is_ready());
}

#[test]
fn root_binary_unexpected_route_is_typed_and_recovers() {
    let mut kernel = Kernel::new();
    assert_eq!(
        kernel.process_event(UnexpectedTensorBinary),
        Err(BinaryError::UnexpectedEvent)
    );

    let layout = layout(2);
    let lhs = [4.0_f32, 6.0];
    let rhs = [2.0_f32, 3.0];
    let mut output = [0.0_f32; 2];
    assert_eq!(
        kernel.process_event(OpDivTensorView::new(
            TensorView::new(&lhs, layout),
            TensorView::new(&rhs, layout),
            TensorViewMut::new(&mut output, layout),
        )),
        Ok(())
    );
    assert_eq!(output, [2.0, 2.0]);
    assert!(kernel.is_ready());
}

#[test]
fn root_binary_dispatch_is_allocation_free() {
    let layout = layout(16);
    let lhs = [4.0_f32; 16];
    let rhs = [2.0_f32; 16];
    let mut output = [0.0_f32; 16];
    let mut kernel = Kernel::new();
    let allocation = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                kernel.process_event(OpMulTensorView::new(
                    TensorView::new(&lhs, layout),
                    TensorView::new(&rhs, layout),
                    TensorViewMut::new(&mut output, layout),
                )),
                Ok(())
            );
        }
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(allocation.bytes_total, 0);
    assert!(kernel.is_ready());
}

#[test]
fn binary_child_exposes_generated_ready_state() {
    assert!(BinaryKernel::new().is_ready());
}
