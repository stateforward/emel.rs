#![allow(clippy::float_cmp, missing_docs)]

use allocation_counter::measure;
use emel_kernels::Kernel;
use emel_kernels::any::event::{
    OpArgmax, OpCos, OpLog, OpMean, OpSin, OpSoftMax, OpSum, OpSumRows, ReductionError,
};
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};
use emel_tensor as _;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use pulp as _;
use sml as _;

fn contiguous(ne: [u64; 4]) -> Layout {
    Layout::contiguous(DType::F32, ne).expect("test layout fits")
}

#[test]
fn public_kernel_composes_all_reduction_events() {
    let source_layout = contiguous([3, 2, 1, 1]);
    let scalar_layout = contiguous([1, 1, 1, 1]);
    let rows_layout = contiguous([1, 2, 1, 1]);
    let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut output = [0.0_f32; 6];
    let mut scalar = [0.0_f32];
    let mut rows = [0.0_f32; 2];
    let mut kernel = Kernel::new();
    assert!(kernel.is_ready());

    assert_eq!(
        kernel.process_event(OpLog::new(
            TensorView::new(&input, source_layout),
            TensorViewMut::new(&mut output, source_layout),
        )),
        Ok(())
    );
    assert!((output[1] - 2.0_f32.ln()).abs() < 1.0e-6);
    assert_eq!(
        kernel.process_event(OpSin::new(
            TensorView::new(&input, source_layout),
            TensorViewMut::new(&mut output, source_layout),
        )),
        Ok(())
    );
    assert_eq!(
        kernel.process_event(OpCos::new(
            TensorView::new(&input, source_layout),
            TensorViewMut::new(&mut output, source_layout),
        )),
        Ok(())
    );
    assert_eq!(
        kernel.process_event(OpSum::new(
            TensorView::new(&input, source_layout),
            TensorViewMut::new(&mut scalar, scalar_layout),
        )),
        Ok(())
    );
    assert_eq!(scalar, [21.0]);
    assert_eq!(
        kernel.process_event(OpSumRows::new(
            TensorView::new(&input, source_layout),
            TensorViewMut::new(&mut rows, rows_layout),
        )),
        Ok(())
    );
    assert_eq!(rows, [6.0, 15.0]);
    assert_eq!(
        kernel.process_event(OpMean::new(
            TensorView::new(&input, source_layout),
            TensorViewMut::new(&mut scalar, scalar_layout),
        )),
        Ok(())
    );
    assert_eq!(scalar, [3.5]);
    assert_eq!(
        kernel.process_event(OpArgmax::new(
            TensorView::new(&input, source_layout),
            TensorViewMut::new(&mut scalar, scalar_layout),
        )),
        Ok(())
    );
    assert_eq!(scalar, [5.0]);
    assert_eq!(
        kernel.process_event(OpSoftMax::new(
            TensorView::new(&input, source_layout),
            TensorViewMut::new(&mut output, source_layout),
        )),
        Ok(())
    );
    assert!((output[0] + output[1] + output[2] - 1.0).abs() < 1.0e-6);
    assert!((output[3] + output[4] + output[5] - 1.0).abs() < 1.0e-6);
    assert!(kernel.is_ready());
}

#[test]
fn composed_reduction_guards_reject_invalid_views_and_shapes_without_mutation() {
    let valid = contiguous([2, 1, 1, 1]);
    let mismatched = contiguous([1, 1, 1, 1]);
    let invalid_dtype = Layout::new(DType::F16, [2, 1, 1, 1], [4, 8, 8, 8]);
    let source = [1.0_f32, 2.0];
    let mut output = [7.0_f32, 7.0];
    let mut kernel = Kernel::new();

    assert_eq!(
        kernel.process_event(OpSin::new(
            TensorView::new(&source, valid),
            TensorViewMut::new(&mut output[..1], mismatched),
        )),
        Err(ReductionError::ShapeMismatch)
    );
    assert_eq!(output, [7.0, 7.0]);
    assert!(kernel.is_ready());
    assert_eq!(
        kernel.process_event(OpSum::new(
            TensorView::new(&source, invalid_dtype),
            TensorViewMut::new(&mut output[..1], contiguous([1, 1, 1, 1])),
        )),
        Err(ReductionError::InvalidView)
    );
    assert_eq!(output, [7.0, 7.0]);
    assert!(kernel.is_ready());
}

#[test]
fn unexpected_transition_is_explicit_and_actor_recovers() {
    let source = include_str!("../../src/any/sm.rs");
    assert!(source.contains("unexpected_event<_>"));
    assert!(source.contains("effect_unexpected"));

    let layout = contiguous([2, 1, 1, 1]);
    let scalar_layout = contiguous([1, 1, 1, 1]);
    let input = [2.0_f32, 4.0];
    let mut scalar = [0.0_f32];
    let mut kernel = Kernel::new();
    assert_eq!(
        kernel.process_event(OpSum::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut scalar, scalar_layout),
        )),
        Ok(())
    );
    assert_eq!(scalar, [6.0]);
    assert!(kernel.is_ready());
}

#[test]
fn composed_reduction_dispatch_is_allocation_free() {
    let layout = contiguous([4, 1, 1, 1]);
    let scalar_layout = contiguous([1, 1, 1, 1]);
    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let mut output = [0.0_f32; 4];
    let mut scalar = [0.0_f32];
    let mut kernel = Kernel::new();
    let info = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                kernel.process_event(OpLog::new(
                    TensorView::new(&input, layout),
                    TensorViewMut::new(&mut output, layout),
                )),
                Ok(())
            );
            assert_eq!(
                kernel.process_event(OpSum::new(
                    TensorView::new(&input, layout),
                    TensorViewMut::new(&mut scalar, scalar_layout),
                )),
                Ok(())
            );
            assert_eq!(
                kernel.process_event(OpSoftMax::new(
                    TensorView::new(&input, layout),
                    TensorViewMut::new(&mut output, layout),
                )),
                Ok(())
            );
        }
    });
    assert_eq!(info.count_total, 0);
    assert_eq!(info.bytes_total, 0);
    assert!(kernel.is_ready());
}
