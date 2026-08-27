#![allow(clippy::float_cmp, missing_docs)]

use allocation_counter::measure;
use emel_kernels::any::reductions::{
    OpArgmax, OpCos, OpLog, OpMean, OpSin, OpSoftMax, OpSum, OpSumRows, ReductionError,
    ReductionKernel,
};
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};
use emel_tensor as _;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use pulp as _;
use sml as _;

const PINNED_EMEL_CPP_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";

fn contiguous(ne: [u64; 4]) -> Layout {
    Layout::contiguous(DType::F32, ne).expect("test layout fits")
}

#[test]
fn unary_events_use_explicit_operation_routes() {
    assert_eq!(PINNED_EMEL_CPP_COMMIT.len(), 40);
    let layout = contiguous([3, 1, 1, 1]);
    let input = [1.0_f32, 2.0, 4.0];
    let mut output = [0.0_f32; 3];
    let mut kernel = ReductionKernel::new();

    assert_eq!(
        kernel.process_event(OpLog::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout),
        )),
        Ok(())
    );
    assert!((output[0] - 0.0).abs() < 1.0e-6);
    assert!((output[1] - 2.0_f32.ln()).abs() < 1.0e-6);

    assert_eq!(
        kernel.process_event(OpSin::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout),
        )),
        Ok(())
    );
    assert!((output[0] - 1.0_f32.sin()).abs() < 1.0e-6);

    assert_eq!(
        kernel.process_event(OpCos::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout),
        )),
        Ok(())
    );
    assert!((output[0] - 1.0_f32.cos()).abs() < 1.0e-6);
}

#[test]
fn scalar_and_row_reductions_match_pinned_dense_contract() {
    let source_layout = contiguous([3, 2, 1, 1]);
    let scalar_layout = contiguous([1, 1, 1, 1]);
    let rows_layout = contiguous([1, 2, 1, 1]);
    let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut scalar = [0.0_f32];
    let mut rows = [0.0_f32; 2];
    let mut kernel = ReductionKernel::new();

    assert_eq!(
        kernel.process_event(OpSum::new(
            TensorView::new(&input, source_layout),
            TensorViewMut::new(&mut scalar, scalar_layout),
        )),
        Ok(())
    );
    assert_eq!(scalar, [21.0]);

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
        kernel.process_event(OpSumRows::new(
            TensorView::new(&input, source_layout),
            TensorViewMut::new(&mut rows, rows_layout),
        )),
        Ok(())
    );
    assert_eq!(rows, [6.0, 15.0]);
}

#[test]
fn softmax_is_stable_and_operates_per_row() {
    let layout = contiguous([3, 2, 1, 1]);
    let input = [1000.0_f32, 1001.0, 1002.0, -1000.0, -999.0, -998.0];
    let mut output = [0.0_f32; 6];
    let mut kernel = ReductionKernel::new();

    assert_eq!(
        kernel.process_event(OpSoftMax::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout),
        )),
        Ok(())
    );
    assert!(output.iter().all(|value| value.is_finite()));
    assert!((output[0] + output[1] + output[2] - 1.0).abs() < 1.0e-6);
    assert!((output[3] + output[4] + output[5] - 1.0).abs() < 1.0e-6);
}

#[test]
fn softmax_accepts_implicit_contiguous_nb0_layouts() {
    let explicit_layout = contiguous([3, 2, 1, 1]);
    let implicit_layout = Layout::new(DType::F32, [3, 2, 1, 1], [0, 0, 0, 0]);
    let input = [0.25_f32, -1.5, 3.0, 8.0, 7.5, -4.0];
    let mut explicit_output = [0.0_f32; 6];
    let mut implicit_output = [0.0_f32; 6];
    let mut kernel = ReductionKernel::new();

    assert_eq!(
        kernel.process_event(OpSoftMax::new(
            TensorView::new(&input, explicit_layout),
            TensorViewMut::new(&mut explicit_output, explicit_layout),
        )),
        Ok(())
    );
    assert_eq!(
        kernel.process_event(OpSoftMax::new(
            TensorView::new(&input, implicit_layout),
            TensorViewMut::new(&mut implicit_output, implicit_layout),
        )),
        Ok(())
    );
    assert_eq!(
        implicit_output.map(f32::to_bits),
        explicit_output.map(f32::to_bits)
    );
}

#[test]
fn guards_separate_view_and_shape_rejections_without_mutation() {
    let valid = contiguous([2, 1, 1, 1]);
    let mismatched = contiguous([1, 1, 1, 1]);
    let invalid_dtype = Layout::new(DType::F16, [2, 1, 1, 1], [4, 8, 8, 8]);
    let invalid_bounds = Layout::new(DType::F32, [2, 1, 1, 1], [8, 8, 8, 8]);
    let source = [1.0_f32, 2.0];
    let mut output = [7.0_f32, 7.0];
    let mut kernel = ReductionKernel::new();

    assert_eq!(
        kernel.process_event(OpSin::new(
            TensorView::new(&source, valid),
            TensorViewMut::new(&mut output[..1], mismatched),
        )),
        Err(ReductionError::ShapeMismatch)
    );
    assert_eq!(output, [7.0, 7.0]);

    assert_eq!(
        kernel.process_event(OpSin::new(
            TensorView::new(&source, invalid_dtype),
            TensorViewMut::new(&mut output, valid),
        )),
        Err(ReductionError::InvalidView)
    );
    assert_eq!(output, [7.0, 7.0]);

    let mut scalar = [9.0_f32];
    assert_eq!(
        kernel.process_event(OpSum::new(
            TensorView::new(&source, invalid_bounds),
            TensorViewMut::new(&mut scalar, contiguous([1, 1, 1, 1])),
        )),
        Err(ReductionError::InvalidView)
    );
    assert_eq!(scalar, [9.0]);
}

#[test]
fn valid_reduction_dispatch_is_allocation_free() {
    let layout = contiguous([4, 1, 1, 1]);
    let scalar_layout = contiguous([1, 1, 1, 1]);
    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let mut scalar = [0.0_f32];
    let mut kernel = ReductionKernel::new();
    let mut result = Err(ReductionError::UnexpectedEvent);
    let info = measure(|| {
        for _ in 0..128 {
            result = kernel.process_event(OpSum::new(
                TensorView::new(&input, layout),
                TensorViewMut::new(&mut scalar, scalar_layout),
            ));
            assert_eq!(result, Ok(()));
            result = kernel.process_event(OpMean::new(
                TensorView::new(&input, layout),
                TensorViewMut::new(&mut scalar, scalar_layout),
            ));
            assert_eq!(result, Ok(()));
        }
    });
    assert_eq!(info.count_total, 0);
    assert_eq!(info.bytes_total, 0);
    assert_eq!(scalar, [2.5]);
}
