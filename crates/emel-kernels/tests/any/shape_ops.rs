#![allow(clippy::float_cmp, missing_docs)]

use allocation_counter::measure;
use emel_kernels::any::shape::{
    OpCont, OpCpy, OpPermute, OpReshape, OpTranspose, OpView, ShapeError, ShapeKernel,
};
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};
use emel_tensor as _;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use pulp as _;
use sml as _;

fn contiguous(ne: [u64; 4]) -> Layout {
    Layout::contiguous(DType::F32, ne).expect("small test layout fits")
}

#[test]
fn cpy_and_cont_materialize_strided_source() {
    let strided = Layout::new(DType::F32, [2, 2, 1, 1], [8, 16, 32, 64]);
    let dense = contiguous([2, 2, 1, 1]);
    let source = [1.0, 99.0, 2.0, 99.0, 3.0, 99.0, 4.0];
    let mut copied = [0.0; 7];
    let mut kernel = ShapeKernel::new();

    assert_eq!(
        kernel.process_event(OpCpy::new(
            TensorView::new(&source, strided),
            TensorViewMut::new(&mut copied, strided),
        )),
        Ok(())
    );
    assert_eq!(copied, [1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0]);

    let mut materialized = [0.0; 4];
    assert_eq!(
        kernel.process_event(OpCont::new(
            TensorView::new(&source, strided),
            TensorViewMut::new(&mut materialized, dense),
        )),
        Ok(())
    );
    assert_eq!(materialized, [1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn metadata_operations_publish_shape_layouts() {
    let source_layout = contiguous([2, 3, 1, 1]);
    let source_data = [0.0; 6];
    let reshaped = contiguous([3, 2, 1, 1]);
    let mut kernel = ShapeKernel::new();

    assert_eq!(
        kernel.process_event(OpReshape::new(
            TensorView::new(&source_data, source_layout),
            reshaped,
        )),
        Ok(reshaped)
    );
    assert_eq!(
        kernel.process_event(OpView::new(
            TensorView::new(&source_data, source_layout),
            reshaped,
        )),
        Ok(reshaped)
    );

    let permuted = source_layout
        .permuted([1, 0, 2, 3])
        .expect("permutation is in range");
    assert_eq!(
        kernel.process_event(OpPermute::new(
            TensorView::new(&source_data, source_layout),
            [1, 0, 2, 3],
        )),
        Ok(permuted)
    );
    assert_eq!(
        kernel.process_event(OpTranspose::new(TensorView::new(
            &source_data,
            source_layout,
        ))),
        Ok(source_layout.transposed())
    );
}

#[test]
fn invalid_guards_reject_without_mutating_destination() {
    let dense = contiguous([2, 2, 1, 1]);
    let strided = Layout::new(DType::F32, [2, 2, 1, 1], [8, 16, 32, 64]);
    let source_data = [1.0, 2.0, 3.0, 4.0];
    let mut output = [7.0; 7];
    let mut kernel = ShapeKernel::new();

    assert_eq!(
        kernel.process_event(OpCont::new(
            TensorView::new(&source_data, dense),
            TensorViewMut::new(&mut output, strided),
        )),
        Err(ShapeError::InvalidLayout)
    );
    assert_eq!(output, [7.0; 7]);

    let mismatched = contiguous([3, 1, 1, 1]);
    assert_eq!(
        kernel.process_event(OpReshape::new(
            TensorView::new(&source_data, dense),
            mismatched,
        )),
        Err(ShapeError::ShapeMismatch)
    );

    assert_eq!(
        kernel.process_event(OpPermute::new(
            TensorView::new(&source_data, dense),
            [0, 0, 1, 2],
        )),
        Err(ShapeError::InvalidPermutation)
    );
}

#[test]
fn valid_dispatch_is_allocation_free() {
    let layout = contiguous([4, 1, 1, 1]);
    let source = [1.0, 2.0, 3.0, 4.0];
    let mut output = [0.0; 4];
    let mut kernel = ShapeKernel::new();
    let mut result = Err(ShapeError::UnexpectedEvent);
    let info = measure(|| {
        result = kernel.process_event(OpCont::new(
            TensorView::new(&source, layout),
            TensorViewMut::new(&mut output, layout),
        ));
    });
    assert_eq!(result, Ok(()));
    assert_eq!(info.count_total, 0);
    assert_eq!(info.bytes_total, 0);
    assert_eq!(output, source);
}

#[test]
fn rejects_copy_and_cont_shape_mismatch() {
    let source_layout = contiguous([4, 1, 1, 1]);
    let destination_layout = contiguous([3, 1, 1, 1]);
    let source = [1.0_f32; 4];
    let mut destination = [9.0_f32; 3];
    let mut kernel = ShapeKernel::default();

    assert_eq!(
        kernel.process_event(OpCpy::new(
            TensorView::new(&source, source_layout),
            TensorViewMut::new(&mut destination, destination_layout),
        )),
        Err(ShapeError::ShapeMismatch)
    );
    assert_eq!(destination, [9.0; 3]);

    assert_eq!(
        kernel.process_event(OpCont::new(
            TensorView::new(&source, source_layout),
            TensorViewMut::new(&mut destination, destination_layout),
        )),
        Err(ShapeError::ShapeMismatch)
    );
    assert_eq!(destination, [9.0; 3]);
}

#[test]
fn rejects_metadata_layout_and_view_failures() {
    let source_layout = contiguous([2, 3, 1, 1]);
    let source = [0.0_f32; 6];
    let target_shape_mismatch = contiguous([5, 1, 1, 1]);
    let target_invalid_dtype = Layout::new(DType::F16, [3, 2, 1, 1], [4, 12, 24, 24]);
    let invalid_source = Layout::new(DType::F16, [2, 3, 1, 1], [4, 8, 24, 24]);
    let mut kernel = ShapeKernel::default();

    assert_eq!(
        kernel.process_event(OpReshape::new(
            TensorView::new(&source, source_layout),
            target_shape_mismatch,
        )),
        Err(ShapeError::ShapeMismatch)
    );
    assert_eq!(
        kernel.process_event(OpReshape::new(
            TensorView::new(&source, source_layout),
            target_invalid_dtype,
        )),
        Err(ShapeError::InvalidLayout)
    );
    assert_eq!(
        kernel.process_event(OpReshape::new(
            TensorView::new(&source, invalid_source),
            source_layout,
        )),
        Err(ShapeError::InvalidView)
    );

    assert_eq!(
        kernel.process_event(OpView::new(
            TensorView::new(&source, source_layout),
            target_shape_mismatch,
        )),
        Err(ShapeError::ShapeMismatch)
    );
    assert_eq!(
        kernel.process_event(OpView::new(
            TensorView::new(&source, source_layout),
            target_invalid_dtype,
        )),
        Err(ShapeError::InvalidLayout)
    );
    assert_eq!(
        kernel.process_event(OpView::new(
            TensorView::new(&source, invalid_source),
            source_layout,
        )),
        Err(ShapeError::InvalidView)
    );

    assert_eq!(
        kernel.process_event(OpPermute::new(
            TensorView::new(&source, source_layout),
            [4, 0, 1, 2],
        )),
        Err(ShapeError::InvalidPermutation)
    );
    assert_eq!(
        kernel.process_event(OpPermute::new(
            TensorView::new(&source, invalid_source),
            [1, 0, 2, 3],
        )),
        Err(ShapeError::InvalidView)
    );
}

#[test]
fn shape_public_accessors_and_errors_are_stable() {
    let source_layout = contiguous([2, 3, 1, 1]);
    let source = [0.0_f32; 6];
    let reshaped = contiguous([3, 2, 1, 1]);
    let reshape = OpReshape::new(TensorView::new(&source, source_layout), reshaped);
    assert_eq!(reshape.target(), reshaped);

    let view = OpView::new(TensorView::new(&source, source_layout), reshaped);
    assert_eq!(view.target(), reshaped);

    let permute = OpPermute::new(TensorView::new(&source, source_layout), [1, 0, 2, 3]);
    assert_eq!(permute.axes(), [1, 0, 2, 3]);
    assert_eq!(
        permute.target(),
        source_layout.permuted([1, 0, 2, 3]).unwrap()
    );

    let transpose = OpTranspose::new(TensorView::new(&source, source_layout));
    assert_eq!(transpose.target(), source_layout.transposed());

    assert_eq!(
        ShapeError::InvalidView.to_string(),
        "invalid shape-operation view"
    );
    assert_eq!(
        ShapeError::ShapeMismatch.to_string(),
        "shape-operation element counts differ"
    );
    assert_eq!(
        ShapeError::InvalidLayout.to_string(),
        "invalid shape-operation layout"
    );
    assert_eq!(
        ShapeError::InvalidPermutation.to_string(),
        "invalid tensor permutation"
    );
    assert_eq!(
        ShapeError::UnexpectedEvent.to_string(),
        "unexpected shape-operation event"
    );
    assert_eq!(
        ShapeError::Internal.to_string(),
        "internal shape-operation dispatch error"
    );
}
