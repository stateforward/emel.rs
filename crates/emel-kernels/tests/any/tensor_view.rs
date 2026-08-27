#![allow(clippy::float_cmp, missing_docs)]

use allocation_counter::measure;
use emel_kernels::any::tensor_view::{
    AddTensor, CopyTensor, DType, Layout, TensorError, TensorKernel, TensorView, TensorViewMut,
    ViewError,
};
use emel_tensor as _;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use pulp as _;
use sml as _;

fn contiguous(ne: [u64; 4]) -> Layout {
    Layout::contiguous(DType::F32, ne).expect("small test layout fits")
}

#[test]
fn validates_contiguous_and_strided_f32_layouts() {
    let contiguous_layout = contiguous([2, 2, 1, 1]);
    assert_eq!(contiguous_layout.validate(4), Ok(4));

    let strided = Layout::new(DType::F32, [2, 2, 1, 1], [8, 16, 32, 64]);
    assert_eq!(strided.validate(7), Ok(4));
    assert_eq!(strided.validate(4), Err(ViewError::OutOfBounds));
}

#[test]
fn rejects_dtype_shape_stride_and_bounds() {
    assert_eq!(
        Layout::new(DType::F16, [1, 1, 1, 1], [4, 4, 4, 4]).validate(1),
        Err(ViewError::UnsupportedDType(DType::F16))
    );
    assert_eq!(
        Layout::new(DType::F32, [0, 1, 1, 1], [4, 4, 4, 4]).validate(1),
        Err(ViewError::InvalidShape)
    );
    assert_eq!(
        Layout::new(DType::F32, [1, 1, 1, 1], [2, 4, 4, 4]).validate(1),
        Err(ViewError::InvalidStride)
    );
    assert_eq!(
        Layout::new(DType::F32, [2, 1, 1, 1], [8, 4, 4, 4]).validate(1),
        Err(ViewError::OutOfBounds)
    );
}

#[test]
fn copies_and_adds_strided_views() {
    let layout = Layout::new(DType::F32, [2, 2, 1, 1], [8, 16, 32, 64]);
    let src = [1.0, 99.0, 2.0, 99.0, 3.0, 99.0, 4.0];
    let mut copied = [0.0; 7];
    let mut kernel = TensorKernel::new();
    assert_eq!(
        kernel.process_event(CopyTensor::new(
            TensorView::new(&src, layout),
            TensorViewMut::new(&mut copied, layout),
        )),
        Ok(())
    );
    assert_eq!(copied, [1.0, 0.0, 2.0, 0.0, 3.0, 0.0, 4.0]);

    let lhs = [1.0, 99.0, 2.0, 99.0, 3.0, 99.0, 4.0];
    let rhs = [5.0, 98.0, 6.0, 98.0, 7.0, 98.0, 8.0];
    let mut output = [0.0; 7];
    assert_eq!(
        kernel.process_event(AddTensor::new(
            TensorView::new(&lhs, layout),
            TensorView::new(&rhs, layout),
            TensorViewMut::new(&mut output, layout),
        )),
        Ok(())
    );
    assert_eq!(output, [6.0, 0.0, 8.0, 0.0, 10.0, 0.0, 12.0]);
}

#[test]
fn dispatch_rejects_invalid_view_and_shape_without_mutating_output() {
    let valid = Layout::new(DType::F32, [2, 1, 1, 1], [4, 8, 8, 8]);
    let mismatched = Layout::new(DType::F32, [1, 2, 1, 1], [4, 4, 8, 8]);
    let invalid = Layout::new(DType::F32, [2, 1, 1, 1], [8, 8, 8, 8]);
    let lhs = [1.0, 2.0];
    let rhs = [3.0, 4.0];
    let mut output = [7.0; 2];
    let mut kernel = TensorKernel::new();

    assert_eq!(
        kernel.process_event(AddTensor::new(
            TensorView::new(&lhs, valid),
            TensorView::new(&rhs, mismatched),
            TensorViewMut::new(&mut output, valid),
        )),
        Err(TensorError::ShapeMismatch)
    );
    assert_eq!(output, [7.0; 2]);

    assert_eq!(
        kernel.process_event(CopyTensor::new(
            TensorView::new(&lhs, invalid),
            TensorViewMut::new(&mut output, valid),
        )),
        Err(TensorError::InvalidView)
    );
    assert_eq!(output, [7.0; 2]);
}

#[test]
fn valid_dispatch_is_allocation_free() {
    let layout = contiguous([4, 1, 1, 1]);
    let src = [1.0, 2.0, 3.0, 4.0];
    let mut output = [0.0; 4];
    let mut kernel = TensorKernel::new();
    let mut result = Err(TensorError::UnexpectedEvent);
    let info = measure(|| {
        result = kernel.process_event(CopyTensor::new(
            TensorView::new(&src, layout),
            TensorViewMut::new(&mut output, layout),
        ));
    });
    assert_eq!(result, Ok(()));
    assert_eq!(info.count_total, 0);
    assert_eq!(info.bytes_total, 0);
    assert_eq!(output, src);
}

#[test]
fn layout_metadata_covers_dtype_stride_and_count_boundaries() {
    let pinned_dtypes = [
        (0, DType::F32),
        (1, DType::F16),
        (2, DType::Q4_0),
        (3, DType::Q4_1),
        (6, DType::Q5_0),
        (7, DType::Q5_1),
        (8, DType::Q8_0),
        (9, DType::Q8_1),
        (10, DType::Q2K),
        (11, DType::Q3K),
        (12, DType::Q4K),
        (13, DType::Q5K),
        (14, DType::Q6K),
        (15, DType::Q8K),
        (16, DType::Iq2Xxs),
        (17, DType::Iq2Xs),
        (18, DType::Iq3Xxs),
        (19, DType::Iq1S),
        (20, DType::Iq4Nl),
        (21, DType::Iq3S),
        (22, DType::Iq2S),
        (23, DType::Iq4Xs),
        (24, DType::I8),
        (25, DType::I16),
        (26, DType::I32),
        (27, DType::I64),
        (28, DType::F64),
        (29, DType::Iq1M),
        (30, DType::Bf16),
        (31, DType::Q4_0_4_4),
        (32, DType::Q4_0_4_8),
        (33, DType::Q4_0_8_8),
        (34, DType::Tq1_0),
        (35, DType::Tq2_0),
        (36, DType::Q6KX8),
        (37, DType::Q6KX8Q8Prepared),
        (38, DType::Q6KX8Q8ArgmaxPrepared),
        (39, DType::Q8_0X4Bl4),
        (40, DType::Q8_0X4Bl8),
        (41, DType::Q4KX8Bl4),
        (42, DType::Q4KX8Bl8),
        (43, DType::Q8KX4),
        (44, DType::Q8KX8),
    ];
    for (code, dtype) in pinned_dtypes {
        assert_eq!(DType::from_code(code), dtype);
        assert_eq!(dtype.code(), code);
    }
    assert_eq!(DType::from_code(255), DType::Other(255));
    assert_eq!(DType::Other(255).code(), 255);

    let i32_layout = Layout::contiguous(DType::I32, [2, 1, 1, 1]).expect("I32 layout fits");
    assert_eq!(i32_layout.validate_i32(2), Ok(2));
    assert_eq!(
        i32_layout.validate(2),
        Err(ViewError::UnsupportedDType(DType::I32))
    );

    assert_eq!(Layout::contiguous(DType::F32, [u64::MAX; 4]), None);
    assert_eq!(
        Layout::new(DType::F32, [0, 1, 1, 1], [4, 4, 4, 4]).element_count(),
        None
    );
    assert_eq!(
        Layout::new(DType::F32, [u64::MAX, 2, 1, 1], [4, 4, 8, 8]).element_count(),
        None
    );

    let dense = contiguous([2, 3, 1, 1]);
    assert!(dense.is_dense_contiguous());
    assert!(!Layout::new(DType::F32, [2, 3, 1, 1], [8, 16, 32, 64]).is_dense_contiguous());
    assert!(!Layout::new(DType::F16, [2, 3, 1, 1], [4, 8, 24, 24]).is_dense_contiguous());
    assert_eq!(dense.dtype(), DType::F32);
    assert_eq!(dense.ne(), [2, 3, 1, 1]);
    assert_eq!(dense.nb(), [4, 8, 24, 24]);

    assert_eq!(dense.permuted([1, 0, 2, 3]).unwrap().ne(), [3, 2, 1, 1]);
    assert_eq!(dense.permuted([4, 0, 1, 2]), None);
    assert_eq!(dense.transposed().ne(), [3, 2, 1, 1]);
}

#[test]
fn tensor_errors_and_debug_defaults_are_observable() {
    assert_eq!(
        ViewError::UnsupportedDType(DType::F16).to_string(),
        "unsupported tensor dtype F16"
    );
    assert_eq!(ViewError::InvalidShape.to_string(), "invalid tensor shape");
    assert_eq!(
        ViewError::InvalidStride.to_string(),
        "invalid tensor stride"
    );
    assert_eq!(
        ViewError::OutOfBounds.to_string(),
        "tensor view is out of bounds"
    );
    assert_eq!(TensorError::InvalidView.to_string(), "invalid tensor view");
    assert_eq!(
        TensorError::ShapeMismatch.to_string(),
        "tensor view shapes differ"
    );
    assert_eq!(
        TensorError::UnexpectedEvent.to_string(),
        "unexpected tensor event"
    );
    assert_eq!(
        TensorError::Internal.to_string(),
        "internal tensor dispatch error"
    );

    let kernel = TensorKernel::default();
    assert!(format!("{kernel:?}").contains("TensorKernel"));
}
