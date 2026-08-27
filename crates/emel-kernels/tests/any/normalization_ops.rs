#![allow(clippy::float_cmp, missing_docs)]

use allocation_counter::measure;
use emel_kernels::any::normalization::{
    NormalizationError, NormalizationKernel, OpNorm, OpRmsNorm,
};
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};
use emel_tensor as _;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use pulp as _;
use sml as _;

const PINNED_EMEL_CPP_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";

fn dense(ne: [u64; 4]) -> Layout {
    Layout::contiguous(DType::F32, ne).expect("small test layout fits")
}

#[test]
fn source_identity_and_unexpected_error_are_explicit() {
    assert_eq!(PINNED_EMEL_CPP_COMMIT.len(), 40);
    assert_eq!(
        NormalizationError::UnexpectedEvent,
        NormalizationError::UnexpectedEvent
    );
}

#[test]
fn norm_matches_reference_row_formula() {
    let layout = dense([3, 2, 1, 1]);
    let input = [1.0, 2.0, 4.0, 2.0, 5.0, 8.0];
    let mut output = [0.0; 6];
    let mut kernel = NormalizationKernel::new();

    assert_eq!(
        kernel.process_event(OpNorm::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout),
            1.0,
        )),
        Ok(())
    );
    assert_eq!(
        output,
        [
            -0.834_057_57,
            -0.208_514_36,
            1.042_572_1,
            -1.133_893_5,
            0.0,
            1.133_893_5,
        ]
    );
}

#[test]
fn rms_norm_matches_reference_row_formula() {
    let layout = dense([2, 2, 1, 1]);
    let input = [3.0, 4.0, 0.0, 2.0];
    let mut output = [0.0; 4];
    let mut kernel = NormalizationKernel::new();

    assert_eq!(
        kernel.process_event(OpRmsNorm::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout),
            1.0,
        )),
        Ok(())
    );
    assert_eq!(output, [0.816_496_6, 1.088_662_1, 0.0, 1.154_700_5]);
}

#[test]
fn norm_preserves_strided_row_storage_contract() {
    let layout = Layout::new(DType::F32, [2, 2, 1, 1], [4, 16, 32, 64]);
    let input = [1.0, 2.0, 99.0, 99.0, 3.0, 4.0, 99.0];
    let mut output = [0.0; 7];
    let mut kernel = NormalizationKernel::new();

    assert_eq!(
        kernel.process_event(OpNorm::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout),
            1.0,
        )),
        Ok(())
    );
    assert!((output[0] + 0.447_213_6).abs() < 1.0e-6);
    assert!((output[1] - 0.447_213_6).abs() < 1.0e-6);
    assert!((output[4] + 0.447_213_6).abs() < 1.0e-6);
    assert!((output[5] - 0.447_213_6).abs() < 1.0e-6);
    assert_eq!(output[2], 0.0);
    assert_eq!(output[3], 0.0);
    assert_eq!(output[6], 0.0);
}

#[test]
fn implicit_contiguous_layout_matches_pinned_contract() {
    let layout = Layout::new(DType::F32, [3, 2, 1, 1], [0, 0, 0, 0]);
    let input = [1.0, 2.0, 4.0, 2.0, 5.0, 8.0];
    let mut output = [0.0; 6];
    let mut kernel = NormalizationKernel::new();

    assert_eq!(
        kernel.process_event(OpRmsNorm::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout),
            1.0,
        )),
        Ok(())
    );
    assert_eq!(
        output.map(f32::to_bits),
        [
            0x3eb5_04f3,
            0x3f35_04f3,
            0x3fb5_04f3,
            0x3eb5_04f3,
            0x3f62_4630,
            0x3fb5_04f3
        ]
    );
}

#[test]
fn zero_outer_extent_is_rejected_without_mutation() {
    let layout = Layout::new(DType::F32, [3, 0, 1, 1], [0, 0, 0, 0]);
    let input = [1.0];
    let mut output = [7.0];
    let mut kernel = NormalizationKernel::new();

    assert_eq!(
        kernel.process_event(OpNorm::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout),
            1.0,
        )),
        Err(NormalizationError::InvalidView)
    );
    assert_eq!(output, [7.0]);
}

#[test]
fn guards_classify_shape_view_and_parameter_failures() {
    let input_layout = dense([2, 1, 1, 1]);
    let shape_layout = dense([3, 1, 1, 1]);
    let invalid_layout = Layout::contiguous(DType::F16, [2, 1, 1, 1]).expect("layout fits");
    let input = [1.0, 2.0, 3.0];
    let mut output = [7.0; 3];
    let mut kernel = NormalizationKernel::new();

    assert_eq!(
        kernel.process_event(OpNorm::new(
            TensorView::new(&input[..2], input_layout),
            TensorViewMut::new(&mut output, shape_layout),
            1.0,
        )),
        Err(NormalizationError::ShapeMismatch)
    );
    assert_eq!(output, [7.0; 3]);

    assert_eq!(
        kernel.process_event(OpRmsNorm::new(
            TensorView::new(&input[..2], invalid_layout),
            TensorViewMut::new(&mut output[..2], input_layout),
            1.0,
        )),
        Err(NormalizationError::InvalidView)
    );
    assert_eq!(output, [7.0; 3]);

    assert_eq!(
        kernel.process_event(OpNorm::new(
            TensorView::new(&input[..2], input_layout),
            TensorViewMut::new(&mut output[..2], input_layout),
            -1.0,
        )),
        Err(NormalizationError::InvalidParameters)
    );
}

#[test]
fn valid_dispatch_is_allocation_free() {
    let layout = dense([4, 1, 1, 1]);
    let input = [1.0, 2.0, 3.0, 4.0];
    let mut output = [0.0; 4];
    let mut kernel = NormalizationKernel::new();
    let mut result = Err(NormalizationError::UnexpectedEvent);
    let info = measure(|| {
        result = kernel.process_event(OpRmsNorm::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout),
            1.0,
        ));
    });
    assert_eq!(result, Ok(()));
    assert_eq!(info.count_total, 0);
    assert_eq!(info.bytes_total, 0);
}

#[test]
fn public_surfaces_and_rms_rejection_paths_are_typed() {
    assert!(format!("{}", NormalizationError::InvalidView).contains("view"));
    assert!(format!("{}", NormalizationError::ShapeMismatch).contains("shapes"));
    assert!(format!("{}", NormalizationError::InvalidParameters).contains("parameters"));
    assert!(format!("{}", NormalizationError::UnexpectedEvent).contains("unexpected"));
    assert!(format!("{}", NormalizationError::Internal).contains("internal"));
    assert!(format!("{:?}", NormalizationKernel::default()).contains("NormalizationKernel"));

    let input_layout = dense([2, 1, 1, 1]);
    let shape_layout = dense([3, 1, 1, 1]);
    let input = [1.0, 2.0, 3.0];
    let mut output = [7.0; 3];
    let mut kernel = NormalizationKernel::default();
    assert_eq!(
        kernel.process_event(OpRmsNorm::new(
            TensorView::new(&input[..2], input_layout),
            TensorViewMut::new(&mut output, shape_layout),
            1.0,
        )),
        Err(NormalizationError::ShapeMismatch)
    );
    assert_eq!(output, [7.0; 3]);
    assert_eq!(
        kernel.process_event(OpRmsNorm::new(
            TensorView::new(&input[..2], input_layout),
            TensorViewMut::new(&mut output[..2], input_layout),
            f32::NAN,
        )),
        Err(NormalizationError::InvalidParameters)
    );
    let invalid_layout = Layout::contiguous(DType::F16, [2, 1, 1, 1]).expect("layout fits");
    assert_eq!(
        kernel.process_event(OpNorm::new(
            TensorView::new(&input[..2], invalid_layout),
            TensorViewMut::new(&mut output[..2], input_layout),
            1.0,
        )),
        Err(NormalizationError::InvalidView)
    );
}
