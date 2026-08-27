#![allow(clippy::float_cmp, missing_docs)]

use allocation_counter::measure;
use emel_kernels::any::reductions::{OpSoftMax, ReductionError, ReductionKernel};
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};
use emel_tensor as _;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use pulp as _;
use sml as _;

const PINNED_EMEL_CPP_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const PINNED_DETAIL_BLOB: &str = "c8a82643eabfe8f2d7883e655955f455794511b0";
const PINNED_X86_SM_BLOB: &str = "0b4d635ebbd0fbd52dbca8a2345547fb571205c8";
const PINNED_AARCH64_SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";

fn contiguous(ne: [u64; 4]) -> Layout {
    Layout::contiguous(DType::F32, ne).expect("test layout fits")
}

#[test]
fn source_identity_is_pinned() {
    assert_eq!(
        PINNED_EMEL_CPP_COMMIT,
        "843a117386ef17dc5a50549bbfc821074c2141d6"
    );
    assert_eq!(PINNED_DETAIL_BLOB.len(), 40);
    assert_eq!(PINNED_X86_SM_BLOB.len(), 40);
    assert_eq!(PINNED_AARCH64_SM_BLOB.len(), 40);
}

#[test]
fn dense_softmax_is_stable_and_row_wise() {
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
    assert!(output[2] > output[1] && output[1] > output[0]);
}

#[test]
fn sparse_softmax_uses_validated_byte_strides() {
    let layout = Layout::new(DType::F32, [3, 2, 1, 1], [8, 24, 48, 48]);
    let input = [
        1.0_f32, 99.0, 2.0, 99.0, 3.0, 77.0, 4.0, 88.0, 5.0, 88.0, 6.0,
    ];
    let mut output = [55.0_f32; 11];
    let mut kernel = ReductionKernel::new();

    assert_eq!(
        kernel.process_event(OpSoftMax::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout),
        )),
        Ok(())
    );
    assert!((output[0] + output[2] + output[4] - 1.0).abs() < 1.0e-6);
    assert!((output[6] + output[8] + output[10] - 1.0).abs() < 1.0e-6);
    assert_eq!(output[1], 55.0);
    assert_eq!(output[3], 55.0);
    assert_eq!(output[5], 55.0);
    assert_eq!(output[7], 55.0);
    assert_eq!(output[9], 55.0);
}

#[test]
fn rejects_zero_width_and_count_mismatch_without_mutation() {
    let zero = Layout::new(DType::F32, [0, 1, 1, 1], [4, 0, 0, 0]);
    let valid = contiguous([2, 1, 1, 1]);
    let source = [1.0_f32, 2.0];
    let mut output = [9.0_f32, 9.0, 9.0];
    let mut kernel = ReductionKernel::new();

    assert_eq!(
        kernel.process_event(OpSoftMax::new(
            TensorView::new(&[], zero),
            TensorViewMut::new(&mut output, zero),
        )),
        Err(ReductionError::InvalidView)
    );
    assert_eq!(output, [9.0, 9.0, 9.0]);

    let mismatched = contiguous([3, 1, 1, 1]);
    assert_eq!(
        kernel.process_event(OpSoftMax::new(
            TensorView::new(&source, valid),
            TensorViewMut::new(&mut output, mismatched),
        )),
        Err(ReductionError::ShapeMismatch)
    );
    assert_eq!(output, [9.0, 9.0, 9.0]);
}

#[test]
fn same_count_different_shape_follows_reference_ordinal_contract() {
    let source_layout = contiguous([3, 2, 1, 1]);
    let destination_layout = contiguous([2, 3, 1, 1]);
    let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut output = [0.0_f32; 6];
    let mut kernel = ReductionKernel::new();

    assert_eq!(
        kernel.process_event(OpSoftMax::new(
            TensorView::new(&input, source_layout),
            TensorViewMut::new(&mut output, destination_layout),
        )),
        Ok(())
    );
    assert!((output[0] + output[1] + output[2] - 1.0).abs() < 1.0e-6);
    assert!((output[3] + output[4] + output[5] - 1.0).abs() < 1.0e-6);
}

#[test]
fn softmax_dispatch_is_allocation_free() {
    let layout = contiguous([4, 1, 1, 1]);
    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let mut output = [0.0_f32; 4];
    let mut kernel = ReductionKernel::new();
    let info = measure(|| {
        for _ in 0..128 {
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
}
