#![allow(clippy::float_cmp, missing_docs)]

use allocation_counter::measure;
use emel_kernels::any::elementwise::{ElementwiseError, ElementwiseKernel, OpAcc, OpAdd1};
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
fn add1_matches_reference_scalar_broadcast() {
    assert_eq!(PINNED_EMEL_CPP_COMMIT.len(), 40);
    let layout = contiguous([4, 1, 1, 1]);
    let input = [1.0_f32, -2.0, 3.5, 4.0];
    let mut output = [0.0_f32; 4];
    let mut kernel = ElementwiseKernel::new();

    assert_eq!(
        kernel.process_event(OpAdd1::new(
            TensorView::new(&input, layout),
            2.0,
            TensorViewMut::new(&mut output, layout),
        )),
        Ok(())
    );
    assert_eq!(output, [3.0, 0.0, 5.5, 6.0]);
}

#[test]
fn acc_copies_base_then_adds_contiguous_update_at_offset() {
    let base_layout = contiguous([6, 1, 1, 1]);
    let update_layout = contiguous([2, 1, 1, 1]);
    let base = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let update = [10.0_f32, 20.0];
    let mut output = [0.0_f32; 6];
    let mut kernel = ElementwiseKernel::new();

    assert_eq!(
        kernel.process_event(OpAcc::new(
            TensorView::new(&base, base_layout),
            TensorView::new(&update, update_layout),
            TensorViewMut::new(&mut output, base_layout),
            2,
        )),
        Ok(())
    );
    assert_eq!(output, [1.0, 2.0, 13.0, 24.0, 5.0, 6.0]);
}

#[test]
fn guards_reject_shape_offset_and_view_without_mutating_output() {
    let base_layout = contiguous([4, 1, 1, 1]);
    let short_layout = contiguous([3, 1, 1, 1]);
    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let update = [5.0_f32, 6.0];
    let mut output = [9.0_f32; 4];
    let mut kernel = ElementwiseKernel::new();

    assert_eq!(
        kernel.process_event(OpAdd1::new(
            TensorView::new(&input, base_layout),
            1.0,
            TensorViewMut::new(&mut output, short_layout),
        )),
        Err(ElementwiseError::ShapeMismatch)
    );
    assert_eq!(output, [9.0; 4]);

    assert_eq!(
        kernel.process_event(OpAcc::new(
            TensorView::new(&input, base_layout),
            TensorView::new(&update, contiguous([2, 1, 1, 1])),
            TensorViewMut::new(&mut output, base_layout),
            3,
        )),
        Err(ElementwiseError::InvalidOffset)
    );
    assert_eq!(output, [9.0; 4]);

    let invalid_layout = Layout::new(DType::F16, [4, 1, 1, 1], [4, 16, 16, 16]);
    assert_eq!(
        kernel.process_event(OpAdd1::new(
            TensorView::new(&input, invalid_layout),
            1.0,
            TensorViewMut::new(&mut output, base_layout),
        )),
        Err(ElementwiseError::InvalidView)
    );
    assert_eq!(output, [9.0; 4]);
}

#[test]
fn acc_invalid_view_and_error_surfaces_are_explicit() {
    let layout = contiguous([4, 1, 1, 1]);
    let update_layout = Layout::new(DType::F16, [2, 1, 1, 1], [4, 8, 8, 8]);
    let base = [1.0_f32; 4];
    let update = [2.0_f32; 2];
    let mut output = [9.0_f32; 4];
    let mut kernel = ElementwiseKernel::default();

    assert_eq!(
        kernel.process_event(OpAcc::new(
            TensorView::new(&base, layout),
            TensorView::new(&update, update_layout),
            TensorViewMut::new(&mut output, layout),
            0,
        )),
        Err(ElementwiseError::InvalidView)
    );
    assert_eq!(output, [9.0; 4]);

    assert_eq!(
        format!("{}", ElementwiseError::InvalidView),
        "invalid elementwise tensor view"
    );
    assert_eq!(
        format!("{}", ElementwiseError::ShapeMismatch),
        "elementwise tensor shapes differ"
    );
    assert_eq!(
        format!("{}", ElementwiseError::InvalidOffset),
        "accumulate update exceeds destination"
    );
    assert_eq!(
        format!("{}", ElementwiseError::UnexpectedEvent),
        "unexpected elementwise event"
    );
    assert_eq!(
        format!("{}", ElementwiseError::Internal),
        "internal elementwise dispatch error"
    );
    assert!(!format!("{kernel:?}").is_empty());
}

#[test]
fn elementwise_dispatch_is_allocation_free() {
    let layout = contiguous([4, 1, 1, 1]);
    let update_layout = contiguous([2, 1, 1, 1]);
    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let update = [5.0_f32, 6.0];
    let mut add_output = [0.0_f32; 4];
    let mut acc_output = [0.0_f32; 4];
    let mut kernel = ElementwiseKernel::new();
    let info = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                kernel.process_event(OpAdd1::new(
                    TensorView::new(&input, layout),
                    1.0,
                    TensorViewMut::new(&mut add_output, layout),
                )),
                Ok(())
            );
            assert_eq!(
                kernel.process_event(OpAcc::new(
                    TensorView::new(&input, layout),
                    TensorView::new(&update, update_layout),
                    TensorViewMut::new(&mut acc_output, layout),
                    1,
                )),
                Ok(())
            );
        }
    });
    assert_eq!(info.count_total, 0);
    assert_eq!(info.bytes_total, 0);
    assert_eq!(add_output, [2.0, 3.0, 4.0, 5.0]);
    assert_eq!(acc_output, [1.0, 7.0, 9.0, 4.0]);
}
