#![allow(clippy::float_cmp, missing_docs)]

use allocation_counter::measure;
use emel_kernels::any::broadcast::{
    BroadcastError, BroadcastKernel, OpAddBroadcastRow, OpMulBroadcastRow, UnexpectedBroadcast,
};
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};
use emel_tensor as _;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use pulp as _;
use sml as _;

const PINNED_EMEL_CPP_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";

fn layout(ne: [u64; 4]) -> Layout {
    Layout::contiguous(DType::F32, ne).expect("test layout fits")
}

#[test]
fn source_identity_and_row_formula_match_pinned_contract() {
    assert_eq!(PINNED_EMEL_CPP_COMMIT.len(), 40);

    let source_layout = layout([3, 2, 1, 1]);
    let row_layout = layout([3, 1, 1, 1]);
    let output_layout = layout([3, 2, 1, 1]);
    let source = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let row = [10.0_f32, 20.0, 30.0];
    let mut output = [0.0_f32; 6];
    let mut kernel = BroadcastKernel::new();

    assert_eq!(
        kernel.process_event(OpAddBroadcastRow::new(
            TensorView::new(&source, source_layout),
            TensorView::new(&row, row_layout),
            TensorViewMut::new(&mut output, output_layout),
        )),
        Ok(())
    );
    assert_eq!(output, [11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
}

#[test]
fn multiply_row_broadcast_matches_pinned_scalar_formula() {
    let source_layout = layout([3, 2, 1, 1]);
    let row_layout = layout([3, 1, 1, 1]);
    let output_layout = source_layout;
    let source = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let row = [10.0_f32, 20.0, 30.0];
    let mut output = [0.0_f32; 6];
    let mut kernel = BroadcastKernel::new();

    assert_eq!(
        kernel.process_event(OpMulBroadcastRow::new(
            TensorView::new(&source, source_layout),
            TensorView::new(&row, row_layout),
            TensorViewMut::new(&mut output, output_layout),
        )),
        Ok(())
    );
    assert_eq!(output, [10.0, 40.0, 90.0, 40.0, 100.0, 180.0]);
}

#[test]
fn strided_views_follow_logical_row_broadcast_ordinals() {
    let source_layout = Layout::new(DType::F32, [3, 2, 1, 1], [8, 32, 64, 128]);
    let row_layout = Layout::new(DType::F32, [3, 1, 1, 1], [8, 24, 24, 24]);
    let output_layout = source_layout;
    let source = [
        1.0_f32, 99.0, 2.0, 99.0, 3.0, 99.0, 99.0, 99.0, 4.0, 99.0, 5.0, 99.0, 6.0,
    ];
    let row = [10.0_f32, 99.0, 20.0, 99.0, 30.0];
    let mut output = [-7.0_f32; 13];
    let mut kernel = BroadcastKernel::new();

    assert_eq!(
        kernel.process_event(OpAddBroadcastRow::new(
            TensorView::new(&source, source_layout),
            TensorView::new(&row, row_layout),
            TensorViewMut::new(&mut output, output_layout),
        )),
        Ok(())
    );
    assert_eq!(
        output,
        [
            11.0, -7.0, 22.0, -7.0, 33.0, -7.0, -7.0, -7.0, 14.0, -7.0, 25.0, -7.0, 36.0,
        ]
    );
}

#[test]
fn singleton_dimensions_may_use_explicit_zero_strides() {
    let source_layout = Layout::new(DType::F32, [3, 2, 1, 1], [4, 12, 0, 0]);
    let row_layout = Layout::new(DType::F32, [3, 1, 1, 1], [4, 12, 0, 0]);
    let source = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let row = [10.0_f32, 20.0, 30.0];
    let mut output = [0.0_f32; 6];
    let mut kernel = BroadcastKernel::new();

    assert_eq!(
        kernel.process_event(OpMulBroadcastRow::new(
            TensorView::new(&source, source_layout),
            TensorView::new(&row, row_layout),
            TensorViewMut::new(&mut output, source_layout),
        )),
        Ok(())
    );
    assert_eq!(output, [10.0, 40.0, 90.0, 40.0, 100.0, 180.0]);
}

#[test]
fn guards_reject_shape_and_view_failures_without_writes() {
    let source_layout = layout([3, 2, 1, 1]);
    let row_layout = layout([3, 1, 1, 1]);
    let one_row_layout = layout([3, 1, 1, 1]);
    let bad_row_layout = layout([2, 1, 1, 1]);
    let output_layout = layout([3, 2, 1, 1]);
    let source = [1.0_f32; 6];
    let row = [2.0_f32; 3];
    let bad_row = [2.0_f32; 2];
    let mut output = [9.0_f32; 6];
    let mut kernel = BroadcastKernel::new();

    assert_eq!(
        kernel.process_event(OpAddBroadcastRow::new(
            TensorView::new(&source, source_layout),
            TensorView::new(&bad_row, bad_row_layout),
            TensorViewMut::new(&mut output, output_layout),
        )),
        Err(BroadcastError::ShapeMismatch)
    );
    assert_eq!(output, [9.0; 6]);

    assert_eq!(
        kernel.process_event(OpMulBroadcastRow::new(
            TensorView::new(&source, source_layout),
            TensorView::new(&bad_row, bad_row_layout),
            TensorViewMut::new(&mut output, output_layout),
        )),
        Err(BroadcastError::ShapeMismatch)
    );
    assert_eq!(output, [9.0; 6]);

    let single_row = [1.0_f32; 3];
    let mut single_output = [9.0_f32; 3];
    assert_eq!(
        kernel.process_event(OpAddBroadcastRow::new(
            TensorView::new(&single_row, one_row_layout),
            TensorView::new(&row, row_layout),
            TensorViewMut::new(&mut single_output, one_row_layout),
        )),
        Err(BroadcastError::ShapeMismatch)
    );
    assert_eq!(single_output, [9.0; 3]);

    let invalid_layout = Layout::new(DType::F16, [3, 2, 1, 1], [4, 12, 24, 24]);
    assert_eq!(
        kernel.process_event(OpAddBroadcastRow::new(
            TensorView::new(&source, invalid_layout),
            TensorView::new(&row, row_layout),
            TensorViewMut::new(&mut output, output_layout),
        )),
        Err(BroadcastError::InvalidView)
    );
    assert_eq!(output, [9.0; 6]);
}

#[test]
fn multiply_invalid_view_and_error_surfaces_are_explicit() {
    let source_layout = layout([3, 2, 1, 1]);
    let row_layout = layout([3, 1, 1, 1]);
    let invalid_layout = Layout::new(DType::F16, [3, 2, 1, 1], [4, 12, 24, 24]);
    let source = [1.0_f32; 6];
    let row = [2.0_f32; 3];
    let mut output = [9.0_f32; 6];
    let mut kernel = BroadcastKernel::default();

    assert_eq!(
        kernel.process_event(OpMulBroadcastRow::new(
            TensorView::new(&source, invalid_layout),
            TensorView::new(&row, row_layout),
            TensorViewMut::new(&mut output, source_layout),
        )),
        Err(BroadcastError::InvalidView)
    );
    assert_eq!(output, [9.0; 6]);

    assert_eq!(
        format!("{}", BroadcastError::InvalidView),
        "invalid broadcast tensor view"
    );
    assert_eq!(
        format!("{}", BroadcastError::ShapeMismatch),
        "broadcast tensor shapes differ"
    );
    assert_eq!(
        format!("{}", BroadcastError::UnexpectedEvent),
        "unexpected broadcast event"
    );
    assert_eq!(
        format!("{}", BroadcastError::Internal),
        "internal broadcast dispatch error"
    );
    assert!(!format!("{kernel:?}").is_empty());
}

#[test]
fn unexpected_event_is_explicit_and_actor_recovers() {
    let source_layout = layout([3, 2, 1, 1]);
    let row_layout = layout([3, 1, 1, 1]);
    let source = [1.0_f32; 6];
    let row = [2.0_f32; 3];
    let mut output = [0.0_f32; 6];
    let mut kernel = BroadcastKernel::new();

    assert_eq!(
        kernel.process_event(UnexpectedBroadcast),
        Err(BroadcastError::UnexpectedEvent)
    );
    assert_eq!(
        kernel.process_event(OpAddBroadcastRow::new(
            TensorView::new(&source, source_layout),
            TensorView::new(&row, row_layout),
            TensorViewMut::new(&mut output, source_layout),
        )),
        Ok(())
    );
    assert_eq!(output, [3.0; 6]);
}

#[test]
fn dispatch_is_allocation_free() {
    let source_layout = layout([3, 2, 1, 1]);
    let row_layout = layout([3, 1, 1, 1]);
    let source = [1.0_f32; 6];
    let row = [2.0_f32; 3];
    let mut output = [0.0_f32; 6];
    let mut kernel = BroadcastKernel::new();

    let allocation = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                kernel.process_event(OpAddBroadcastRow::new(
                    TensorView::new(&source, source_layout),
                    TensorView::new(&row, row_layout),
                    TensorViewMut::new(&mut output, source_layout),
                )),
                Ok(())
            );
            assert_eq!(
                kernel.process_event(OpMulBroadcastRow::new(
                    TensorView::new(&source, source_layout),
                    TensorView::new(&row, row_layout),
                    TensorViewMut::new(&mut output, source_layout),
                )),
                Ok(())
            );
        }
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(allocation.bytes_total, 0);
}
