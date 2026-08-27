#![allow(clippy::float_cmp, missing_docs)]

use allocation_counter::measure;
use emel_kernels::any::sequence::{
    OpConcat, OpCumsum, OpRepeat, OpRepeatBack, SequenceError, SequenceKernel, UnexpectedSequence,
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
fn cumsum_matches_pinned_dimension_zero_rows() {
    assert_eq!(PINNED_EMEL_CPP_COMMIT.len(), 40);
    let layout = contiguous([3, 2, 1, 1]);
    let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut output = [0.0_f32; 6];
    let mut kernel = SequenceKernel::new();

    assert_eq!(
        kernel.process_event(OpCumsum::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout),
        )),
        Ok(())
    );
    assert_eq!(output, [1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);
}

#[test]
fn repeat_tiles_each_source_dimension() {
    let source_layout = contiguous([2, 2, 1, 1]);
    let target_layout = contiguous([4, 4, 1, 1]);
    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let mut output = [0.0_f32; 16];
    let mut kernel = SequenceKernel::new();

    assert_eq!(
        kernel.process_event(OpRepeat::new(
            TensorView::new(&input, source_layout),
            TensorViewMut::new(&mut output, target_layout),
        )),
        Ok(())
    );
    assert_eq!(
        output,
        [
            1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0,
        ]
    );
}

#[test]
fn repeat_back_sums_repeated_source_coordinates() {
    let source_layout = contiguous([4, 2, 1, 1]);
    let target_layout = contiguous([2, 1, 1, 1]);
    let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mut output = [9.0_f32; 2];
    let mut kernel = SequenceKernel::new();

    assert_eq!(
        kernel.process_event(OpRepeatBack::new(
            TensorView::new(&input, source_layout),
            TensorViewMut::new(&mut output, target_layout),
        )),
        Ok(())
    );
    assert_eq!(output, [16.0, 20.0]);
}

#[test]
fn concat_supports_axis_zero_and_axis_one() {
    let mut kernel = SequenceKernel::new();
    let left_layout = contiguous([2, 2, 1, 1]);
    let right_layout = contiguous([1, 2, 1, 1]);
    let left = [1.0_f32, 2.0, 3.0, 4.0];
    let right = [5.0_f32, 6.0];
    let mut axis_zero = [0.0_f32; 6];
    assert_eq!(
        kernel.process_event(OpConcat::new(
            TensorView::new(&left, left_layout),
            TensorView::new(&right, right_layout),
            TensorViewMut::new(&mut axis_zero, contiguous([3, 2, 1, 1])),
            0,
        )),
        Ok(())
    );
    assert_eq!(axis_zero, [1.0, 2.0, 5.0, 3.0, 4.0, 6.0]);

    let right_axis_one_layout = contiguous([2, 1, 1, 1]);
    let mut axis_one = [0.0_f32; 6];
    assert_eq!(
        kernel.process_event(OpConcat::new(
            TensorView::new(&left, left_layout),
            TensorView::new(&right, right_axis_one_layout),
            TensorViewMut::new(&mut axis_one, contiguous([2, 3, 1, 1])),
            1,
        )),
        Ok(())
    );
    assert_eq!(axis_one, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let upper_layout = contiguous([1, 1, 2, 1]);
    let upper_rhs_layout = contiguous([1, 1, 1, 1]);
    let upper = [7.0_f32, 8.0];
    let upper_rhs = [9.0_f32];
    let mut axis_two = [0.0_f32; 3];
    assert_eq!(
        kernel.process_event(OpConcat::new(
            TensorView::new(&upper, upper_layout),
            TensorView::new(&upper_rhs, upper_rhs_layout),
            TensorViewMut::new(&mut axis_two, contiguous([1, 1, 3, 1])),
            2,
        )),
        Ok(())
    );
    assert_eq!(axis_two, [7.0, 8.0, 9.0]);

    let depth_layout = contiguous([1, 1, 1, 1]);
    let mut axis_three = [0.0_f32; 2];
    assert_eq!(
        kernel.process_event(OpConcat::new(
            TensorView::new(&upper_rhs, depth_layout),
            TensorView::new(&upper_rhs, depth_layout),
            TensorViewMut::new(&mut axis_three, contiguous([1, 1, 1, 2])),
            3,
        )),
        Ok(())
    );
    assert_eq!(axis_three, [9.0, 9.0]);
}

#[test]
fn sequence_invalid_and_unexpected_events_do_not_mutate_output() {
    let mut kernel = SequenceKernel::new();
    let input_layout = contiguous([2, 1, 1, 1]);
    let input = [1.0_f32, 2.0];
    let mut output = [7.0_f32; 4];
    assert_eq!(
        kernel.process_event(OpConcat::new(
            TensorView::new(&input, input_layout),
            TensorView::new(&input, input_layout),
            TensorViewMut::new(&mut output, contiguous([4, 1, 1, 1])),
            4,
        )),
        Err(SequenceError::InvalidAxis)
    );
    assert_eq!(output, [7.0; 4]);

    assert_eq!(
        kernel.process_event(OpRepeatBack::new(
            TensorView::new(&[1.0_f32, 2.0, 3.0], contiguous([3, 1, 1, 1])),
            TensorViewMut::new(&mut output, contiguous([2, 1, 1, 1])),
        )),
        Err(SequenceError::ShapeMismatch)
    );
    assert_eq!(output, [7.0; 4]);

    assert_eq!(
        kernel.process_event(OpConcat::new(
            TensorView::new(&input, input_layout),
            TensorView::new(&input, input_layout),
            TensorViewMut::new(&mut output, contiguous([3, 1, 1, 1])),
            0,
        )),
        Err(SequenceError::ShapeMismatch)
    );
    assert_eq!(output, [7.0; 4]);

    let invalid_source = Layout::new(DType::F16, [2, 1, 1, 1], [4, 8, 8, 8]);
    assert_eq!(
        kernel.process_event(OpRepeatBack::new(
            TensorView::new(&input, invalid_source),
            TensorViewMut::new(&mut output, input_layout),
        )),
        Err(SequenceError::InvalidView)
    );
    assert_eq!(output, [7.0; 4]);

    assert_eq!(
        kernel.process_event(UnexpectedSequence),
        Err(SequenceError::UnexpectedEvent)
    );
}

#[test]
fn guards_reject_invalid_views_and_nondivisible_repeat_shapes() {
    let source_layout = contiguous([2, 1, 1, 1]);
    let invalid_target = contiguous([3, 1, 1, 1]);
    let input = [1.0_f32, 2.0];
    let mut output = [7.0_f32; 3];
    let mut kernel = SequenceKernel::new();

    assert_eq!(
        kernel.process_event(OpRepeat::new(
            TensorView::new(&input, source_layout),
            TensorViewMut::new(&mut output, invalid_target),
        )),
        Err(SequenceError::ShapeMismatch)
    );
    assert_eq!(output, [7.0; 3]);

    let invalid_source = Layout::new(DType::F16, [2, 1, 1, 1], [4, 8, 8, 8]);
    let mut cumsum_output = [9.0_f32; 2];
    assert_eq!(
        kernel.process_event(OpCumsum::new(
            TensorView::new(&input, invalid_source),
            TensorViewMut::new(&mut cumsum_output, source_layout),
        )),
        Err(SequenceError::InvalidView)
    );
    assert_eq!(cumsum_output, [9.0; 2]);
}

#[test]
fn cumsum_shape_and_repeat_view_rejections_are_explicit() {
    let source_layout = contiguous([3, 1, 1, 1]);
    let short_layout = contiguous([2, 1, 1, 1]);
    let input = [1.0_f32, 2.0, 3.0];
    let mut output = [9.0_f32; 2];
    let mut kernel = SequenceKernel::default();

    assert_eq!(
        kernel.process_event(OpCumsum::new(
            TensorView::new(&input, source_layout),
            TensorViewMut::new(&mut output, short_layout),
        )),
        Err(SequenceError::ShapeMismatch)
    );
    assert_eq!(output, [9.0; 2]);

    let invalid_layout = Layout::new(DType::F16, [3, 1, 1, 1], [4, 12, 12, 12]);
    let mut repeat_output = [9.0_f32; 3];
    assert_eq!(
        kernel.process_event(OpRepeat::new(
            TensorView::new(&input, invalid_layout),
            TensorViewMut::new(&mut repeat_output, source_layout),
        )),
        Err(SequenceError::InvalidView)
    );
    assert_eq!(repeat_output, [9.0; 3]);

    assert_eq!(
        format!("{}", SequenceError::InvalidView),
        "invalid sequence tensor view"
    );
    assert_eq!(
        format!("{}", SequenceError::ShapeMismatch),
        "sequence tensor shapes differ"
    );
    assert_eq!(
        format!("{}", SequenceError::InvalidAxis),
        "sequence concatenation axis is invalid"
    );
    assert_eq!(
        format!("{}", SequenceError::UnexpectedEvent),
        "unexpected sequence event"
    );
    assert_eq!(
        format!("{}", SequenceError::Internal),
        "internal sequence dispatch error"
    );
    assert!(!format!("{kernel:?}").is_empty());
}

#[test]
fn sequence_dispatch_is_allocation_free() {
    let source_layout = contiguous([4, 1, 1, 1]);
    let target_layout = contiguous([8, 1, 1, 1]);
    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let mut output = [0.0_f32; 8];
    let mut cumsum_output = [0.0_f32; 4];
    let mut repeat_back_output = [0.0_f32; 4];
    let mut kernel = SequenceKernel::new();
    let mut result = Err(SequenceError::UnexpectedEvent);
    let info = measure(|| {
        for _ in 0..128 {
            result = kernel.process_event(OpCumsum::new(
                TensorView::new(&input, source_layout),
                TensorViewMut::new(&mut cumsum_output, source_layout),
            ));
            assert_eq!(result, Ok(()));
            result = kernel.process_event(OpRepeat::new(
                TensorView::new(&input, source_layout),
                TensorViewMut::new(&mut output, target_layout),
            ));
            assert_eq!(result, Ok(()));
            result = kernel.process_event(OpRepeatBack::new(
                TensorView::new(&output, target_layout),
                TensorViewMut::new(&mut repeat_back_output, source_layout),
            ));
            assert_eq!(result, Ok(()));
        }
    });
    assert_eq!(info.count_total, 0);
    assert_eq!(info.bytes_total, 0);
    assert_eq!(cumsum_output, [1.0, 3.0, 6.0, 10.0]);
    assert_eq!(output, [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]);
}
