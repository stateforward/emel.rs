#![allow(clippy::float_cmp, missing_docs)]

use allocation_counter::measure;
use emel_kernels::any::event::{GroupNormError, GroupNormKernel, OpGroupNorm, OpL2Norm};
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};
use emel_kernels::{Kernel, any::event};
use emel_tensor as _;
use sml as _;

fn dense(ne: [u64; 4]) -> Layout {
    Layout::contiguous(DType::F32, ne).expect("small test layout fits")
}

#[test]
fn group_norm_normalizes_each_channel_group() {
    let layout = dense([2, 1, 4, 1]);
    let input = [1.0, 3.0, 1.0, 3.0, 10.0, 14.0, 10.0, 14.0];
    let mut output = [0.0; 8];
    let mut kernel = GroupNormKernel::new();

    assert_eq!(
        kernel.process_event(OpGroupNorm::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout),
            2,
            0.0,
        )),
        Ok(())
    );
    assert_eq!(output, [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]);
}

#[test]
fn l2_norm_matches_pinned_f32_sum_and_rejects_zero() {
    let layout = dense([3, 1, 1, 1]);
    let input = [3.0, 4.0, 0.0];
    let mut output = [0.0; 3];
    let mut kernel = GroupNormKernel::new();
    assert_eq!(
        kernel.process_event(OpL2Norm::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout),
        )),
        Ok(())
    );
    assert_eq!(output, [0.6, 0.8, 0.0]);

    let zero = [0.0; 3];
    assert_eq!(
        kernel.process_event(OpL2Norm::new(
            TensorView::new(&zero, layout),
            TensorViewMut::new(&mut output, layout),
        )),
        Err(GroupNormError::ZeroNorm)
    );
}

#[test]
fn parameter_and_shape_rejections_preserve_output() {
    let layout = dense([2, 1, 2, 1]);
    let mismatched = dense([2, 1, 1, 1]);
    let input = [1.0, 2.0, 3.0, 4.0];
    let mut output = [7.0; 4];
    let mut kernel = GroupNormKernel::new();
    assert_eq!(
        kernel.process_event(OpGroupNorm::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, mismatched),
            1,
            0.0,
        )),
        Err(GroupNormError::ShapeMismatch)
    );
    assert_eq!(output, [7.0; 4]);

    assert_eq!(
        kernel.process_event(OpGroupNorm::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout),
            3,
            0.0,
        )),
        Err(GroupNormError::InvalidParameters)
    );
    assert_eq!(output, [7.0; 4]);
}

#[test]
fn dispatch_is_allocation_free_after_actor_construction() {
    let layout = dense([4, 1, 2, 1]);
    let input = [1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0];
    let mut output = [0.0; 8];
    let mut kernel = GroupNormKernel::new();
    let allocation = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                kernel.process_event(OpGroupNorm::new(
                    TensorView::new(&input, layout),
                    TensorViewMut::new(&mut output, layout),
                    2,
                    1.0,
                )),
                Ok(())
            );
        }
    });
    assert_eq!(allocation.count_total, 0);
}

#[test]
fn root_kernel_dispatches_both_public_events() {
    let layout = dense([2, 1, 2, 1]);
    let input = [1.0, 3.0, 1.0, 3.0];
    let mut output = [0.0; 4];
    let mut kernel = Kernel::new();
    assert_eq!(
        kernel.process_event(event::OpGroupNorm::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout),
            2,
            0.0,
        )),
        Ok(())
    );
    assert_eq!(output, [-1.0, 1.0, -1.0, 1.0]);
    assert_eq!(
        kernel.process_event(event::OpL2Norm::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout),
        )),
        Ok(())
    );
}
