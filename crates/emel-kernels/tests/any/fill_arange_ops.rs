#![allow(clippy::float_cmp, missing_docs)]

use allocation_counter::measure;
use emel_kernels::Kernel;
use emel_kernels::any::event;
use emel_kernels::any::fill_arange::{
    FillArangeError, FillArangeKernel, OpArange, OpFill, UnexpectedFillArange,
};
use emel_kernels::any::tensor_view::{DType, Layout, TensorViewMut};
use emel_tensor as _;
use sml as _;

const PINNED_EMEL_CPP_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";

fn contiguous(ne: [u64; 4]) -> Layout {
    Layout::contiguous(DType::F32, ne).expect("test layout fits")
}

#[test]
fn fill_writes_every_logical_element() {
    let mut output = [0.0_f32; 6];
    let mut actor = FillArangeKernel::new();
    assert_eq!(
        actor.process_event(OpFill::new(
            TensorViewMut::new(&mut output, contiguous([3, 2, 1, 1])),
            -2.5,
        )),
        Ok(())
    );
    assert_eq!(output, [-2.5; 6]);
}

#[test]
fn arange_writes_start_and_step_progression() {
    let mut output = [0.0_f32; 5];
    let mut kernel = Kernel::new();
    assert_eq!(
        kernel.process_event(event::OpArange::new(
            TensorViewMut::new(&mut output, contiguous([5, 1, 1, 1])),
            1.5,
            -0.25,
        )),
        Ok(())
    );
    assert_eq!(output, [1.5, 1.25, 1.0, 0.75, 0.5]);
}

#[test]
fn invalid_views_are_rejected_without_mutation() {
    let mut output = [7.0_f32; 2];
    let mut actor = FillArangeKernel::new();
    let invalid = Layout::new(DType::F16, [2, 1, 1, 1], [2, 4, 8, 8]);
    assert_eq!(
        actor.process_event(OpFill::new(TensorViewMut::new(&mut output, invalid), 3.0)),
        Err(FillArangeError::InvalidView)
    );
    assert_eq!(output, [7.0; 2]);
}

#[test]
fn unexpected_events_are_explicit() {
    let mut actor = FillArangeKernel::new();
    assert_eq!(
        actor.process_event(UnexpectedFillArange),
        Err(FillArangeError::UnexpectedEvent)
    );
}

#[test]
fn valid_dispatch_does_not_allocate() {
    let layout = contiguous([64, 1, 1, 1]);
    let mut output = [0.0_f32; 64];
    let mut actor = FillArangeKernel::new();
    let allocations = measure(|| {
        assert_eq!(
            actor.process_event(OpArange::new(
                TensorViewMut::new(&mut output, layout),
                0.0,
                1.0,
            )),
            Ok(())
        );
    });
    assert_eq!(allocations.count_total, 0);
    assert_eq!(PINNED_EMEL_CPP_COMMIT.len(), 40);
}

#[test]
fn public_kernel_dispatches_fill() {
    let mut output = [0.0_f32; 3];
    assert_eq!(
        Kernel::new().process_event(event::OpFill::new(
            TensorViewMut::new(&mut output, contiguous([3, 1, 1, 1])),
            4.5,
        )),
        Ok(())
    );
    assert_eq!(output, [4.5; 3]);
    assert_eq!(
        FillArangeError::InvalidView.to_string(),
        "invalid fill/arange tensor view"
    );
    let _ = format!("{:?}", FillArangeKernel::default());
    let mut invalid_out = [7.0_f32; 2];
    let invalid = Layout::new(
        emel_kernels::any::tensor_view::DType::F16,
        [2, 1, 1, 1],
        [2, 4, 8, 8],
    );
    assert_eq!(
        FillArangeKernel::new().process_event(OpArange::new(
            TensorViewMut::new(&mut invalid_out, invalid),
            0.0,
            1.0,
        )),
        Err(FillArangeError::InvalidView)
    );
}
