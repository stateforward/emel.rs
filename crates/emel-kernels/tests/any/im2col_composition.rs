#![allow(clippy::float_cmp, missing_docs)]

use allocation_counter::measure;
use emel_kernels::Kernel;
use emel_kernels::any::event::{
    F16OutputViewMut, Im2ColError, Im2ColParams, OpIm2Col, OpIm2ColF16, UnexpectedIm2Col,
};
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};
use emel_tensor as _;
use pulp as _;
use sml as _;

fn f32_layout(ne: [u64; 4]) -> Layout {
    Layout::contiguous(DType::F32, ne).expect("test layout fits")
}

const fn f16_layout(ne: [u64; 4]) -> Layout {
    let mut nb = [2_u64; 4];
    nb[1] = nb[0] * ne[0];
    nb[2] = nb[1] * ne[1];
    nb[3] = nb[2] * ne[2];
    Layout::new(DType::F16, ne, nb)
}

const PARAMS: Im2ColParams = Im2ColParams {
    stride: 1,
    padding: 1,
    dilation: 1,
    is_2d: 0,
};

#[test]
fn root_forwards_f32_im2col_to_owned_child() {
    let kernel_layout = f32_layout([3, 1, 1, 1]);
    let input_layout = f32_layout([4, 1, 1, 1]);
    let output_layout = f32_layout([3, 4, 1, 1]);
    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let mut output = [0.0_f32; 12];
    let mut actor = Kernel::new();

    assert_eq!(
        actor.process_event(OpIm2Col::new(
            kernel_layout,
            TensorView::new(&input, input_layout),
            TensorViewMut::new(&mut output, output_layout),
            PARAMS,
        )),
        Ok(())
    );
    assert_eq!(
        output,
        [0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 0.0]
    );
    assert!(actor.is_ready());
}

#[test]
fn root_forwards_f16_im2col_to_owned_child() {
    let kernel_layout = f32_layout([2, 1, 1, 1]);
    let input_layout = f32_layout([3, 1, 1, 1]);
    let output_layout = f16_layout([2, 2, 1, 1]);
    let input = [1.0_f32, 2.0, 3.0];
    let mut output = [0_u16; 6];
    let mut actor = Kernel::new();

    assert_eq!(
        actor.process_event(OpIm2ColF16::new(
            kernel_layout,
            TensorView::new(&input, input_layout),
            F16OutputViewMut::new(&mut output, output_layout),
            Im2ColParams {
                stride: 1,
                padding: 0,
                dilation: 1,
                is_2d: 0,
            },
        )),
        Ok(())
    );
    assert_eq!(output, [0x3c00, 0x4000, 0x4000, 0x4200, 0, 0]);
    assert!(actor.is_ready());
}

#[test]
fn root_typed_unexpected_im2col_is_explicit_and_recovers() {
    let mut actor = Kernel::new();
    assert_eq!(
        actor.process_event(UnexpectedIm2Col),
        Err(Im2ColError::UnexpectedEvent)
    );
    assert!(actor.is_ready());
}

#[test]
fn root_rejects_im2col_shape_without_mutating_and_recovers() {
    let kernel_layout = f32_layout([2, 1, 1, 1]);
    let input_layout = f32_layout([3, 1, 1, 1]);
    let output_layout = f32_layout([2, 1, 1, 1]);
    let input = [1.0_f32, 2.0, 3.0];
    let mut output = [9.0_f32; 2];
    let mut actor = Kernel::new();

    assert_eq!(
        actor.process_event(OpIm2Col::new(
            kernel_layout,
            TensorView::new(&input, input_layout),
            TensorViewMut::new(&mut output, output_layout),
            Im2ColParams {
                stride: 1,
                padding: 0,
                dilation: 1,
                is_2d: 0,
            },
        )),
        Err(Im2ColError::ShapeMismatch)
    );
    assert_eq!(output, [9.0, 9.0]);
    assert!(actor.is_ready());
}

#[test]
fn root_im2col_dispatch_is_allocation_free() {
    let kernel_layout = f32_layout([2, 1, 1, 1]);
    let input_layout = f32_layout([3, 1, 1, 1]);
    let output_layout = f32_layout([2, 2, 1, 1]);
    let input = [1.0_f32, 2.0, 3.0];
    let mut output = [0.0_f32; 4];
    let mut actor = Kernel::new();
    let event = OpIm2Col::new(
        kernel_layout,
        TensorView::new(&input, input_layout),
        TensorViewMut::new(&mut output, output_layout),
        Im2ColParams {
            stride: 1,
            padding: 0,
            dilation: 1,
            is_2d: 0,
        },
    );

    let allocations = measure(|| assert_eq!(actor.process_event(event), Ok(())));
    assert_eq!(allocations.count_total, 0);
}
