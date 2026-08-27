#![allow(clippy::float_cmp, missing_docs)]

use allocation_counter::measure;
use emel_kernels::any::im2col::{
    F16OutputViewMut, Im2ColError, Im2ColKernel, Im2ColParams, OpIm2Col, OpIm2ColF16,
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

const fn f16_layout(ne: [u64; 4]) -> Layout {
    let mut nb = [2_u64; 4];
    nb[1] = nb[0] * ne[0];
    nb[2] = nb[1] * ne[1];
    nb[3] = nb[2] * ne[2];
    Layout::new(DType::F16, ne, nb)
}

const fn implicit_layout(ne: [u64; 4]) -> Layout {
    Layout::new(DType::F32, ne, [0, 0, 0, 0])
}

const fn params(stride: i32, padding: i32, dilation: i32, is_2d: i32) -> Im2ColParams {
    Im2ColParams {
        stride,
        padding,
        dilation,
        is_2d,
    }
}

#[test]
fn source_identity_and_zero_padding_match_pinned_formula() {
    assert_eq!(PINNED_EMEL_CPP_COMMIT.len(), 40);

    // kernel=3, channels=1, length=4, stride=1, padding=1, dilation=1.
    // out_length=(4 + 2*1 - 1*(3-1) - 1)/1 + 1=4.
    let kernel_layout = layout([3, 1, 1, 1]);
    let input_layout = layout([4, 1, 1, 1]);
    let output_layout = layout([3, 4, 1, 1]);
    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let mut output = [0.0_f32; 12];
    let mut actor = Im2ColKernel::new();

    assert_eq!(
        actor.process_event(OpIm2Col::new(
            kernel_layout,
            TensorView::new(&input, input_layout),
            TensorViewMut::new(&mut output, output_layout),
            params(1, 1, 1, 0),
        )),
        Ok(())
    );
    assert_eq!(
        output,
        [0.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 0.0]
    );
}

#[test]
fn multiple_channels_batches_stride_and_dilation_follow_layout() {
    // kernel=2, channels=2, length=5, batches=2, stride=2, dilation=2.
    // out_length=(5 - 2*(2-1) - 1)/2 + 1=2.
    let kernel_layout = layout([2, 2, 1, 1]);
    let input_layout = layout([5, 2, 2, 1]);
    let output_layout = layout([4, 2, 2, 1]);
    let input = [
        1.0, 2.0, 3.0, 4.0, 5.0, // channel 0, batch 0
        10.0, 20.0, 30.0, 40.0, 50.0, // channel 1, batch 0
        101.0, 102.0, 103.0, 104.0, 105.0, // channel 0, batch 1
        110.0, 120.0, 130.0, 140.0, 150.0, // channel 1, batch 1
    ];
    let mut output = [0.0_f32; 16];
    let mut actor = Im2ColKernel::new();

    assert_eq!(
        actor.process_event(OpIm2Col::new(
            kernel_layout,
            TensorView::new(&input, input_layout),
            TensorViewMut::new(&mut output, output_layout),
            params(2, 0, 2, 0),
        )),
        Ok(())
    );
    assert_eq!(
        output,
        [
            1.0, 3.0, 10.0, 30.0, 3.0, 5.0, 30.0, 50.0, 101.0, 103.0, 110.0, 130.0, 103.0, 105.0,
            130.0, 150.0
        ]
    );
}

#[test]
fn strided_input_is_read_by_logical_coordinates() {
    let kernel_layout = layout([2, 1, 1, 1]);
    let input_layout = Layout::new(DType::F32, [3, 1, 1, 1], [8, 24, 24, 24]);
    let output_layout = layout([2, 2, 1, 1]);
    let input = [1.0_f32, 99.0, 2.0, 88.0, 3.0];
    let mut output = [0.0_f32; 4];
    let mut actor = Im2ColKernel::new();

    assert_eq!(
        actor.process_event(OpIm2Col::new(
            kernel_layout,
            TensorView::new(&input, input_layout),
            TensorViewMut::new(&mut output, output_layout),
            params(1, 0, 1, 0),
        )),
        Ok(())
    );
    assert_eq!(output, [1.0, 2.0, 2.0, 3.0]);
}

#[test]
fn implicit_contiguous_input_and_output_match_reference_layout_semantics() {
    let kernel_layout = layout([2, 1, 1, 1]);
    let input_layout = implicit_layout([3, 1, 1, 1]);
    let output_layout = implicit_layout([2, 2, 1, 1]);
    let input = [7.0_f32, 8.0, 9.0];
    let mut output = [0.0_f32; 4];
    let mut actor = Im2ColKernel::new();

    assert_eq!(
        actor.process_event(OpIm2Col::new(
            kernel_layout,
            TensorView::new(&input, input_layout),
            TensorViewMut::new(&mut output, output_layout),
            params(1, 0, 1, 0),
        )),
        Ok(())
    );
    assert_eq!(output, [7.0, 8.0, 8.0, 9.0]);
}

#[test]
fn zero_batch_is_rejected_by_the_pinned_guard() {
    let kernel_layout = layout([2, 1, 1, 1]);
    let input_layout = implicit_layout([3, 1, 0, 1]);
    let output_layout = implicit_layout([2, 2, 0, 1]);
    let input = [0.0_f32];
    let mut output = [];
    let mut actor = Im2ColKernel::new();

    assert_eq!(
        actor.process_event(OpIm2Col::new(
            kernel_layout,
            TensorView::new(&input, input_layout),
            TensorViewMut::new(&mut output, output_layout),
            params(1, 0, 1, 0),
        )),
        Err(Im2ColError::InvalidView)
    );
    assert!(output.is_empty());
}

#[test]
fn f16_destination_matches_pinned_run_im2col_as_with_zero_padding() {
    let kernel_layout = layout([3, 1, 1, 1]);
    let input_layout = layout([4, 1, 1, 1]);
    let output_layout = f16_layout([3, 4, 1, 1]);
    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let mut output = [0_u16; 12];
    let mut actor = Im2ColKernel::new();

    assert_eq!(
        actor.process_event(OpIm2ColF16::new(
            kernel_layout,
            TensorView::new(&input, input_layout),
            F16OutputViewMut::new(&mut output, output_layout),
            params(1, 1, 1, 0),
        )),
        Ok(())
    );
    assert_eq!(
        output,
        [
            0x0000, 0x3c00, 0x4000, 0x3c00, 0x4000, 0x4200, 0x4000, 0x4200, 0x4400, 0x4200, 0x4400,
            0x0000,
        ]
    );
}

#[test]
fn f16_destination_rejects_shape_and_view_without_writing() {
    let kernel_layout = layout([2, 1, 1, 1]);
    let input_layout = layout([3, 1, 1, 1]);
    let input = [1.0_f32, 2.0, 3.0];
    let mut output = [0xffff_u16; 6];
    let mut actor = Im2ColKernel::new();

    assert_eq!(
        actor.process_event(OpIm2ColF16::new(
            kernel_layout,
            TensorView::new(&input, input_layout),
            F16OutputViewMut::new(&mut output, f16_layout([2, 3, 1, 1])),
            params(1, 0, 1, 0),
        )),
        Err(Im2ColError::ShapeMismatch)
    );
    assert_eq!(output, [0xffff; 6]);

    assert_eq!(
        actor.process_event(OpIm2ColF16::new(
            kernel_layout,
            TensorView::new(&input, input_layout),
            F16OutputViewMut::new(
                &mut output,
                Layout::new(DType::F16, [2, 2, 1, 1], [4, 8, 16, 16]),
            ),
            params(1, 0, 1, 0),
        )),
        Err(Im2ColError::InvalidView)
    );
    assert_eq!(output, [0xffff; 6]);
}

#[test]
fn f16_zero_batch_is_rejected_by_the_pinned_guard() {
    let kernel_layout = layout([2, 1, 1, 1]);
    let input_layout = implicit_layout([3, 1, 0, 1]);
    let output_layout = f16_layout([2, 2, 0, 1]);
    let input = [0.0_f32];
    let mut output = [];
    let mut actor = Im2ColKernel::new();

    assert_eq!(
        actor.process_event(OpIm2ColF16::new(
            kernel_layout,
            TensorView::new(&input, input_layout),
            F16OutputViewMut::new(&mut output, output_layout),
            params(1, 0, 1, 0),
        )),
        Err(Im2ColError::InvalidView)
    );
    assert!(output.is_empty());
}

#[test]
fn f16_dispatch_is_allocation_free() {
    let kernel_layout = layout([2, 1, 1, 1]);
    let input_layout = layout([3, 1, 1, 1]);
    let output_layout = f16_layout([2, 2, 1, 1]);
    let input = [1.0_f32, 2.0, 3.0];
    let mut output = [0_u16; 4];
    let mut actor = Im2ColKernel::new();

    let allocation = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                actor.process_event(OpIm2ColF16::new(
                    kernel_layout,
                    TensorView::new(&input, input_layout),
                    F16OutputViewMut::new(&mut output, output_layout),
                    params(1, 0, 1, 0),
                )),
                Ok(())
            );
        }
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(allocation.bytes_total, 0);
}

#[test]
fn guards_classify_parameters_shapes_and_views_without_writes() {
    let kernel_layout = layout([3, 1, 1, 1]);
    let input_layout = layout([4, 1, 1, 1]);
    let output_layout = layout([3, 4, 1, 1]);
    let bad_output_layout = layout([3, 3, 1, 1]);
    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let mut output = [9.0_f32; 12];
    let mut actor = Im2ColKernel::new();

    for invalid in [
        params(0, 0, 1, 0),
        params(1, -1, 1, 0),
        params(1, 0, 0, 0),
        params(1, 0, 1, 1),
    ] {
        assert_eq!(
            actor.process_event(OpIm2Col::new(
                kernel_layout,
                TensorView::new(&input, input_layout),
                TensorViewMut::new(&mut output, output_layout),
                invalid,
            )),
            Err(Im2ColError::InvalidParameters)
        );
        assert_eq!(output, [9.0; 12]);
    }

    assert_eq!(
        actor.process_event(OpIm2Col::new(
            kernel_layout,
            TensorView::new(&input, input_layout),
            TensorViewMut::new(&mut output, bad_output_layout),
            params(1, 1, 1, 0),
        )),
        Err(Im2ColError::ShapeMismatch)
    );
    assert_eq!(output, [9.0; 12]);

    let invalid_kernel_layout = Layout::new(DType::Other(99), [3, 1, 1, 1], [4, 12, 12, 12]);
    assert_eq!(
        actor.process_event(OpIm2Col::new(
            invalid_kernel_layout,
            TensorView::new(&input, input_layout),
            TensorViewMut::new(&mut output, output_layout),
            params(1, 1, 1, 0),
        )),
        Err(Im2ColError::InvalidView)
    );
    assert_eq!(output, [9.0; 12]);
}

#[test]
fn dispatch_is_allocation_free() {
    let kernel_layout = layout([3, 1, 1, 1]);
    let input_layout = layout([4, 1, 1, 1]);
    let output_layout = layout([3, 4, 1, 1]);
    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let mut output = [0.0_f32; 12];
    let mut actor = Im2ColKernel::new();

    let allocation = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                actor.process_event(OpIm2Col::new(
                    kernel_layout,
                    TensorView::new(&input, input_layout),
                    TensorViewMut::new(&mut output, output_layout),
                    params(1, 1, 1, 0),
                )),
                Ok(())
            );
        }
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(allocation.bytes_total, 0);
}

#[test]
fn public_surfaces_and_metadata_guards_are_typed() {
    assert!(format!("{}", Im2ColError::InvalidView).contains("view"));
    assert!(format!("{}", Im2ColError::ShapeMismatch).contains("shapes"));
    assert!(format!("{}", Im2ColError::InvalidParameters).contains("parameters"));
    assert!(format!("{}", Im2ColError::UnexpectedEvent).contains("unexpected"));
    assert!(format!("{}", Im2ColError::Internal).contains("internal"));
    assert!(format!("{:?}", Im2ColKernel::default()).contains("Im2ColKernel"));
}

#[test]
fn generated_machine_is_ready_after_every_public_dispatch() {
    let kernel_layout = layout([2, 1, 1, 1]);
    let input_layout = layout([3, 1, 1, 1]);
    let output_layout = layout([2, 2, 1, 1]);
    let input = [1.0_f32, 2.0, 3.0];
    let mut output = [0.0_f32; 4];
    let mut actor = Im2ColKernel::new();

    assert!(actor.is_ready());
    assert_eq!(
        actor.process_event(OpIm2Col::new(
            kernel_layout,
            TensorView::new(&input, input_layout),
            TensorViewMut::new(&mut output, output_layout),
            params(1, 0, 1, 0),
        )),
        Ok(())
    );
    assert!(actor.is_ready());
}
