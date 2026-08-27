#![allow(clippy::float_cmp, missing_docs)]

use allocation_counter::measure;
use emel_kernels::Kernel;
use emel_kernels::any::conv_transpose_1d::{
    ConvTranspose1dError, ConvTranspose1dKernel, ConvTranspose1dParams, OpConvTranspose1d,
};
use emel_kernels::any::event::UnexpectedConvTranspose1d;
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};
use emel_tensor as _;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use pulp as _;
use sml as _;

const PINNED_EMEL_CPP_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const PINNED_EVENTS_BLOB: &str = "4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9";
const PINNED_DETAIL_BLOB: &str = "c8a82643eabfe8f2d7883e655955f455794511b0";
const PINNED_X86_SM_BLOB: &str = "0b4d635ebbd0fbd52dbca8a2345547fb571205c8";
const PINNED_AARCH64_SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";
const MAX_CONV_GUARD_EXTENT: u64 = 1 << 31;

fn layout(ne: [u64; 4]) -> Layout {
    Layout::contiguous(DType::F32, ne).expect("test layout fits")
}

const fn params(stride: i32) -> ConvTranspose1dParams {
    ConvTranspose1dParams {
        stride,
        padding: 0,
        dilation: 1,
    }
}

#[test]
fn source_identity_and_f32_scatter_formula_match_pinned_contract() {
    assert_eq!(PINNED_EMEL_CPP_COMMIT.len(), 40);
    assert_eq!(PINNED_EVENTS_BLOB.len(), 40);
    assert_eq!(PINNED_DETAIL_BLOB.len(), 40);
    assert_eq!(PINNED_X86_SM_BLOB.len(), 40);
    assert_eq!(PINNED_AARCH64_SM_BLOB.len(), 40);

    // kernel=3, out_channels=1, in_channels=1, length=2, stride=2.
    // out_length=(2-1)*2+3=5; the second input overlaps tap 0 of the first.
    let weights_layout = layout([3, 1, 1, 1]);
    let input_layout = layout([2, 1, 1, 1]);
    let output_layout = layout([5, 1, 1, 1]);
    let weights = [1.0_f32, 2.0, 3.0];
    let input = [1.0_f32, 2.0];
    let mut output = [0.0_f32; 5];
    let mut kernel = ConvTranspose1dKernel::new();

    assert_eq!(
        kernel.process_event(OpConvTranspose1d::new(
            TensorView::new(&weights, weights_layout),
            TensorView::new(&input, input_layout),
            TensorViewMut::new(&mut output, output_layout),
            params(2),
        )),
        Ok(())
    );
    assert_eq!(output, [1.0, 2.0, 5.0, 4.0, 6.0]);
}

#[test]
fn multiple_channels_follow_pinned_weight_and_output_layouts() {
    // weights [kernel=2, out_channels=2, in_channels=1], input [length=2, 1].
    let weights_layout = layout([2, 2, 1, 1]);
    let input_layout = layout([2, 1, 1, 1]);
    let output_layout = layout([3, 2, 1, 1]);
    let weights = [1.0_f32, 2.0, 3.0, 4.0];
    let input = [5.0_f32, 6.0];
    let mut output = [0.0_f32; 6];
    let mut kernel = ConvTranspose1dKernel::new();

    assert_eq!(
        kernel.process_event(OpConvTranspose1d::new(
            TensorView::new(&weights, weights_layout),
            TensorView::new(&input, input_layout),
            TensorViewMut::new(&mut output, output_layout),
            params(1),
        )),
        Ok(())
    );
    // Output is [out_length, out_channels] with channel rows contiguous.
    assert_eq!(output, [5.0, 16.0, 12.0, 15.0, 38.0, 24.0]);
}

#[test]
fn strided_weights_and_input_are_read_safely() {
    let weights_layout = Layout::new(DType::F32, [2, 1, 1, 1], [8, 16, 16, 16]);
    let input_layout = Layout::new(DType::F32, [2, 1, 1, 1], [8, 16, 16, 16]);
    let output_layout = layout([3, 1, 1, 1]);
    let weights = [2.0_f32, -9.0, 3.0];
    let input = [4.0_f32, -9.0, 5.0];
    let mut output = [0.0_f32; 3];
    let mut kernel = ConvTranspose1dKernel::new();

    assert_eq!(
        kernel.process_event(OpConvTranspose1d::new(
            TensorView::new(&weights, weights_layout),
            TensorView::new(&input, input_layout),
            TensorViewMut::new(&mut output, output_layout),
            params(1),
        )),
        Ok(())
    );
    assert_eq!(output, [8.0, 22.0, 15.0]);
}

#[test]
fn implicit_contiguous_nb0_is_resolved_by_the_f32_view_contract() {
    let weights_layout = Layout::new(DType::F32, [2, 1, 1, 1], [0, 8, 8, 8]);
    let input_layout = Layout::new(DType::F32, [2, 1, 1, 1], [0, 8, 8, 8]);
    let output_layout = layout([3, 1, 1, 1]);
    let weights = [2.0_f32, 3.0];
    let input = [4.0_f32, 5.0];
    let mut output = [0.0_f32; 3];
    let mut kernel = ConvTranspose1dKernel::new();

    assert_eq!(
        kernel.process_event(OpConvTranspose1d::new(
            TensorView::new(&weights, weights_layout),
            TensorView::new(&input, input_layout),
            TensorViewMut::new(&mut output, output_layout),
            params(1),
        )),
        Ok(())
    );
    assert_eq!(output, [8.0, 22.0, 15.0]);
}

#[test]
fn singleton_zero_outer_stride_is_rejected_without_writing() {
    let weights_layout = layout([2, 1, 1, 1]);
    let input_layout = Layout::new(DType::F32, [2, 1, 1, 1], [4, 0, 8, 8]);
    let output_layout = layout([3, 1, 1, 1]);
    let weights = [2.0_f32, 3.0];
    let input = [4.0_f32, 5.0];
    let mut output = [9.0_f32; 3];
    let mut kernel = ConvTranspose1dKernel::new();

    assert_eq!(
        kernel.process_event(OpConvTranspose1d::new(
            TensorView::new(&weights, weights_layout),
            TensorView::new(&input, input_layout),
            TensorViewMut::new(&mut output, output_layout),
            params(1),
        )),
        Err(ConvTranspose1dError::InvalidView)
    );
    assert_eq!(output, [9.0; 3]);
}

#[test]
fn implicit_dense_output_nb0_is_rejected_without_writing() {
    let weights_layout = layout([2, 1, 1, 1]);
    let input_layout = layout([2, 1, 1, 1]);
    let output_layout = Layout::new(DType::F32, [3, 1, 1, 1], [0, 12, 12, 12]);
    let weights = [2.0_f32, 3.0];
    let input = [4.0_f32, 5.0];
    let mut output = [9.0_f32; 3];
    let mut kernel = ConvTranspose1dKernel::new();

    assert_eq!(
        kernel.process_event(OpConvTranspose1d::new(
            TensorView::new(&weights, weights_layout),
            TensorView::new(&input, input_layout),
            TensorViewMut::new(&mut output, output_layout),
            params(1),
        )),
        Err(ConvTranspose1dError::InvalidView)
    );
    assert_eq!(output, [9.0; 3]);
}

#[test]
fn guards_classify_parameters_shapes_and_views_without_writes() {
    let weights_layout = layout([3, 1, 1, 1]);
    let input_layout = layout([2, 1, 1, 1]);
    let output_layout = layout([5, 1, 1, 1]);
    let bad_output_layout = layout([4, 1, 1, 1]);
    let weights = [1.0_f32, 2.0, 3.0];
    let input = [1.0_f32, 2.0];
    let mut output = [9.0_f32; 5];
    let mut kernel = ConvTranspose1dKernel::new();

    assert_eq!(
        kernel.process_event(OpConvTranspose1d::new(
            TensorView::new(&weights, weights_layout),
            TensorView::new(&input, input_layout),
            TensorViewMut::new(&mut output, output_layout),
            ConvTranspose1dParams {
                stride: 0,
                padding: 0,
                dilation: 1,
            },
        )),
        Err(ConvTranspose1dError::InvalidParameters)
    );
    assert_eq!(output, [9.0; 5]);

    assert_eq!(
        kernel.process_event(OpConvTranspose1d::new(
            TensorView::new(&weights, weights_layout),
            TensorView::new(&input, input_layout),
            TensorViewMut::new(&mut output, bad_output_layout),
            params(2),
        )),
        Err(ConvTranspose1dError::ShapeMismatch)
    );
    assert_eq!(output, [9.0; 5]);

    let invalid_weights_layout = Layout::new(DType::F16, [3, 1, 1, 1], [4, 12, 12, 12]);
    assert_eq!(
        kernel.process_event(OpConvTranspose1d::new(
            TensorView::new(&weights, invalid_weights_layout),
            TensorView::new(&input, input_layout),
            TensorViewMut::new(&mut output, output_layout),
            params(2),
        )),
        Err(ConvTranspose1dError::InvalidView)
    );
    assert_eq!(output, [9.0; 5]);

    let invalid_output_layout = Layout::new(DType::F32, [5, 1, 1, 1], [8, 32, 32, 32]);
    assert_eq!(
        kernel.process_event(OpConvTranspose1d::new(
            TensorView::new(&weights, weights_layout),
            TensorView::new(&input, input_layout),
            TensorViewMut::new(&mut output, invalid_output_layout),
            params(2),
        )),
        Err(ConvTranspose1dError::InvalidView)
    );
    assert_eq!(output, [9.0; 5]);
}

#[test]
fn oversized_extent_cap_rejects_without_mutating_output() {
    let weights_layout = Layout::new(
        DType::F32,
        [MAX_CONV_GUARD_EXTENT + 1, 1, 1, 1],
        [4, 4, 4, 4],
    );
    let input_layout = layout([1, 1, 1, 1]);
    let output_layout = layout([1, 1, 1, 1]);
    let weights = [1.0_f32];
    let input = [2.0_f32];
    let mut output = [9.0_f32];
    let mut kernel = ConvTranspose1dKernel::new();

    assert_eq!(
        kernel.process_event(OpConvTranspose1d::new(
            TensorView::new(&weights, weights_layout),
            TensorView::new(&input, input_layout),
            TensorViewMut::new(&mut output, output_layout),
            params(1),
        )),
        Err(ConvTranspose1dError::InvalidView)
    );
    assert_eq!(output, [9.0]);
    assert!(kernel.is_ready());
}

#[test]
fn oversized_destination_product_rejects_without_mutating_output() {
    let weights_layout = Layout::new(DType::F32, [1, MAX_CONV_GUARD_EXTENT, 1, 1], [4, 4, 4, 4]);
    let input_layout = layout([1, 1, 1, 1]);
    let output_layout = Layout::new(DType::F32, [1, MAX_CONV_GUARD_EXTENT, 1, 1], [4, 4, 4, 4]);
    let weights = [1.0_f32];
    let input = [2.0_f32];
    let mut output = [9.0_f32];
    let mut kernel = ConvTranspose1dKernel::new();

    assert_eq!(
        kernel.process_event(OpConvTranspose1d::new(
            TensorView::new(&weights, weights_layout),
            TensorView::new(&input, input_layout),
            TensorViewMut::new(&mut output, output_layout),
            params(1),
        )),
        Err(ConvTranspose1dError::InvalidView)
    );
    assert_eq!(output, [9.0]);
    assert!(kernel.is_ready());
}

#[test]
fn dispatch_is_allocation_free() {
    let weights_layout = layout([3, 1, 1, 1]);
    let input_layout = layout([2, 1, 1, 1]);
    let output_layout = layout([5, 1, 1, 1]);
    let weights = [1.0_f32, 2.0, 3.0];
    let input = [1.0_f32, 2.0];
    let mut output = [0.0_f32; 5];
    let mut kernel = ConvTranspose1dKernel::new();

    let allocation = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                kernel.process_event(OpConvTranspose1d::new(
                    TensorView::new(&weights, weights_layout),
                    TensorView::new(&input, input_layout),
                    TensorViewMut::new(&mut output, output_layout),
                    params(2),
                )),
                Ok(())
            );
        }
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(allocation.bytes_total, 0);
}

#[test]
fn public_surfaces_are_typed_and_debuggable() {
    assert!(format!("{}", ConvTranspose1dError::InvalidView).contains("view"));
    assert!(format!("{}", ConvTranspose1dError::ShapeMismatch).contains("shapes"));
    assert!(format!("{}", ConvTranspose1dError::InvalidParameters).contains("parameters"));
    assert!(format!("{}", ConvTranspose1dError::UnexpectedEvent).contains("unexpected"));
    assert!(format!("{}", ConvTranspose1dError::Internal).contains("internal"));
    assert!(format!("{:?}", ConvTranspose1dKernel::default()).contains("ConvTranspose1dKernel"));
}

#[test]
fn root_kernel_composes_transposed_convolution_and_recovers() {
    let weights_layout = layout([3, 1, 1, 1]);
    let input_layout = layout([2, 1, 1, 1]);
    let output_layout = layout([5, 1, 1, 1]);
    let weights = [1.0_f32, 2.0, 3.0];
    let input = [1.0_f32, 2.0];
    let mut output = [0.0_f32; 5];
    let mut kernel = Kernel::new();

    assert!(kernel.is_ready());
    assert_eq!(
        kernel.process_event(OpConvTranspose1d::new(
            TensorView::new(&weights, weights_layout),
            TensorView::new(&input, input_layout),
            TensorViewMut::new(&mut output, output_layout),
            params(2),
        )),
        Ok(())
    );
    assert_eq!(output, [1.0, 2.0, 5.0, 4.0, 6.0]);
    assert!(kernel.is_ready());

    assert_eq!(
        kernel.process_event(UnexpectedConvTranspose1d),
        Err(ConvTranspose1dError::UnexpectedEvent)
    );
    assert!(kernel.is_ready());
}

#[test]
fn root_kernel_rejects_invalid_transposed_convolution_without_writing() {
    let weights_layout = layout([3, 1, 1, 1]);
    let input_layout = layout([2, 1, 1, 1]);
    let output_layout = layout([4, 1, 1, 1]);
    let weights = [1.0_f32, 2.0, 3.0];
    let input = [1.0_f32, 2.0];
    let mut output = [9.0_f32; 4];
    let mut kernel = Kernel::new();

    assert_eq!(
        kernel.process_event(OpConvTranspose1d::new(
            TensorView::new(&weights, weights_layout),
            TensorView::new(&input, input_layout),
            TensorViewMut::new(&mut output, output_layout),
            params(2),
        )),
        Err(ConvTranspose1dError::ShapeMismatch)
    );
    assert_eq!(output, [9.0; 4]);
    assert!(kernel.is_ready());
}

#[test]
fn root_transposed_convolution_dispatch_is_allocation_free() {
    let weights_layout = layout([3, 1, 1, 1]);
    let input_layout = layout([2, 1, 1, 1]);
    let output_layout = layout([5, 1, 1, 1]);
    let weights = [1.0_f32, 2.0, 3.0];
    let input = [1.0_f32, 2.0];
    let mut output = [0.0_f32; 5];
    let mut kernel = Kernel::new();

    let allocation = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                kernel.process_event(OpConvTranspose1d::new(
                    TensorView::new(&weights, weights_layout),
                    TensorView::new(&input, input_layout),
                    TensorViewMut::new(&mut output, output_layout),
                    params(2),
                )),
                Ok(())
            );
        }
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(allocation.bytes_total, 0);
}
