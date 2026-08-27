#![allow(clippy::float_cmp, missing_docs)]

use allocation_counter::measure;
use emel_kernels::any::activation::{
    ActivationError, ActivationKernel, OpClamp, OpLeakyRelu, OpScale, OpSiluBack,
};
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};
use emel_tensor as _;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use pulp as _;
use sml as _;

const PINNED_EMEL_CPP: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";

fn layout(count: u64) -> Layout {
    Layout::contiguous(DType::F32, [count, 1, 1, 1]).expect("test layout fits")
}

#[test]
fn source_identity_and_activation_formulas_match_pinned_contract() {
    assert_eq!(PINNED_EMEL_CPP, "843a117386ef17dc5a50549bbfc821074c2141d6");
    let source = [-2.0_f32, -0.5, 0.0, 2.0];
    let mut output = [0.0_f32; 4];
    let view = layout(4);
    let mut kernel = ActivationKernel::new();

    assert_eq!(
        kernel.process_event(OpScale::new(
            TensorView::new(&source, view),
            TensorViewMut::new(&mut output, view),
            2.0,
        )),
        Ok(())
    );
    assert_eq!(output, [-4.0, -1.0, 0.0, 4.0]);

    assert_eq!(
        kernel.process_event(OpClamp::new(
            TensorView::new(&source, view),
            TensorViewMut::new(&mut output, view),
            -1.0,
            1.0,
        )),
        Ok(())
    );
    assert_eq!(output, [-1.0, -0.5, 0.0, 1.0]);

    assert_eq!(
        kernel.process_event(OpLeakyRelu::new(
            TensorView::new(&source, view),
            TensorViewMut::new(&mut output, view),
            0.1,
        )),
        Ok(())
    );
    assert_eq!(output, [-0.2, -0.05, 0.0, 2.0]);
}

#[test]
fn clamp_preserves_nan_like_pinned_comparison_clamp() {
    let view = layout(1);
    let input = [f32::NAN];
    let mut output = [0.0_f32];
    let mut kernel = ActivationKernel::new();

    assert_eq!(
        kernel.process_event(OpClamp::new(
            TensorView::new(&input, view),
            TensorViewMut::new(&mut output, view),
            -1.0,
            1.0,
        )),
        Ok(())
    );
    assert!(output[0].is_nan());
}

#[test]
fn silu_backward_matches_reference_derivative() {
    let view = layout(3);
    let input = [-1.0_f32, 0.0, 1.0];
    let gradient = [2.0_f32; 3];
    let mut output = [0.0_f32; 3];
    let mut kernel = ActivationKernel::new();
    assert_eq!(
        kernel.process_event(OpSiluBack::new(
            TensorView::new(&input, view),
            TensorView::new(&gradient, view),
            TensorViewMut::new(&mut output, view),
        )),
        Ok(())
    );
    for (index, value) in input.iter().copied().enumerate() {
        let sigmoid = 1.0 / (1.0 + (-value).exp());
        let expected = 2.0 * sigmoid * value.mul_add(1.0 - sigmoid, 1.0);
        assert!((output[index] - expected).abs() < 1.0e-6);
    }
}

#[test]
fn guards_classify_shape_and_view_without_mutating_output() {
    let source_layout = layout(2);
    let mismatch = Layout::contiguous(DType::F32, [1, 2, 1, 1]).expect("mismatch fits");
    let invalid = Layout::new(DType::F16, [2, 1, 1, 1], [2, 4, 4, 4]);
    let source = [1.0_f32, 2.0];
    let mut output = [7.0_f32, 7.0];
    let mut kernel = ActivationKernel::new();

    assert_eq!(
        kernel.process_event(OpScale::new(
            TensorView::new(&source, source_layout),
            TensorViewMut::new(&mut output, mismatch),
            2.0,
        )),
        Err(ActivationError::ShapeMismatch)
    );
    assert_eq!(output, [7.0, 7.0]);

    assert_eq!(
        kernel.process_event(OpClamp::new(
            TensorView::new(&source, invalid),
            TensorViewMut::new(&mut output, source_layout),
            -1.0,
            1.0,
        )),
        Err(ActivationError::InvalidView)
    );
    assert_eq!(output, [7.0, 7.0]);
}

#[test]
fn activation_dispatch_is_allocation_free() {
    let view = layout(4);
    let source = [1.0_f32, -2.0, 3.0, -4.0];
    let mut output = [0.0_f32; 4];
    let mut kernel = ActivationKernel::new();
    let result = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                kernel.process_event(OpScale::new(
                    TensorView::new(&source, view),
                    TensorViewMut::new(&mut output, view),
                    0.5,
                )),
                Ok(())
            );
        }
    });
    assert_eq!(result.count_total, 0);
    assert_eq!(result.bytes_total, 0);
}

#[test]
fn silu_and_leaky_guards_classify_shape_and_view_failures() {
    let view = layout(3);
    let mismatch = Layout::contiguous(DType::F32, [2, 1, 1, 1]).expect("mismatch fits");
    let invalid = Layout::new(DType::F16, [3, 1, 1, 1], [4, 12, 12, 12]);
    let input = [1.0_f32; 3];
    let gradient = [2.0_f32; 3];
    let mut output = [7.0_f32; 3];
    let mut kernel = ActivationKernel::default();

    assert_eq!(
        kernel.process_event(OpSiluBack::new(
            TensorView::new(&input, view),
            TensorView::new(&gradient, mismatch),
            TensorViewMut::new(&mut output, view),
        )),
        Err(ActivationError::ShapeMismatch)
    );
    assert_eq!(output, [7.0; 3]);

    assert_eq!(
        kernel.process_event(OpSiluBack::new(
            TensorView::new(&input, invalid),
            TensorView::new(&gradient, view),
            TensorViewMut::new(&mut output, view),
        )),
        Err(ActivationError::InvalidView)
    );
    assert_eq!(output, [7.0; 3]);

    assert_eq!(
        kernel.process_event(OpLeakyRelu::new(
            TensorView::new(&input, view),
            TensorViewMut::new(&mut output, mismatch),
            0.1,
        )),
        Err(ActivationError::ShapeMismatch)
    );
    assert_eq!(output, [7.0; 3]);

    assert_eq!(
        kernel.process_event(OpLeakyRelu::new(
            TensorView::new(&input, invalid),
            TensorViewMut::new(&mut output, view),
            0.1,
        )),
        Err(ActivationError::InvalidView)
    );
    assert_eq!(output, [7.0; 3]);
}

#[test]
fn activation_errors_and_debug_default_are_observable() {
    assert_eq!(
        ActivationError::InvalidView.to_string(),
        "invalid activation-operation view"
    );
    assert_eq!(
        ActivationError::ShapeMismatch.to_string(),
        "activation-operation shapes differ"
    );
    assert_eq!(
        ActivationError::UnexpectedEvent.to_string(),
        "unexpected activation-operation event"
    );
    assert_eq!(
        ActivationError::Internal.to_string(),
        "internal activation-operation dispatch error"
    );
    let kernel = ActivationKernel::default();
    assert!(format!("{kernel:?}").contains("ActivationKernel"));
}
