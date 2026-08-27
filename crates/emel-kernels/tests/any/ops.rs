#![allow(missing_docs)]
#![allow(clippy::float_cmp)]

use allocation_counter::measure;
use emel_kernels::any::event;
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};
use emel_kernels::{Error, Kernel, KernelKind, dispatch_name, kernel_kind};
use emel_tensor as _;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use pulp as _;
use sml as _;

#[test]
fn dup_copies_f32_values() {
    let input = [1.0_f32, -2.5, 4.0];
    let mut output = [0.0_f32; 3];
    let mut kernel = Kernel::new();

    assert_eq!(
        kernel.process_event(event::OpDup::new(&input, &mut output)),
        Ok(())
    );
    assert_eq!(output, input);
}

#[test]
fn add_computes_elementwise_f32_values() {
    let lhs = [1.0_f32, -2.5, 4.0];
    let rhs = [2.0_f32, 0.5, -1.0];
    let mut output = [0.0_f32; 3];
    let mut kernel = Kernel::new();

    assert_eq!(
        kernel.process_event(event::OpAdd::new(&lhs, &rhs, &mut output)),
        Ok(())
    );
    assert_eq!(output, [3.0, -2.0, 3.0]);
}

#[test]
fn binary_and_unary_extensions_match_scalar_semantics() {
    let lhs = [9.0_f32, -4.0, 2.0];
    let rhs = [3.0_f32, 2.0, -4.0];
    let mut output = [0.0_f32; 3];
    let mut kernel = Kernel::new();

    assert_eq!(
        kernel.process_event(event::OpSub::new(&lhs, &rhs, &mut output)),
        Ok(())
    );
    assert_eq!(output, [6.0, -6.0, 6.0]);
    assert_eq!(
        kernel.process_event(event::OpMul::new(&lhs, &rhs, &mut output)),
        Ok(())
    );
    assert_eq!(output, [27.0, -8.0, -8.0]);
    assert_eq!(
        kernel.process_event(event::OpDiv::new(&lhs, &rhs, &mut output)),
        Ok(())
    );
    assert_eq!(output, [3.0, -2.0, -0.5]);
    assert_eq!(
        kernel.process_event(event::OpSqr::new(&rhs, &mut output)),
        Ok(())
    );
    assert_eq!(output, [9.0, 4.0, 16.0]);
    let mut sqrt_output = [0.0_f32; 3];
    assert_eq!(
        kernel.process_event(event::OpSqrt::new(&output, &mut sqrt_output)),
        Ok(())
    );
    assert_eq!(sqrt_output, [3.0, 2.0, 4.0]);
}

#[test]
fn invalid_shapes_are_explicitly_rejected() {
    let input = [1.0_f32, 2.0];
    let mut short_output = [0.0_f32];
    let mut kernel = Kernel::new();
    assert_eq!(
        kernel.process_event(event::OpDup::new(&input, &mut short_output)),
        Err(Error::InvalidShape)
    );

    let rhs = [3.0_f32];
    let mut output = [0.0_f32; 2];
    assert_eq!(
        kernel.process_event(event::OpAdd::new(&input, &rhs, &mut output)),
        Err(Error::InvalidShape)
    );

    let mut empty_output = [];
    assert_eq!(
        kernel.process_event(event::OpDup::new(&[], &mut empty_output)),
        Err(Error::InvalidShape)
    );
}

#[test]
fn every_maintained_rejection_route_and_error_display_is_covered() {
    let input = [1.0_f32, 2.0];
    let short = [0.0_f32];
    let mut output = [0.0_f32; 2];
    let mut kernel = Kernel::default();

    assert_eq!(
        kernel.process_event(event::OpSub::new(&input, &short, &mut output)),
        Err(Error::InvalidShape)
    );
    assert_eq!(
        kernel.process_event(event::OpMul::new(&input, &short, &mut output)),
        Err(Error::InvalidShape)
    );
    assert_eq!(
        kernel.process_event(event::OpDiv::new(&input, &short, &mut output)),
        Err(Error::InvalidShape)
    );
    assert_eq!(
        kernel.process_event(event::OpSqr::new(&input, &mut output[..1])),
        Err(Error::InvalidShape)
    );
    assert_eq!(
        kernel.process_event(event::OpSqrt::new(&input, &mut output[..1])),
        Err(Error::InvalidShape)
    );

    assert_eq!(format!("{}", Error::InvalidShape), "invalid kernel shape");
    assert_eq!(
        format!("{}", Error::UnexpectedEvent),
        "unexpected kernel event"
    );
    assert_eq!(
        format!("{}", Error::Internal),
        "internal kernel dispatch error"
    );
    assert!(!format!("{kernel:?}").is_empty());
}

#[test]
fn unsupported_operations_are_typed_rejections() {
    let mut kernel = Kernel::new();
    assert_eq!(
        kernel.process_event(event::Unsupported::new(event::UnsupportedOperation::MulMat)),
        Err(Error::UnsupportedOperation(
            event::UnsupportedOperation::MulMat
        ))
    );
}

#[test]
fn dispatch_is_allocation_free() {
    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let rhs = [4.0_f32, 3.0, 2.0, 1.0];
    let mut kernel = Kernel::new();
    let mut output = [0.0_f32; 4];
    let allocation = measure(|| {
        for _ in 0..256 {
            assert_eq!(
                kernel.process_event(event::OpDup::new(&input, &mut output)),
                Ok(())
            );
            assert_eq!(
                kernel.process_event(event::OpAdd::new(&input, &rhs, &mut output)),
                Ok(())
            );
            assert_eq!(
                kernel.process_event(event::OpSub::new(&input, &rhs, &mut output)),
                Ok(())
            );
            assert_eq!(
                kernel.process_event(event::OpMul::new(&input, &rhs, &mut output)),
                Ok(())
            );
            assert_eq!(
                kernel.process_event(event::OpDiv::new(&input, &rhs, &mut output)),
                Ok(())
            );
            assert_eq!(
                kernel.process_event(event::OpSqr::new(&input, &mut output)),
                Ok(())
            );
            assert_eq!(
                kernel.process_event(event::OpSqrt::new(&input, &mut output)),
                Ok(())
            );
        }
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(allocation.bytes_total, 0);
}

#[test]
fn kernel_kind_and_dispatch_name_are_explicit() {
    let expected = if cfg!(target_arch = "x86_64") {
        "x86_64"
    } else if cfg!(target_arch = "aarch64") {
        "aarch64"
    } else {
        "any"
    };
    assert_eq!(dispatch_name(), expected);
    assert!(matches!(
        kernel_kind(),
        KernelKind::X86_64 | KernelKind::Aarch64
    ));
}

#[test]
fn portable_kernel_composes_maintained_activation_and_elementwise_actors() {
    let layout = Layout::contiguous(DType::F32, [4, 1, 1, 1]).expect("layout fits");
    let input = [-2.0_f32, -0.5, 0.0, 2.0];
    let mut output = [0.0_f32; 4];
    let mut kernel = Kernel::new();

    assert_eq!(
        kernel.process_event(event::OpScale::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout),
            2.0,
        )),
        Ok(())
    );
    assert_eq!(output, [-4.0, -1.0, 0.0, 4.0]);

    assert_eq!(
        kernel.process_event(event::OpAdd1::new(
            TensorView::new(&input, layout),
            3.0,
            TensorViewMut::new(&mut output, layout),
        )),
        Ok(())
    );
    assert_eq!(output, [1.0, 2.5, 3.0, 5.0]);

    assert_eq!(
        kernel.process_event(event::OpClamp::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout),
            -1.0,
            1.0,
        )),
        Ok(())
    );
    assert_eq!(output, [-1.0, -0.5, 0.0, 1.0]);

    let gradient = [2.0_f32; 4];
    assert_eq!(
        kernel.process_event(event::OpSiluBack::new(
            TensorView::new(&input, layout),
            TensorView::new(&gradient, layout),
            TensorViewMut::new(&mut output, layout),
        )),
        Ok(())
    );

    assert_eq!(
        kernel.process_event(event::OpLeakyRelu::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout),
            0.1,
        )),
        Ok(())
    );
    assert_eq!(output, [-0.2, -0.05, 0.0, 2.0]);

    let base = [1.0_f32, 2.0, 3.0, 4.0];
    let update = [10.0_f32, 20.0];
    assert_eq!(
        kernel.process_event(event::OpAcc::new(
            TensorView::new(&base, layout),
            TensorView::new(
                &update,
                Layout::contiguous(DType::F32, [2, 1, 1, 1]).unwrap()
            ),
            TensorViewMut::new(&mut output, layout),
            1,
        )),
        Ok(())
    );
    assert_eq!(output, [1.0, 12.0, 23.0, 4.0]);
}

#[test]
fn portable_kernel_child_errors_are_typed_and_preserve_output() {
    let layout = Layout::contiguous(DType::F32, [2, 1, 1, 1]).expect("layout fits");
    let invalid = Layout::new(DType::F16, [2, 1, 1, 1], [2, 4, 4, 4]);
    let input = [1.0_f32, 2.0];
    let mut output = [7.0_f32; 2];
    let mut kernel = Kernel::new();

    assert_eq!(
        kernel.process_event(event::OpScale::new(
            TensorView::new(&input, invalid),
            TensorViewMut::new(&mut output, layout),
            2.0,
        )),
        Err(event::ActivationError::InvalidView)
    );
    assert_eq!(output, [7.0, 7.0]);

    assert_eq!(
        kernel.process_event(event::OpAdd1::new(
            TensorView::new(&input, layout),
            1.0,
            TensorViewMut::new(&mut output, invalid),
        )),
        Err(event::ElementwiseError::InvalidView)
    );
    assert_eq!(output, [7.0, 7.0]);
}

#[test]
fn portable_kernel_composed_dispatch_is_allocation_free() {
    let layout = Layout::contiguous(DType::F32, [8, 1, 1, 1]).expect("layout fits");
    let input = [1.0_f32; 8];
    let mut output = [0.0_f32; 8];
    let mut kernel = Kernel::new();
    let allocation = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                kernel.process_event(event::OpScale::new(
                    TensorView::new(&input, layout),
                    TensorViewMut::new(&mut output, layout),
                    0.5,
                )),
                Ok(())
            );
            assert_eq!(
                kernel.process_event(event::OpAdd1::new(
                    TensorView::new(&input, layout),
                    0.5,
                    TensorViewMut::new(&mut output, layout),
                )),
                Ok(())
            );
        }
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(allocation.bytes_total, 0);
}

#[test]
fn portable_kernel_composes_shape_sequence_and_normalization_actors() {
    let line = Layout::contiguous(DType::F32, [4, 1, 1, 1]).expect("layout fits");
    let matrix = Layout::contiguous(DType::F32, [2, 2, 1, 1]).expect("layout fits");
    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let mut output = [0.0_f32; 4];
    let mut kernel = Kernel::new();

    assert_eq!(
        kernel.process_event(event::OpCpy::new(
            TensorView::new(&input, line),
            TensorViewMut::new(&mut output, line),
        )),
        Ok(())
    );
    assert_eq!(output, input);

    assert_eq!(
        kernel.process_event(event::OpCumsum::new(
            TensorView::new(&input, line),
            TensorViewMut::new(&mut output, line),
        )),
        Ok(())
    );
    assert_eq!(output, [1.0, 3.0, 6.0, 10.0]);

    assert_eq!(
        kernel.process_event(event::OpConcat::new(
            TensorView::new(
                &input[..2],
                Layout::contiguous(DType::F32, [2, 1, 1, 1]).unwrap()
            ),
            TensorView::new(
                &input[2..],
                Layout::contiguous(DType::F32, [2, 1, 1, 1]).unwrap()
            ),
            TensorViewMut::new(&mut output, line),
            0,
        )),
        Ok(())
    );
    assert_eq!(output, input);

    let repeated_source = [5.0_f32, 7.0];
    let repeated_layout = Layout::contiguous(DType::F32, [2, 1, 1, 1]).unwrap();
    assert_eq!(
        kernel.process_event(event::OpRepeat::new(
            TensorView::new(&repeated_source, repeated_layout),
            TensorViewMut::new(&mut output, line),
        )),
        Ok(())
    );
    assert_eq!(output, [5.0, 7.0, 5.0, 7.0]);

    let mut normalized = [0.0_f32; 4];
    assert_eq!(
        kernel.process_event(event::OpNorm::new(
            TensorView::new(&input, matrix),
            TensorViewMut::new(&mut normalized, matrix),
            0.0,
        )),
        Ok(())
    );
    assert!((normalized[0] + 1.0).abs() < 1.0e-6);
    assert!((normalized[1] - 1.0).abs() < 1.0e-6);
    assert!((normalized[2] + 1.0).abs() < 1.0e-6);
    assert!((normalized[3] - 1.0).abs() < 1.0e-6);

    let reshaped = kernel
        .process_event(event::OpReshape::new(TensorView::new(&input, line), matrix))
        .expect("reshape preserves count");
    assert_eq!(reshaped.ne(), matrix.ne());
    let viewed = kernel
        .process_event(event::OpView::new(TensorView::new(&input, line), line))
        .expect("view preserves count");
    assert_eq!(viewed, line);
    let transposed = kernel
        .process_event(event::OpTranspose::new(TensorView::new(&input, matrix)))
        .expect("transpose preserves metadata");
    assert_eq!(transposed.ne(), [2, 2, 1, 1]);
}
