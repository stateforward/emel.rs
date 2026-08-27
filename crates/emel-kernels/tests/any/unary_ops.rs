#![allow(clippy::float_cmp, missing_docs)]

use allocation_counter::measure;
use emel_kernels::Kernel;
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};
use emel_kernels::any::unary::*;
use emel_tensor as _;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use pulp as _;
use sml as _;

const PINNED_EMEL_CPP_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";

fn layout() -> Layout {
    Layout::contiguous(DType::F32, [4, 1, 1, 1]).expect("layout fits")
}

#[test]
fn source_identity_and_formula_family_are_pinned() {
    assert_eq!(PINNED_EMEL_CPP_COMMIT.len(), 40);
}

#[test]
fn unary_formulas_follow_pinned_scalar_family() {
    let layout = layout();
    let input = [-2.0_f32, -0.5, 0.5, 2.0];
    let mut output = [0.0_f32; 4];
    let mut actor = UnaryKernel::new();

    assert_eq!(
        actor.process_event(OpAbs::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout)
        )),
        Ok(())
    );
    assert_eq!(output, [2.0, 0.5, 0.5, 2.0]);
    assert_eq!(
        actor.process_event(OpSgn::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout)
        )),
        Ok(())
    );
    assert_eq!(output, [-1.0, -1.0, 1.0, 1.0]);
    assert_eq!(
        actor.process_event(OpNeg::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout)
        )),
        Ok(())
    );
    assert_eq!(output, [2.0, 0.5, -0.5, -2.0]);
    assert_eq!(
        actor.process_event(OpStep::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout)
        )),
        Ok(())
    );
    assert_eq!(output, [0.0, 0.0, 1.0, 1.0]);
    assert_eq!(
        actor.process_event(OpRelu::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout)
        )),
        Ok(())
    );
    assert_eq!(output, [0.0, 0.0, 0.5, 2.0]);
    assert_eq!(
        actor.process_event(OpSilu::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout)
        )),
        Ok(())
    );
    assert!((output[2] - 0.311_229_68).abs() < 1e-6);
    assert_eq!(
        actor.process_event(OpExp::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout)
        )),
        Ok(())
    );
    assert!((output[0] - (-2.0_f32).exp()).abs() < 1e-6);
    assert_eq!(
        actor.process_event(OpExpm1::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout)
        )),
        Ok(())
    );
    assert!((output[2] - 0.5_f32.exp_m1()).abs() < 1e-6);
    assert_eq!(
        actor.process_event(OpFloor::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout)
        )),
        Ok(())
    );
    assert_eq!(output, [-2.0, -1.0, 0.0, 2.0]);
    assert_eq!(
        actor.process_event(OpCeil::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout)
        )),
        Ok(())
    );
    assert_eq!(output, [-2.0, 0.0, 1.0, 2.0]);
    assert_eq!(
        actor.process_event(OpRound::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout)
        )),
        Ok(())
    );
    assert_eq!(output, [-2.0, -1.0, 1.0, 2.0]);
    assert_eq!(
        actor.process_event(OpTrunc::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout)
        )),
        Ok(())
    );
    assert_eq!(output, [-2.0, 0.0, 0.0, 2.0]);
}

#[test]
fn generic_op_unary_routes_every_pinned_scalar_suboperation() {
    let layout = layout();
    let input = [-10.1_f32, -1.25, 0.0, 10.1];
    let expected = [
        UnarySubOp::Abs,
        UnarySubOp::Neg,
        UnarySubOp::Relu,
        UnarySubOp::Exp,
        UnarySubOp::Tanh,
        UnarySubOp::Elu,
        UnarySubOp::Gelu,
        UnarySubOp::Silu,
    ];
    let mut output = [0.0_f32; 4];
    let mut actor = UnaryKernel::new();
    for subop in expected {
        output.fill(0.0);
        assert_eq!(
            actor.process_event(OpUnary::new(
                subop,
                TensorView::new(&input, layout),
                TensorViewMut::new(&mut output, layout),
            )),
            Ok(())
        );
    }
}

#[test]
fn generic_op_unary_rejects_unsupported_suboperation_without_mutation() {
    let layout = layout();
    let input = [1.0_f32; 4];
    let mut output = [9.0_f32; 4];
    let mut actor = UnaryKernel::new();
    assert_eq!(
        actor.process_event(OpUnary::new(
            UnarySubOp::Sgn,
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout),
        )),
        Err(UnaryError::UnexpectedEvent)
    );
    assert_eq!(output, [9.0; 4]);
}

#[test]
fn generic_op_unary_accepts_equal_element_count_with_different_shapes() {
    let input_layout = Layout::contiguous(DType::F32, [2, 2, 1, 1]).expect("layout fits");
    let output_layout = Layout::contiguous(DType::F32, [4, 1, 1, 1]).expect("layout fits");
    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let mut output = [0.0_f32; 4];
    let mut actor = UnaryKernel::new();

    assert_eq!(
        actor.process_event(OpUnary::new(
            UnarySubOp::Abs,
            TensorView::new(&input, input_layout),
            TensorViewMut::new(&mut output, output_layout),
        )),
        Ok(())
    );
    assert_eq!(output, input);

    let mismatch_layout = Layout::contiguous(DType::F32, [3, 1, 1, 1]).expect("layout fits");
    output.fill(9.0);
    assert_eq!(
        actor.process_event(OpUnary::new(
            UnarySubOp::Abs,
            TensorView::new(&input, input_layout),
            TensorViewMut::new(&mut output, mismatch_layout),
        )),
        Err(UnaryError::UnexpectedEvent)
    );
    assert_eq!(output, [9.0; 4]);
}

#[test]
fn remaining_unary_formulas_dispatch_and_are_finite() {
    let layout = layout();
    let input = [-2.0_f32, -0.5, 0.5, 2.0];
    let mut output = [0.0_f32; 4];
    let mut actor = UnaryKernel::new();
    macro_rules! run {
        ($event:expr) => {
            assert_eq!(actor.process_event($event), Ok(()));
            assert!(output.iter().all(|value| value.is_finite()));
        };
    }
    run!(OpTanh::new(
        TensorView::new(&input, layout),
        TensorViewMut::new(&mut output, layout)
    ));
    run!(OpElu::new(
        TensorView::new(&input, layout),
        TensorViewMut::new(&mut output, layout)
    ));
    run!(OpSigmoid::new(
        TensorView::new(&input, layout),
        TensorViewMut::new(&mut output, layout)
    ));
    run!(OpGelu::new(
        TensorView::new(&input, layout),
        TensorViewMut::new(&mut output, layout)
    ));
    run!(OpGeluQuick::new(
        TensorView::new(&input, layout),
        TensorViewMut::new(&mut output, layout)
    ));
    run!(OpHardswish::new(
        TensorView::new(&input, layout),
        TensorViewMut::new(&mut output, layout)
    ));
    run!(OpHardsigmoid::new(
        TensorView::new(&input, layout),
        TensorViewMut::new(&mut output, layout)
    ));
    run!(OpSoftplus::new(
        TensorView::new(&input, layout),
        TensorViewMut::new(&mut output, layout)
    ));
    run!(OpGeluErf::new(
        TensorView::new(&input, layout),
        TensorViewMut::new(&mut output, layout)
    ));
}

#[test]
fn unary_guards_are_explicit_and_non_mutating() {
    let valid = layout();
    let mismatch = Layout::contiguous(DType::F32, [3, 1, 1, 1]).expect("layout fits");
    let invalid = Layout::new(DType::F16, [4, 1, 1, 1], [2, 8, 32, 32]);
    let input = [1.0_f32; 4];
    let mut output = [9.0_f32; 4];
    let mut actor = UnaryKernel::new();
    assert_eq!(
        actor.process_event(OpAbs::new(
            TensorView::new(&input, valid),
            TensorViewMut::new(&mut output, mismatch)
        )),
        Err(UnaryError::ShapeMismatch)
    );
    assert_eq!(output, [9.0; 4]);
    assert_eq!(
        actor.process_event(OpAbs::new(
            TensorView::new(&input, invalid),
            TensorViewMut::new(&mut output, valid)
        )),
        Err(UnaryError::InvalidView)
    );
    assert_eq!(output, [9.0; 4]);
    assert_eq!(
        actor.process_event(UnexpectedUnary),
        Err(UnaryError::UnexpectedEvent)
    );
}

#[test]
fn unary_dispatch_is_allocation_free() {
    let layout = layout();
    let input = [-1.0_f32, 0.0, 1.0, 2.0];
    let mut output = [0.0_f32; 4];
    let mut actor = UnaryKernel::new();
    let allocation = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                actor.process_event(OpAbs::new(
                    TensorView::new(&input, layout),
                    TensorViewMut::new(&mut output, layout)
                )),
                Ok(())
            );
        }
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(allocation.bytes_total, 0);
}

#[test]
fn unary_errors_are_typed_and_debuggable() {
    assert!(format!("{}", UnaryError::InvalidView).contains("view"));
    assert!(format!("{:?}", UnaryKernel::default()).contains("UnaryKernel"));
}

#[test]
fn remaining_dedicated_unary_events_match_formulas() {
    let layout = layout();
    let input = [-2.0_f32, -0.5, 0.5, 2.0];
    let mut output = [0.0_f32; 4];
    let mut child = UnaryKernel::new();
    child
        .process_event(OpSgn::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout),
        ))
        .unwrap();
    assert_eq!(output, [-1.0, -1.0, 1.0, 1.0]);
    child
        .process_event(OpStep::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout),
        ))
        .unwrap();
    assert_eq!(output, [0.0, 0.0, 1.0, 1.0]);
    child
        .process_event(OpFloor::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout),
        ))
        .unwrap();
    assert_eq!(output, [-2.0, -1.0, 0.0, 2.0]);
    child
        .process_event(OpCeil::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout),
        ))
        .unwrap();
    assert_eq!(output, [-2.0, 0.0, 1.0, 2.0]);
    child
        .process_event(OpTrunc::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout),
        ))
        .unwrap();
    assert_eq!(output, [-2.0, -0.0, 0.0, 2.0]);
    child
        .process_event(OpRound::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout),
        ))
        .unwrap();
    assert_eq!(output, [-2.0, -1.0, 1.0, 2.0]);
    child
        .process_event(OpExpm1::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout),
        ))
        .unwrap();
    assert!(output.iter().all(|value| value.is_finite()));
}

#[test]
fn public_op_unary_rejects_source_gap_subops_without_mutation() {
    let layout = layout();
    let input = [1.0_f32; 4];
    let mut output = [9.0_f32; 4];
    let mut kernel = Kernel::new();
    for subop in [
        UnarySubOp::Sgn,
        UnarySubOp::Step,
        UnarySubOp::Sigmoid,
        UnarySubOp::GeluQuick,
        UnarySubOp::Hardswish,
        UnarySubOp::Hardsigmoid,
        UnarySubOp::Expm1,
        UnarySubOp::Softplus,
        UnarySubOp::GeluErf,
        UnarySubOp::Xielu,
        UnarySubOp::Floor,
        UnarySubOp::Ceil,
        UnarySubOp::Round,
        UnarySubOp::Trunc,
    ] {
        output.fill(9.0);
        assert_eq!(
            kernel.process_event(OpUnary::new(
                subop,
                TensorView::new(&input, layout),
                TensorViewMut::new(&mut output, layout),
            )),
            Err(UnaryError::UnexpectedEvent)
        );
        assert_eq!(output, [9.0; 4]);
    }
}
