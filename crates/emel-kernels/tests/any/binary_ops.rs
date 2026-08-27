#![allow(clippy::float_cmp, missing_docs)]

use allocation_counter::measure;
use emel_kernels::any::binary::*;
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};
use emel_tensor as _;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use pulp as _;
use sml as _;

const PINNED_EMEL_CPP_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";

fn layout(count: u64) -> Layout {
    Layout::contiguous(DType::F32, [count, 1, 1, 1]).expect("layout fits")
}

#[test]
fn source_identity_and_formulas_match_pinned_scalar_contract() {
    assert_eq!(PINNED_EMEL_CPP_COMMIT.len(), 40);
    let layout = layout(4);
    let lhs = [1.0_f32, -2.0, 3.0, -4.0];
    let rhs = [0.5_f32, 2.0, -3.0, -2.0];
    let mut output = [0.0_f32; 4];
    let mut actor = BinaryKernel::new();
    assert_eq!(
        actor.process_event(OpAdd::new(
            TensorView::new(&lhs, layout),
            TensorView::new(&rhs, layout),
            TensorViewMut::new(&mut output, layout),
        )),
        Ok(())
    );
    assert_eq!(output, [1.5, 0.0, 0.0, -6.0]);
    assert_eq!(
        actor.process_event(OpSub::new(
            TensorView::new(&lhs, layout),
            TensorView::new(&rhs, layout),
            TensorViewMut::new(&mut output, layout),
        )),
        Ok(())
    );
    assert_eq!(output, [0.5, -4.0, 6.0, -2.0]);
    assert_eq!(
        actor.process_event(OpMul::new(
            TensorView::new(&lhs, layout),
            TensorView::new(&rhs, layout),
            TensorViewMut::new(&mut output, layout),
        )),
        Ok(())
    );
    assert_eq!(output, [0.5, -4.0, -9.0, 8.0]);
    assert_eq!(
        actor.process_event(OpDiv::new(
            TensorView::new(&lhs, layout),
            TensorView::new(&rhs, layout),
            TensorViewMut::new(&mut output, layout),
        )),
        Ok(())
    );
    assert_eq!(output, [2.0, -1.0, -1.0, 2.0]);
}

#[test]
fn strided_views_follow_logical_ordinal_order() {
    let layout = Layout::new(DType::F32, [2, 2, 1, 1], [4, 8, 16, 16]);
    let lhs = [1.0_f32, 2.0, 3.0, 4.0];
    let rhs = [10.0_f32, 20.0, 30.0, 40.0];
    let mut output = [0.0_f32; 4];
    let mut actor = BinaryKernel::new();
    assert_eq!(
        actor.process_event(OpAdd::new(
            TensorView::new(&lhs, layout),
            TensorView::new(&rhs, layout),
            TensorViewMut::new(&mut output, layout),
        )),
        Ok(())
    );
    assert_eq!(output, [11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn same_count_different_shapes_follow_pinned_binary_eligibility() {
    // detail.hpp::can_run_binary compares element counts, not the four
    // individual extents.  The scalar run_binary path then consumes each
    // operand by logical ordinal.
    let lhs_layout = Layout::contiguous(DType::F32, [2, 2, 1, 1]).expect("layout fits");
    let rhs_layout = Layout::contiguous(DType::F32, [4, 1, 1, 1]).expect("layout fits");
    let output_layout = rhs_layout;
    let lhs = [1.0_f32, 2.0, 3.0, 4.0];
    let rhs = [10.0_f32, 20.0, 30.0, 40.0];
    let mut output = [0.0_f32; 4];
    let mut actor = BinaryKernel::new();

    assert_eq!(
        actor.process_event(OpAdd::new(
            TensorView::new(&lhs, lhs_layout),
            TensorView::new(&rhs, rhs_layout),
            TensorViewMut::new(&mut output, output_layout),
        )),
        Ok(())
    );
    assert_eq!(output, [11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn implicit_contiguous_f32_stride_is_accepted() {
    // The pinned tensor_stride_bytes() derives all strides when nb[0] == 0;
    // the remaining nb fields are ignored in that representation.
    let implicit = Layout::new(DType::F32, [2, 2, 1, 1], [0, 999, 0, 0]);
    let lhs = [1.0_f32, 2.0, 3.0, 4.0];
    let rhs = [10.0_f32, 20.0, 30.0, 40.0];
    let mut output = [0.0_f32; 4];
    let mut actor = BinaryKernel::new();

    assert_eq!(
        actor.process_event(OpMul::new(
            TensorView::new(&lhs, implicit),
            TensorView::new(&rhs, implicit),
            TensorViewMut::new(&mut output, implicit),
        )),
        Ok(())
    );
    assert_eq!(output, [10.0, 40.0, 90.0, 160.0]);
}

#[test]
fn mixed_explicit_and_implicit_layouts_use_their_guarded_route() {
    let explicit = Layout::new(DType::F32, [4, 1, 1, 1], [4, 16, 64, 64]);
    let implicit = Layout::new(DType::F32, [4, 1, 1, 1], [0, 999, 0, 0]);
    let lhs = [10.0_f32, 20.0, 30.0, 40.0];
    let rhs = [1.0_f32, 2.0, 3.0, 4.0];
    let mut output = [0.0_f32; 4];
    let mut actor = BinaryKernel::new();

    assert_eq!(
        actor.process_event(OpSub::new(
            TensorView::new(&lhs, explicit),
            TensorView::new(&rhs, implicit),
            TensorViewMut::new(&mut output, explicit),
        )),
        Ok(())
    );
    assert_eq!(output, [9.0, 18.0, 27.0, 36.0]);
}

#[test]
fn singleton_dimensions_may_use_explicit_zero_strides() {
    // detail.hpp::has_valid_tensor_layout permits nb[d] == 0 when ne[d] == 1.
    let layout = Layout::new(DType::F32, [4, 1, 1, 1], [4, 16, 0, 0]);
    let lhs = [1.0_f32, 2.0, 3.0, 4.0];
    let rhs = [10.0_f32, 20.0, 30.0, 40.0];
    let mut output = [0.0_f32; 4];
    let mut actor = BinaryKernel::new();

    assert_eq!(
        actor.process_event(OpAdd::new(
            TensorView::new(&lhs, layout),
            TensorView::new(&rhs, layout),
            TensorViewMut::new(&mut output, layout),
        )),
        Ok(())
    );
    assert_eq!(output, [11.0, 22.0, 33.0, 44.0]);
}

#[test]
fn guards_reject_shape_and_view_without_mutation() {
    let valid = layout(4);
    let mismatch = layout(3);
    let invalid = Layout::new(DType::F16, [4, 1, 1, 1], [2, 8, 32, 32]);
    let lhs = [1.0_f32; 4];
    let rhs = [2.0_f32; 4];
    let mut output = [9.0_f32; 4];
    let mut actor = BinaryKernel::new();
    assert_eq!(
        actor.process_event(OpMul::new(
            TensorView::new(&lhs, valid),
            TensorView::new(&rhs, mismatch),
            TensorViewMut::new(&mut output, valid),
        )),
        Err(BinaryError::ShapeMismatch)
    );
    assert_eq!(output, [9.0; 4]);
    assert_eq!(
        actor.process_event(OpDiv::new(
            TensorView::new(&lhs, invalid),
            TensorView::new(&rhs, valid),
            TensorViewMut::new(&mut output, valid),
        )),
        Err(BinaryError::InvalidView)
    );
    assert_eq!(output, [9.0; 4]);
}

#[test]
fn unexpected_event_is_explicit_and_actor_recovers() {
    let valid = layout(2);
    let lhs = [4.0_f32, 6.0];
    let rhs = [2.0_f32, 3.0];
    let mut output = [0.0_f32; 2];
    let mut actor = BinaryKernel::new();
    assert_eq!(
        actor.process_event(UnexpectedBinary),
        Err(BinaryError::UnexpectedEvent)
    );
    assert_eq!(
        actor.process_event(OpDiv::new(
            TensorView::new(&lhs, valid),
            TensorView::new(&rhs, valid),
            TensorViewMut::new(&mut output, valid),
        )),
        Ok(())
    );
    assert_eq!(output, [2.0, 2.0]);
}

#[test]
fn dispatch_is_allocation_free() {
    let valid = layout(16);
    let lhs = [4.0_f32; 16];
    let rhs = [2.0_f32; 16];
    let mut output = [0.0_f32; 16];
    let mut actor = BinaryKernel::new();
    let allocation = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                actor.process_event(OpMul::new(
                    TensorView::new(&lhs, valid),
                    TensorView::new(&rhs, valid),
                    TensorViewMut::new(&mut output, valid),
                )),
                Ok(())
            );
        }
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(allocation.bytes_total, 0);
}

#[test]
fn errors_are_typed_and_debuggable() {
    assert_eq!(
        format!("{:?}", BinaryKernel::default()),
        "BinaryKernel { .. }"
    );
    assert_eq!(format!("{:?}", BinaryError::Internal), "Internal");
    assert_eq!(
        BinaryError::InvalidView.to_string(),
        "invalid binary tensor view"
    );
    assert_eq!(
        BinaryError::ShapeMismatch.to_string(),
        "binary tensor element counts differ"
    );
    assert_eq!(
        BinaryError::UnexpectedEvent.to_string(),
        "unexpected binary event"
    );
    assert_eq!(
        BinaryError::Internal.to_string(),
        "internal binary dispatch error"
    );
}
