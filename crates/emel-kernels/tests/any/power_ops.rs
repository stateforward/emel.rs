#![allow(clippy::float_cmp, missing_docs)]

use allocation_counter::measure;
use emel_kernels::any::power::*;
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
    assert_eq!(
        PINNED_EMEL_CPP_COMMIT,
        "843a117386ef17dc5a50549bbfc821074c2141d6"
    );
    let layout = layout(4);
    let input = [-2.0_f32, -0.5, 0.5, 2.0];
    let mut output = [0.0_f32; 4];
    let mut actor = PowerKernel::new();
    assert_eq!(
        actor.process_event(OpSqr::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout),
        )),
        Ok(())
    );
    assert_eq!(output, [4.0, 0.25, 0.25, 4.0]);
    let squared = output;
    assert_eq!(
        actor.process_event(OpSqrt::new(
            TensorView::new(&squared, layout),
            TensorViewMut::new(&mut output, layout),
        )),
        Ok(())
    );
    assert_eq!(output, [2.0, 0.5, 0.5, 2.0]);
}

#[test]
fn guards_reject_shape_and_view_without_mutation() {
    let valid = layout(4);
    let mismatch = layout(3);
    let invalid = Layout::new(DType::F16, [4, 1, 1, 1], [2, 8, 32, 32]);
    let input = [1.0_f32; 4];
    let mut output = [9.0_f32; 4];
    let mut actor = PowerKernel::new();
    assert_eq!(
        actor.process_event(OpSqr::new(
            TensorView::new(&input, valid),
            TensorViewMut::new(&mut output, mismatch),
        )),
        Err(PowerError::ShapeMismatch)
    );
    assert_eq!(output, [9.0; 4]);
    assert_eq!(
        actor.process_event(OpSqrt::new(
            TensorView::new(&input, invalid),
            TensorViewMut::new(&mut output, valid),
        )),
        Err(PowerError::InvalidView)
    );
    assert_eq!(output, [9.0; 4]);
    assert_eq!(
        actor.process_event(OpSqr::new(
            TensorView::new(&input, invalid),
            TensorViewMut::new(&mut output, valid),
        )),
        Err(PowerError::InvalidView)
    );
    assert_eq!(output, [9.0; 4]);
}

#[test]
fn equal_element_count_with_different_dimensions_is_accepted() {
    let input_layout = Layout::contiguous(DType::F32, [2, 2, 1, 1]).expect("layout fits");
    let output_layout = Layout::contiguous(DType::F32, [4, 1, 1, 1]).expect("layout fits");
    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let mut output = [0.0_f32; 4];
    let mut actor = PowerKernel::new();
    assert_eq!(
        actor.process_event(OpSqr::new(
            TensorView::new(&input, input_layout),
            TensorViewMut::new(&mut output, output_layout),
        )),
        Ok(())
    );
    assert_eq!(output, [1.0, 4.0, 9.0, 16.0]);
}

#[test]
fn implicit_contiguous_f32_layouts_are_accepted() {
    let layout = Layout::new(DType::F32, [4, 1, 1, 1], [0, 0, 0, 0]);
    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let mut output = [0.0_f32; 4];
    let mut actor = PowerKernel::new();
    assert_eq!(
        actor.process_event(OpSqrt::new(
            TensorView::new(&input, layout),
            TensorViewMut::new(&mut output, layout),
        )),
        Ok(())
    );
    assert_eq!(output, [1.0, 2.0_f32.sqrt(), 3.0_f32.sqrt(), 2.0]);
}

#[test]
fn unexpected_event_is_explicit_and_actor_recovers() {
    let valid = layout(2);
    let input = [1.0_f32, 4.0];
    let mut output = [0.0_f32; 2];
    let mut actor = PowerKernel::new();
    assert_eq!(
        actor.process_event(UnexpectedPower),
        Err(PowerError::UnexpectedEvent)
    );
    assert_eq!(
        actor.process_event(OpSqrt::new(
            TensorView::new(&input, valid),
            TensorViewMut::new(&mut output, valid),
        )),
        Ok(())
    );
    assert_eq!(output, [1.0, 2.0]);
}

#[test]
fn readiness_is_observable_before_and_after_every_rtc_dispatch() {
    let valid = layout(2);
    let input = [1.0_f32, 4.0];
    let mut output = [0.0_f32; 2];
    let mut actor = PowerKernel::new();
    assert!(actor.is_ready());
    assert_eq!(
        actor.process_event(OpSqr::new(
            TensorView::new(&input, valid),
            TensorViewMut::new(&mut output, valid),
        )),
        Ok(())
    );
    assert!(actor.is_ready());
    assert_eq!(
        actor.process_event(UnexpectedPower),
        Err(PowerError::UnexpectedEvent)
    );
    assert!(actor.is_ready());
}

#[test]
fn dispatch_is_allocation_free() {
    let valid = layout(16);
    let input = [4.0_f32; 16];
    let mut output = [0.0_f32; 16];
    let mut actor = PowerKernel::new();
    let allocation = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                actor.process_event(OpSqrt::new(
                    TensorView::new(&input, valid),
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
        format!("{:?}", PowerKernel::default()),
        "PowerKernel { .. }"
    );
    assert_eq!(format!("{:?}", PowerError::Internal), "Internal");
    assert_eq!(
        PowerError::InvalidView.to_string(),
        "invalid power-operation tensor view"
    );
    assert_eq!(
        PowerError::ShapeMismatch.to_string(),
        "power-operation tensor shapes differ"
    );
    assert_eq!(
        PowerError::UnexpectedEvent.to_string(),
        "unexpected power-operation event"
    );
    assert_eq!(
        PowerError::Internal.to_string(),
        "internal power-operation dispatch error"
    );
}
