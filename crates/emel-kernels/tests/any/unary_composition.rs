#![allow(clippy::float_cmp, missing_docs)]

use allocation_counter::measure;
use emel_kernels::Kernel;
use emel_kernels::any::event::{Event, OpUnary, UnaryError, UnarySubOp, UnexpectedUnary};
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};
use emel_tensor as _;
use pulp as _;
use sml as _;

fn layout() -> Layout {
    Layout::contiguous(DType::F32, [4, 1, 1, 1]).expect("F32 layout fits")
}

fn request<'a>(subop: UnarySubOp, input: &'a [f32], output: &'a mut [f32]) -> OpUnary<'a> {
    let layout = layout();
    OpUnary::new(
        subop,
        TensorView::new(input, layout),
        TensorViewMut::new(output, layout),
    )
}

#[test]
fn root_forwards_generic_unary_to_the_public_child_actor() {
    let input = [-2.0_f32, -0.5, 0.5, 2.0];
    let mut output = [0.0_f32; 4];
    let mut kernel = Kernel::new();

    assert_eq!(
        kernel.process_event(request(UnarySubOp::Abs, &input, &mut output)),
        Ok(())
    );
    assert_eq!(output, [2.0, 0.5, 0.5, 2.0]);
}

#[test]
fn root_rejects_unimplemented_generic_suboperation_without_mutation() {
    let input = [-2.0_f32, -0.5, 0.5, 2.0];
    let mut output = [9.0_f32; 4];
    let mut kernel = Kernel::new();

    assert_eq!(
        kernel.process_event(request(UnarySubOp::Sgn, &input, &mut output)),
        Err(UnaryError::UnexpectedEvent)
    );
    assert_eq!(output, [9.0; 4]);
}

#[test]
fn root_unexpected_unary_is_typed_and_recovers() {
    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let mut output = [0.0_f32; 4];
    let mut kernel = Kernel::new();

    assert_eq!(
        kernel.process_event(UnexpectedUnary),
        Err(UnaryError::UnexpectedEvent)
    );
    assert_eq!(
        kernel.process_event(request(UnarySubOp::Neg, &input, &mut output)),
        Ok(())
    );
    assert_eq!(output, [-1.0, -2.0, -3.0, -4.0]);
}

#[test]
fn root_unary_dispatch_is_allocation_free_after_construction() {
    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let mut output = [0.0_f32; 4];
    let mut kernel = Kernel::new();

    let allocation = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                kernel.process_event(request(UnarySubOp::Relu, &input, &mut output)),
                Ok(())
            );
        }
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(allocation.bytes_total, 0);
}

#[test]
fn root_unary_event_is_publicly_dispatchable() {
    fn dispatches<E: Event>(event: E, kernel: &mut Kernel) -> E::Output {
        event.dispatch(kernel)
    }

    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let mut output = [0.0_f32; 4];
    let mut kernel = Kernel::new();
    assert_eq!(
        dispatches(request(UnarySubOp::Exp, &input, &mut output), &mut kernel,),
        Ok(())
    );
}
