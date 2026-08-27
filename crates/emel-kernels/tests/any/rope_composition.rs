#![allow(clippy::float_cmp, missing_docs)]

use allocation_counter::measure;
use emel_kernels::Kernel;
use emel_kernels::any::event::{
    Event, I32View, OpRope, ROPE_MODE_NORM, RopeError, RopeParams, UnexpectedRope,
};
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};
use emel_tensor as _;
use pulp as _;
use sml as _;

fn layout(ne: [u64; 4]) -> Layout {
    Layout::contiguous(DType::F32, ne).expect("F32 layout fits")
}

fn position_layout(ne: [u64; 4]) -> Layout {
    Layout::contiguous(DType::I32, ne).expect("I32 layout fits")
}

const fn params(mode: i32) -> RopeParams {
    RopeParams {
        n_dims: 4,
        mode,
        freq_base: 10_000.0,
        freq_scale: 1.0,
        ext_factor: 0.0,
        attn_factor: 1.0,
    }
}

fn request<'a>(
    source: &'a [f32],
    positions: &'a [i32],
    destination: &'a mut [f32],
    mode: i32,
) -> OpRope<'a> {
    OpRope::new(
        TensorView::new(source, layout([4, 1, 1, 1])),
        I32View::new(positions, position_layout([1, 1, 1, 1])),
        TensorViewMut::new(destination, layout([4, 1, 1, 1])),
        params(mode),
    )
}

#[test]
fn root_forwards_rope_to_the_public_child_actor() {
    let source = [1.0_f32, 2.0, 3.0, 4.0];
    let positions = [1_i32];
    let mut destination = [0.0_f32; 4];
    let mut kernel = Kernel::new();

    assert_eq!(
        kernel.process_event(request(
            &source,
            &positions,
            &mut destination,
            ROPE_MODE_NORM,
        )),
        Ok(())
    );
    assert_ne!(destination, [0.0; 4]);
}

#[test]
fn root_rejects_invalid_rope_mode_without_mutating_output() {
    let source = [1.0_f32, 2.0, 3.0, 4.0];
    let positions = [1_i32];
    let mut destination = [9.0_f32; 4];
    let mut kernel = Kernel::new();

    assert_eq!(
        kernel.process_event(request(&source, &positions, &mut destination, 99)),
        Err(RopeError::InvalidMode)
    );
    assert_eq!(destination, [9.0; 4]);
}

#[test]
fn root_unexpected_rope_is_typed_and_recovers() {
    let source = [1.0_f32, 2.0, 3.0, 4.0];
    let positions = [1_i32];
    let mut destination = [0.0_f32; 4];
    let mut kernel = Kernel::new();

    assert_eq!(
        kernel.process_event(UnexpectedRope),
        Err(RopeError::UnexpectedEvent)
    );
    assert_eq!(
        kernel.process_event(request(
            &source,
            &positions,
            &mut destination,
            ROPE_MODE_NORM,
        )),
        Ok(())
    );
}

#[test]
fn root_rope_dispatch_is_allocation_free_after_construction() {
    let source = [1.0_f32, 2.0, 3.0, 4.0];
    let positions = [1_i32];
    let mut destination = [0.0_f32; 4];
    let mut kernel = Kernel::new();

    let allocation = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                kernel.process_event(request(
                    &source,
                    &positions,
                    &mut destination,
                    ROPE_MODE_NORM,
                )),
                Ok(())
            );
        }
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(allocation.bytes_total, 0);
}

#[test]
fn root_rope_event_is_publicly_dispatchable() {
    fn dispatches<E: Event>(event: E, kernel: &mut Kernel) -> E::Output {
        event.dispatch(kernel)
    }

    let source = [1.0_f32, 2.0, 3.0, 4.0];
    let positions = [1_i32];
    let mut destination = [0.0_f32; 4];
    let mut kernel = Kernel::new();
    assert_eq!(
        dispatches(
            request(&source, &positions, &mut destination, ROPE_MODE_NORM),
            &mut kernel,
        ),
        Ok(())
    );
}
