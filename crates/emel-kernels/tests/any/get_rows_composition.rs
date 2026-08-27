#![allow(clippy::float_cmp, missing_docs)]

use allocation_counter::measure;
use emel_kernels::Kernel;
use emel_kernels::any::event::{Event, GetRowsError, IndexView, OpGetRows, UnexpectedGetRows};
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};
use emel_tensor as _;
use pulp as _;
use sml as _;

fn layout(ne: [u64; 4]) -> Layout {
    Layout::contiguous(DType::F32, ne).expect("F32 layout fits")
}

fn request<'a>(source: &'a [f32], indices: &'a [i32], destination: &'a mut [f32]) -> OpGetRows<'a> {
    OpGetRows::new(
        TensorView::new(source, layout([2, 2, 1, 1])),
        IndexView::new(indices, [2, 1, 1]),
        TensorViewMut::new(destination, layout([2, 2, 1, 1])),
    )
}

#[test]
fn root_forwards_f32_get_rows_to_the_public_child_actor() {
    let source = [1.0_f32, 2.0, 3.0, 4.0];
    let indices = [1_i32, 0];
    let mut destination = [0.0_f32; 4];
    let mut kernel = Kernel::new();

    assert_eq!(
        kernel.process_event(request(&source, &indices, &mut destination)),
        Ok(())
    );
    assert_eq!(destination, [3.0, 4.0, 1.0, 2.0]);
}

#[test]
fn root_get_rows_rejects_invalid_views_without_mutating_output() {
    let source = [1.0_f32, 2.0, 3.0, 4.0];
    let indices = [2_i32, 0];
    let mut destination = [9.0_f32; 4];
    let mut kernel = Kernel::new();

    assert_eq!(
        kernel.process_event(request(&source, &indices, &mut destination)),
        Err(GetRowsError::IndexOutOfBounds)
    );
    assert_eq!(destination, [9.0; 4]);
}

#[test]
fn root_unexpected_get_rows_is_typed_and_recovers() {
    let source = [1.0_f32, 2.0, 3.0, 4.0];
    let indices = [0_i32, 1];
    let mut destination = [0.0_f32; 4];
    let mut kernel = Kernel::new();

    assert_eq!(
        kernel.process_event(UnexpectedGetRows),
        Err(GetRowsError::UnexpectedEvent)
    );
    assert_eq!(
        kernel.process_event(request(&source, &indices, &mut destination)),
        Ok(())
    );
    assert_eq!(destination, source);
}

#[test]
fn root_get_rows_dispatch_is_allocation_free_after_construction() {
    let source = [1.0_f32, 2.0, 3.0, 4.0];
    let indices = [1_i32, 0];
    let mut destination = [0.0_f32; 4];
    let mut kernel = Kernel::new();

    let allocation = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                kernel.process_event(request(&source, &indices, &mut destination)),
                Ok(())
            );
        }
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(allocation.bytes_total, 0);
}

#[test]
fn all_root_get_rows_event_implementations_are_publicly_dispatchable() {
    fn dispatches<E: Event>(event: E, kernel: &mut Kernel) -> E::Output {
        event.dispatch(kernel)
    }

    let source = [1.0_f32, 2.0, 3.0, 4.0];
    let indices = [0_i32, 1];
    let mut destination = [0.0_f32; 4];
    let mut kernel = Kernel::new();

    assert_eq!(
        dispatches(request(&source, &indices, &mut destination), &mut kernel),
        Ok(())
    );
}
