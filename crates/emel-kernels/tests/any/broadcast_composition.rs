#![allow(clippy::float_cmp, missing_docs)]

use allocation_counter::measure;
use emel_kernels::Kernel;
use emel_kernels::any::event::{OpAddBroadcastRow, OpMulBroadcastRow, UnexpectedBroadcast};
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};

fn layout(ne: [u64; 4]) -> Layout {
    Layout::contiguous(DType::F32, ne).expect("test layout fits")
}

#[test]
fn root_composes_add_and_mul_broadcast_actors() {
    let source_layout = layout([3, 2, 1, 1]);
    let row_layout = layout([3, 1, 1, 1]);
    let source = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let row = [10.0_f32, 20.0, 30.0];
    let mut output = [0.0_f32; 6];
    let mut kernel = Kernel::new();

    assert_eq!(
        kernel.process_event(OpAddBroadcastRow::new(
            TensorView::new(&source, source_layout),
            TensorView::new(&row, row_layout),
            TensorViewMut::new(&mut output, source_layout),
        )),
        Ok(())
    );
    assert_eq!(output, [11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);

    assert_eq!(
        kernel.process_event(OpMulBroadcastRow::new(
            TensorView::new(&source, source_layout),
            TensorView::new(&row, row_layout),
            TensorViewMut::new(&mut output, source_layout),
        )),
        Ok(())
    );
    assert_eq!(output, [10.0, 40.0, 90.0, 40.0, 100.0, 180.0]);
    assert!(kernel.is_ready());
}

#[test]
fn root_broadcast_rejections_do_not_mutate_output() {
    let source_layout = layout([3, 2, 1, 1]);
    let bad_row_layout = layout([2, 1, 1, 1]);
    let source = [1.0_f32; 6];
    let bad_row = [2.0_f32; 2];
    let mut output = [9.0_f32; 6];
    let mut kernel = Kernel::new();

    assert_eq!(
        kernel.process_event(OpAddBroadcastRow::new(
            TensorView::new(&source, source_layout),
            TensorView::new(&bad_row, bad_row_layout),
            TensorViewMut::new(&mut output, source_layout),
        )),
        Err(emel_kernels::any::broadcast::BroadcastError::ShapeMismatch)
    );
    assert_eq!(output, [9.0; 6]);
    assert!(kernel.is_ready());
}

#[test]
fn root_broadcast_unexpected_event_recovers() {
    let mut kernel = Kernel::new();
    assert_eq!(
        kernel.process_event(UnexpectedBroadcast),
        Err(emel_kernels::any::broadcast::BroadcastError::UnexpectedEvent)
    );
    assert!(kernel.is_ready());
}

#[test]
fn root_broadcast_dispatch_is_allocation_free() {
    let source_layout = layout([3, 2, 1, 1]);
    let row_layout = layout([3, 1, 1, 1]);
    let source = [1.0_f32; 6];
    let row = [2.0_f32; 3];
    let mut output = [0.0_f32; 6];
    let mut kernel = Kernel::new();

    let allocation = measure(|| {
        assert_eq!(
            kernel.process_event(OpAddBroadcastRow::new(
                TensorView::new(&source, source_layout),
                TensorView::new(&row, row_layout),
                TensorViewMut::new(&mut output, source_layout),
            )),
            Ok(())
        );
    });
    assert_eq!(allocation.count_total, 0);
}
