//! Public observer for the maintained F32 `get_rows` root route.

#![allow(clippy::float_cmp)]

use emel_kernels::Kernel;
use emel_kernels::any::event::{IndexView, OpGetRows};
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};
use std::time::Instant;

const SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";

fn layout(ne: [u64; 4]) -> Layout {
    Layout::contiguous(DType::F32, ne).expect("fixture layout fits")
}

fn bits(values: &[f32]) -> String {
    values
        .iter()
        .map(|value| format!("{value:08x}", value = value.to_bits()))
        .collect::<Vec<_>>()
        .join(",")
}

fn main() {
    if std::env::args().nth(1).as_deref() == Some("--benchmark") {
        benchmark();
        return;
    }

    let source = [1.0_f32, 2.0, 3.0, 4.0];
    let indices = [1_i32, 0];
    let mut destination = [0.0_f32; 4];
    let source_layout = layout([2, 2, 1, 1]);
    let indices_view = IndexView::new(&indices, [2, 1, 1]);
    let destination_layout = layout([2, 2, 1, 1]);
    let mut actor = Kernel::new();
    let result = actor.process_event(OpGetRows::new(
        TensorView::new(&source, source_layout),
        indices_view,
        TensorViewMut::new(&mut destination, destination_layout),
    ));
    assert_eq!(result, Ok(()));

    println!("kernel-get-rows-parity/v1");
    println!("source_repository=stateforward/emel.cpp");
    println!("source_commit={SOURCE_COMMIT}");
    println!("source_kernel_events_blob=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9");
    println!("source_kernel_detail_blob=c8a82643eabfe8f2d7883e655955f455794511b0");
    println!("source_kernel_x86_sm_blob=0b4d635ebbd0fbd52dbca8a2345547fb571205c8");
    println!("scope=f32_public_root_get_rows_dense_rank4");
    println!("case=rows status=ok output_bits={}", bits(&destination));
}

fn benchmark() {
    const ITERATIONS: u32 = 10_000;
    const WARMUP: usize = 1_000;
    let source = [1.0_f32, 2.0, 3.0, 4.0];
    let indices = [1_i32, 0];
    let source_layout = layout([2, 2, 1, 1]);
    let indices_view = IndexView::new(&indices, [2, 1, 1]);
    let destination_layout = layout([2, 2, 1, 1]);
    let mut destination = [0.0_f32; 4];
    let mut actor = Kernel::new();

    for _ in 0..WARMUP {
        let result = actor.process_event(OpGetRows::new(
            TensorView::new(&source, source_layout),
            indices_view,
            TensorViewMut::new(&mut destination, destination_layout),
        ));
        assert_eq!(result, Ok(()));
    }
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let result = actor.process_event(OpGetRows::new(
            TensorView::new(&source, source_layout),
            indices_view,
            TensorViewMut::new(&mut destination, destination_layout),
        ));
        assert_eq!(result, Ok(()));
    }
    let ns_per_dispatch = start.elapsed().as_secs_f64() * 1e9 / f64::from(ITERATIONS);
    assert_eq!(destination, [3.0, 4.0, 1.0, 2.0]);
    println!("case=op_get_rows_f32 rust_ns_per_dispatch={ns_per_dispatch:.3} output=4");
}
