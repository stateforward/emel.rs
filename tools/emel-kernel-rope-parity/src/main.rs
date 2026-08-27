//! Public-actor observer for the bounded F32 `RoPE` routes.

use emel_kernels::Kernel;
use emel_kernels::any::event::{
    I32View, OpRope, ROPE_MODE_NEOX, ROPE_MODE_NORM, ROPE_MODE_TIMESTEP, RopeParams,
};
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};
use std::time::Instant;

const SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";

const fn layout(ne: [u64; 4]) -> Layout {
    Layout::new(
        DType::F32,
        ne,
        [4, 4 * ne[0], 4 * ne[0] * ne[1], 4 * ne[0] * ne[1] * ne[2]],
    )
}

const fn position_layout(ne: [u64; 4]) -> Layout {
    Layout::new(
        DType::I32,
        ne,
        [4, 4 * ne[0], 4 * ne[0] * ne[1], 4 * ne[0] * ne[1] * ne[2]],
    )
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

fn bits(values: &[f32]) -> String {
    values
        .iter()
        .map(|value| format!("{:08x}", value.to_bits()))
        .collect::<Vec<_>>()
        .join(",")
}

fn main() {
    if std::env::args().nth(1).as_deref() == Some("--benchmark") {
        benchmark();
        return;
    }

    println!("kernel-rope-parity/v2");
    println!("source_repository=stateforward/emel.cpp");
    println!("source_commit={SOURCE_COMMIT}");
    println!("source_kernel_events_blob=4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9");
    println!("source_kernel_detail_blob=c8a82643eabfe8f2d7883e655955f455794511b0");
    println!("source_kernel_x86_sm_blob=0b4d635ebbd0fbd52dbca8a2345547fb571205c8");
    println!("source_kernel_x86_guards_blob=cb3dac8253f8417c9b44acff1de414f6d0a3a3cf");
    println!("source_kernel_x86_actions_blob=d45558f5eb96950f43c16a09d768cb4f382d6d61");
    println!("source_detail_guard_span=4650-4702");
    println!("source_detail_rotation_span=4704-4761");
    println!("source_detail_timestep_span=4763-4830");
    println!("scope=op_rope_f32_norm_neox_timestep_positive_contiguous");

    let source = [
        1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let positions = [1_i32, 2_i32];
    let mut output = [0.0_f32; 12];
    let mut kernel = Kernel::new();
    let result = kernel.process_event(OpRope::new(
        TensorView::new(&source, layout([6, 1, 2, 1])),
        I32View::new(&positions, position_layout([2, 1, 1, 1])),
        TensorViewMut::new(&mut output, layout([6, 1, 2, 1])),
        params(ROPE_MODE_NORM),
    ));
    println!(
        "case=norm status={} output_bits={}",
        if result.is_ok() { "ok" } else { "rejected" },
        bits(&output)
    );

    let source = [1.0_f32, 2.0, 3.0, 4.0];
    let positions = [1_i32];
    let mut output = [0.0_f32; 4];
    let result = kernel.process_event(OpRope::new(
        TensorView::new(&source, layout([4, 1, 1, 1])),
        I32View::new(&positions, position_layout([1, 1, 1, 1])),
        TensorViewMut::new(&mut output, layout([4, 1, 1, 1])),
        params(ROPE_MODE_NEOX),
    ));
    println!(
        "case=neox status={} output_bits={}",
        if result.is_ok() { "ok" } else { "rejected" },
        bits(&output)
    );

    let mut output = [0.0_f32; 4];
    let result = kernel.process_event(OpRope::new(
        TensorView::new(&source, layout([4, 1, 1, 1])),
        I32View::new(&positions, position_layout([1, 1, 1, 1])),
        TensorViewMut::new(&mut output, layout([4, 1, 1, 1])),
        params(ROPE_MODE_TIMESTEP),
    ));
    println!(
        "case=timestep status={} output_bits={}",
        if result.is_ok() { "ok" } else { "rejected" },
        bits(&output)
    );

    let mut output = [9.0_f32; 4];
    let result = kernel.process_event(OpRope::new(
        TensorView::new(&source, layout([4, 1, 1, 1])),
        I32View::new(&positions, position_layout([1, 1, 1, 1])),
        TensorViewMut::new(&mut output, layout([4, 1, 1, 1])),
        params(99),
    ));
    println!(
        "case=invalid_mode status={} output_bits={}",
        if result.is_ok() { "ok" } else { "rejected" },
        bits(&output)
    );
}

fn benchmark() {
    const ITERATIONS: u32 = 10_000;
    const WARMUP: usize = 1_000;
    let source = [1.0_f32, 2.0, 3.0, 4.0];
    let positions = [1_i32];
    let source_layout = layout([4, 1, 1, 1]);
    let positions_layout = position_layout([1, 1, 1, 1]);
    let destination_layout = layout([4, 1, 1, 1]);
    let mut destination = [0.0_f32; 4];
    let mut actor = Kernel::new();

    for _ in 0..WARMUP {
        let result = actor.process_event(OpRope::new(
            TensorView::new(&source, source_layout),
            I32View::new(&positions, positions_layout),
            TensorViewMut::new(&mut destination, destination_layout),
            params(ROPE_MODE_NORM),
        ));
        assert_eq!(result, Ok(()));
    }
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let result = actor.process_event(OpRope::new(
            TensorView::new(&source, source_layout),
            I32View::new(&positions, positions_layout),
            TensorViewMut::new(&mut destination, destination_layout),
            params(ROPE_MODE_NORM),
        ));
        assert_eq!(result, Ok(()));
    }
    let ns_per_dispatch = start.elapsed().as_secs_f64() * 1e9 / f64::from(ITERATIONS);
    assert!(destination.iter().any(|value| value.to_bits() != 0));
    println!("case=op_rope_norm rust_ns_per_dispatch={ns_per_dispatch:.3} output=4");
}
