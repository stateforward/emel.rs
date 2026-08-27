//! Public observer for the maintained row-broadcast actors.

use emel_kernels::any::broadcast::{BroadcastKernel, OpAddBroadcastRow, OpMulBroadcastRow};
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};

const COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";

fn bits(values: &[f32]) -> String {
    values
        .iter()
        .map(|v| format!("{v:08x}", v = v.to_bits()))
        .collect::<Vec<_>>()
        .join(",")
}
fn logical(data: &[f32], ne: [u64; 4], nb: [u64; 4]) -> Vec<f32> {
    let mut strides = nb;
    if strides[0] == 0 {
        strides[0] = 4;
        for d in 1..4 {
            strides[d] = strides[d - 1] * ne[d - 1];
        }
    }
    let count = ne.iter().product::<u64>() as usize;
    (0..count)
        .map(|ordinal| {
            let mut rem = ordinal as u64;
            let mut offset = 0;
            for d in 0..4 {
                offset += (rem % ne[d]) * strides[d];
                rem /= ne[d];
            }
            data[(offset / 4) as usize]
        })
        .collect()
}
fn layout(ne: [u64; 4], nb: [u64; 4]) -> Layout {
    Layout::new(DType::F32, ne, nb)
}

fn main() {
    println!("kernel-broadcast-parity/v1");
    println!("source_repository=stateforward/emel.cpp");
    println!("source_commit={COMMIT}");
    println!("source_sml_commit=49207123cd3f39767764bae774932cb48623f92f");
    println!("source_detail_guard_span=3788-3836");
    println!("source_detail_run_span=3823-3836");
    println!("source_x86_add_sm_span=46-63");
    println!("source_x86_mul_sm_span=111-128");
    println!("scope=portable_f32_row_broadcast_add_mul");
    let source = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let row = [10.0_f32, 20.0, 30.0];
    let ne = [3, 2, 1, 1];
    let row_ne = [3, 1, 1, 1];
    let nb = [4, 12, 24, 24];
    let row_nb = [4, 12, 12, 12];
    let cases = [
        ("row_add", true),
        ("row_mul", false),
        ("row_add_implicit", true),
    ];
    for (name, add) in cases {
        let input_nb = if name.ends_with("implicit") {
            [0, 0, 0, 0]
        } else {
            nb
        };
        let input_row_nb = if name.ends_with("implicit") {
            [0, 0, 0, 0]
        } else {
            row_nb
        };
        let mut output = [0.0_f32; 6];
        let mut actor = BroadcastKernel::new();
        let result = if add {
            actor.process_event(OpAddBroadcastRow::new(
                TensorView::new(&source, layout(ne, input_nb)),
                TensorView::new(&row, layout(row_ne, input_row_nb)),
                TensorViewMut::new(&mut output, layout(ne, input_nb)),
            ))
        } else {
            actor.process_event(OpMulBroadcastRow::new(
                TensorView::new(&source, layout(ne, input_nb)),
                TensorView::new(&row, layout(row_ne, input_row_nb)),
                TensorViewMut::new(&mut output, layout(ne, input_nb)),
            ))
        };
        println!(
            "case={name} status={} output_bits={}",
            if result.is_ok() { "ok" } else { "error" },
            bits(&logical(&output, ne, input_nb))
        );
    }
}
