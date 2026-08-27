//! Split-lane observer for the public root `FlashAttention` route.

use emel_kernels::Kernel;
use emel_kernels::any::event::{F16View, FlashAttnOptions, OpFlashAttnExt};
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};
use std::time::Instant;

const SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const SML_COMMIT: &str = "49207123cd3f39767764bae774932cb48623f92f";
const EVENTS_BLOB: &str = "4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9";
const DETAIL_BLOB: &str = "c8a82643eabfe8f2d7883e655955f455794511b0";
const X86_SM_BLOB: &str = "0b4d635ebbd0fbd52dbca8a2345547fb571205c8";
const X86_GUARDS_BLOB: &str = "cb3dac8253f8417c9b44acff1de414f6d0a3a3cf";
const X86_ACTIONS_BLOB: &str = "d45558f5eb96950f43c16a09d768cb4f382d6d61";

fn f32_layout(shape: [u64; 4]) -> Layout {
    Layout::contiguous(DType::F32, shape).expect("F32 parity layout fits")
}

const fn f16_layout(shape: [u64; 4]) -> Layout {
    let nb0 = 2_u64;
    let nb1 = nb0 * shape[0];
    let nb2 = nb1 * shape[1];
    let nb3 = nb2 * shape[2];
    Layout::new(DType::F16, shape, [nb0, nb1, nb2, nb3])
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

    println!("kernel-flash-attn-parity/v1");
    println!("source_repository=stateforward/emel.cpp");
    println!("source_commit={SOURCE_COMMIT}");
    println!("source_sml_commit={SML_COMMIT}");
    println!("source_kernel_events_blob={EVENTS_BLOB}");
    println!("source_kernel_detail_blob={DETAIL_BLOB}");
    println!("source_kernel_x86_sm_blob={X86_SM_BLOB}");
    println!("source_kernel_x86_guards_blob={X86_GUARDS_BLOB}");
    println!("source_kernel_x86_actions_blob={X86_ACTIONS_BLOB}");
    println!("source_detail_spans=170-185,2154-2214,4225-4248,4254-4327,4339-4389,5239-5311");
    println!("scope=op_flash_attn_ext_f32_q_f32_dst_f16_k_f16_v_query_count_1_head_replication");

    let query = [1.0_f32, 0.0, 1.0, 0.0];
    let key = [0x3c00_u16, 0, 0, 0, 0, 0x3c00, 0, 0];
    let value = [0x4000_u16, 0x4400, 0x4600, 0x4800];
    let mut output = [0.0_f32; 4];
    let mut kernel = Kernel::new();
    let result = kernel.process_event(OpFlashAttnExt::new(
        TensorView::new(&query, f32_layout([2, 1, 2, 1])),
        F16View::new(&key, f16_layout([2, 2, 1, 1])),
        F16View::new(&value, f16_layout([2, 2, 1, 1])),
        TensorViewMut::new(&mut output, f32_layout([2, 1, 2, 1])),
        FlashAttnOptions {
            scale: Some(1.0),
            masked_total_tokens: Some(2),
        },
    ));
    println!(
        "case=canonical status={} output_bits={}",
        if result.is_ok() { "ok" } else { "rejected" },
        bits(&output)
    );

    let mut invalid_output = [9.0_f32; 4];
    let invalid = kernel.process_event(OpFlashAttnExt::new(
        TensorView::new(&query, f32_layout([2, 1, 2, 1])),
        F16View::new(&key, f16_layout([2, 2, 1, 1])),
        F16View::new(&value, f16_layout([2, 2, 1, 1])),
        TensorViewMut::new(&mut invalid_output, f32_layout([2, 1, 2, 1])),
        FlashAttnOptions {
            scale: Some(0.0),
            masked_total_tokens: Some(2),
        },
    ));
    println!(
        "case=invalid_scale status={} output_bits={}",
        if invalid.is_ok() { "ok" } else { "rejected" },
        bits(&invalid_output)
    );
}

fn benchmark() {
    const ITERATIONS: u32 = 10_000;
    const WARMUP: u32 = 1_000;
    let query = [1.0_f32, 0.0];
    let key = [0x3c00_u16, 0];
    let value = [0x4000_u16, 0];
    let query_layout = f32_layout([2, 1, 1, 1]);
    let kv_layout = f16_layout([2, 1, 1, 1]);
    let mut output = [0.0_f32; 2];
    let mut kernel = Kernel::new();
    let mut sink = 0.0_f32;

    for _ in 0..WARMUP {
        let result = kernel.process_event(OpFlashAttnExt::new(
            TensorView::new(&query, query_layout),
            F16View::new(&key, kv_layout),
            F16View::new(&value, kv_layout),
            TensorViewMut::new(&mut output, query_layout),
            FlashAttnOptions {
                scale: Some(1.0),
                masked_total_tokens: Some(1),
            },
        ));
        assert_eq!(result, Ok(()));
        sink = output[0];
    }
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let result = kernel.process_event(OpFlashAttnExt::new(
            TensorView::new(&query, query_layout),
            F16View::new(&key, kv_layout),
            F16View::new(&value, kv_layout),
            TensorViewMut::new(&mut output, query_layout),
            FlashAttnOptions {
                scale: Some(1.0),
                masked_total_tokens: Some(1),
            },
        ));
        assert_eq!(result, Ok(()));
        sink = output[0];
    }
    let ns_per_dispatch = start.elapsed().as_secs_f64() * 1e9 / f64::from(ITERATIONS);
    assert!(sink.is_finite());
    println!("case=op_flash_attn_ext_canonical rust_ns_per_dispatch={ns_per_dispatch:.3} output=2");
}
