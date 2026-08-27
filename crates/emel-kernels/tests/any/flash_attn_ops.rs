#![allow(clippy::float_cmp, missing_docs)]

use allocation_counter::measure;
use emel_kernels::any::f16_matmul::F16View;
use emel_kernels::any::flash_attn::*;
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};
use emel_tensor as _;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use pulp as _;
use sml as _;

const PINNED_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const FLASH_ATTN_MANIFEST: &str =
    include_str!("../../../../snapshots/parity/kernel-flash-attn/manifest.txt");

fn f32_layout(shape: [u64; 4]) -> Layout {
    Layout::contiguous(DType::F32, shape).expect("F32 layout fits")
}

const fn f16_layout(shape: [u64; 4]) -> Layout {
    let nb0: u64 = 2;
    let nb1 = nb0.checked_mul(shape[0]).expect("F16 stride fits");
    let nb2 = nb1.checked_mul(shape[1]).expect("F16 stride fits");
    let nb3 = nb2.checked_mul(shape[2]).expect("F16 stride fits");
    Layout::new(DType::F16, shape, [nb0, nb1, nb2, nb3])
}

#[test]
fn source_identity_and_online_attention_formula_match_bounded_contract() {
    assert!(
        FLASH_ATTN_MANIFEST
            .lines()
            .any(|line| line == format!("source_commit={PINNED_COMMIT}"))
    );
    assert!(FLASH_ATTN_MANIFEST.lines().any(|line| {
        line == "source_kernel_detail_blob=c8a82643eabfe8f2d7883e655955f455794511b0"
    }));
    assert!(FLASH_ATTN_MANIFEST.lines().any(|line| {
        line == "source_kernel_x86_sm_blob=0b4d635ebbd0fbd52dbca8a2345547fb571205c8"
    }));
    assert!(FLASH_ATTN_MANIFEST.lines().any(|line| {
        line == "source_kernel_x86_actions_blob=d45558f5eb96950f43c16a09d768cb4f382d6d61"
    }));
    assert!(FLASH_ATTN_MANIFEST.lines().any(|line| {
        line == "source_kernel_x86_guards_blob=cb3dac8253f8417c9b44acff1de414f6d0a3a3cf"
    }));
    assert!(
        FLASH_ATTN_MANIFEST
            .lines()
            .any(|line| {
                line == "source_detail_spans=detail.hpp:170-185,2154-2214,4225-4248,4254-4327,4339-4389,5239-5311;x86_64/sm.hpp:911-923"
            })
    );
    let q_layout = f32_layout([2, 1, 2, 1]);
    let kv_layout = f16_layout([2, 2, 1, 1]);
    let q = [1.0_f32, 0.0, 1.0, 0.0];
    // K rows: [1, 0], [0, 1]. V rows: [2, 4], [6, 8].
    let k = [0x3c00, 0x0000, 0x0000, 0x3c00];
    let v = [0x4000, 0x4400, 0x4600, 0x4800];
    let mut output = [0.0_f32; 4];
    let mut actor = FlashAttnKernel::new();
    assert_eq!(
        actor.process_event(OpFlashAttnExt::new(
            TensorView::new(&q, q_layout),
            F16View::new(&k, kv_layout),
            F16View::new(&v, kv_layout),
            TensorViewMut::new(&mut output, q_layout),
            FlashAttnOptions {
                scale: Some(1.0),
                masked_total_tokens: None,
            },
        )),
        Ok(())
    );
    // Online stable attention with scores [1, 0], rounded through F16
    // accumulation, is close to softmax([1,0]) * V.
    assert!((output[0] - 3.0758).abs() < 0.01);
    assert!((output[1] - 5.0758).abs() < 0.01);
    assert!((output[2] - 3.0758).abs() < 0.01);
    assert!((output[3] - 5.0758).abs() < 0.01);
    assert_eq!(actor.prepared_tokens(), 2);
    assert_eq!(actor.reuse_count(), 0);
}

#[test]
fn head_replication_and_workspace_reuse_are_preserved() {
    let q_layout = f32_layout([1, 1, 2, 1]);
    let kv_layout = f16_layout([1, 2, 1, 1]);
    let q = [1.0_f32, 1.0];
    let k = [0x3c00, 0x3c00];
    let v = [0x4000, 0x4400];
    let mut output = [0.0_f32; 2];
    let mut actor = FlashAttnKernel::new();
    for _ in 0..2 {
        assert_eq!(
            actor.process_event(OpFlashAttnExt::new(
                TensorView::new(&q, q_layout),
                F16View::new(&k, kv_layout),
                F16View::new(&v, kv_layout),
                TensorViewMut::new(&mut output, q_layout),
                FlashAttnOptions {
                    scale: Some(1.0),
                    masked_total_tokens: Some(2),
                },
            )),
            Ok(())
        );
    }
    assert_eq!(actor.prepared_tokens(), 2);
    assert_eq!(actor.reuse_count(), 1);
    assert!((output[0] - 3.0).abs() < 0.02);
    assert!((output[1] - 3.0).abs() < 0.02);
}

#[test]
fn implicit_contiguous_layouts_match_pinned_stride_resolution() {
    let q_layout = Layout::new(DType::F32, [1, 1, 1, 1], [0, 99, 0, 0]);
    let kv_layout = Layout::new(DType::F16, [1, 1, 1, 1], [0, 99, 0, 0]);
    let q = [1.0_f32];
    let k = [0x3c00_u16];
    let v = [0x4000_u16];
    let mut output = [0.0_f32];
    let mut actor = FlashAttnKernel::new();

    assert_eq!(
        actor.process_event(OpFlashAttnExt::new(
            TensorView::new(&q, q_layout),
            F16View::new(&k, kv_layout),
            F16View::new(&v, kv_layout),
            TensorViewMut::new(&mut output, q_layout),
            FlashAttnOptions {
                scale: Some(1.0),
                masked_total_tokens: Some(1),
            },
        )),
        Ok(())
    );
    assert!((output[0] - 2.0).abs() < 0.01);
}

#[test]
fn singleton_f16_zero_and_unaligned_strides_are_ignored_safely() {
    let q_layout = f32_layout([1, 1, 1, 1]);
    // The pinned layout predicate permits arbitrary strides on singleton
    // dimensions. Only the first F16 element is addressed by this request.
    let key_layout = Layout::new(DType::F16, [1, 1, 1, 1], [2, 0, 1, 3]);
    let value_layout = Layout::new(DType::F16, [1, 1, 1, 1], [2, 1, 3, 5]);
    let q = [1.0_f32];
    let key = [0x3c00_u16];
    let value = [0x4000_u16];
    let mut output = [0.0_f32];
    let mut actor = FlashAttnKernel::new();

    assert_eq!(
        actor.process_event(OpFlashAttnExt::new(
            TensorView::new(&q, q_layout),
            F16View::new(&key, key_layout),
            F16View::new(&value, value_layout),
            TensorViewMut::new(&mut output, q_layout),
            FlashAttnOptions {
                scale: Some(1.0),
                masked_total_tokens: Some(1),
            },
        )),
        Ok(())
    );
    assert!((output[0] - 2.0).abs() < 0.01);
}

#[test]
fn multi_element_unaligned_f16_stride_remains_invalid() {
    let q_layout = f32_layout([2, 1, 1, 1]);
    let invalid_kv_layout = Layout::new(DType::F16, [2, 1, 1, 1], [3, 6, 12, 24]);
    let valid_v_layout = f16_layout([2, 1, 1, 1]);
    let q = [1.0_f32, 0.0];
    let key = [0x3c00_u16, 0x0000];
    let value = [0x4000_u16, 0x4000];
    let mut output = [9.0_f32; 2];
    let mut actor = FlashAttnKernel::new();

    assert_eq!(
        actor.process_event(OpFlashAttnExt::new(
            TensorView::new(&q, q_layout),
            F16View::new(&key, invalid_kv_layout),
            F16View::new(&value, valid_v_layout),
            TensorViewMut::new(&mut output, q_layout),
            FlashAttnOptions::default(),
        )),
        Err(FlashAttnError::InvalidRequest)
    );
    assert_eq!(output, [9.0; 2]);
}

#[test]
fn active_kv_tokens_are_not_limited_by_workspace_vector_capacity() {
    let q_layout = f32_layout([1, 1, 1, 1]);
    let kv_layout = f16_layout([1, 4097, 1, 1]);
    let q = [1.0_f32];
    let k = [0x3c00_u16; 4097];
    let v = [0x4000_u16; 4097];
    let mut output = [0.0_f32];
    let mut actor = FlashAttnKernel::new();
    assert_eq!(
        actor.process_event(OpFlashAttnExt::new(
            TensorView::new(&q, q_layout),
            F16View::new(&k, kv_layout),
            F16View::new(&v, kv_layout),
            TensorViewMut::new(&mut output, q_layout),
            FlashAttnOptions {
                scale: Some(1.0),
                masked_total_tokens: Some(4097),
            },
        )),
        Ok(())
    );
    assert!(output[0].is_finite());
    assert_eq!(actor.prepared_tokens(), 4097);
}

#[test]
fn guards_reject_wrong_dtype_shape_and_params_without_mutation() {
    let q_layout = f32_layout([2, 1, 1, 1]);
    let kv_layout = f16_layout([2, 2, 1, 1]);
    let wrong_k_layout = f32_layout([2, 2, 1, 1]);
    let q = [1.0_f32, 0.0];
    let k = [0_u16; 4];
    let v = [0_u16; 4];
    let mut output = [9.0_f32; 2];
    let mut actor = FlashAttnKernel::new();
    assert_eq!(
        actor.process_event(OpFlashAttnExt::new(
            TensorView::new(&q, q_layout),
            F16View::new(&k, wrong_k_layout),
            F16View::new(&v, kv_layout),
            TensorViewMut::new(&mut output, q_layout),
            FlashAttnOptions::default(),
        )),
        Err(FlashAttnError::InvalidRequest)
    );
    assert_eq!(output, [9.0; 2]);
    assert_eq!(
        actor.process_event(OpFlashAttnExt::new(
            TensorView::new(&q, q_layout),
            F16View::new(&k, kv_layout),
            F16View::new(&v, kv_layout),
            TensorViewMut::new(&mut output, q_layout),
            FlashAttnOptions {
                scale: Some(0.0),
                masked_total_tokens: Some(1),
            },
        )),
        Err(FlashAttnError::InvalidRequest)
    );
    assert_eq!(output, [9.0; 2]);
}

#[test]
fn unexpected_event_is_explicit_and_actor_recovers() {
    let mut actor = FlashAttnKernel::new();
    assert_eq!(
        actor.process_event(UnexpectedFlashAttn),
        Err(FlashAttnError::UnexpectedEvent)
    );
    let q_layout = f32_layout([1, 1, 1, 1]);
    let kv_layout = f16_layout([1, 1, 1, 1]);
    let q = [1.0_f32];
    let k = [0x3c00];
    let v = [0x4000];
    let mut output = [0.0_f32];
    assert_eq!(
        actor.process_event(OpFlashAttnExt::new(
            TensorView::new(&q, q_layout),
            F16View::new(&k, kv_layout),
            F16View::new(&v, kv_layout),
            TensorViewMut::new(&mut output, q_layout),
            FlashAttnOptions::default(),
        )),
        Ok(())
    );
    assert!((output[0] - 2.0).abs() < 0.01);
}

#[test]
fn dispatch_is_allocation_free_after_actor_construction() {
    let q_layout = f32_layout([2, 1, 1, 1]);
    let kv_layout = f16_layout([2, 1, 1, 1]);
    let q = [1.0_f32, 0.0];
    let k = [0x3c00, 0x0000];
    let v = [0x4000, 0x4400];
    let mut output = [0.0_f32; 2];
    let mut actor = FlashAttnKernel::new();
    let allocation = measure(|| {
        for _ in 0..32 {
            assert_eq!(
                actor.process_event(OpFlashAttnExt::new(
                    TensorView::new(&q, q_layout),
                    F16View::new(&k, kv_layout),
                    F16View::new(&v, kv_layout),
                    TensorViewMut::new(&mut output, q_layout),
                    FlashAttnOptions::default(),
                )),
                Ok(())
            );
        }
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(allocation.bytes_total, 0);
}

#[test]
fn error_and_actor_debug_contract_are_typed() {
    assert_eq!(
        format!("{:?}", FlashAttnKernel::default()),
        "FlashAttnKernel { prepared_tokens: 0, reuse_count: 0, .. }"
    );
    assert_eq!(
        FlashAttnError::InvalidRequest.to_string(),
        "invalid flash-attention request"
    );
}
