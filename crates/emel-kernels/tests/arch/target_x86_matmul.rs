#![allow(clippy::float_cmp, missing_docs)]
#![cfg(target_arch = "x86_64")]

use allocation_counter::measure;
use emel_kernels::any::event::{F16View, FlashAttnError, FlashAttnOptions};
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};
use emel_kernels::x86_64::{
    MatmulError, OpMulMat, OpMulMatArgmaxQ4_0, OpMulMatQ4_0, QuantizedView, X86Kernel,
    X86OpFlashAttnExt,
};

fn f32_layout(ne: [u64; 4]) -> Layout {
    Layout::contiguous(DType::F32, ne).expect("F32 layout fits")
}

const fn packed_layout(code: u8, k: u64, rows: u64, row_bytes: u64) -> Layout {
    Layout::new(
        DType::Other(code),
        [k, rows, 1, 1],
        [1, row_bytes, row_bytes * rows, row_bytes * rows],
    )
}

fn q4_0_block(nibble: u8) -> [u8; 18] {
    let mut block = [0_u8; 18];
    block[0] = 0;
    block[1] = 0x3c;
    block[2..].fill(nibble | (nibble << 4));
    block
}

fn x86_kernel() -> Option<X86Kernel> {
    X86Kernel::try_new()
}

#[test]
fn target_router_executes_dense_f32_matmul() {
    let Some(mut kernel) = x86_kernel() else {
        return;
    };
    let lhs = [1.0_f32, 2.0, 3.0, 4.0];
    let rhs = [5.0_f32, 6.0, 7.0, 8.0];
    let mut destination = [0.0_f32; 4];

    assert_eq!(
        kernel.process_event(
            OpMulMat::new(
                TensorView::new(&lhs, f32_layout([2, 2, 1, 1])),
                TensorView::new(&rhs, f32_layout([2, 2, 1, 1])),
                TensorViewMut::new(&mut destination, f32_layout([2, 2, 1, 1])),
            ),
            &mut [],
        ),
        Ok(())
    );
    assert_eq!(destination, [17.0, 39.0, 23.0, 53.0]);
}

#[test]
fn target_router_executes_packed_q4_matmul_and_argmax() {
    let Some(mut kernel) = x86_kernel() else {
        return;
    };
    let mut lhs = [0_u8; 36];
    lhs[..18].copy_from_slice(&q4_0_block(9));
    lhs[18..].copy_from_slice(&q4_0_block(8));
    let rhs = [1.0_f32; 32];
    let mut destination = [0.0_f32; 2];
    let lhs_view = QuantizedView::new(&lhs, packed_layout(2, 32, 2, 18));
    assert_eq!(
        kernel.process_event(
            OpMulMatQ4_0::new(
                lhs_view,
                TensorView::new(&rhs, f32_layout([1, 32, 1, 1])),
                TensorViewMut::new(&mut destination, f32_layout([1, 2, 1, 1])),
            ),
            &mut [],
        ),
        Ok(())
    );
    assert!(destination[0].is_finite());
    assert!(destination[0] > 0.0);
    assert_eq!(destination[1], 0.0);
}

#[test]
fn target_router_executes_packed_q4_argmax() {
    let Some(mut kernel) = x86_kernel() else {
        return;
    };
    let mut lhs = [0_u8; 36];
    lhs[..18].copy_from_slice(&q4_0_block(9));
    lhs[18..].copy_from_slice(&q4_0_block(8));
    let rhs = [1.0_f32; 32];
    let mut destination = [0.0_f32; 1];

    assert_eq!(
        kernel.process_event(
            OpMulMatArgmaxQ4_0::new(
                QuantizedView::new(&lhs, packed_layout(2, 32, 2, 18)),
                TensorView::new(&rhs, f32_layout([32, 1, 1, 1])),
                TensorViewMut::new(&mut destination, f32_layout([1, 1, 1, 1])),
            ),
            &mut [],
        ),
        Ok(0)
    );
}

#[test]
fn target_router_rejects_invalid_packed_matmul_without_mutation() {
    let Some(mut kernel) = x86_kernel() else {
        return;
    };
    let rhs = [1.0_f32; 32];
    let mut destination = [7.0_f32; 1];
    assert_eq!(
        kernel.process_event(
            OpMulMatQ4_0::new(
                QuantizedView::new(&[0_u8; 18], packed_layout(2, 16, 1, 18)),
                TensorView::new(&rhs, f32_layout([1, 32, 1, 1])),
                TensorViewMut::new(&mut destination, f32_layout([1, 1, 1, 1])),
            ),
            &mut [],
        ),
        Err(MatmulError::ShapeMismatch)
    );
    assert_eq!(destination, [7.0]);
}

#[test]
fn target_matmul_dispatch_is_allocation_free_after_construction() {
    let Some(mut kernel) = x86_kernel() else {
        return;
    };
    let lhs = [1.0_f32, 2.0, 3.0, 4.0];
    let rhs = [5.0_f32, 6.0, 7.0, 8.0];
    let mut destination = [0.0_f32; 4];
    let allocation = measure(|| {
        for _ in 0..64 {
            assert_eq!(
                kernel.process_event(
                    OpMulMat::new(
                        TensorView::new(&lhs, f32_layout([2, 2, 1, 1])),
                        TensorView::new(&rhs, f32_layout([2, 2, 1, 1])),
                        TensorViewMut::new(&mut destination, f32_layout([2, 2, 1, 1])),
                    ),
                    &mut [],
                ),
                Ok(())
            );
        }
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(allocation.bytes_total, 0);
}

#[test]
fn target_router_exposes_f16kv_flash_attention_route() {
    let Some(mut kernel) = x86_kernel() else {
        return;
    };
    let query = [1.0_f32, 0.0, 1.0, 0.0];
    let key = [0x3c00_u16, 0, 0, 0, 0, 0x3c00, 0, 0];
    let value = [
        0x4000_u16, 0x4400, 0x4600, 0x4800, 0x4000, 0x4400, 0x4600, 0x4800,
    ];
    let mut output = [0.0_f32; 4];
    let result = kernel.process_event(
        X86OpFlashAttnExt::new(
            TensorView::new(&query, f32_layout([2, 1, 2, 1])),
            F16View::new(&key, f16_layout([2, 2, 1, 1])),
            F16View::new(&value, f16_layout([2, 2, 1, 1])),
            TensorViewMut::new(&mut output, f32_layout([2, 1, 2, 1])),
            FlashAttnOptions {
                scale: Some(1.0),
                masked_total_tokens: Some(2),
            },
        ),
        &mut [],
    );
    assert_eq!(result, Ok(()));
    assert!(output.iter().all(|value| value.is_finite()));
}

#[test]
fn target_router_classifies_invalid_flash_attention_explicitly() {
    let Some(mut kernel) = x86_kernel() else {
        return;
    };
    let query = [1.0_f32, 0.0, 1.0, 0.0];
    let key = [0x3c00_u16, 0, 0, 0, 0, 0x3c00, 0, 0];
    let value = [
        0x4000_u16, 0x4400, 0x4600, 0x4800, 0x4000, 0x4400, 0x4600, 0x4800,
    ];
    let mut output = [9.0_f32; 4];
    let result = kernel.process_event(
        X86OpFlashAttnExt::new(
            TensorView::new(&query, f32_layout([2, 1, 2, 1])),
            F16View::new(&key, f16_layout([2, 2, 1, 1])),
            F16View::new(&value, f16_layout([2, 2, 1, 1])),
            TensorViewMut::new(&mut output, f32_layout([2, 1, 2, 1])),
            FlashAttnOptions {
                scale: Some(0.0),
                masked_total_tokens: Some(2),
            },
        ),
        &mut [],
    );
    assert_eq!(result, Err(FlashAttnError::InvalidRequest));
    assert_eq!(output, [9.0; 4]);
}

const fn f16_layout(shape: [u64; 4]) -> Layout {
    let nb0 = 2;
    let nb1 = nb0 * shape[0];
    let nb2 = nb1 * shape[1];
    let nb3 = nb2 * shape[2];
    Layout::new(DType::F16, shape, [nb0, nb1, nb2, nb3])
}
