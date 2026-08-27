#![allow(clippy::float_cmp)]
#![allow(missing_docs)]
#![cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]

use allocation_counter::measure;
use emel_kernels::any::f16_matmul::F16View;
use emel_kernels::any::get_rows::{
    Bf16View, GetRowsKernel, OpGetRowsBf16, OpGetRowsF16, OpGetRowsQ4_0, OpGetRowsQ4K,
    OpGetRowsQ8_0,
};
use emel_kernels::any::matmul::QuantizedView;
use emel_kernels::any::quant::{Q4_0_BLOCK_BYTES, Q8_0_BLOCK_BYTES};
use emel_kernels::any::quant_more::{Q4_K_BLOCK_BYTES, QK_K_VALUES};
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};

#[cfg(target_arch = "aarch64")]
use emel_kernels::aarch64::{
    GetRowsError, IndexView, OpGetRows, TARGET_GET_ROWS_SOURCE_COMMIT,
    TARGET_GET_ROWS_SOURCE_SM_BLOB, UnexpectedGetRows,
};
#[cfg(target_arch = "x86_64")]
use emel_kernels::x86_64::{
    GetRowsError, IndexView, OpGetRows, TARGET_GET_ROWS_SOURCE_COMMIT,
    TARGET_GET_ROWS_SOURCE_SM_BLOB, UnexpectedGetRows,
};

#[cfg(target_arch = "aarch64")]
type ArchKernel = emel_kernels::aarch64::Kernel;
#[cfg(target_arch = "x86_64")]
type ArchKernel = emel_kernels::x86_64::X86Kernel;

fn q4_k_block() -> [u8; Q4_K_BLOCK_BYTES] {
    let mut block = [0_u8; Q4_K_BLOCK_BYTES];
    block[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
    block[2..4].copy_from_slice(&0x3800_u16.to_le_bytes());
    block[4..8].fill(1);
    block[8..12].fill(1);
    block[12..16].fill(0x11);
    block[16..].fill(0x21);
    block
}

fn q4_0_block() -> [u8; Q4_0_BLOCK_BYTES] {
    let mut block = [0_u8; Q4_0_BLOCK_BYTES];
    block[0..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
    block[2..].fill(0x11);
    block
}

fn q8_0_block() -> [u8; Q8_0_BLOCK_BYTES] {
    let mut block = [0_u8; Q8_0_BLOCK_BYTES];
    block[0..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
    for (index, value) in block[2..].iter_mut().enumerate() {
        *value = u8::try_from(index + 1).expect("fixture byte");
    }
    block
}

fn f32_event<'a>(source: &'a [f32], indices: &'a [i32], output: &'a mut [f32]) -> OpGetRows<'a> {
    let source_layout = Layout::contiguous(DType::F32, [2, 2, 1, 1]).expect("layout");
    let destination_layout = Layout::contiguous(DType::F32, [2, 2, 1, 1]).expect("layout");
    OpGetRows::new(
        TensorView::new(source, source_layout),
        IndexView::new(indices, [2, 1, 1]),
        TensorViewMut::new(output, destination_layout),
    )
}

#[test]
fn target_get_rows_dispatches_f32_and_recovers_after_invalid_input() {
    let Some(mut kernel) = ArchKernel::try_new() else {
        return;
    };
    let source = [1.0_f32, 2.0, 3.0, 4.0];
    let indices = [1_i32, 0];
    let mut output = [0.0_f32; 4];
    assert_eq!(
        kernel.process_get_rows(f32_event(&source, &indices, &mut output)),
        Ok(())
    );
    assert_eq!(output, [3.0, 4.0, 1.0, 2.0]);

    let before = output;
    let invalid_indices = [2_i32, 0];
    assert_eq!(
        kernel.process_get_rows(f32_event(&source, &invalid_indices, &mut output)),
        Err(GetRowsError::IndexOutOfBounds)
    );
    assert_eq!(output, before);
    assert!(kernel.is_ready());
}

#[test]
fn target_get_rows_unexpected_event_is_explicit_and_recovers() {
    let Some(mut kernel) = ArchKernel::try_new() else {
        return;
    };
    assert_eq!(
        kernel.process_get_rows(UnexpectedGetRows),
        Err(GetRowsError::UnexpectedEvent)
    );

    let source = [1.0_f32, 2.0, 3.0, 4.0];
    let indices = [0_i32, 1];
    let mut output = [0.0_f32; 4];
    assert_eq!(
        kernel.process_get_rows(f32_event(&source, &indices, &mut output)),
        Ok(())
    );
}

#[test]
fn target_get_rows_dispatch_is_allocation_free_after_construction() {
    let Some(mut kernel) = ArchKernel::try_new() else {
        return;
    };
    let source = [1.0_f32, 2.0, 3.0, 4.0];
    let indices = [1_i32, 0];
    let mut output = [0.0_f32; 4];
    let allocation = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                kernel.process_get_rows(f32_event(&source, &indices, &mut output)),
                Ok(())
            );
        }
    });
    assert_eq!(allocation.count_total, 0);
}

#[test]
fn target_get_rows_source_identity_is_pinned() {
    assert_eq!(
        TARGET_GET_ROWS_SOURCE_COMMIT,
        "843a117386ef17dc5a50549bbfc821074c2141d6"
    );
    #[cfg(target_arch = "aarch64")]
    assert_eq!(
        TARGET_GET_ROWS_SOURCE_SM_BLOB,
        "865a9cc6ba6115382ed043c464f3d62bcd851357"
    );
    #[cfg(target_arch = "x86_64")]
    assert_eq!(
        TARGET_GET_ROWS_SOURCE_SM_BLOB,
        "0b4d635ebbd0fbd52dbca8a2345547fb571205c8"
    );
}

#[test]
fn target_get_rows_routes_all_six_typed_variants() {
    let Some(mut kernel) = ArchKernel::try_new() else {
        return;
    };
    let indices = IndexView::new(&[0_i32], [1, 1, 1]);

    let mut f16_output = [0.0_f32; 2];
    assert_eq!(
        kernel.process_get_rows(OpGetRowsF16::new(
            F16View::new(&[], Layout::new(DType::F16, [2, 2, 1, 1], [2, 4, 8, 8])),
            indices,
            TensorViewMut::new(
                &mut f16_output,
                Layout::contiguous(DType::F32, [2, 1, 1, 1]).expect("layout"),
            ),
        )),
        Err(GetRowsError::InvalidView)
    );

    let mut bf16_output = [0.0_f32; 2];
    assert_eq!(
        kernel.process_get_rows(OpGetRowsBf16::new(
            Bf16View::new(&[], Layout::new(DType::Bf16, [2, 2, 1, 1], [2, 4, 8, 8])),
            indices,
            TensorViewMut::new(
                &mut bf16_output,
                Layout::contiguous(DType::F32, [2, 1, 1, 1]).expect("layout"),
            ),
        )),
        Err(GetRowsError::InvalidView)
    );

    let mut q4_output = [0.0_f32; 32];
    assert_eq!(
        kernel.process_get_rows(OpGetRowsQ4_0::new(
            QuantizedView::new(
                &[],
                Layout::new(DType::Other(2), [32, 1, 1, 1], [1, 18, 18, 18])
            ),
            indices,
            TensorViewMut::new(
                &mut q4_output,
                Layout::contiguous(DType::F32, [32, 1, 1, 1]).expect("layout"),
            ),
        )),
        Err(GetRowsError::InvalidView)
    );

    let mut q8_output = [0.0_f32; 32];
    assert_eq!(
        kernel.process_get_rows(OpGetRowsQ8_0::new(
            QuantizedView::new(
                &[],
                Layout::new(DType::Other(8), [32, 1, 1, 1], [1, 34, 34, 34])
            ),
            indices,
            TensorViewMut::new(
                &mut q8_output,
                Layout::contiguous(DType::F32, [32, 1, 1, 1]).expect("layout"),
            ),
        )),
        Err(GetRowsError::InvalidView)
    );

    let mut q4k_result = [0.0_f32; 256];
    assert_eq!(
        kernel.process_get_rows(OpGetRowsQ4K::new(
            QuantizedView::new(
                &[],
                Layout::new(DType::Other(12), [256, 1, 1, 1], [1, 144, 144, 144])
            ),
            indices,
            TensorViewMut::new(
                &mut q4k_result,
                Layout::contiguous(DType::F32, [256, 1, 1, 1]).expect("layout"),
            ),
        )),
        Err(GetRowsError::InvalidView)
    );
}

#[test]
fn target_get_rows_q4k_delegates_successfully() {
    let Some(mut kernel) = ArchKernel::try_new() else {
        return;
    };
    let source = q4_k_block();
    let source_layout = Layout::new(
        DType::Other(12),
        [QK_K_VALUES as u64, 1, 1, 1],
        [
            1,
            Q4_K_BLOCK_BYTES as u64,
            Q4_K_BLOCK_BYTES as u64,
            Q4_K_BLOCK_BYTES as u64,
        ],
    );
    let indices = IndexView::new(&[0_i32], [1, 1, 1]);
    let mut child = GetRowsKernel::new();
    let mut child_output = [0.0_f32; QK_K_VALUES];
    assert_eq!(
        child.process_event(OpGetRowsQ4K::new(
            QuantizedView::new(&source, source_layout),
            indices,
            TensorViewMut::new(
                &mut child_output,
                Layout::contiguous(DType::F32, [QK_K_VALUES as u64, 1, 1, 1]).expect("layout"),
            ),
        )),
        Ok(())
    );
    for chunk in child_output.as_chunks::<64>().0 {
        assert_eq!(&chunk[..32], &[0.5; 32]);
        assert_eq!(&chunk[32..], &[1.5; 32]);
    }
    let mut output = [0.0_f32; QK_K_VALUES];

    assert_eq!(
        kernel.process_get_rows(OpGetRowsQ4K::new(
            QuantizedView::new(&source, source_layout),
            indices,
            TensorViewMut::new(
                &mut output,
                Layout::contiguous(DType::F32, [QK_K_VALUES as u64, 1, 1, 1]).expect("layout"),
            ),
        )),
        Ok(())
    );
    for chunk in output.as_chunks::<64>().0 {
        assert_eq!(&chunk[..32], &[0.5; 32]);
        assert_eq!(&chunk[32..], &[1.5; 32]);
    }
    assert!(kernel.is_ready());
}

#[test]
fn target_get_rows_packed_dispatch_is_allocation_free_after_construction() {
    let Some(mut kernel) = ArchKernel::try_new() else {
        return;
    };
    let q4_source = q4_0_block();
    let q8_source = q8_0_block();
    let wide_fixture = q4_k_block();
    let indices = IndexView::new(&[0_i32], [1, 1, 1]);
    let q4_layout = Layout::new(
        DType::Other(2),
        [32, 1, 1, 1],
        [
            1,
            Q4_0_BLOCK_BYTES as u64,
            Q4_0_BLOCK_BYTES as u64,
            Q4_0_BLOCK_BYTES as u64,
        ],
    );
    let q8_layout = Layout::new(
        DType::Other(8),
        [32, 1, 1, 1],
        [
            1,
            Q8_0_BLOCK_BYTES as u64,
            Q8_0_BLOCK_BYTES as u64,
            Q8_0_BLOCK_BYTES as u64,
        ],
    );
    let wide_layout = Layout::new(
        DType::Other(12),
        [QK_K_VALUES as u64, 1, 1, 1],
        [
            1,
            Q4_K_BLOCK_BYTES as u64,
            Q4_K_BLOCK_BYTES as u64,
            Q4_K_BLOCK_BYTES as u64,
        ],
    );
    let f32_32 = Layout::contiguous(DType::F32, [32, 1, 1, 1]).expect("layout");
    let f32_256 = Layout::contiguous(DType::F32, [QK_K_VALUES as u64, 1, 1, 1]).expect("layout");
    let mut q4_output = [0.0_f32; 32];
    let mut q8_output = [0.0_f32; 32];
    let mut wide_output = [0.0_f32; QK_K_VALUES];
    let allocation = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                kernel.process_get_rows(OpGetRowsQ4_0::new(
                    QuantizedView::new(&q4_source, q4_layout),
                    indices,
                    TensorViewMut::new(&mut q4_output, f32_32),
                )),
                Ok(())
            );
            assert_eq!(
                kernel.process_get_rows(OpGetRowsQ8_0::new(
                    QuantizedView::new(&q8_source, q8_layout),
                    indices,
                    TensorViewMut::new(&mut q8_output, f32_32),
                )),
                Ok(())
            );
            assert_eq!(
                kernel.process_get_rows(OpGetRowsQ4K::new(
                    QuantizedView::new(&wide_fixture, wide_layout),
                    indices,
                    TensorViewMut::new(&mut wide_output, f32_256),
                )),
                Ok(())
            );
        }
    });
    assert_eq!(allocation.count_total, 0);
}
