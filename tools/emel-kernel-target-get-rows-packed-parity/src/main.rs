//! Split Rust observer for packed target `get_rows` routes.

#![cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]

use emel_kernels::aarch64 as target;
use emel_kernels::any::matmul::QuantizedView;
use emel_kernels::any::quant::{Q4_0_BLOCK_BYTES, Q8_0_BLOCK_BYTES};
use emel_kernels::any::quant_more::{Q4_K_BLOCK_BYTES, QK_K_VALUES};
use emel_kernels::any::tensor_view::{DType, Layout, TensorViewMut};

const SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const EVENTS_BLOB: &str = "4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9";
const DETAIL_BLOB: &str = "c8a82643eabfe8f2d7883e655955f455794511b0";
const AARCH64_SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";
const SENTINEL: f32 = f32::from_bits(0x7fc0_0001);

#[cfg(not(target_arch = "aarch64"))]
fn main() {
    eprintln!("packed observer requires an AArch64 build");
    std::process::exit(2);
}

#[cfg(target_arch = "aarch64")]
fn bits(values: &[f32]) -> String {
    values
        .iter()
        .map(|value| format!("{:08x}", value.to_bits()))
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(target_arch = "aarch64")]
fn source_layout(code: u8, columns: usize, row_bytes: usize, stride_0: u64) -> Layout {
    Layout::new(
        DType::Other(code),
        [columns as u64, 1, 1, 1],
        [
            stride_0,
            row_bytes as u64,
            row_bytes as u64,
            row_bytes as u64,
        ],
    )
}

#[cfg(target_arch = "aarch64")]
fn destination_layout(columns: usize) -> Layout {
    Layout::contiguous(DType::F32, [columns as u64, 1, 1, 1]).expect("fixture layout")
}

#[cfg(target_arch = "aarch64")]
fn q4_source() -> [u8; Q4_0_BLOCK_BYTES] {
    let mut source = [0_u8; Q4_0_BLOCK_BYTES];
    source[0] = 0x00;
    source[1] = 0x3c;
    source[2..].fill(0x11);
    source
}

#[cfg(target_arch = "aarch64")]
fn q8_source() -> [u8; Q8_0_BLOCK_BYTES] {
    let mut source = [0_u8; Q8_0_BLOCK_BYTES];
    source[0] = 0x00;
    source[1] = 0x3c;
    for (index, byte) in source[2..].iter_mut().enumerate() {
        *byte = (index + 1) as u8;
    }
    source
}

#[cfg(target_arch = "aarch64")]
fn q4_k_source() -> [u8; Q4_K_BLOCK_BYTES] {
    let mut source = [0_u8; Q4_K_BLOCK_BYTES];
    source[0] = 0x00;
    source[1] = 0x3c;
    source[2] = 0x00;
    source[3] = 0x38;
    source[4..8].fill(1);
    source[8..12].fill(1);
    source[12..16].fill(0x11);
    source[16..].fill(0x21);
    source
}

#[cfg(target_arch = "aarch64")]
fn emit_rejection(name: &str, error: target::GetRowsError, output: &[f32]) {
    assert!(
        output
            .iter()
            .all(|value| value.to_bits() == SENTINEL.to_bits())
    );
    println!(
        "case={name} status=reject error={error:?} output_bits={}",
        bits(output)
    );
}

#[cfg(target_arch = "aarch64")]
fn run_q4_0(actor: &mut target::Kernel, source: &[u8; Q4_0_BLOCK_BYTES]) {
    let indices = [0_i32];
    let layout = source_layout(2, 32, Q4_0_BLOCK_BYTES, 1);
    let mut output = [0.0_f32; 32];
    assert_eq!(
        actor.process_get_rows(target::OpGetRowsQ4_0::new(
            QuantizedView::new(source, layout),
            target::IndexView::new(&indices, [1, 1, 1]),
            TensorViewMut::new(&mut output, destination_layout(32)),
        )),
        Ok(())
    );
    println!("case=q4_0 status=ok output_bits={}", bits(&output));

    let mut invalid_index = [SENTINEL; 32];
    let negative = [-1_i32];
    assert_eq!(
        actor.process_get_rows(target::OpGetRowsQ4_0::new(
            QuantizedView::new(source, layout),
            target::IndexView::new(&negative, [1, 1, 1]),
            TensorViewMut::new(&mut invalid_index, destination_layout(32)),
        )),
        Err(target::GetRowsError::IndexOutOfBounds)
    );
    emit_rejection(
        "q4_0_invalid_index",
        target::GetRowsError::IndexOutOfBounds,
        &invalid_index,
    );

    let mut invalid_layout = [SENTINEL; 32];
    let malformed = source_layout(2, 32, Q4_0_BLOCK_BYTES, 2);
    assert_eq!(
        actor.process_get_rows(target::OpGetRowsQ4_0::new(
            QuantizedView::new(source, malformed),
            target::IndexView::new(&indices, [1, 1, 1]),
            TensorViewMut::new(&mut invalid_layout, destination_layout(32)),
        )),
        Err(target::GetRowsError::InvalidView)
    );
    emit_rejection(
        "q4_0_invalid_layout",
        target::GetRowsError::InvalidView,
        &invalid_layout,
    );
}

#[cfg(target_arch = "aarch64")]
fn run_q8_0(actor: &mut target::Kernel, source: &[u8; Q8_0_BLOCK_BYTES]) {
    let indices = [0_i32];
    let layout = source_layout(8, 32, Q8_0_BLOCK_BYTES, 1);
    let mut output = [0.0_f32; 32];
    assert_eq!(
        actor.process_get_rows(target::OpGetRowsQ8_0::new(
            QuantizedView::new(source, layout),
            target::IndexView::new(&indices, [1, 1, 1]),
            TensorViewMut::new(&mut output, destination_layout(32)),
        )),
        Ok(())
    );
    println!("case=q8_0 status=ok output_bits={}", bits(&output));

    let mut invalid_index = [SENTINEL; 32];
    let negative = [-1_i32];
    assert_eq!(
        actor.process_get_rows(target::OpGetRowsQ8_0::new(
            QuantizedView::new(source, layout),
            target::IndexView::new(&negative, [1, 1, 1]),
            TensorViewMut::new(&mut invalid_index, destination_layout(32)),
        )),
        Err(target::GetRowsError::IndexOutOfBounds)
    );
    emit_rejection(
        "q8_0_invalid_index",
        target::GetRowsError::IndexOutOfBounds,
        &invalid_index,
    );

    let mut invalid_layout = [SENTINEL; 32];
    let malformed = source_layout(8, 32, Q8_0_BLOCK_BYTES, 2);
    assert_eq!(
        actor.process_get_rows(target::OpGetRowsQ8_0::new(
            QuantizedView::new(source, malformed),
            target::IndexView::new(&indices, [1, 1, 1]),
            TensorViewMut::new(&mut invalid_layout, destination_layout(32)),
        )),
        Err(target::GetRowsError::InvalidView)
    );
    emit_rejection(
        "q8_0_invalid_layout",
        target::GetRowsError::InvalidView,
        &invalid_layout,
    );
}

#[cfg(target_arch = "aarch64")]
fn run_q4_k(actor: &mut target::Kernel, source: &[u8; Q4_K_BLOCK_BYTES]) {
    let indices = [0_i32];
    let layout = source_layout(12, QK_K_VALUES, Q4_K_BLOCK_BYTES, 1);
    let mut output = [0.0_f32; QK_K_VALUES];
    assert_eq!(
        actor.process_get_rows(target::OpGetRowsQ4K::new(
            QuantizedView::new(source, layout),
            target::IndexView::new(&indices, [1, 1, 1]),
            TensorViewMut::new(&mut output, destination_layout(QK_K_VALUES)),
        )),
        Ok(())
    );
    println!("case=q4_k status=ok output_bits={}", bits(&output));

    let mut invalid_index = [SENTINEL; QK_K_VALUES];
    let negative = [-1_i32];
    assert_eq!(
        actor.process_get_rows(target::OpGetRowsQ4K::new(
            QuantizedView::new(source, layout),
            target::IndexView::new(&negative, [1, 1, 1]),
            TensorViewMut::new(&mut invalid_index, destination_layout(QK_K_VALUES)),
        )),
        Err(target::GetRowsError::IndexOutOfBounds)
    );
    emit_rejection(
        "q4_k_invalid_index",
        target::GetRowsError::IndexOutOfBounds,
        &invalid_index,
    );

    let mut invalid_layout = [SENTINEL; QK_K_VALUES];
    let malformed = source_layout(12, QK_K_VALUES, Q4_K_BLOCK_BYTES, 2);
    assert_eq!(
        actor.process_get_rows(target::OpGetRowsQ4K::new(
            QuantizedView::new(source, malformed),
            target::IndexView::new(&indices, [1, 1, 1]),
            TensorViewMut::new(&mut invalid_layout, destination_layout(QK_K_VALUES)),
        )),
        Err(target::GetRowsError::InvalidView)
    );
    emit_rejection(
        "q4_k_invalid_layout",
        target::GetRowsError::InvalidView,
        &invalid_layout,
    );
}

#[cfg(target_arch = "aarch64")]
fn main() {
    let mut actor = target::Kernel::try_new().expect("AArch64 router");
    run_q4_0(&mut actor, &q4_source());
    run_q8_0(&mut actor, &q8_source());
    run_q4_k(&mut actor, &q4_k_source());
    println!("kernel-target-get-rows-live-packed/v1");
    println!("source_repository=stateforward/emel.cpp");
    println!("source_commit={SOURCE_COMMIT}");
    println!("source_kernel_events_blob={EVENTS_BLOB}");
    println!("source_kernel_detail_blob={DETAIL_BLOB}");
    println!("source_kernel_aarch64_sm_blob={AARCH64_SM_BLOB}");
    println!("target_arch=aarch64");
    println!("scope=target_router_get_rows_q4_0_q8_0_q4_k_positive_and_typed_rejection");
    println!("execution=split_reference_and_public_target_router");
}
