//! Rust-side split observer for target-router `get_rows` variants.
//!
//! The reference executable owns independent C++ fixtures. This process uses
//! only public target-router events; no reference object or state crosses the
//! process boundary.

use emel_kernels::any::f16_matmul::F16View;
use emel_kernels::any::tensor_view::{DType, Layout, TensorViewMut};

#[cfg(target_arch = "aarch64")]
use emel_kernels::aarch64 as target;
#[cfg(target_arch = "x86_64")]
use emel_kernels::x86_64 as target;

#[cfg(target_arch = "aarch64")]
type Kernel = target::Kernel;
#[cfg(target_arch = "x86_64")]
type Kernel = target::X86Kernel;

const SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const EVENTS_BLOB: &str = "4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9";
const DETAIL_BLOB: &str = "c8a82643eabfe8f2d7883e655955f455794511b0";
const X86_SM_BLOB: &str = "0b4d635ebbd0fbd52dbca8a2345547fb571205c8";
const AARCH64_SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";

#[cfg(target_arch = "aarch64")]
const TARGET_ARCH: &str = "aarch64";
#[cfg(target_arch = "x86_64")]
const TARGET_ARCH: &str = "x86_64";

fn layout(dtype: DType, ne: [u64; 4]) -> Layout {
    Layout::contiguous(dtype, ne).expect("fixture layout fits")
}

fn bits(values: &[f32]) -> String {
    values
        .iter()
        .map(|value| format!("{:08x}", value.to_bits()))
        .collect::<Vec<_>>()
        .join(",")
}

fn print_valid(name: &str, output: &[f32]) {
    println!("case={name} status=ok output_bits={}", bits(output));
}

fn print_rejected(name: &str, output: &[f32]) {
    println!(
        "case={name} status=reject error=IndexOutOfBounds output_bits={}",
        bits(output)
    );
}

fn main() {
    let Some(mut kernel) = Kernel::try_new() else {
        panic!("maintained target router unavailable");
    };
    let source_f32 = [1.0_f32, 2.0, 3.0, 4.0];
    let f16_values = [0x3c00_u16, 0x4000, 0x4200, 0x4400];
    let bf16_values = [0x3f80_u16, 0x4000, 0x4040, 0x4080];
    let indices = [1_i32, 0];
    let invalid_indices = [-1_i32, 0];
    let source_f32_layout = layout(DType::F32, [2, 2, 1, 1]);
    let f16_layout = Layout::new(DType::F16, [2, 2, 1, 1], [2, 4, 8, 8]);
    let bf16_layout = Layout::new(DType::Bf16, [2, 2, 1, 1], [2, 4, 8, 8]);
    let destination_layout = layout(DType::F32, [2, 2, 1, 1]);

    let mut f32_output = [0.0_f32; 4];
    assert_eq!(
        kernel.process_get_rows(target::OpGetRows::new(
            emel_kernels::any::tensor_view::TensorView::new(&source_f32, source_f32_layout),
            target::IndexView::new(&indices, [2, 1, 1]),
            TensorViewMut::new(&mut f32_output, destination_layout),
        )),
        Ok(())
    );

    let mut f16_output = [0.0_f32; 4];
    assert_eq!(
        kernel.process_get_rows(target::OpGetRowsF16::new(
            F16View::new(&f16_values, f16_layout),
            target::IndexView::new(&indices, [2, 1, 1]),
            TensorViewMut::new(&mut f16_output, destination_layout),
        )),
        Ok(())
    );

    let mut bf16_output = [0.0_f32; 4];
    assert_eq!(
        kernel.process_get_rows(target::OpGetRowsBf16::new(
            target::Bf16View::new(&bf16_values, bf16_layout),
            target::IndexView::new(&indices, [2, 1, 1]),
            TensorViewMut::new(&mut bf16_output, destination_layout),
        )),
        Ok(())
    );

    let mut f32_invalid = [f32::from_bits(0x7fc0_0001); 4];
    assert_eq!(
        kernel.process_get_rows(target::OpGetRows::new(
            emel_kernels::any::tensor_view::TensorView::new(&source_f32, source_f32_layout),
            target::IndexView::new(&invalid_indices, [2, 1, 1]),
            TensorViewMut::new(&mut f32_invalid, destination_layout),
        )),
        Err(target::GetRowsError::IndexOutOfBounds)
    );

    let mut f16_invalid = [f32::from_bits(0x7fc0_0001); 4];
    assert_eq!(
        kernel.process_get_rows(target::OpGetRowsF16::new(
            F16View::new(&f16_values, f16_layout),
            target::IndexView::new(&invalid_indices, [2, 1, 1]),
            TensorViewMut::new(&mut f16_invalid, destination_layout),
        )),
        Err(target::GetRowsError::IndexOutOfBounds)
    );

    let mut bf16_invalid = [f32::from_bits(0x7fc0_0001); 4];
    assert_eq!(
        kernel.process_get_rows(target::OpGetRowsBf16::new(
            target::Bf16View::new(&bf16_values, bf16_layout),
            target::IndexView::new(&invalid_indices, [2, 1, 1]),
            TensorViewMut::new(&mut bf16_invalid, destination_layout),
        )),
        Err(target::GetRowsError::IndexOutOfBounds)
    );

    println!("kernel-target-get-rows-live/v1");
    println!("source_repository=stateforward/emel.cpp");
    println!("source_commit={SOURCE_COMMIT}");
    println!("source_kernel_events_blob={EVENTS_BLOB}");
    println!("source_kernel_detail_blob={DETAIL_BLOB}");
    println!("source_kernel_x86_sm_blob={X86_SM_BLOB}");
    println!("source_kernel_aarch64_sm_blob={AARCH64_SM_BLOB}");
    println!("target_arch={TARGET_ARCH}");
    println!("scope=target_router_get_rows_f32_f16_bf16_dense_positive_and_typed_rejection");
    println!("execution=split_reference_and_public_target_router");
    print_valid("f32", &f32_output);
    print_valid("f16", &f16_output);
    print_valid("bf16", &bf16_output);
    print_rejected("f32_invalid_index", &f32_invalid);
    print_rejected("f16_invalid_index", &f16_invalid);
    print_rejected("bf16_invalid_index", &bf16_invalid);
}
