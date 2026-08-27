#![allow(clippy::float_cmp)]
#![allow(missing_docs)]
#![cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]

use allocation_counter::measure;
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};

#[cfg(target_arch = "aarch64")]
use emel_kernels::aarch64::{
    F16OutputViewMut, Im2ColError, Im2ColParams, OpIm2Col, OpIm2ColF16,
    TARGET_IM2COL_SOURCE_COMMIT, TARGET_IM2COL_SOURCE_SM_BLOB, UnexpectedIm2Col,
};
#[cfg(target_arch = "x86_64")]
use emel_kernels::x86_64::{
    F16OutputViewMut, Im2ColError, Im2ColParams, OpIm2Col, OpIm2ColF16,
    TARGET_IM2COL_SOURCE_COMMIT, TARGET_IM2COL_SOURCE_SM_BLOB, UnexpectedIm2Col,
};

#[cfg(target_arch = "aarch64")]
type ArchKernel = emel_kernels::aarch64::Kernel;
#[cfg(target_arch = "x86_64")]
type ArchKernel = emel_kernels::x86_64::X86Kernel;

const PARAMS: Im2ColParams = Im2ColParams {
    stride: 1,
    padding: 0,
    dilation: 1,
    is_2d: 0,
};

const fn f16_layout(ne: [u64; 4]) -> Layout {
    let mut nb = [2_u64; 4];
    nb[1] = nb[0] * ne[0];
    nb[2] = nb[1] * ne[1];
    nb[3] = nb[2] * ne[2];
    Layout::new(DType::F16, ne, nb)
}

fn f32_event(output: &mut [f32]) -> OpIm2Col<'_> {
    let kernel = Layout::contiguous(DType::F32, [2, 1, 1, 1]).expect("layout");
    let input_layout = Layout::contiguous(DType::F32, [3, 1, 1, 1]).expect("layout");
    let output_layout = Layout::contiguous(DType::F32, [2, 2, 1, 1]).expect("layout");
    OpIm2Col::new(
        kernel,
        TensorView::new(&[1.0, 2.0, 3.0], input_layout),
        TensorViewMut::new(output, output_layout),
        PARAMS,
    )
}

#[test]
fn target_im2col_dispatches_f32_and_recovers_after_invalid_shape() {
    let Some(mut kernel) = ArchKernel::try_new() else {
        return;
    };
    let mut output = [0.0_f32; 4];
    assert_eq!(kernel.process_im2col(f32_event(&mut output)), Ok(()));
    assert_eq!(output, [1.0, 2.0, 2.0, 3.0]);

    let mut invalid_output = [9.0_f32; 2];
    let before = invalid_output;
    let event = OpIm2Col::new(
        Layout::contiguous(DType::F32, [2, 1, 1, 1]).expect("layout"),
        TensorView::new(
            &[1.0, 2.0, 3.0],
            Layout::contiguous(DType::F32, [3, 1, 1, 1]).expect("layout"),
        ),
        TensorViewMut::new(
            &mut invalid_output,
            Layout::contiguous(DType::F32, [2, 1, 1, 1]).expect("layout"),
        ),
        PARAMS,
    );
    assert_eq!(
        kernel.process_im2col(event),
        Err(Im2ColError::ShapeMismatch)
    );
    assert_eq!(invalid_output, before);
    assert!(kernel.is_ready());
}

#[test]
fn target_im2col_dispatches_f16_destination() {
    let Some(mut kernel) = ArchKernel::try_new() else {
        return;
    };
    let mut output = [0_u16; 4];
    let event = OpIm2ColF16::new(
        Layout::contiguous(DType::F32, [2, 1, 1, 1]).expect("layout"),
        TensorView::new(
            &[1.0, 2.0, 3.0],
            Layout::contiguous(DType::F32, [3, 1, 1, 1]).expect("layout"),
        ),
        F16OutputViewMut::new(&mut output, f16_layout([2, 2, 1, 1])),
        PARAMS,
    );
    assert_eq!(kernel.process_im2col(event), Ok(()));
    assert_eq!(output, [0x3c00, 0x4000, 0x4000, 0x4200]);
}

#[test]
fn target_im2col_unexpected_event_is_explicit_and_recovers() {
    let Some(mut kernel) = ArchKernel::try_new() else {
        return;
    };
    assert_eq!(
        kernel.process_im2col(UnexpectedIm2Col),
        Err(Im2ColError::UnexpectedEvent)
    );
    let mut output = [0.0_f32; 4];
    assert_eq!(kernel.process_im2col(f32_event(&mut output)), Ok(()));
}

#[test]
fn target_im2col_dispatch_is_allocation_free_after_construction() {
    let Some(mut kernel) = ArchKernel::try_new() else {
        return;
    };
    let mut output = [0.0_f32; 4];
    let allocations = measure(|| {
        for _ in 0..128 {
            assert_eq!(kernel.process_im2col(f32_event(&mut output)), Ok(()));
        }
    });
    assert_eq!(allocations.count_total, 0);
}

#[test]
fn target_im2col_source_identity_is_pinned() {
    assert_eq!(
        TARGET_IM2COL_SOURCE_COMMIT,
        "843a117386ef17dc5a50549bbfc821074c2141d6"
    );
    #[cfg(target_arch = "aarch64")]
    assert_eq!(
        TARGET_IM2COL_SOURCE_SM_BLOB,
        "865a9cc6ba6115382ed043c464f3d62bcd851357"
    );
    #[cfg(target_arch = "x86_64")]
    assert_eq!(
        TARGET_IM2COL_SOURCE_SM_BLOB,
        "0b4d635ebbd0fbd52dbca8a2345547fb571205c8"
    );
}
