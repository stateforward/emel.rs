#![allow(clippy::float_cmp)]
#![allow(missing_docs)]
#![cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]

use allocation_counter::measure;
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};

#[cfg(target_arch = "aarch64")]
use emel_kernels::aarch64::{
    OpSoftMax, ReductionError, TARGET_SOFTMAX_SOURCE_COMMIT, TARGET_SOFTMAX_SOURCE_SM_BLOB,
    UnexpectedSoftMax,
};
#[cfg(target_arch = "x86_64")]
use emel_kernels::x86_64::{
    OpSoftMax, ReductionError, TARGET_SOFTMAX_SOURCE_COMMIT, TARGET_SOFTMAX_SOURCE_SM_BLOB,
    UnexpectedSoftMax,
};

#[cfg(target_arch = "aarch64")]
type ArchKernel = emel_kernels::aarch64::Kernel;
#[cfg(target_arch = "x86_64")]
type ArchKernel = emel_kernels::x86_64::X86Kernel;

fn event(output: &mut [f32]) -> OpSoftMax<'_> {
    let layout = Layout::contiguous(DType::F32, [3, 2, 1, 1]).expect("layout");
    OpSoftMax::new(
        TensorView::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], layout),
        TensorViewMut::new(output, layout),
    )
}

fn implicit_event(output: &mut [f32]) -> OpSoftMax<'_> {
    let layout = Layout::new(DType::F32, [3, 2, 1, 1], [0, 0, 0, 0]);
    OpSoftMax::new(
        TensorView::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], layout),
        TensorViewMut::new(output, layout),
    )
}

#[test]
fn target_softmax_dispatches_rows_and_recovers_after_shape_error() {
    let Some(mut kernel) = ArchKernel::try_new() else {
        return;
    };
    let mut output = [0.0_f32; 6];
    assert_eq!(kernel.process_soft_max(event(&mut output)), Ok(()));
    assert!((output[..3].iter().sum::<f32>() - 1.0).abs() < 1.0e-6);
    assert!((output[3..].iter().sum::<f32>() - 1.0).abs() < 1.0e-6);

    let mut invalid_output = [7.0_f32; 2];
    let before = invalid_output;
    let source_layout = Layout::contiguous(DType::F32, [3, 2, 1, 1]).expect("layout");
    let output_layout = Layout::contiguous(DType::F32, [2, 1, 1, 1]).expect("layout");
    assert_eq!(
        kernel.process_soft_max(OpSoftMax::new(
            TensorView::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], source_layout),
            TensorViewMut::new(&mut invalid_output, output_layout),
        )),
        Err(ReductionError::ShapeMismatch)
    );
    assert_eq!(invalid_output, before);
    assert!(kernel.is_ready());
}

#[test]
fn target_softmax_accepts_implicit_contiguous_nb0_layouts() {
    let Some(mut kernel) = ArchKernel::try_new() else {
        return;
    };
    let mut output = [0.0_f32; 6];
    assert_eq!(kernel.process_soft_max(implicit_event(&mut output)), Ok(()));
    assert!((output[..3].iter().sum::<f32>() - 1.0).abs() < 1.0e-6);
    assert!((output[3..].iter().sum::<f32>() - 1.0).abs() < 1.0e-6);
    assert!(kernel.is_ready());
}

#[test]
fn target_softmax_unexpected_event_is_explicit_and_recovers() {
    let Some(mut kernel) = ArchKernel::try_new() else {
        return;
    };
    assert_eq!(
        kernel.process_soft_max(UnexpectedSoftMax),
        Err(ReductionError::UnexpectedEvent)
    );
    let mut output = [0.0_f32; 6];
    assert_eq!(kernel.process_soft_max(event(&mut output)), Ok(()));
}

#[test]
fn target_softmax_dispatch_is_allocation_free_after_construction() {
    let Some(mut kernel) = ArchKernel::try_new() else {
        return;
    };
    let mut output = [0.0_f32; 6];
    let allocations = measure(|| {
        for _ in 0..128 {
            assert_eq!(kernel.process_soft_max(event(&mut output)), Ok(()));
        }
    });
    assert_eq!(allocations.count_total, 0);
}

#[test]
fn target_softmax_source_identity_is_pinned() {
    assert_eq!(
        TARGET_SOFTMAX_SOURCE_COMMIT,
        "843a117386ef17dc5a50549bbfc821074c2141d6"
    );
    #[cfg(target_arch = "aarch64")]
    assert_eq!(
        TARGET_SOFTMAX_SOURCE_SM_BLOB,
        "865a9cc6ba6115382ed043c464f3d62bcd851357"
    );
    #[cfg(target_arch = "x86_64")]
    assert_eq!(
        TARGET_SOFTMAX_SOURCE_SM_BLOB,
        "0b4d635ebbd0fbd52dbca8a2345547fb571205c8"
    );
}
