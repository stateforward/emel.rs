#![allow(clippy::float_cmp)]
#![allow(missing_docs)]
#![cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]

use allocation_counter::measure;
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};

#[cfg(target_arch = "aarch64")]
use emel_kernels::aarch64::{
    NormalizationError, OpNorm, OpRmsNorm, TARGET_NORMALIZATION_SOURCE_COMMIT,
    TARGET_NORMALIZATION_SOURCE_SM_BLOB, UnexpectedNormalization,
};
#[cfg(target_arch = "x86_64")]
use emel_kernels::x86_64::{
    NormalizationError, OpNorm, OpRmsNorm, TARGET_NORMALIZATION_SOURCE_COMMIT,
    TARGET_NORMALIZATION_SOURCE_SM_BLOB, UnexpectedNormalization,
};

#[cfg(target_arch = "aarch64")]
type ArchKernel = emel_kernels::aarch64::Kernel;
#[cfg(target_arch = "x86_64")]
type ArchKernel = emel_kernels::x86_64::X86Kernel;

fn norm_event(output: &mut [f32]) -> OpNorm<'_> {
    let layout = Layout::contiguous(DType::F32, [4, 2, 1, 1]).expect("layout");
    OpNorm::new(
        TensorView::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], layout),
        TensorViewMut::new(output, layout),
        1.0e-5,
    )
}

fn rms_event(output: &mut [f32]) -> OpRmsNorm<'_> {
    let layout = Layout::contiguous(DType::F32, [4, 2, 1, 1]).expect("layout");
    OpRmsNorm::new(
        TensorView::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], layout),
        TensorViewMut::new(output, layout),
        1.0e-5,
    )
}

#[test]
fn target_normalization_dispatches_norm_and_rms_norm() {
    let Some(mut kernel) = ArchKernel::try_new() else {
        return;
    };
    let mut norm_output = [0.0_f32; 8];
    let mut rms_output = [0.0_f32; 8];
    assert_eq!(
        kernel.process_normalization(norm_event(&mut norm_output)),
        Ok(())
    );
    assert_eq!(
        kernel.process_normalization(rms_event(&mut rms_output)),
        Ok(())
    );
    assert_ne!(norm_output, [0.0; 8]);
    assert_ne!(rms_output, [0.0; 8]);
    assert!(kernel.is_ready());
}

#[test]
fn target_normalization_rejects_shape_without_mutation_and_recovers() {
    let Some(mut kernel) = ArchKernel::try_new() else {
        return;
    };
    let mut output = [7.0_f32; 2];
    let before = output;
    let source_layout = Layout::contiguous(DType::F32, [4, 2, 1, 1]).expect("layout");
    let output_layout = Layout::contiguous(DType::F32, [2, 1, 1, 1]).expect("layout");
    assert_eq!(
        kernel.process_normalization(OpNorm::new(
            TensorView::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], source_layout),
            TensorViewMut::new(&mut output, output_layout),
            1.0e-5,
        )),
        Err(NormalizationError::ShapeMismatch)
    );
    assert_eq!(output, before);
    let mut recovered = [0.0_f32; 8];
    assert_eq!(
        kernel.process_normalization(norm_event(&mut recovered)),
        Ok(())
    );
}

#[test]
fn target_normalization_unexpected_event_is_explicit_and_recovers() {
    let Some(mut kernel) = ArchKernel::try_new() else {
        return;
    };
    assert_eq!(
        kernel.process_normalization(UnexpectedNormalization),
        Err(NormalizationError::UnexpectedEvent)
    );
    let mut output = [0.0_f32; 8];
    assert_eq!(
        kernel.process_normalization(norm_event(&mut output)),
        Ok(())
    );
}

#[test]
fn target_normalization_dispatch_is_allocation_free_after_construction() {
    let Some(mut kernel) = ArchKernel::try_new() else {
        return;
    };
    let mut output = [0.0_f32; 8];
    let allocations = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                kernel.process_normalization(norm_event(&mut output)),
                Ok(())
            );
        }
    });
    assert_eq!(allocations.count_total, 0);
}

#[test]
fn target_normalization_source_identity_is_pinned() {
    assert_eq!(
        TARGET_NORMALIZATION_SOURCE_COMMIT,
        "843a117386ef17dc5a50549bbfc821074c2141d6"
    );
    #[cfg(target_arch = "aarch64")]
    assert_eq!(
        TARGET_NORMALIZATION_SOURCE_SM_BLOB,
        "865a9cc6ba6115382ed043c464f3d62bcd851357"
    );
    #[cfg(target_arch = "x86_64")]
    assert_eq!(
        TARGET_NORMALIZATION_SOURCE_SM_BLOB,
        "0b4d635ebbd0fbd52dbca8a2345547fb571205c8"
    );
}
