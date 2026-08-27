#![allow(clippy::float_cmp)]
#![allow(missing_docs)]
#![cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]

use allocation_counter::measure;
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};

#[cfg(target_arch = "aarch64")]
use emel_kernels::aarch64::{
    OpScalarCos, OpScalarLog, OpScalarSin, ReductionError, TARGET_SCALAR_TRIG_SOURCE_COMMIT,
    TARGET_SCALAR_TRIG_SOURCE_SM_BLOB, UnexpectedScalarTrig,
};
#[cfg(target_arch = "x86_64")]
use emel_kernels::x86_64::{
    OpScalarCos, OpScalarLog, OpScalarSin, ReductionError, TARGET_SCALAR_TRIG_SOURCE_COMMIT,
    TARGET_SCALAR_TRIG_SOURCE_SM_BLOB, UnexpectedScalarTrig,
};

#[cfg(target_arch = "aarch64")]
type ArchKernel = emel_kernels::aarch64::Kernel;
#[cfg(target_arch = "x86_64")]
type ArchKernel = emel_kernels::x86_64::X86Kernel;

fn layout(ne: [u64; 4]) -> Layout {
    Layout::contiguous(DType::F32, ne).expect("layout")
}

#[test]
fn target_scalar_trig_dispatches_log_sin_and_cos() {
    let Some(mut kernel) = ArchKernel::try_new() else {
        return;
    };
    let source_layout = layout([3, 1, 1, 1]);
    let source = [1.0_f32, 2.0, 4.0];
    let mut output = [0.0_f32; 3];

    assert_eq!(
        kernel.process_scalar_trig(OpScalarLog::new(
            TensorView::new(&source, source_layout),
            TensorViewMut::new(&mut output, source_layout),
        )),
        Ok(())
    );
    assert!((output[0] - 1.0_f32.ln()).abs() < 1.0e-6);

    assert_eq!(
        kernel.process_scalar_trig(OpScalarSin::new(
            TensorView::new(&source, source_layout),
            TensorViewMut::new(&mut output, source_layout),
        )),
        Ok(())
    );
    assert!((output[1] - 2.0_f32.sin()).abs() < 1.0e-6);

    assert_eq!(
        kernel.process_scalar_trig(OpScalarCos::new(
            TensorView::new(&source, source_layout),
            TensorViewMut::new(&mut output, source_layout),
        )),
        Ok(())
    );
    assert!((output[2] - 4.0_f32.cos()).abs() < 1.0e-6);
    assert!(kernel.is_ready());
}

#[test]
fn target_scalar_trig_accepts_valid_differing_layouts_with_equal_count() {
    let Some(mut kernel) = ArchKernel::try_new() else {
        return;
    };
    let source_layout = layout([2, 2, 1, 1]);
    let destination_layout = Layout::new(DType::F32, [4, 1, 1, 1], [4, 16, 16, 16]);
    let source = [1.0_f32, 2.0, 4.0, 8.0];
    let mut output = [0.0_f32; 4];

    assert_ne!(source_layout, destination_layout);
    assert_eq!(
        kernel.process_scalar_trig(OpScalarLog::new(
            TensorView::new(&source, source_layout),
            TensorViewMut::new(&mut output, destination_layout),
        )),
        Ok(())
    );
    assert!((output[0] - 1.0_f32.ln()).abs() < 1.0e-6);
    assert!((output[3] - 8.0_f32.ln()).abs() < 1.0e-6);

    assert_eq!(
        kernel.process_scalar_trig(OpScalarSin::new(
            TensorView::new(&source, source_layout),
            TensorViewMut::new(&mut output, destination_layout),
        )),
        Ok(())
    );
    assert!((output[1] - 2.0_f32.sin()).abs() < 1.0e-6);

    assert_eq!(
        kernel.process_scalar_trig(OpScalarCos::new(
            TensorView::new(&source, source_layout),
            TensorViewMut::new(&mut output, destination_layout),
        )),
        Ok(())
    );
    assert!((output[2] - 4.0_f32.cos()).abs() < 1.0e-6);
    assert!(kernel.is_ready());
}

#[test]
fn target_scalar_trig_preserves_reduction_validation_and_recovers() {
    let Some(mut kernel) = ArchKernel::try_new() else {
        return;
    };
    let source_layout = layout([3, 1, 1, 1]);
    let mismatched_layout = layout([2, 1, 1, 1]);
    let source = [1.0_f32, 2.0, 4.0];
    let mut invalid_shape_output = [7.0_f32; 2];
    assert_eq!(
        kernel.process_scalar_trig(OpScalarSin::new(
            TensorView::new(&source, source_layout),
            TensorViewMut::new(&mut invalid_shape_output, mismatched_layout),
        )),
        Err(ReductionError::ShapeMismatch)
    );
    assert_eq!(invalid_shape_output, [7.0, 7.0]);

    let invalid_dtype = Layout::new(DType::F16, [3, 1, 1, 1], [4, 12, 12, 12]);
    let mut invalid_view_output = [9.0_f32; 3];
    assert_eq!(
        kernel.process_scalar_trig(OpScalarCos::new(
            TensorView::new(&source, invalid_dtype),
            TensorViewMut::new(&mut invalid_view_output, source_layout),
        )),
        Err(ReductionError::InvalidView)
    );
    assert_eq!(invalid_view_output, [9.0, 9.0, 9.0]);
    assert_eq!(
        kernel.process_scalar_trig(UnexpectedScalarTrig),
        Err(ReductionError::UnexpectedEvent)
    );
    assert!(kernel.is_ready());
}

#[test]
fn target_scalar_trig_dispatch_is_allocation_free_after_construction() {
    let Some(mut kernel) = ArchKernel::try_new() else {
        return;
    };
    let source_layout = layout([17, 1, 1, 1]);
    let source = [0.25_f32; 17];
    let mut output = [0.0_f32; 17];
    let allocations = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                kernel.process_scalar_trig(OpScalarCos::new(
                    TensorView::new(&source, source_layout),
                    TensorViewMut::new(&mut output, source_layout),
                )),
                Ok(())
            );
        }
    });
    assert_eq!(allocations.count_total, 0);
}

#[test]
fn target_scalar_trig_source_identity_is_pinned() {
    assert_eq!(
        TARGET_SCALAR_TRIG_SOURCE_COMMIT,
        "843a117386ef17dc5a50549bbfc821074c2141d6"
    );
    #[cfg(target_arch = "aarch64")]
    assert_eq!(
        TARGET_SCALAR_TRIG_SOURCE_SM_BLOB,
        "865a9cc6ba6115382ed043c464f3d62bcd851357"
    );
    #[cfg(target_arch = "x86_64")]
    assert_eq!(
        TARGET_SCALAR_TRIG_SOURCE_SM_BLOB,
        "0b4d635ebbd0fbd52dbca8a2345547fb571205c8"
    );
}
