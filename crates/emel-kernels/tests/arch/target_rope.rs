#![allow(clippy::float_cmp)]
#![allow(missing_docs)]
#![cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]

use allocation_counter::measure;
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};

#[cfg(target_arch = "aarch64")]
use emel_kernels::aarch64::{
    I32View, OpRope, RopeError, RopeParams, TARGET_ROPE_SOURCE_COMMIT, TARGET_ROPE_SOURCE_SM_BLOB,
    UnexpectedRope,
};
#[cfg(target_arch = "x86_64")]
use emel_kernels::x86_64::{
    I32View, OpRope, RopeError, RopeParams, TARGET_ROPE_SOURCE_COMMIT, TARGET_ROPE_SOURCE_SM_BLOB,
    UnexpectedRope,
};

#[cfg(target_arch = "aarch64")]
type ArchKernel = emel_kernels::aarch64::Kernel;
#[cfg(target_arch = "x86_64")]
type ArchKernel = emel_kernels::x86_64::X86Kernel;

const fn params(mode: i32) -> RopeParams {
    RopeParams {
        n_dims: 4,
        mode,
        freq_base: 10_000.0,
        freq_scale: 1.0,
        ext_factor: 0.0,
        attn_factor: 1.0,
    }
}

fn event(mode: i32, output: &mut [f32]) -> OpRope<'_> {
    let source_layout = Layout::contiguous(DType::F32, [4, 1, 1, 1]).expect("layout");
    let positions_layout = Layout::contiguous(DType::I32, [1, 1, 1, 1]).expect("layout");
    let destination_layout = Layout::contiguous(DType::F32, [4, 1, 1, 1]).expect("layout");
    OpRope::new(
        TensorView::new(&[1.0, 2.0, 3.0, 4.0], source_layout),
        I32View::new(&[0], positions_layout),
        TensorViewMut::new(output, destination_layout),
        params(mode),
    )
}

#[test]
fn target_rope_routes_all_three_modes_and_recovers() {
    let Some(mut kernel) = ArchKernel::try_new() else {
        return;
    };
    let mut norm = [0.0_f32; 4];
    let mut neox = [0.0_f32; 4];
    let mut timestep = [0.0_f32; 4];

    assert_eq!(
        kernel.process_rope(event(emel_kernels::any::rope::ROPE_MODE_NORM, &mut norm)),
        Ok(())
    );
    assert_eq!(
        kernel.process_rope(event(emel_kernels::any::rope::ROPE_MODE_NEOX, &mut neox)),
        Ok(())
    );
    assert_eq!(
        kernel.process_rope(event(
            emel_kernels::any::rope::ROPE_MODE_TIMESTEP,
            &mut timestep,
        )),
        Ok(())
    );
    assert_ne!(norm, [0.0; 4]);
    assert_ne!(neox, [0.0; 4]);
    assert_ne!(timestep, [0.0; 4]);
    assert!(kernel.is_ready());
}

#[test]
fn target_rope_rejects_invalid_mode_without_mutation_and_recovers() {
    let Some(mut kernel) = ArchKernel::try_new() else {
        return;
    };
    let mut output = [7.0_f32; 4];
    let before = output;
    assert_eq!(
        kernel.process_rope(event(99, &mut output)),
        Err(RopeError::InvalidMode)
    );
    assert_eq!(output, before);
    assert_eq!(
        kernel.process_rope(event(emel_kernels::any::rope::ROPE_MODE_NORM, &mut output)),
        Ok(())
    );
}

#[test]
fn target_rope_unexpected_event_is_explicit_and_recovers() {
    let Some(mut kernel) = ArchKernel::try_new() else {
        return;
    };
    assert_eq!(
        kernel.process_rope(UnexpectedRope),
        Err(RopeError::UnexpectedEvent)
    );
    let mut output = [0.0_f32; 4];
    assert_eq!(
        kernel.process_rope(event(emel_kernels::any::rope::ROPE_MODE_NORM, &mut output)),
        Ok(())
    );
}

#[test]
fn target_rope_dispatch_is_allocation_free_after_construction() {
    let Some(mut kernel) = ArchKernel::try_new() else {
        return;
    };
    let mut output = [0.0_f32; 4];
    let allocations = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                kernel.process_rope(event(emel_kernels::any::rope::ROPE_MODE_NORM, &mut output)),
                Ok(())
            );
        }
    });
    assert_eq!(allocations.count_total, 0);
}

#[test]
fn target_rope_source_identity_is_pinned() {
    assert_eq!(
        TARGET_ROPE_SOURCE_COMMIT,
        "843a117386ef17dc5a50549bbfc821074c2141d6"
    );
    #[cfg(target_arch = "aarch64")]
    assert_eq!(
        TARGET_ROPE_SOURCE_SM_BLOB,
        "865a9cc6ba6115382ed043c464f3d62bcd851357"
    );
    #[cfg(target_arch = "x86_64")]
    assert_eq!(
        TARGET_ROPE_SOURCE_SM_BLOB,
        "0b4d635ebbd0fbd52dbca8a2345547fb571205c8"
    );
}
