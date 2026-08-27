#![allow(clippy::float_cmp)]
#![allow(missing_docs)]
#![cfg(any(target_arch = "aarch64", target_arch = "x86_64"))]

use allocation_counter::measure;
use emel_kernels::any::f16_matmul::{F16MatmulError, F16View};
use emel_kernels::any::tensor_view::{DType, Layout, TensorViewMut};

#[cfg(target_arch = "aarch64")]
use emel_kernels::aarch64::{
    OpScalarMulMatF16, TARGET_F16_MATMUL_SOURCE_COMMIT, TARGET_F16_MATMUL_SOURCE_SM_BLOB,
    UnexpectedF16Matmul,
};
#[cfg(target_arch = "x86_64")]
use emel_kernels::x86_64::{
    OpScalarMulMatF16, TARGET_F16_MATMUL_SOURCE_COMMIT, TARGET_F16_MATMUL_SOURCE_SM_BLOB,
    UnexpectedF16Matmul,
};

#[cfg(target_arch = "aarch64")]
type ArchKernel = emel_kernels::aarch64::Kernel;
#[cfg(target_arch = "x86_64")]
type ArchKernel = emel_kernels::x86_64::X86Kernel;

const fn f16_layout(ne: [u64; 4]) -> Layout {
    let mut nb = [2_u64; 4];
    nb[1] = 2 * ne[0];
    nb[2] = nb[1] * ne[1];
    nb[3] = nb[2] * ne[2];
    Layout::new(DType::F16, ne, nb)
}

fn f32_layout(ne: [u64; 4]) -> Layout {
    Layout::contiguous(DType::F32, ne).expect("F32 layout fits")
}

fn request(destination: &mut [f32]) -> OpScalarMulMatF16<'_> {
    let lhs_layout = f16_layout([3, 2, 1, 1]);
    let rhs_layout = f16_layout([3, 2, 1, 1]);
    OpScalarMulMatF16::new(
        F16View::new(
            &[0x3c00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600],
            lhs_layout,
        ),
        F16View::new(
            &[0x3c00, 0x0000, 0x0000, 0x3c00, 0x3c00, 0x3c00],
            rhs_layout,
        ),
        TensorViewMut::new(destination, f32_layout([2, 2, 1, 1])),
    )
}

#[test]
fn target_f16_matmul_preserves_pinned_orientation() {
    let Some(mut kernel) = ArchKernel::try_new() else {
        return;
    };
    let mut destination = [0.0_f32; 4];
    assert_eq!(kernel.process_f16_matmul(request(&mut destination)), Ok(()));
    assert_eq!(destination, [1.0, 4.0, 6.0, 15.0]);
    assert!(kernel.is_ready());
}

#[test]
fn target_f16_matmul_rejects_invalid_shape_without_mutation() {
    let Some(mut kernel) = ArchKernel::try_new() else {
        return;
    };
    let lhs_layout = f16_layout([3, 2, 1, 1]);
    let rhs_layout = f16_layout([2, 2, 1, 1]);
    let mut destination = [9.0_f32; 4];
    let before = destination;
    assert_eq!(
        kernel.process_f16_matmul(OpScalarMulMatF16::new(
            F16View::new(&[0x3c00; 6], lhs_layout),
            F16View::new(&[0x3c00; 4], rhs_layout),
            TensorViewMut::new(&mut destination, f32_layout([2, 2, 1, 1])),
        )),
        Err(F16MatmulError::ShapeMismatch)
    );
    assert_eq!(destination, before);

    let mut invalid_destination = [8.0_f32; 4];
    let invalid_rhs = Layout::new(DType::F16, [3, 2, 1, 1], [4, 12, 24, 48]);
    assert_eq!(
        kernel.process_f16_matmul(OpScalarMulMatF16::new(
            F16View::new(&[0x3c00; 6], lhs_layout),
            F16View::new(&[0x3c00; 6], invalid_rhs),
            TensorViewMut::new(&mut invalid_destination, f32_layout([2, 2, 1, 1])),
        )),
        Err(F16MatmulError::InvalidView)
    );
    assert_eq!(invalid_destination, [8.0; 4]);
    assert!(kernel.is_ready());
}

#[test]
fn target_f16_matmul_unexpected_recovers_and_dispatch_is_allocation_free() {
    let Some(mut kernel) = ArchKernel::try_new() else {
        return;
    };
    assert_eq!(
        kernel.process_f16_matmul(UnexpectedF16Matmul),
        Err(F16MatmulError::UnexpectedEvent)
    );
    let mut destination = [0.0_f32; 4];
    let allocations = measure(|| {
        for _ in 0..128 {
            assert_eq!(kernel.process_f16_matmul(request(&mut destination)), Ok(()));
        }
    });
    assert_eq!(allocations.count_total, 0);
    assert_eq!(allocations.bytes_total, 0);
    assert!(kernel.is_ready());
}

#[test]
fn target_f16_matmul_source_identity_is_pinned() {
    assert_eq!(
        TARGET_F16_MATMUL_SOURCE_COMMIT,
        "843a117386ef17dc5a50549bbfc821074c2141d6"
    );
    #[cfg(target_arch = "aarch64")]
    assert_eq!(
        TARGET_F16_MATMUL_SOURCE_SM_BLOB,
        "865a9cc6ba6115382ed043c464f3d62bcd851357"
    );
    #[cfg(target_arch = "x86_64")]
    assert_eq!(
        TARGET_F16_MATMUL_SOURCE_SM_BLOB,
        "0b4d635ebbd0fbd52dbca8a2345547fb571205c8"
    );
}
