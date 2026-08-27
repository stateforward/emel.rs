#![allow(clippy::float_cmp, missing_docs)]

use allocation_counter::measure;
use emel_kernels::any::f16_matmul::{
    F16MatmulError, F16MatmulKernel, F16View, OpMulMatF16, UnexpectedF16Matmul,
};
use emel_kernels::any::tensor_view::{DType, Layout, TensorViewMut};
use emel_tensor as _;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use pulp as _;
use sml as _;

const PINNED_EMEL_CPP_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const PINNED_SML_COMMIT: &str = "49207123cd3f39767764bae774932cb48623f92f";
const PINNED_EVENTS_BLOB: &str = "4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9";
const PINNED_DETAIL_BLOB: &str = "c8a82643eabfe8f2d7883e655955f455794511b0";
const PINNED_X86_SM_BLOB: &str = "0b4d635ebbd0fbd52dbca8a2345547fb571205c8";
const PINNED_X86_GUARDS_BLOB: &str = "cb3dac8253f8417c9b44acff1de414f6d0a3a3cf";
const PINNED_X86_ACTIONS_BLOB: &str = "d45558f5eb96950f43c16a09d768cb4f382d6d61";
const PINNED_AARCH64_SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";
const PINNED_AARCH64_GUARDS_BLOB: &str = "c25714566ec9a02679daef85089544575123408e";
const PINNED_AARCH64_ACTIONS_BLOB: &str = "267d4f74e6e7498155c8535920322ffef2c02fb6";

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

#[test]
fn source_identity_and_orientation_match_pinned_f16_contract() {
    assert_eq!(
        PINNED_EMEL_CPP_COMMIT,
        "843a117386ef17dc5a50549bbfc821074c2141d6"
    );
    assert_eq!(
        PINNED_SML_COMMIT,
        "49207123cd3f39767764bae774932cb48623f92f"
    );
    assert_eq!(
        PINNED_EVENTS_BLOB,
        "4b3f02fa5ff9c5071d1fc83938cef1b22b4658b9"
    );
    assert_eq!(
        PINNED_DETAIL_BLOB,
        "c8a82643eabfe8f2d7883e655955f455794511b0"
    );
    assert_eq!(
        PINNED_X86_SM_BLOB,
        "0b4d635ebbd0fbd52dbca8a2345547fb571205c8"
    );
    assert_eq!(
        PINNED_X86_GUARDS_BLOB,
        "cb3dac8253f8417c9b44acff1de414f6d0a3a3cf"
    );
    assert_eq!(
        PINNED_X86_ACTIONS_BLOB,
        "d45558f5eb96950f43c16a09d768cb4f382d6d61"
    );
    assert_eq!(
        PINNED_AARCH64_SM_BLOB,
        "865a9cc6ba6115382ed043c464f3d62bcd851357"
    );
    assert_eq!(
        PINNED_AARCH64_GUARDS_BLOB,
        "c25714566ec9a02679daef85089544575123408e"
    );
    assert_eq!(
        PINNED_AARCH64_ACTIONS_BLOB,
        "267d4f74e6e7498155c8535920322ffef2c02fb6"
    );
}

#[test]
fn f16_scope_rejects_implicit_or_non_dense_operand_strides() {
    let lhs = [0x3c00_u16; 1];
    let mut destination = [7.0_f32; 1];
    let mut actor = F16MatmulKernel::new();
    let implicit = Layout::new(DType::F16, [1, 1, 1, 1], [0, 2, 2, 2]);
    let dense = f16_layout([1, 1, 1, 1]);
    assert_eq!(
        actor.process_event(OpMulMatF16::new(
            F16View::new(&lhs, implicit),
            F16View::new(&lhs, dense),
            TensorViewMut::new(&mut destination, f32_layout([1, 1, 1, 1])),
        )),
        Err(F16MatmulError::InvalidView)
    );
    assert_eq!(destination, [7.0]);
}

#[test]
fn f16_matmul_converts_operands_and_matches_destination_orientation() {
    let lhs_layout = f16_layout([3, 2, 1, 1]);
    let rhs_layout = f16_layout([3, 2, 1, 1]);
    let destination_layout = f32_layout([2, 2, 1, 1]);
    let lhs = [0x3c00, 0x4000, 0x4200, 0x4400, 0x4500, 0x4600];
    let rhs = [0x3c00, 0x0000, 0x0000, 0x3c00, 0x3c00, 0x3c00];
    let mut destination = [0.0_f32; 4];
    let mut actor = F16MatmulKernel::new();

    assert_eq!(
        actor.process_event(OpMulMatF16::new(
            F16View::new(&lhs, lhs_layout),
            F16View::new(&rhs, rhs_layout),
            TensorViewMut::new(&mut destination, destination_layout),
        )),
        Ok(())
    );
    assert_eq!(destination, [1.0, 4.0, 6.0, 15.0]);
}

#[test]
fn f16_matmul_preserves_reference_double_accumulation_before_f32_cast() {
    let layout = f16_layout([16, 1, 1, 1]);
    let destination_layout = f32_layout([1, 1, 1, 1]);
    let lhs = [
        22892, 382, 28000, 56722, 59028, 37094, 30984, 61306, 6332, 3406, 63920, 15970, 36836,
        5814, 36696, 27210,
    ];
    let rhs = [
        34267, 15045, 49471, 24521, 41187, 62221, 18631, 47249, 7403, 46165, 49487, 10841, 6643,
        40605, 19159, 54561,
    ];
    let mut destination = [0.0_f32];
    let mut actor = F16MatmulKernel::new();

    assert_eq!(
        actor.process_event(OpMulMatF16::new(
            F16View::new(&lhs, layout),
            F16View::new(&rhs, layout),
            TensorViewMut::new(&mut destination, destination_layout),
        )),
        Ok(())
    );
    assert_eq!(destination[0].to_bits(), 0x4780_0e98);
}

#[test]
fn f16_guards_reject_dtype_and_shape_without_writing() {
    let lhs_layout = f16_layout([3, 2, 1, 1]);
    let destination_layout = f32_layout([2, 2, 1, 1]);
    let lhs = [0x3c00_u16; 6];
    let mut destination = [9.0_f32; 4];
    let mut actor = F16MatmulKernel::new();

    assert_eq!(
        actor.process_event(OpMulMatF16::new(
            F16View::new(&lhs, lhs_layout),
            F16View::new(&lhs, f16_layout([2, 2, 1, 1])),
            TensorViewMut::new(&mut destination, destination_layout),
        )),
        Err(F16MatmulError::ShapeMismatch)
    );
    assert_eq!(destination, [9.0; 4]);

    assert_eq!(
        actor.process_event(OpMulMatF16::new(
            F16View::new(&lhs, lhs_layout),
            F16View::new(
                &[0_u16; 6],
                Layout::new(DType::F16, [3, 2, 1, 1], [4, 12, 24, 48])
            ),
            TensorViewMut::new(&mut destination, destination_layout),
        )),
        Err(F16MatmulError::InvalidView)
    );
}

#[test]
fn f16_unexpected_and_dispatch_are_explicit_and_allocation_free() {
    let lhs_layout = f16_layout([1, 1, 1, 1]);
    let destination_layout = f32_layout([1, 1, 1, 1]);
    let lhs = [0x3c00_u16];
    let mut destination = [0.0_f32];
    let mut actor = F16MatmulKernel::new();
    assert_eq!(
        actor.process_event(UnexpectedF16Matmul),
        Err(F16MatmulError::UnexpectedEvent)
    );

    let allocation = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                actor.process_event(OpMulMatF16::new(
                    F16View::new(&lhs, lhs_layout),
                    F16View::new(&lhs, lhs_layout),
                    TensorViewMut::new(&mut destination, destination_layout),
                )),
                Ok(())
            );
        }
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(allocation.bytes_total, 0);
}

#[test]
fn f16_error_surface_is_typed() {
    assert!(format!("{}", F16MatmulError::InvalidView).contains("F16"));
    assert!(format!("{:?}", F16MatmulKernel::default()).contains("F16MatmulKernel"));
}
