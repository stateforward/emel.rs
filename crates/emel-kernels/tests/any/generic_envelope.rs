#![allow(clippy::float_cmp, missing_docs)]

use emel_kernels::any::event::{
    GENERIC_OPERATION_COUNT, GenericEvent, GenericLayout, GenericRequest, GenericViewError,
    KernelOperation, OpParams, RawTensorView, RawTensorViewMut,
};
use emel_kernels::{Error, Kernel};

const fn layout() -> GenericLayout {
    GenericLayout::new(
        emel_kernels::any::event::GenericDType::F32,
        [4, 1, 1, 1],
        [4, 16, 64, 256],
    )
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}

const fn read_f32(bytes: &[u8], index: usize) -> f32 {
    let start = index * 4;
    f32::from_ne_bytes([
        bytes[start],
        bytes[start + 1],
        bytes[start + 2],
        bytes[start + 3],
    ])
}

fn f16_bytes(values: &[u16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * 2);
    for value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}

const fn f32_layout(ne: [u64; 4]) -> GenericLayout {
    let mut nb = [4_u64; 4];
    let mut dimension = 1;
    while dimension < 4 {
        nb[dimension] = nb[dimension - 1] * ne[dimension - 1];
        dimension += 1;
    }
    GenericLayout::new(emel_kernels::any::event::GenericDType::F32, ne, nb)
}

fn i32_params(slot: usize, value: i32) -> OpParams {
    let mut bytes = [0_u8; 64];
    bytes[slot * 4..slot * 4 + 4].copy_from_slice(&value.to_ne_bytes());
    OpParams::new(bytes, slot * 4 + 4).expect("i32 parameter fits")
}

fn f32_params(slot: usize, value: f32) -> OpParams {
    let mut bytes = [0_u8; 64];
    bytes[slot * 4..slot * 4 + 4].copy_from_slice(&value.to_ne_bytes());
    OpParams::new(bytes, slot * 4 + 4).expect("f32 parameter fits")
}

#[test]
fn pinned_operation_inventory_round_trips_all_entries() {
    assert_eq!(GENERIC_OPERATION_COUNT, 95);
    for code in 0..GENERIC_OPERATION_COUNT {
        let operation = KernelOperation::from_code(
            u8::try_from(code).expect("pinned operation ordinal fits in u8"),
        )
        .expect("pinned operation");
        assert_eq!(operation as usize, code);
        assert!(operation.name().starts_with("op_"));
    }
    assert!(
        KernelOperation::from_code(
            u8::try_from(GENERIC_OPERATION_COUNT).expect("operation count fits in u8"),
        )
        .is_none()
    );
}

#[test]
fn generic_envelope_preserves_safe_views_params_and_argmax_destination() {
    let source = f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let rhs = f32_bytes(&[1.0, 1.0]);
    let mut destination = [0_u8; 4];
    let source_layout = GenericLayout::new(
        emel_kernels::any::event::GenericDType::F32,
        [2, 3, 1, 1],
        [4, 8, 24, 24],
    );
    let rhs_layout = GenericLayout::new(
        emel_kernels::any::event::GenericDType::F32,
        [2, 1, 1, 1],
        [4, 8, 8, 8],
    );
    let destination_layout = GenericLayout::new(
        emel_kernels::any::event::GenericDType::F32,
        [1, 1, 1, 1],
        [4, 4, 4, 4],
    );
    let params = OpParams::new([7_u8; 64], 3).expect("parameter prefix");
    let mut index = -1_i32;
    let request = GenericRequest::new(
        KernelOperation::MulMatArgmax,
        Some(RawTensorView::new(&source, source_layout)),
        Some(RawTensorView::new(&rhs, rhs_layout)),
        None,
        Some(RawTensorViewMut::new(&mut destination, destination_layout)),
        params,
    )
    .with_index_out(&mut index);

    assert_eq!(request.params().as_slice(), &[7, 7, 7]);
    assert_eq!(request.params().len(), 3);
    assert!(request.has_index_out());
    assert!(request.validate().is_ok());

    let mut kernel = Kernel::new();
    assert_eq!(kernel.process_event(GenericEvent::new(request)), Ok(()));
    assert!((read_f32(&destination, 0) - 11.0).abs() < f32::EPSILON);
    assert_eq!(index, 2);
    assert!(kernel.is_ready());
}

#[test]
fn generic_envelope_executes_pinned_portable_scalar_family() {
    let source = f32_bytes(&[1.0, -2.0, 3.0, 4.0]);
    let rhs = f32_bytes(&[2.0, 4.0, 6.0, 8.0]);
    let layout = layout();
    let mut destination = [0_u8; 16];
    let request = GenericRequest::new(
        KernelOperation::Add,
        Some(RawTensorView::new(&source, layout)),
        Some(RawTensorView::new(&rhs, layout)),
        None,
        Some(RawTensorViewMut::new(&mut destination, layout)),
        i32_params(0, 0),
    );
    let mut kernel = Kernel::new();
    assert_eq!(kernel.process_event(GenericEvent::new(request)), Ok(()));
    assert_eq!(
        (0..4)
            .map(|index| read_f32(&destination, index))
            .collect::<Vec<_>>(),
        [3.0, 2.0, 9.0, 12.0]
    );

    let mut destination = [0_u8; 16];
    let request = GenericRequest::new(
        KernelOperation::Unary,
        Some(RawTensorView::new(&source, layout)),
        None,
        None,
        Some(RawTensorViewMut::new(&mut destination, layout)),
        OpParams::empty(),
    )
    .with_unary(emel_kernels::any::event::UnarySubOp::Abs);
    assert_eq!(kernel.process_event(GenericEvent::new(request)), Ok(()));
    assert_eq!(
        (0..4)
            .map(|index| read_f32(&destination, index))
            .collect::<Vec<_>>(),
        [1.0, 2.0, 3.0, 4.0]
    );
}

#[test]
fn generic_envelope_executes_rows_and_double_accumulation_norm() {
    let source = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
    let layout = GenericLayout::new(
        emel_kernels::any::event::GenericDType::F32,
        [2, 2, 1, 1],
        [4, 8, 16, 16],
    );
    let mut destination = [0_u8; 16];
    let request = GenericRequest::new(
        KernelOperation::SoftMax,
        Some(RawTensorView::new(&source, layout)),
        None,
        None,
        Some(RawTensorViewMut::new(&mut destination, layout)),
        OpParams::empty(),
    );
    let mut kernel = Kernel::new();
    assert_eq!(kernel.process_event(GenericEvent::new(request)), Ok(()));
    assert!((read_f32(&destination, 0) - 0.268_941_43).abs() < 1.0e-6);
    assert!((read_f32(&destination, 1) - 0.731_058_6).abs() < 1.0e-6);
    assert!((read_f32(&destination, 2) - 0.268_941_43).abs() < 1.0e-6);
    assert!((read_f32(&destination, 3) - 0.731_058_6).abs() < 1.0e-6);

    let epsilon = 1.0e-5_f32.to_ne_bytes();
    let mut params = [0_u8; 64];
    params[..4].copy_from_slice(&epsilon);
    let mut normalized = [0_u8; 16];
    let request = GenericRequest::new(
        KernelOperation::RmsNorm,
        Some(RawTensorView::new(&source, layout)),
        None,
        None,
        Some(RawTensorViewMut::new(&mut normalized, layout)),
        OpParams::new(params, 4).expect("epsilon parameter"),
    );
    assert_eq!(kernel.process_event(GenericEvent::new(request)), Ok(()));
    let scale = 1.0 / (2.5 + 1.0e-5_f32).sqrt();
    assert!((read_f32(&normalized, 0) - scale).abs() < 1.0e-6);
    assert!((read_f32(&normalized, 1) - scale - scale).abs() < 1.0e-6);
}

#[test]
fn generic_envelope_executes_pinned_f16_matmul() {
    let lhs = f16_bytes(&[0x3c00, 0x4000, 0x4200, 0x4400]);
    let rhs = f16_bytes(&[0x4500, 0x4600, 0x4700, 0x4800]);
    let lhs_layout = GenericLayout::new(
        emel_kernels::any::event::GenericDType::F16,
        [2, 2, 1, 1],
        [2, 4, 8, 8],
    );
    let rhs_layout = GenericLayout::new(
        emel_kernels::any::event::GenericDType::F16,
        [2, 2, 1, 1],
        [2, 4, 8, 8],
    );
    let destination_layout = GenericLayout::new(
        emel_kernels::any::event::GenericDType::F32,
        [2, 2, 1, 1],
        [4, 8, 16, 16],
    );
    let mut destination = [0_u8; 16];
    let request = GenericRequest::new(
        KernelOperation::MulMat,
        Some(RawTensorView::new(&lhs, lhs_layout)),
        Some(RawTensorView::new(&rhs, rhs_layout)),
        None,
        Some(RawTensorViewMut::new(&mut destination, destination_layout)),
        OpParams::empty(),
    );
    assert!(request.validate().is_ok());

    let mut kernel = Kernel::new();
    assert_eq!(kernel.process_event(GenericEvent::new(request)), Ok(()));
    assert!((read_f32(&destination, 0) - 17.0).abs() < 1.0e-6);
    assert!((read_f32(&destination, 1) - 39.0).abs() < 1.0e-6);
    assert!((read_f32(&destination, 2) - 23.0).abs() < 1.0e-6);
    assert!((read_f32(&destination, 3) - 53.0).abs() < 1.0e-6);
}

#[test]
fn generic_envelope_executes_scalar_fill_and_range_routes() {
    let source = f32_bytes(&[1.0, -2.0, 3.0]);
    let layout = f32_layout([3, 1, 1, 1]);
    let mut destination = [0_u8; 12];
    let request = GenericRequest::new(
        KernelOperation::Add1,
        Some(RawTensorView::new(&source, layout)),
        None,
        None,
        Some(RawTensorViewMut::new(&mut destination, layout)),
        f32_params(0, 2.0),
    );
    let mut kernel = Kernel::new();
    assert_eq!(kernel.process_event(GenericEvent::new(request)), Ok(()));
    assert_eq!(read_f32(&destination, 0), 3.0);
    assert_eq!(read_f32(&destination, 1), 0.0);
    assert_eq!(read_f32(&destination, 2), 5.0);

    let mut destination = [0_u8; 12];
    let request = GenericRequest::new(
        KernelOperation::Clamp,
        Some(RawTensorView::new(&source, layout)),
        None,
        None,
        Some(RawTensorViewMut::new(&mut destination, layout)),
        {
            let mut params = [0_u8; 64];
            params[..4].copy_from_slice(&(-1.0_f32).to_ne_bytes());
            params[4..8].copy_from_slice(&1.0_f32.to_ne_bytes());
            OpParams::new(params, 8).expect("clamp parameters")
        },
    );
    assert_eq!(kernel.process_event(GenericEvent::new(request)), Ok(()));
    assert_eq!(
        (0..3)
            .map(|index| read_f32(&destination, index))
            .collect::<Vec<_>>(),
        [1.0, -1.0, 1.0]
    );

    let mut destination = [0_u8; 12];
    let request = GenericRequest::new(
        KernelOperation::Fill,
        None,
        None,
        None,
        Some(RawTensorViewMut::new(&mut destination, layout)),
        f32_params(0, 7.5),
    );
    assert_eq!(kernel.process_event(GenericEvent::new(request)), Ok(()));
    assert_eq!(
        (0..3)
            .map(|index| read_f32(&destination, index))
            .collect::<Vec<_>>(),
        [7.5, 7.5, 7.5]
    );

    let mut destination = [0_u8; 12];
    let request = GenericRequest::new(
        KernelOperation::Arange,
        None,
        None,
        None,
        Some(RawTensorViewMut::new(&mut destination, layout)),
        {
            let mut params = [0_u8; 64];
            params[..4].copy_from_slice(&1.5_f32.to_ne_bytes());
            params[4..8].copy_from_slice(&0.5_f32.to_ne_bytes());
            OpParams::new(params, 8).expect("arange parameters")
        },
    );
    assert_eq!(kernel.process_event(GenericEvent::new(request)), Ok(()));
    assert_eq!(
        (0..3)
            .map(|index| read_f32(&destination, index))
            .collect::<Vec<_>>(),
        [1.5, 2.0, 2.5]
    );
}

#[test]
fn generic_envelope_executes_sequence_diag_and_mask_routes() {
    let source = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
    let layout = f32_layout([2, 2, 1, 1]);
    let mut destination = [0_u8; 16];
    let request = GenericRequest::new(
        KernelOperation::Cumsum,
        Some(RawTensorView::new(&source, layout)),
        None,
        None,
        Some(RawTensorViewMut::new(&mut destination, layout)),
        OpParams::empty(),
    );
    let mut kernel = Kernel::new();
    assert_eq!(kernel.process_event(GenericEvent::new(request)), Ok(()));
    assert_eq!(
        (0..4)
            .map(|index| read_f32(&destination, index))
            .collect::<Vec<_>>(),
        [1.0, 3.0, 3.0, 7.0]
    );

    let diagonal_source = f32_bytes(&[3.0, 4.0]);
    let mut diagonal = [0_u8; 16];
    let request = GenericRequest::new(
        KernelOperation::Diag,
        Some(RawTensorView::new(
            &diagonal_source,
            f32_layout([2, 1, 1, 1]),
        )),
        None,
        None,
        Some(RawTensorViewMut::new(
            &mut diagonal,
            f32_layout([2, 2, 1, 1]),
        )),
        OpParams::empty(),
    );
    assert_eq!(kernel.process_event(GenericEvent::new(request)), Ok(()));
    assert_eq!(
        (0..4)
            .map(|index| read_f32(&diagonal, index))
            .collect::<Vec<_>>(),
        [3.0, 0.0, 0.0, 4.0]
    );

    let mask_source = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
    let mut masked = [0_u8; 16];
    let request = GenericRequest::new(
        KernelOperation::DiagMaskInf,
        Some(RawTensorView::new(&mask_source, layout)),
        None,
        None,
        Some(RawTensorViewMut::new(&mut masked, layout)),
        i32_params(0, 0),
    );
    assert_eq!(kernel.process_event(GenericEvent::new(request)), Ok(()));
    assert_eq!(read_f32(&masked, 0), 1.0);
    assert!(read_f32(&masked, 1).is_infinite());
    assert_eq!(read_f32(&masked, 2), 3.0);
    assert_eq!(read_f32(&masked, 3), 4.0);

    let lhs = f32_bytes(&[1.0, 2.0]);
    let rhs = f32_bytes(&[3.0]);
    let mut concatenated = [0_u8; 12];
    let request = GenericRequest::new(
        KernelOperation::Concat,
        Some(RawTensorView::new(&lhs, f32_layout([2, 1, 1, 1]))),
        Some(RawTensorView::new(&rhs, f32_layout([1, 1, 1, 1]))),
        None,
        Some(RawTensorViewMut::new(
            &mut concatenated,
            f32_layout([3, 1, 1, 1]),
        )),
        i32_params(0, 0),
    );
    assert_eq!(kernel.process_event(GenericEvent::new(request)), Ok(()));
    assert_eq!(
        (0..3)
            .map(|index| read_f32(&concatenated, index))
            .collect::<Vec<_>>(),
        [1.0, 2.0, 3.0]
    );
}

#[test]
fn generic_envelope_executes_backward_norm_and_comparison_routes() {
    let layout = f32_layout([2, 2, 1, 1]);
    let values = f32_bytes(&[0.25, 0.75, 0.5, 0.5]);
    let gradients = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
    let mut softmax_gradient = [0_u8; 16];
    let request = GenericRequest::new(
        KernelOperation::SoftMaxBack,
        Some(RawTensorView::new(&values, layout)),
        Some(RawTensorView::new(&gradients, layout)),
        None,
        Some(RawTensorViewMut::new(&mut softmax_gradient, layout)),
        i32_params(0, 2),
    );
    let mut kernel = Kernel::new();
    assert_eq!(kernel.process_event(GenericEvent::new(request)), Ok(()));
    assert!((read_f32(&softmax_gradient, 0) + 0.1875).abs() < 1.0e-6);
    assert!((read_f32(&softmax_gradient, 1) - 0.1875).abs() < 1.0e-6);
    assert!((read_f32(&softmax_gradient, 2) + 0.25).abs() < 1.0e-6);
    assert!((read_f32(&softmax_gradient, 3) - 0.25).abs() < 1.0e-6);

    let source = f32_bytes(&[3.0, 4.0]);
    let mut normalized = [0_u8; 8];
    let vector_layout = f32_layout([2, 1, 1, 1]);
    let request = GenericRequest::new(
        KernelOperation::L2Norm,
        Some(RawTensorView::new(&source, vector_layout)),
        None,
        None,
        Some(RawTensorViewMut::new(&mut normalized, vector_layout)),
        OpParams::empty(),
    );
    assert_eq!(kernel.process_event(GenericEvent::new(request)), Ok(()));
    assert!((read_f32(&normalized, 0) - 0.6).abs() < 1.0e-6);
    assert!((read_f32(&normalized, 1) - 0.8).abs() < 1.0e-6);

    let mut equal = [0_u8; 4];
    let request = GenericRequest::new(
        KernelOperation::CountEqual,
        Some(RawTensorView::new(&source, vector_layout)),
        Some(RawTensorView::new(&source, vector_layout)),
        None,
        Some(RawTensorViewMut::new(&mut equal, f32_layout([1, 1, 1, 1]))),
        OpParams::empty(),
    );
    assert_eq!(kernel.process_event(GenericEvent::new(request)), Ok(()));
    assert_eq!(read_f32(&equal, 0), 2.0);
}

#[test]
fn generic_envelope_rejects_bad_span_before_unsupported_outcome() {
    let source = [0_u8; 4];
    let mut destination = [0_u8; 4];
    let layout = GenericLayout::new(
        emel_kernels::any::event::GenericDType::F32,
        [4, 1, 1, 1],
        [4, 16, 64, 256],
    );
    let request = GenericRequest::new(
        KernelOperation::Add,
        Some(RawTensorView::new(&source, layout)),
        Some(RawTensorView::new(&source, layout)),
        Some(RawTensorView::new(&source, layout)),
        Some(RawTensorViewMut::new(&mut destination, layout)),
        OpParams::empty(),
    );
    assert_eq!(request.validate(), Err(GenericViewError::OutOfBounds));

    let mut kernel = Kernel::new();
    assert_eq!(
        kernel.process_event(GenericEvent::new(request)),
        Err(Error::InvalidShape)
    );
    assert!(kernel.is_ready());
}

#[test]
fn generic_envelope_requires_safe_argmax_destination() {
    let source = [0_u8; 16];
    let mut destination = [0_u8; 16];
    let layout = layout();
    let request = GenericRequest::new(
        KernelOperation::MulMatArgmax,
        Some(RawTensorView::new(&source, layout)),
        Some(RawTensorView::new(&source, layout)),
        Some(RawTensorView::new(&source, layout)),
        Some(RawTensorViewMut::new(&mut destination, layout)),
        OpParams::empty(),
    );
    assert_eq!(
        request.validate(),
        Err(GenericViewError::MissingArgmaxDestination)
    );
}
