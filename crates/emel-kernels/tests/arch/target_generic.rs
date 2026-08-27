#![allow(clippy::float_cmp, missing_docs)]

#[cfg(target_arch = "x86_64")]
#[test]
fn x86_target_generic_event_joins_portable_child_rtc() {
    use emel_kernels::any::event::{GenericDType, GenericEvent, GenericLayout, GenericRequest};
    use emel_kernels::any::event::{KernelOperation, OpParams, RawTensorView, RawTensorViewMut};
    use emel_kernels::x86_64::{X86Generic, X86Kernel};

    let input = [1.0_f32.to_ne_bytes(), 2.0_f32.to_ne_bytes()]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let rhs = [3.0_f32.to_ne_bytes(), 4.0_f32.to_ne_bytes()]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let mut output = [0_u8; 8];
    let layout = GenericLayout::new(GenericDType::F32, [2, 1, 1, 1], [4, 8, 8, 8]);
    let request = GenericRequest::new(
        KernelOperation::Add,
        Some(RawTensorView::new(&input, layout)),
        Some(RawTensorView::new(&rhs, layout)),
        None,
        Some(RawTensorViewMut::new(&mut output, layout)),
        OpParams::empty(),
    );
    let mut actor = X86Kernel::try_new().expect("x86 target router");
    assert_eq!(
        actor.process_generic(X86Generic::new(GenericEvent::new(request))),
        Ok(())
    );
    assert_eq!(
        f32::from_ne_bytes(output[..4].try_into().expect("f32 bytes")),
        4.0
    );
    assert_eq!(
        f32::from_ne_bytes(output[4..8].try_into().expect("f32 bytes")),
        6.0
    );
    assert!(actor.is_ready());
}

#[cfg(target_arch = "x86_64")]
#[test]
fn x86_target_generic_router_classifies_every_pinned_operation() {
    use emel_kernels::any::event::{
        GENERIC_OPERATION_COUNT, GenericEvent, GenericRequest, KernelOperation, OpParams,
    };
    use emel_kernels::x86_64::{X86Generic, X86Kernel};

    let mut actor = X86Kernel::try_new().expect("x86 target router");
    let mut code = 0;
    while code < GENERIC_OPERATION_COUNT {
        let operation = KernelOperation::from_code(u8::try_from(code).expect("ordinal fits in u8"))
            .expect("pinned operation");
        let request = GenericRequest::new(operation, None, None, None, None, OpParams::empty());
        let result = actor.process_generic(X86Generic::new(GenericEvent::new(request)));
        assert!(
            result.is_err(),
            "empty envelope unexpectedly executed {operation:?}"
        );
        assert!(
            actor.is_ready(),
            "router did not return to ready after {operation:?}"
        );
        code += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn aarch64_target_generic_event_joins_portable_child_rtc() {
    use emel_kernels::aarch64::{Aarch64Generic, Kernel};
    use emel_kernels::any::event::{
        GenericDType, GenericEvent, GenericLayout, GenericRequest, KernelOperation, OpParams,
        RawTensorView, RawTensorViewMut,
    };

    let input = [1.0_f32.to_ne_bytes(), 2.0_f32.to_ne_bytes()]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let rhs = [3.0_f32.to_ne_bytes(), 4.0_f32.to_ne_bytes()]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let mut output = [0_u8; 8];
    let layout = GenericLayout::new(GenericDType::F32, [2, 1, 1, 1], [4, 8, 8, 8]);
    let request = GenericRequest::new(
        KernelOperation::Add,
        Some(RawTensorView::new(&input, layout)),
        Some(RawTensorView::new(&rhs, layout)),
        None,
        Some(RawTensorViewMut::new(&mut output, layout)),
        OpParams::empty(),
    );
    let mut actor = Kernel::try_new().expect("AArch64 target router");
    assert_eq!(
        actor.process_generic(Aarch64Generic::new(GenericEvent::new(request))),
        Ok(())
    );
    assert_eq!(
        f32::from_ne_bytes(output[..4].try_into().expect("f32 bytes")),
        4.0
    );
    assert_eq!(
        f32::from_ne_bytes(output[4..8].try_into().expect("f32 bytes")),
        6.0
    );
    assert!(actor.is_ready());
}

#[cfg(target_arch = "aarch64")]
#[test]
fn aarch64_target_generic_router_classifies_every_pinned_operation() {
    use emel_kernels::aarch64::{Aarch64Generic, Kernel};
    use emel_kernels::any::event::{
        GENERIC_OPERATION_COUNT, GenericEvent, GenericRequest, KernelOperation, OpParams,
    };

    let mut actor = Kernel::try_new().expect("AArch64 target router");
    let mut code = 0;
    while code < GENERIC_OPERATION_COUNT {
        let operation = KernelOperation::from_code(u8::try_from(code).expect("ordinal fits in u8"))
            .expect("pinned operation");
        let request = GenericRequest::new(operation, None, None, None, None, OpParams::empty());
        let result = actor.process_generic(Aarch64Generic::new(GenericEvent::new(request)));
        assert!(
            result.is_err(),
            "empty envelope unexpectedly executed {operation:?}"
        );
        assert!(
            actor.is_ready(),
            "router did not return to ready after {operation:?}"
        );
        code += 1;
    }
}
