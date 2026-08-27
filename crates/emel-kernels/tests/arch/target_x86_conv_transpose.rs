#![allow(clippy::float_cmp, missing_docs)]
#![cfg(target_arch = "x86_64")]

use allocation_counter::measure;
use emel_kernels::x86_64::{
    UnexpectedX86ConvTranspose, X86ConvTranspose1dF32Error, X86ConvTranspose1dF32Shape,
    X86ConvTransposeF32, X86Kernel,
};

const PINNED_EMEL_CPP_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const PINNED_GUARD_BLOB: &str = "cb3dac8253f8417c9b44acff1de414f6d0a3a3cf";
const PINNED_ACTION_BLOB: &str = "d45558f5eb96950f43c16a09d768cb4f382d6d61";
const PINNED_SM_BLOB: &str = "0b4d635ebbd0fbd52dbca8a2345547fb571205c8";
const PINNED_GUARD_SPAN: &str = "src/emel/kernel/x86_64/guards.hpp:182-198,399-404";
const PINNED_ACTION_SPAN: &str = "src/emel/kernel/x86_64/actions.hpp:2642-2645,2733-2734";
const PINNED_TRANSITION_SPAN: &str = "src/emel/kernel/x86_64/sm.hpp:680-693";

const fn shape(
    kernel: usize,
    out_channels: usize,
    in_channels: usize,
    input_length: usize,
    stride: i32,
) -> X86ConvTranspose1dF32Shape {
    X86ConvTranspose1dF32Shape {
        kernel,
        out_channels,
        in_channels,
        input_length,
        stride,
        padding: 0,
        dilation: 1,
    }
}

fn kernel() -> Option<X86Kernel> {
    // Cross-compilation can execute on emulated hosts that do not expose the
    // pinned AVX2/FMA feature set; those runtime checks self-skip.
    X86Kernel::try_new()
}

#[test]
fn pinned_source_identity_is_exact() {
    assert_eq!(
        PINNED_EMEL_CPP_COMMIT,
        "843a117386ef17dc5a50549bbfc821074c2141d6"
    );
    assert_eq!(
        PINNED_GUARD_BLOB,
        "cb3dac8253f8417c9b44acff1de414f6d0a3a3cf"
    );
    assert_eq!(
        PINNED_ACTION_BLOB,
        "d45558f5eb96950f43c16a09d768cb4f382d6d61"
    );
    assert_eq!(PINNED_SM_BLOB, "0b4d635ebbd0fbd52dbca8a2345547fb571205c8");
    assert_eq!(
        PINNED_GUARD_SPAN,
        "src/emel/kernel/x86_64/guards.hpp:182-198,399-404"
    );
    assert_eq!(
        PINNED_ACTION_SPAN,
        "src/emel/kernel/x86_64/actions.hpp:2642-2645,2733-2734"
    );
    assert_eq!(
        PINNED_TRANSITION_SPAN,
        "src/emel/kernel/x86_64/sm.hpp:680-693"
    );
}

#[test]
fn target_router_executes_f32_transposed_convolution() {
    let weights = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let input = [1.0_f32, 2.0];
    let mut output = [99.0_f32; 7];
    let Some(mut actor) = kernel() else {
        return;
    };
    assert_eq!(
        actor.process_event(
            X86ConvTransposeF32::new(&weights, &input, shape(5, 1, 1, 2, 2)),
            &mut output
        ),
        Ok(())
    );
    assert!(actor.is_ready());
    assert_eq!(output, [1.0, 2.0, 5.0, 8.0, 11.0, 8.0, 10.0]);
}

#[test]
fn multiple_channels_follow_the_pinned_layout() {
    let weights = [1.0_f32, 2.0, 3.0, 4.0];
    let input = [5.0_f32, 6.0];
    let mut output = [99.0_f32; 6];
    let Some(mut actor) = kernel() else {
        return;
    };
    assert_eq!(
        actor.process_event(
            X86ConvTransposeF32::new(&weights, &input, shape(2, 2, 1, 2, 1)),
            &mut output
        ),
        Ok(())
    );
    assert_eq!(output, [5.0, 16.0, 12.0, 15.0, 38.0, 24.0]);
}

#[test]
fn invalid_shape_does_not_mutate_output_and_recovers() {
    let weights = [1.0_f32, 2.0, 3.0];
    let input = [1.0_f32, 2.0];
    let mut output = [7.0_f32; 5];
    let Some(mut actor) = kernel() else {
        return;
    };
    assert_eq!(
        actor.process_event(
            X86ConvTransposeF32::new(&weights, &input, shape(3, 1, 1, 2, 0)),
            &mut output
        ),
        Err(X86ConvTranspose1dF32Error::InvalidShape)
    );
    assert_eq!(
        actor.process_event(
            X86ConvTransposeF32::new(&weights, &input, shape(3, 1, 1, 3, 2)),
            &mut output
        ),
        Err(X86ConvTranspose1dF32Error::InvalidShape)
    );
    assert_eq!(output, [7.0; 5]);
    assert!(actor.is_ready());
}

#[test]
fn unexpected_event_is_explicit_and_recovers() {
    let Some(mut actor) = kernel() else {
        return;
    };
    assert_eq!(
        actor.process_event(UnexpectedX86ConvTranspose, &mut []),
        Err(X86ConvTranspose1dF32Error::UnexpectedEvent)
    );
    assert!(actor.is_ready());
}

#[test]
fn composed_dispatch_is_allocation_free_after_construction() {
    let weights = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let input = [1.0_f32, 2.0];
    let mut output = [0.0_f32; 7];
    let Some(mut actor) = kernel() else {
        return;
    };
    let allocation = measure(|| {
        for _ in 0..64 {
            assert_eq!(
                actor.process_event(
                    X86ConvTransposeF32::new(&weights, &input, shape(5, 1, 1, 2, 2)),
                    &mut output
                ),
                Ok(())
            );
        }
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(allocation.bytes_total, 0);
}
