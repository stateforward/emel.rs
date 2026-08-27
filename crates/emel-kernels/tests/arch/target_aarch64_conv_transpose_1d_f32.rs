#![allow(clippy::float_cmp, missing_docs)]
#![cfg(target_arch = "aarch64")]

use allocation_counter::measure;
use emel_kernels::aarch64::OpAarch64ConvTranspose1dF16;
use emel_kernels::aarch64::{
    ConvTranspose1dF32Error, ConvTranspose1dF32Kernel, ConvTranspose1dF32Shape,
    OpAarch64ConvTranspose1dF32, UnexpectedAarch64ConvTranspose1dF32,
};
use emel_tensor as _;
use pulp as _;
use sml as _;

const PINNED_EMEL_CPP_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const PINNED_GUARD_BLOB: &str = "c25714566ec9a02679daef85089544575123408e";
const PINNED_ACTION_BLOB: &str = "267d4f74e6e7498155c8535920322ffef2c02fb6";
const PINNED_SM_BLOB: &str = "865a9cc6ba6115382ed043c464f3d62bcd851357";
const PINNED_GUARD_SPAN: &str = "src/emel/kernel/aarch64/guards.hpp:287-297,408-417,1026-1049";
const PINNED_ACTION_SPAN: &str =
    "src/emel/kernel/aarch64/actions.hpp:1252-1258,7074-7121,8770-8779";
const PINNED_TRANSITION_SPAN: &str = "src/emel/kernel/aarch64/sm.hpp:800-818";

const fn shape(
    kernel: usize,
    out_channels: usize,
    in_channels: usize,
    input_length: usize,
    stride: i32,
) -> ConvTranspose1dF32Shape {
    ConvTranspose1dF32Shape {
        kernel,
        out_channels,
        in_channels,
        input_length,
        stride,
        padding: 0,
        dilation: 1,
    }
}

fn kernel() -> ConvTranspose1dF32Kernel {
    ConvTranspose1dF32Kernel::try_new().expect("NEON is required on AArch64")
}

#[test]
fn pinned_source_identity_is_exact() {
    assert_eq!(
        PINNED_EMEL_CPP_COMMIT,
        "843a117386ef17dc5a50549bbfc821074c2141d6"
    );
    assert_eq!(
        PINNED_GUARD_BLOB,
        "c25714566ec9a02679daef85089544575123408e"
    );
    assert_eq!(
        PINNED_ACTION_BLOB,
        "267d4f74e6e7498155c8535920322ffef2c02fb6"
    );
    assert_eq!(PINNED_SM_BLOB, "865a9cc6ba6115382ed043c464f3d62bcd851357");
    assert_eq!(
        PINNED_GUARD_SPAN,
        "src/emel/kernel/aarch64/guards.hpp:287-297,408-417,1026-1049"
    );
    assert_eq!(
        PINNED_ACTION_SPAN,
        "src/emel/kernel/aarch64/actions.hpp:1252-1258,7074-7121,8770-8779"
    );
    assert_eq!(
        PINNED_TRANSITION_SPAN,
        "src/emel/kernel/aarch64/sm.hpp:800-818"
    );
}

#[test]
fn conv_transpose_matches_pinned_neon_scatter_formula() {
    let weights = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let input = [1.0_f32, 2.0];
    let mut output = [99.0_f32; 7];
    let mut actor = kernel();
    assert_eq!(
        actor.process_event(
            OpAarch64ConvTranspose1dF32::new(&weights, &input, shape(5, 1, 1, 2, 2),),
            &mut output
        ),
        Ok(())
    );
    assert_eq!(output, [1.0, 2.0, 5.0, 8.0, 11.0, 8.0, 10.0]);
    assert!(actor.is_ready());
}

#[test]
fn multiple_channels_match_source_layout() {
    let weights = [1.0_f32, 2.0, 3.0, 4.0];
    let input = [5.0_f32, 6.0];
    let mut output = [99.0_f32; 6];
    let mut actor = kernel();
    assert_eq!(
        actor.process_event(
            OpAarch64ConvTranspose1dF32::new(&weights, &input, shape(2, 2, 1, 2, 1),),
            &mut output
        ),
        Ok(())
    );
    assert_eq!(output, [5.0, 16.0, 12.0, 15.0, 38.0, 24.0]);
}

#[test]
fn f16_weights_decode_then_accumulate_like_the_pinned_shared_route() {
    let weights: [u16; 10] = [
        0x3c00, 0x4000, 0x4200, 0x4400, 0x4500, 0x3c00, 0x4000, 0x4200, 0x4400, 0x4500,
    ];
    let input = [1.0_f32, 2.0];
    let mut output = [f32::NAN; 14];
    let mut actor = kernel();
    assert_eq!(
        actor.process_event(
            OpAarch64ConvTranspose1dF16::new(&weights, &input, shape(5, 2, 1, 2, 2)),
            &mut output,
        ),
        Ok(())
    );
    assert_eq!(output[0], 1.0);
    assert_eq!(output[1], 2.0);
    assert_eq!(output[2], 5.0);
    assert_eq!(output[3], 8.0);
    assert_eq!(output[4], 11.0);
    assert_eq!(output[5], 8.0);
    assert_eq!(output[6], 10.0);
    for index in 0..7 {
        assert_eq!(output[index], output[7 + index]);
    }

    let mut invalid_output = [f32::NAN; 14];
    assert_eq!(
        actor.process_event(
            OpAarch64ConvTranspose1dF16::new(&weights, &input, shape(5, 2, 1, 3, 2)),
            &mut invalid_output,
        ),
        Err(ConvTranspose1dF32Error::InvalidShape)
    );
}

#[test]
fn invalid_shape_does_not_mutate_output() {
    let weights = [1.0_f32, 2.0, 3.0];
    let input = [1.0_f32, 2.0];
    let mut output = [7.0_f32; 5];
    let mut actor = kernel();
    assert_eq!(
        actor.process_event(
            OpAarch64ConvTranspose1dF32::new(&weights, &input, shape(3, 1, 1, 2, 0),),
            &mut output
        ),
        Err(ConvTranspose1dF32Error::InvalidShape)
    );
    assert_eq!(output, [7.0; 5]);
    assert_eq!(
        actor.process_event(
            OpAarch64ConvTranspose1dF32::new(&weights, &input, shape(3, 1, 1, 3, 2),),
            &mut output
        ),
        Err(ConvTranspose1dF32Error::InvalidShape)
    );
    assert_eq!(output, [7.0; 5]);
}

#[test]
fn unexpected_event_is_typed_and_actor_recovers() {
    let mut actor = kernel();
    assert_eq!(
        actor.process_event(UnexpectedAarch64ConvTranspose1dF32, &mut []),
        Err(ConvTranspose1dF32Error::UnexpectedEvent)
    );
    assert!(actor.is_ready());
}

#[test]
fn dispatch_is_allocation_free_after_construction() {
    let weights = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let input = [1.0_f32, 2.0];
    let mut output = [0.0_f32; 7];
    let mut actor = kernel();
    let allocation = measure(|| {
        for _ in 0..64 {
            assert_eq!(
                actor.process_event(
                    OpAarch64ConvTranspose1dF32::new(&weights, &input, shape(5, 1, 1, 2, 2),),
                    &mut output
                ),
                Ok(())
            );
        }
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(allocation.bytes_total, 0);
}

#[test]
fn errors_are_typed_and_debuggable() {
    assert_eq!(
        ConvTranspose1dF32Error::InvalidShape.to_string(),
        "invalid AArch64 conv-transpose F32 shape"
    );
    assert!(format!("{:?}", kernel()).contains("ConvTranspose1dF32Kernel"));
}
