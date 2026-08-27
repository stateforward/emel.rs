#![allow(clippy::cast_precision_loss, clippy::float_cmp, missing_docs)]

use allocation_counter::measure;
use emel_kernels::any::rope::{
    I32View, OpRope, ROPE_MODE_NEOX, ROPE_MODE_NORM, ROPE_MODE_TIMESTEP, RopeError, RopeKernel,
    RopeParams, UnexpectedRope,
};
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};
use emel_tensor as _;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use pulp as _;
use sml as _;

const PINNED_EMEL_CPP_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";
const PINNED_DETAIL_BLOB: &str = "c8a82643eabfe8f2d7883e655955f455794511b0";
const PINNED_X86_SM_BLOB: &str = "0b4d635ebbd0fbd52dbca8a2345547fb571205c8";
const PINNED_X86_GUARDS_BLOB: &str = "cb3dac8253f8417c9b44acff1de414f6d0a3a3cf";

fn f32_layout(ne: [u64; 4]) -> Layout {
    Layout::contiguous(DType::F32, ne).expect("F32 layout fits")
}

fn i32_layout(ne: [u64; 4]) -> Layout {
    Layout::contiguous(DType::I32, ne).expect("I32 layout fits")
}

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

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
    }
}

// Keep the pinned rotation oracle's separate products and subtraction order.
#[allow(clippy::suboptimal_flops)]
fn expected_rotation<const COLUMNS: usize, const NEOX: bool>(
    input: [f32; COLUMNS],
    position: f32,
) -> [f32; COLUMNS] {
    let mut output = input;
    let theta_scale = 10_000.0_f32.powf(-2.0 / 4.0);
    let mut theta = position;
    let mut pair = 0;
    while pair < 2 {
        let angle = theta;
        let cosine = angle.cos();
        let sine = angle.sin();
        let (a, b) = if NEOX {
            (pair, pair + 2)
        } else {
            (2 * pair, 2 * pair + 1)
        };
        output[a] = input[a] * cosine - input[b] * sine;
        output[b] = input[a] * sine + input[b] * cosine;
        theta *= theta_scale;
        pair += 1;
    }
    output
}

#[test]
fn source_identity_and_mode_rows_match_pinned_contract() {
    assert_eq!(PINNED_EMEL_CPP_COMMIT.len(), 40);
    assert_eq!(PINNED_DETAIL_BLOB.len(), 40);
    assert_eq!(PINNED_X86_SM_BLOB.len(), 40);
    assert_eq!(PINNED_X86_GUARDS_BLOB.len(), 40);
    assert_eq!(ROPE_MODE_NORM, 0);
    assert_eq!(ROPE_MODE_NEOX, 2);
    assert_eq!(ROPE_MODE_TIMESTEP, 4);
}

#[test]
fn norm_mode_matches_pinned_pairing_and_copies_tail() {
    let source_layout = f32_layout([6, 1, 2, 1]);
    let position_layout = i32_layout([2, 1, 1, 1]);
    let destination_layout = f32_layout([6, 1, 2, 1]);
    let source = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let positions = [1_i32, 2_i32];
    let mut destination = [0.0_f32; 12];
    let expected0 = expected_rotation::<6, false>([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1.0);
    let expected1 = expected_rotation::<6, false>([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 2.0);
    let mut actor = RopeKernel::new();

    assert_eq!(
        actor.process_event(OpRope::new(
            TensorView::new(&source, source_layout),
            I32View::new(&positions, position_layout),
            TensorViewMut::new(&mut destination, destination_layout),
            params(ROPE_MODE_NORM),
        )),
        Ok(())
    );
    assert_close(&destination[..6], &expected0);
    assert_close(&destination[6..], &expected1);
}

#[test]
fn neox_mode_matches_split_half_pairing() {
    let layout = f32_layout([4, 1, 1, 1]);
    let position_layout = i32_layout([1, 1, 1, 1]);
    let source = [1.0_f32, 2.0, 3.0, 4.0];
    let positions = [1_i32];
    let mut destination = [0.0_f32; 4];
    let expected = expected_rotation::<4, true>(source, 1.0);
    let mut actor = RopeKernel::new();

    assert_eq!(
        actor.process_event(OpRope::new(
            TensorView::new(&source, layout),
            I32View::new(&positions, position_layout),
            TensorViewMut::new(&mut destination, layout),
            params(ROPE_MODE_NEOX),
        )),
        Ok(())
    );
    assert_close(&destination, &expected);
}

#[test]
fn timestep_mode_writes_split_real_and_imaginary_halves() {
    let layout = f32_layout([4, 1, 1, 1]);
    let position_layout = i32_layout([1, 1, 1, 1]);
    let source = [1.0_f32, 2.0, 3.0, 4.0];
    let positions = [1_i32];
    let mut destination = [0.0_f32; 4];
    let mut expected = [0.0_f32; 4];
    let half = 2;
    let mut pair = 0;
    while pair < half {
        let frequency = (-(10_000.0_f32).ln() * pair as f32 / half as f32).exp();
        let angle = frequency;
        let cosine = angle.cos();
        let sine = angle.sin();
        let real = source[2 * pair];
        let imaginary = source[2 * pair + 1];
        expected[pair] = real * cosine - imaginary * sine;
        expected[half + pair] = real * sine + imaginary * cosine;
        pair += 1;
    }
    let mut actor = RopeKernel::new();

    assert_eq!(
        actor.process_event(OpRope::new(
            TensorView::new(&source, layout),
            I32View::new(&positions, position_layout),
            TensorViewMut::new(&mut destination, layout),
            params(ROPE_MODE_TIMESTEP),
        )),
        Ok(())
    );
    assert_close(&destination, &expected);
}

#[test]
fn guards_classify_mode_parameters_shapes_and_views() {
    let layout = f32_layout([4, 1, 1, 1]);
    let position_layout = i32_layout([1, 1, 1, 1]);
    let source = [1.0_f32, 2.0, 3.0, 4.0];
    let positions = [1_i32];
    let mut destination = [9.0_f32; 4];
    let mut actor = RopeKernel::new();

    let mut invalid_mode = params(99);
    assert_eq!(
        actor.process_event(OpRope::new(
            TensorView::new(&source, layout),
            I32View::new(&positions, position_layout),
            TensorViewMut::new(&mut destination, layout),
            invalid_mode,
        )),
        Err(RopeError::InvalidMode)
    );
    assert_eq!(destination, [9.0; 4]);

    invalid_mode = params(ROPE_MODE_NORM);
    invalid_mode.freq_base = 0.0;
    assert_eq!(
        actor.process_event(OpRope::new(
            TensorView::new(&source, layout),
            I32View::new(&positions, position_layout),
            TensorViewMut::new(&mut destination, layout),
            invalid_mode,
        )),
        Err(RopeError::InvalidParameters)
    );
    assert_eq!(destination, [9.0; 4]);

    let mismatch = f32_layout([2, 1, 1, 1]);
    assert_eq!(
        actor.process_event(OpRope::new(
            TensorView::new(&source, layout),
            I32View::new(&positions, position_layout),
            TensorViewMut::new(&mut destination[..2], mismatch),
            params(ROPE_MODE_NORM),
        )),
        Err(RopeError::ShapeMismatch)
    );
    assert_eq!(destination, [9.0; 4]);

    let bad_positions = Layout::contiguous(DType::F32, [1, 1, 1, 1]).expect("layout fits");
    assert_eq!(
        actor.process_event(OpRope::new(
            TensorView::new(&source, layout),
            I32View::new(&positions, bad_positions),
            TensorViewMut::new(&mut destination, layout),
            params(ROPE_MODE_NORM),
        )),
        Err(RopeError::InvalidView)
    );
    assert_eq!(destination, [9.0; 4]);
}

#[test]
fn timestep_requires_the_pinned_dense_layout() {
    let source_layout = Layout::new(DType::F32, [4, 1, 1, 1], [8, 8, 32, 32]);
    let destination_layout = f32_layout([4, 1, 1, 1]);
    let position_layout = i32_layout([1, 1, 1, 1]);
    let source = [1.0_f32, 2.0, 3.0, 4.0, 99.0, 98.0, 97.0, 96.0];
    let positions = [1_i32];
    let mut destination = [9.0_f32; 4];
    let mut actor = RopeKernel::new();

    assert_eq!(
        actor.process_event(OpRope::new(
            TensorView::new(&source, source_layout),
            I32View::new(&positions, position_layout),
            TensorViewMut::new(&mut destination, destination_layout),
            params(ROPE_MODE_TIMESTEP),
        )),
        Err(RopeError::ShapeMismatch)
    );
    assert_eq!(destination, [9.0; 4]);
}

#[test]
fn unexpected_event_is_typed_and_actor_recovers() {
    let layout = f32_layout([4, 1, 1, 1]);
    let position_layout = i32_layout([1, 1, 1, 1]);
    let source = [1.0_f32, 2.0, 3.0, 4.0];
    let positions = [1_i32];
    let mut destination = [0.0_f32; 4];
    let mut actor = RopeKernel::new();

    assert_eq!(
        actor.process_event(UnexpectedRope),
        Err(RopeError::UnexpectedEvent)
    );
    assert_eq!(
        actor.process_event(OpRope::new(
            TensorView::new(&source, layout),
            I32View::new(&positions, position_layout),
            TensorViewMut::new(&mut destination, layout),
            params(ROPE_MODE_NORM),
        )),
        Ok(())
    );
}

#[test]
fn dispatch_is_allocation_free() {
    let layout = f32_layout([4, 1, 1, 1]);
    let position_layout = i32_layout([1, 1, 1, 1]);
    let source = [1.0_f32, 2.0, 3.0, 4.0];
    let positions = [1_i32];
    let mut destination = [0.0_f32; 4];
    let mut actor = RopeKernel::new();

    let allocation = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                actor.process_event(OpRope::new(
                    TensorView::new(&source, layout),
                    I32View::new(&positions, position_layout),
                    TensorViewMut::new(&mut destination, layout),
                    params(ROPE_MODE_NORM),
                )),
                Ok(())
            );
        }
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(allocation.bytes_total, 0);
}

#[test]
fn public_surfaces_are_typed_and_debuggable() {
    assert!(format!("{}", RopeError::InvalidView).contains("view"));
    assert!(format!("{}", RopeError::ShapeMismatch).contains("shapes"));
    assert!(format!("{}", RopeError::InvalidParameters).contains("parameters"));
    assert!(format!("{}", RopeError::InvalidMode).contains("mode"));
    assert!(format!("{}", RopeError::UnexpectedEvent).contains("unexpected"));
    assert!(format!("{:?}", RopeKernel::default()).contains("RopeKernel"));
}
