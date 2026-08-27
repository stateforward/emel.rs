#![allow(clippy::float_cmp, missing_docs)]

use allocation_counter::measure;
use emel_kernels::Kernel;
use emel_kernels::any::event::{
    Event, F16View, FlashAttnError, FlashAttnOptions, OpFlashAttnExt, UnexpectedFlashAttn,
};
use emel_kernels::any::tensor_view::{DType, Layout, TensorView, TensorViewMut};
use emel_tensor as _;
use pulp as _;
use sml as _;

fn f32_layout(shape: [u64; 4]) -> Layout {
    Layout::contiguous(DType::F32, shape).expect("F32 layout fits")
}

const fn f16_layout(shape: [u64; 4]) -> Layout {
    let nb0: u64 = 2;
    let nb1 = nb0.checked_mul(shape[0]).expect("F16 stride fits");
    let nb2 = nb1.checked_mul(shape[1]).expect("F16 stride fits");
    let nb3 = nb2.checked_mul(shape[2]).expect("F16 stride fits");
    Layout::new(DType::F16, shape, [nb0, nb1, nb2, nb3])
}

fn request<'a>(
    query: &'a [f32],
    key: &'a [u16],
    value: &'a [u16],
    output: &'a mut [f32],
) -> OpFlashAttnExt<'a> {
    OpFlashAttnExt::new(
        TensorView::new(query, f32_layout([2, 1, 2, 1])),
        F16View::new(key, f16_layout([2, 2, 1, 1])),
        F16View::new(value, f16_layout([2, 2, 1, 1])),
        TensorViewMut::new(output, f32_layout([2, 1, 2, 1])),
        FlashAttnOptions {
            scale: Some(1.0),
            masked_total_tokens: Some(2),
        },
    )
}

#[test]
fn root_forwards_flash_attention_to_the_public_child_actor() {
    let query = [1.0_f32, 0.0, 1.0, 0.0];
    let key = [0x3c00_u16, 0, 0, 0, 0, 0x3c00, 0, 0];
    let value = [
        0x4000_u16, 0x4400, 0x4600, 0x4800, 0x4000, 0x4400, 0x4600, 0x4800,
    ];
    let mut output = [0.0_f32; 4];
    let mut kernel = Kernel::new();

    assert_eq!(
        kernel.process_event(request(&query, &key, &value, &mut output)),
        Ok(())
    );
    assert!(output.iter().all(|value| value.is_finite()));
    assert!(output.iter().any(|value| *value != 0.0));
}

#[test]
fn root_rejects_invalid_flash_attention_without_mutating_output() {
    let query = [1.0_f32, 0.0, 1.0, 0.0];
    let key = [0x3c00_u16, 0, 0, 0, 0, 0x3c00, 0, 0];
    let value = [
        0x4000_u16, 0x4400, 0x4600, 0x4800, 0x4000, 0x4400, 0x4600, 0x4800,
    ];
    let mut output = [9.0_f32; 4];
    let mut kernel = Kernel::new();
    let event = OpFlashAttnExt::new(
        TensorView::new(&query, f32_layout([2, 1, 2, 1])),
        F16View::new(&key, f16_layout([2, 2, 1, 1])),
        F16View::new(&value, f16_layout([2, 2, 1, 1])),
        TensorViewMut::new(&mut output, f32_layout([2, 1, 2, 1])),
        FlashAttnOptions {
            scale: Some(0.0),
            masked_total_tokens: Some(2),
        },
    );

    assert_eq!(
        kernel.process_event(event),
        Err(FlashAttnError::InvalidRequest)
    );
    assert_eq!(output, [9.0; 4]);
}

#[test]
fn root_unexpected_flash_attention_is_typed_and_recovers() {
    let mut kernel = Kernel::new();
    assert_eq!(
        kernel.process_event(UnexpectedFlashAttn),
        Err(FlashAttnError::UnexpectedEvent)
    );

    let query = [1.0_f32, 0.0, 1.0, 0.0];
    let key = [0x3c00_u16, 0, 0, 0, 0, 0x3c00, 0, 0];
    let value = [
        0x4000_u16, 0x4400, 0x4600, 0x4800, 0x4000, 0x4400, 0x4600, 0x4800,
    ];
    let mut output = [0.0_f32; 4];
    assert_eq!(
        kernel.process_event(request(&query, &key, &value, &mut output)),
        Ok(())
    );
}

#[test]
fn root_flash_attention_dispatch_is_allocation_free_after_construction() {
    let query = [1.0_f32, 0.0, 1.0, 0.0];
    let key = [0x3c00_u16, 0, 0, 0, 0, 0x3c00, 0, 0];
    let value = [
        0x4000_u16, 0x4400, 0x4600, 0x4800, 0x4000, 0x4400, 0x4600, 0x4800,
    ];
    let mut output = [0.0_f32; 4];
    let mut kernel = Kernel::new();

    let allocation = measure(|| {
        for _ in 0..64 {
            assert_eq!(
                kernel.process_event(request(&query, &key, &value, &mut output)),
                Ok(())
            );
        }
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(allocation.bytes_total, 0);
}

#[test]
fn root_flash_attention_event_is_publicly_dispatchable() {
    fn dispatches<E: Event>(event: E, kernel: &mut Kernel) -> E::Output {
        event.dispatch(kernel)
    }

    let query = [1.0_f32, 0.0, 1.0, 0.0];
    let key = [0x3c00_u16, 0, 0, 0, 0, 0x3c00, 0, 0];
    let value = [
        0x4000_u16, 0x4400, 0x4600, 0x4800, 0x4000, 0x4400, 0x4600, 0x4800,
    ];
    let mut output = [0.0_f32; 4];
    let mut kernel = Kernel::new();

    assert_eq!(
        dispatches(request(&query, &key, &value, &mut output), &mut kernel),
        Ok(())
    );
}
