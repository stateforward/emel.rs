#![no_main]

use emel_kernels::any::event;
use emel_kernels::{Error, Kernel};
use libfuzzer_sys::fuzz_target;

const MAX_ELEMENTS: usize = 64;

fuzz_target!(|data: &[u8]| {
    let lhs_len = length(data, 0);
    let rhs_len = length(data, 1);
    let output_len = length(data, 2);
    let input_len = length(data, 3);

    let lhs = values(data, 4, lhs_len);
    let rhs = values(data, 4 + lhs_len * 4, rhs_len);
    let input = values(data, 4 + (lhs_len + rhs_len) * 4, input_len);
    let mut output = values(data, 4 + (lhs_len + rhs_len + input_len) * 4, output_len);
    let mut kernel = Kernel::new();

    let binary_valid = lhs_len > 0 && lhs_len == rhs_len && lhs_len == output_len;
    assert_eq!(
        kernel
            .process_event(event::OpAdd::new(&lhs, &rhs, &mut output))
            .is_ok(),
        binary_valid
    );
    assert_eq!(
        kernel
            .process_event(event::OpSub::new(&lhs, &rhs, &mut output))
            .is_ok(),
        binary_valid
    );
    assert_eq!(
        kernel
            .process_event(event::OpMul::new(&lhs, &rhs, &mut output))
            .is_ok(),
        binary_valid
    );
    assert_eq!(
        kernel
            .process_event(event::OpDiv::new(&lhs, &rhs, &mut output))
            .is_ok(),
        binary_valid
    );

    let unary_valid = input_len > 0 && input_len == output_len;
    assert_eq!(
        kernel
            .process_event(event::OpDup::new(&input, &mut output))
            .is_ok(),
        unary_valid
    );
    assert_eq!(
        kernel
            .process_event(event::OpSqr::new(&input, &mut output))
            .is_ok(),
        unary_valid
    );
    assert_eq!(
        kernel
            .process_event(event::OpSqrt::new(&input, &mut output))
            .is_ok(),
        unary_valid
    );

    let operation = event::UnsupportedOperation::MulMat;
    assert!(matches!(
        kernel.process_event(event::Unsupported::new(operation)),
        Err(Error::UnsupportedOperation(reported)) if reported == operation
    ));
});

fn length(data: &[u8], index: usize) -> usize {
    usize::from(data.get(index).copied().unwrap_or_default()) % (MAX_ELEMENTS + 1)
}

fn values(data: &[u8], offset: usize, count: usize) -> Vec<f32> {
    (0..count)
        .map(|index| {
            let start = offset.saturating_add(index.saturating_mul(4));
            let mut bits = [0_u8; 4];
            for (byte_index, byte) in bits.iter_mut().enumerate() {
                *byte = data
                    .get(start.saturating_add(byte_index))
                    .copied()
                    .unwrap_or_default();
            }
            f32::from_bits(u32::from_le_bytes(bits))
        })
        .collect()
}
