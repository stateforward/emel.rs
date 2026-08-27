#![allow(missing_docs)]
#![allow(clippy::float_cmp)]

use allocation_counter::measure;
use emel_kernels::any::quant::{
    BLOCK_VALUES, Format, Q4_0_BLOCK_BYTES, Q4_0Row, Q8_0_BLOCK_BYTES, Q8_0Row, QuantError,
    dot_q4_0_q8_0, dot_q8_0_q8_0,
};
use emel_tensor as _;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use pulp as _;
use sml as _;

const PINNED_EMEL_CPP: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";

fn q4_block(scale: u16, low: u8, high: u8) -> [u8; Q4_0_BLOCK_BYTES] {
    let mut block = [0_u8; Q4_0_BLOCK_BYTES];
    block[..2].copy_from_slice(&scale.to_le_bytes());
    for packed in &mut block[2..] {
        *packed = (low & 0x0f) | ((high & 0x0f) << 4);
    }
    block
}

fn q8_block(scale: u16, value: i8) -> [u8; Q8_0_BLOCK_BYTES] {
    let mut block = [0_u8; Q8_0_BLOCK_BYTES];
    block[..2].copy_from_slice(&scale.to_le_bytes());
    for byte in &mut block[2..] {
        *byte = value.to_ne_bytes()[0];
    }
    block
}

#[test]
fn q4_decode_matches_reference_nibble_order() {
    assert_eq!(PINNED_EMEL_CPP, "843a117386ef17dc5a50549bbfc821074c2141d6");
    let mut encoded = q4_block(0x3c00, 0, 15).to_vec();
    let row = Q4_0Row::from_bytes(&encoded).expect("valid q4 row");
    let mut decoded = [0.0_f32; BLOCK_VALUES];
    row.decode_block(0, &mut decoded).expect("valid q4 block");
    assert_eq!(&decoded[..16], &[-8.0; 16]);
    assert_eq!(&decoded[16..], &[7.0; 16]);
    encoded.push(0);
    assert!(matches!(
        Q4_0Row::from_bytes(&encoded),
        Err(QuantError::InvalidBlockLength {
            format: Format::Q4_0,
            bytes
        }) if bytes == Q4_0_BLOCK_BYTES + 1
    ));
}

#[test]
fn q8_decode_uses_little_endian_binary16_scale() {
    let encoded = q8_block(0x3800, -2);
    let row = Q8_0Row::from_bytes(&encoded).expect("valid q8 row");
    let mut decoded = [0.0_f32; BLOCK_VALUES];
    row.decode_block(0, &mut decoded).expect("valid q8 block");
    assert_eq!(decoded, [-1.0; BLOCK_VALUES]);
}

#[test]
fn q4_q8_dot_matches_integer_accumulation_and_scale() {
    let lhs = q4_block(0x3c00, 9, 7);
    let rhs = q8_block(0x3800, 2);
    let result = dot_q4_0_q8_0(
        Q4_0Row::from_bytes(&lhs).unwrap(),
        Q8_0Row::from_bytes(&rhs).unwrap(),
    )
    .unwrap();
    assert_eq!(result, 0.0);
}

#[test]
fn q8_q8_dot_matches_reference_scalar_kernel() {
    let lhs = q8_block(0x3c00, 3);
    let rhs = q8_block(0x3800, -2);
    let result = dot_q8_0_q8_0(
        Q8_0Row::from_bytes(&lhs).unwrap(),
        Q8_0Row::from_bytes(&rhs).unwrap(),
    )
    .unwrap();
    assert_eq!(result, -96.0);
}

#[test]
fn dots_reject_mismatched_rows_before_reading_blocks() {
    let lhs = [q8_block(0x3c00, 1), q8_block(0x3c00, 1)].concat();
    let rhs = q8_block(0x3c00, 1);
    assert_eq!(
        dot_q8_0_q8_0(
            Q8_0Row::from_bytes(&lhs).unwrap(),
            Q8_0Row::from_bytes(&rhs).unwrap(),
        ),
        Err(QuantError::MismatchedBlockCount { lhs: 2, rhs: 1 })
    );
}

#[test]
fn quantized_hot_paths_are_allocation_free() {
    let lhs = q8_block(0x3c00, 3);
    let rhs = q8_block(0x3800, -2);
    let lhs = Q8_0Row::from_bytes(&lhs).unwrap();
    let rhs = Q8_0Row::from_bytes(&rhs).unwrap();
    let allocation = measure(|| {
        for _ in 0..256 {
            assert_eq!(dot_q8_0_q8_0(lhs, rhs), Ok(-96.0));
        }
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(allocation.bytes_total, 0);
}

#[test]
fn quant_public_surfaces_and_invalid_blocks_are_typed() {
    let q4 = q4_block(0x3c00, 0, 15);
    let q8 = q8_block(0x3c00, 2);
    let q4_row = Q4_0Row::from_bytes(&q4).unwrap();
    let q8_row = Q8_0Row::from_bytes(&q8).unwrap();
    assert_eq!(q4_row.block_count(), 1);
    assert_eq!(q4_row.as_bytes(), q4.as_slice());
    assert_eq!(q8_row.block_count(), 1);
    assert_eq!(q8_row.as_bytes(), q8.as_slice());

    let mut output = [0.0_f32; BLOCK_VALUES];
    assert_eq!(
        q4_row.decode_block(1, &mut output),
        Err(QuantError::InvalidBlockLength {
            format: Format::Q4_0,
            bytes: Q4_0_BLOCK_BYTES,
        })
    );
    assert_eq!(
        q4_row.decode_block(usize::MAX, &mut output),
        Err(QuantError::InvalidBlockLength {
            format: Format::Q4_0,
            bytes: Q4_0_BLOCK_BYTES,
        })
    );
    assert_eq!(
        q8_row.decode_block(1, &mut output),
        Err(QuantError::InvalidBlockLength {
            format: Format::Q8_0,
            bytes: Q8_0_BLOCK_BYTES,
        })
    );
    assert_eq!(
        q8_row.decode_block(usize::MAX, &mut output),
        Err(QuantError::InvalidBlockLength {
            format: Format::Q8_0,
            bytes: Q8_0_BLOCK_BYTES,
        })
    );

    let invalid_q8 = [0_u8; Q8_0_BLOCK_BYTES - 1];
    assert!(matches!(
        Q8_0Row::from_bytes(&invalid_q8),
        Err(QuantError::InvalidBlockLength {
            format: Format::Q8_0,
            bytes,
        }) if bytes == Q8_0_BLOCK_BYTES - 1
    ));
    let two_q4 = [q4, q4].concat();
    assert_eq!(
        dot_q4_0_q8_0(Q4_0Row::from_bytes(&two_q4).unwrap(), q8_row,),
        Err(QuantError::MismatchedBlockCount { lhs: 2, rhs: 1 })
    );

    let zero_scale = q8_block(0, 1);
    let zero_row = Q8_0Row::from_bytes(&zero_scale).unwrap();
    zero_row.decode_block(0, &mut output).unwrap();
    assert_eq!(output, [0.0; BLOCK_VALUES]);
    assert!(
        format!(
            "{}",
            QuantError::InvalidBlockLength {
                format: Format::Q4_0,
                bytes: 1,
            }
        )
        .contains("Q4_0")
    );
    assert!(format!("{}", QuantError::MismatchedBlockCount { lhs: 1, rhs: 2 }).contains("1 != 2"));
}
