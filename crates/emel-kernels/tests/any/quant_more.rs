#![allow(missing_docs)]
#![allow(clippy::float_cmp)]

use allocation_counter::measure;
use emel_kernels::any::quant::{Q8_0_BLOCK_BYTES, Q8_0Row};
use emel_kernels::any::quant_more::{
    BLOCK_VALUES, Format, Q2_K_BLOCK_BYTES, Q2KRow, Q3_K_BLOCK_BYTES, Q3KRow, Q4_1_BLOCK_BYTES,
    Q4_1Row, Q4_K_BLOCK_BYTES, Q4KRow, Q5_0_BLOCK_BYTES, Q5_0Row, Q5_1_BLOCK_BYTES, Q5_1Row,
    Q5_K_BLOCK_BYTES, Q5KRow, Q8_1_BLOCK_BYTES, Q8_1Row, Q8_K_BLOCK_BYTES, Q8KRow, QK_K_VALUES,
    QuantMoreError, dot_q2_k_q8_k, dot_q3_k_q8_k, dot_q4_1_q8_0, dot_q4_k_q8_k, dot_q5_0_q8_0,
    dot_q5_1_q8_0, dot_q5_k_q8_k,
};
use emel_tensor as _;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use pulp as _;
use sml as _;

const PINNED_EMEL_CPP: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";

fn q4_1_block(scale: u16, minimum: u16, low: u8, high: u8) -> [u8; Q4_1_BLOCK_BYTES] {
    let mut block = [0_u8; Q4_1_BLOCK_BYTES];
    block[..2].copy_from_slice(&scale.to_le_bytes());
    block[2..4].copy_from_slice(&minimum.to_le_bytes());
    for packed in &mut block[4..] {
        *packed = (low & 0x0f) | ((high & 0x0f) << 4);
    }
    block
}

fn q5_0_block(scale: u16, high_bits: u32, low: u8, high: u8) -> [u8; Q5_0_BLOCK_BYTES] {
    let mut block = [0_u8; Q5_0_BLOCK_BYTES];
    block[..2].copy_from_slice(&scale.to_le_bytes());
    block[2..6].copy_from_slice(&high_bits.to_le_bytes());
    for packed in &mut block[6..] {
        *packed = (low & 0x0f) | ((high & 0x0f) << 4);
    }
    block
}

fn q5_1_block(
    scale: u16,
    minimum: u16,
    high_bits: u32,
    low: u8,
    high: u8,
) -> [u8; Q5_1_BLOCK_BYTES] {
    let mut block = [0_u8; Q5_1_BLOCK_BYTES];
    block[..2].copy_from_slice(&scale.to_le_bytes());
    block[2..4].copy_from_slice(&minimum.to_le_bytes());
    block[4..8].copy_from_slice(&high_bits.to_le_bytes());
    for packed in &mut block[8..] {
        *packed = (low & 0x0f) | ((high & 0x0f) << 4);
    }
    block
}

fn q8_0_block(scale: u16, value: i8) -> [u8; Q8_0_BLOCK_BYTES] {
    let mut block = [0_u8; Q8_0_BLOCK_BYTES];
    block[..2].copy_from_slice(&scale.to_le_bytes());
    for byte in &mut block[2..] {
        *byte = value.to_ne_bytes()[0];
    }
    block
}

fn q8_1_block(scale: u16, sum: u16, value: i8) -> [u8; Q8_1_BLOCK_BYTES] {
    let mut block = [0_u8; Q8_1_BLOCK_BYTES];
    block[..2].copy_from_slice(&scale.to_le_bytes());
    block[2..4].copy_from_slice(&sum.to_le_bytes());
    for byte in &mut block[4..] {
        *byte = value.to_ne_bytes()[0];
    }
    block
}

fn q5_k_block(scale: u16, minimum: u16, low: u8, high: u8) -> [u8; Q5_K_BLOCK_BYTES] {
    let mut block = [0_u8; Q5_K_BLOCK_BYTES];
    block[..2].copy_from_slice(&scale.to_le_bytes());
    block[2..4].copy_from_slice(&minimum.to_le_bytes());
    block[4..12].fill(1);
    block[12..16].fill(0x11);
    for packed in &mut block[48..] {
        *packed = (low & 0x0f) | ((high & 0x0f) << 4);
    }
    block
}

fn q8_k_block(scale: f32, value: i8) -> [u8; Q8_K_BLOCK_BYTES] {
    let mut block = [0_u8; Q8_K_BLOCK_BYTES];
    block[..4].copy_from_slice(&scale.to_le_bytes());
    for byte in &mut block[4..260] {
        *byte = value.to_ne_bytes()[0];
    }
    block
}

fn q3_k_block(d: u16, q3: u8, high_mask: u8, scales: [u8; 12]) -> [u8; Q3_K_BLOCK_BYTES] {
    let mut block = [0_u8; Q3_K_BLOCK_BYTES];
    block[..32].fill(high_mask);
    block[32..96].fill(q3);
    block[96..108].copy_from_slice(&scales);
    block[108..].copy_from_slice(&d.to_le_bytes());
    block
}

fn q4_k_block(d: u16, dmin: u16, scales: [u8; 12], packed: u8) -> [u8; Q4_K_BLOCK_BYTES] {
    let mut block = [0_u8; Q4_K_BLOCK_BYTES];
    block[..2].copy_from_slice(&d.to_le_bytes());
    block[2..4].copy_from_slice(&dmin.to_le_bytes());
    block[4..16].copy_from_slice(&scales);
    block[16..].fill(packed);
    block
}

fn varied_q4_k_block() -> [u8; Q4_K_BLOCK_BYTES] {
    let mut block = q4_k_block(
        0x3c00,
        0x3800,
        [
            0x81, 0x42, 0xc3, 0x24, 0x85, 0x46, 0xc7, 0x28, 0x39, 0x7a, 0x5b, 0x6c,
        ],
        0,
    );
    for (index, packed) in block[16..].iter_mut().enumerate() {
        let low = u8::try_from((5 * index + 1) % 16).unwrap();
        let high = u8::try_from((7 * index + 3) % 16).unwrap();
        *packed = low | (high << 4);
    }
    block
}

fn varied_q8_k_oracle_block() -> [u8; Q8_K_BLOCK_BYTES] {
    let mut block = q8_k_block(-0.75, 0);
    for (index, byte) in block[4..260].iter_mut().enumerate() {
        let value = i8::try_from((13 * index + 17) % 127).unwrap() - 63;
        *byte = value.to_ne_bytes()[0];
    }
    for (group, bytes) in block[260..].as_chunks_mut::<2>().0.iter_mut().enumerate() {
        let sum = i16::try_from(37 * group).unwrap() - 250;
        bytes.copy_from_slice(&sum.to_le_bytes());
    }
    block
}

fn reference_q4_k_scale_min(group: usize, scales: &[u8; 12]) -> (i32, i32) {
    if group < 4 {
        (
            i32::from(scales[group] & 0x3f),
            i32::from(scales[group + 4] & 0x3f),
        )
    } else {
        (
            i32::from((scales[group + 4] & 0x0f) | ((scales[group - 4] >> 6) << 4)),
            i32::from((scales[group + 4] >> 4) | ((scales[group] >> 6) << 4)),
        )
    }
}

#[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
fn reference_q4_k_dequant(block: &[u8; Q4_K_BLOCK_BYTES]) -> [f32; QK_K_VALUES] {
    assert_eq!(&block[..4], &[0x00, 0x3c, 0x00, 0x38]);
    let d = 1.0_f32;
    let minimum = 0.5_f32;
    let scales_data: [u8; 12] = block[4..16].try_into().unwrap();
    let mut output = [0.0_f32; QK_K_VALUES];
    let mut scale_index = 0;
    let mut output_offset = 0;
    while output_offset < QK_K_VALUES {
        let (scale_a, minimum_a) = reference_q4_k_scale_min(scale_index, &scales_data);
        let (scale_b, minimum_b) = reference_q4_k_scale_min(scale_index + 1, &scales_data);
        let packed_offset = 16 + output_offset / 2;
        for lane in 0..QK_K_VALUES / 8 {
            let packed = block[packed_offset + lane];
            output[output_offset + lane] =
                d * scale_a as f32 * f32::from(packed & 0x0f) - minimum * minimum_a as f32;
            output[output_offset + lane + QK_K_VALUES / 8] =
                d * scale_b as f32 * f32::from(packed >> 4) - minimum * minimum_b as f32;
        }
        scale_index += 2;
        output_offset += QK_K_VALUES / 4;
    }
    output
}

// Preserve the reference's separate products and accumulation order so the
// test remains a parity oracle rather than a fused-arithmetic variant.
#[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
fn reference_q4_k_q8_k_dot(lhs: &[u8; Q4_K_BLOCK_BYTES], rhs: &[u8; Q8_K_BLOCK_BYTES]) -> f32 {
    let scales: [u8; 12] = lhs[4..16].try_into().unwrap();
    let rhs_scale = f32::from_le_bytes(rhs[..4].try_into().unwrap());
    let mut lane_sums = [0_i32; 8];
    for value_index in 0..256 {
        let group = value_index / 32;
        let packed_index = (value_index / 64) * 32 + (value_index % 32);
        let packed = lhs[16 + packed_index];
        let q4 = if value_index % 64 < 32 {
            packed & 0x0f
        } else {
            packed >> 4
        };
        let rhs_value = i32::from(i8::from_ne_bytes([rhs[4 + value_index]]));
        let (scale, _) = reference_q4_k_scale_min(group, &scales);
        lane_sums[value_index % 8] += scale * i32::from(q4) * rhs_value;
    }

    let mut minimum_sum = 0_i32;
    for group in 0..16 {
        let rhs_sum = i16::from_le_bytes([rhs[260 + 2 * group], rhs[261 + 2 * group]]);
        let (_, minimum) = reference_q4_k_scale_min(group / 2, &scales);
        minimum_sum += i32::from(rhs_sum) * minimum;
    }

    let d = 1.0_f32 * rhs_scale;
    let dmin = 0.5_f32 * rhs_scale;
    let mut output = 0.0_f32;
    output -= dmin * minimum_sum as f32;
    for lane in lane_sums {
        output += d * lane as f32;
    }
    output
}

// Preserve the pinned oracle's separate scale products and accumulation.
#[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
fn reference_q3_k_q8_k_dot(lhs: &[u8; Q3_K_BLOCK_BYTES], rhs: &[u8; Q8_K_BLOCK_BYTES]) -> f32 {
    let s0 = u32::from_le_bytes(lhs[96..100].try_into().unwrap());
    let s1 = u32::from_le_bytes(lhs[100..104].try_into().unwrap());
    let s2 = u32::from_le_bytes(lhs[104..108].try_into().unwrap());
    let words = [
        (s0 & 0x0f0f_0f0f) | ((s2 & 0x0303_0303) << 4),
        (s1 & 0x0f0f_0f0f) | (((s2 >> 2) & 0x0303_0303) << 4),
        ((s0 >> 4) & 0x0f0f_0f0f) | (((s2 >> 4) & 0x0303_0303) << 4),
        ((s1 >> 4) & 0x0f0f_0f0f) | (((s2 >> 6) & 0x0303_0303) << 4),
    ];
    let mut scales = [0_i32; 16];
    for (index, scale) in scales.iter_mut().enumerate() {
        *scale = i32::from(words[index / 4].to_le_bytes()[index % 4]) - 32;
    }

    let mut accumulators = [0_i32; 8];
    for value_index in 0..256 {
        let half = value_index / 128;
        let within_half = value_index % 128;
        let plane = within_half / 32;
        let lane = within_half % 32;
        let low_bits = (lhs[32 + half * 32 + lane] >> (plane * 2)) & 0x03;
        let high_mask = 1_u8 << (plane + half * 4);
        let high = i32::from(u8::from((lhs[lane] & high_mask) == 0)) * 4;
        let value = i32::from(low_bits) - high;
        let rhs_value = i32::from(i8::from_ne_bytes([rhs[4 + value_index]]));
        accumulators[value_index & 7] += scales[value_index / 16] * (value * rhs_value);
    }

    let rhs_scale = f32::from_le_bytes(rhs[..4].try_into().unwrap());
    assert_eq!(u16::from_le_bytes([lhs[108], lhs[109]]), 0x3800);
    let d = 0.5_f32 * rhs_scale;
    let mut sum = 0.0_f32;
    for accumulator in accumulators {
        sum += d * accumulator as f32;
    }
    sum
}

fn q2_k_block(d: u16, dmin: u16, scale: u8, minimum: u8, value: u8) -> [u8; Q2_K_BLOCK_BYTES] {
    let mut block = [0_u8; Q2_K_BLOCK_BYTES];
    block[..16].fill((minimum << 4) | (scale & 0x0f));
    block[16..80].fill(value & 0x55);
    block[80..82].copy_from_slice(&d.to_le_bytes());
    block[82..84].copy_from_slice(&dmin.to_le_bytes());
    block
}

fn q8_k_block_with_sums(scale: f32, value: i8) -> [u8; Q8_K_BLOCK_BYTES] {
    let mut block = q8_k_block(scale, value);
    let sum = i16::from(value) * 16;
    for bytes in block[260..].as_chunks_mut::<2>().0.iter_mut() {
        bytes.copy_from_slice(&sum.to_le_bytes());
    }
    block
}

fn varied_q2_k_block(d: u16, dmin: u16, seed: usize) -> [u8; Q2_K_BLOCK_BYTES] {
    let mut block = [0_u8; Q2_K_BLOCK_BYTES];
    for (index, byte) in block[..16].iter_mut().enumerate() {
        let scale = u8::try_from((index + seed) % 16).unwrap();
        let minimum = u8::try_from((3 * index + seed) % 16).unwrap();
        *byte = (minimum << 4) | scale;
    }
    for (index, byte) in block[16..80].iter_mut().enumerate() {
        let value = u8::try_from((index + seed) % 4).unwrap();
        *byte = value | (value << 2) | (value << 4) | (value << 6);
    }
    block[80..82].copy_from_slice(&d.to_le_bytes());
    block[82..84].copy_from_slice(&dmin.to_le_bytes());
    block
}

fn varied_q8_k_block(scale: f32, seed: usize) -> [u8; Q8_K_BLOCK_BYTES] {
    let mut block = q8_k_block(scale, 0);
    for index in 0..256 {
        let value = i8::try_from((index + seed) % 31).unwrap() - 15;
        block[4 + index] = value.to_ne_bytes()[0];
    }
    for group in 0..16 {
        let mut sum = 0_i16;
        for lane in 0..16 {
            let value = i8::from_ne_bytes([block[4 + group * 16 + lane]]);
            sum += i16::from(value);
        }
        block[260 + group * 2..262 + group * 2].copy_from_slice(&sum.to_le_bytes());
    }
    block
}

#[test]
fn source_identity_and_q4_1_layout_are_pinned() {
    assert_eq!(PINNED_EMEL_CPP, "843a117386ef17dc5a50549bbfc821074c2141d6");
    let encoded = q4_1_block(0x3800, 0x3c00, 2, 3);
    let row = Q4_1Row::from_bytes(&encoded).expect("valid q4_1 row");
    let mut decoded = [0.0_f32; BLOCK_VALUES];
    row.decode_block(0, &mut decoded).expect("valid q4_1 block");
    assert_eq!(&decoded[..16], &[2.0; 16]);
    assert_eq!(&decoded[16..], &[2.5; 16]);
}

#[test]
fn q5_layouts_preserve_high_bit_order_and_binary16_fields() {
    let q5_0 = q5_0_block(0x3c00, 1 | (1 << 16), 0, 15);
    let mut decoded_0 = [0.0_f32; BLOCK_VALUES];
    Q5_0Row::from_bytes(&q5_0)
        .unwrap()
        .decode_block(0, &mut decoded_0)
        .unwrap();
    assert_eq!(decoded_0[0], 0.0);
    assert_eq!(decoded_0[16], 15.0);

    let q5_1 = q5_1_block(0x3800, 0x3c00, 1 | (1 << 16), 0, 15);
    let mut decoded_1 = [0.0_f32; BLOCK_VALUES];
    Q5_1Row::from_bytes(&q5_1)
        .unwrap()
        .decode_block(0, &mut decoded_1)
        .unwrap();
    assert_eq!(decoded_1[0], 9.0);
    assert_eq!(decoded_1[16], 16.5);
}

#[test]
fn q8_1_decode_preserves_d_and_s_fields() {
    let encoded = q8_1_block(0x3800, 0x3e00, -2);
    let row = Q8_1Row::from_bytes(&encoded).expect("valid q8_1 row");
    assert_eq!(row.sum_bits(0), Ok(0x3e00));
    let mut decoded = [0.0_f32; BLOCK_VALUES];
    row.decode_block(0, &mut decoded).expect("valid q8_1 block");
    assert_eq!(decoded, [-1.0; BLOCK_VALUES]);
}

#[test]
fn pinned_direct_q8_0_dot_paths_match_integer_accumulation() {
    let rhs = q8_0_block(0x3800, 2);
    let rhs_row = Q8_0Row::from_bytes(&rhs).unwrap();

    let q4 = q4_1_block(0x3c00, 0x3c00, 2, 3);
    assert_eq!(
        dot_q4_1_q8_0(Q4_1Row::from_bytes(&q4).unwrap(), rhs_row),
        Ok(112.0)
    );

    let q5_0 = q5_0_block(0x3c00, 1 | (1 << 16), 0, 15);
    assert_eq!(
        dot_q5_0_q8_0(Q5_0Row::from_bytes(&q5_0).unwrap(), rhs_row),
        Ok(-240.0)
    );

    let q5_1 = q5_1_block(0x3800, 0x3c00, 1 | (1 << 16), 0, 15);
    assert_eq!(
        dot_q5_1_q8_0(Q5_1Row::from_bytes(&q5_1).unwrap(), rhs_row),
        Ok(168.0)
    );
}

#[test]
fn malformed_rows_and_mismatched_dots_are_rejected_typed() {
    let invalid_bytes = Q5_0_BLOCK_BYTES - 1;
    assert!(matches!(
        Q5_0Row::from_bytes(&[0; Q5_0_BLOCK_BYTES - 1]),
        Err(QuantMoreError::InvalidBlockLength {
            format: Format::Q5_0,
            bytes,
        }) if bytes == invalid_bytes
    ));
    let q4 = q4_1_block(0x3c00, 0x3c00, 0, 0);
    let q8 = [q8_0_block(0x3c00, 1), q8_0_block(0x3c00, 1)].concat();
    assert_eq!(
        dot_q4_1_q8_0(
            Q4_1Row::from_bytes(&q4).unwrap(),
            Q8_0Row::from_bytes(&q8).unwrap(),
        ),
        Err(QuantMoreError::MismatchedBlockCount { lhs: 1, rhs: 2 })
    );
}

#[test]
fn packed_decode_and_dot_hot_paths_are_allocation_free() {
    let q5_1 = q5_1_block(0x3800, 0x3c00, 1 | (1 << 16), 0, 15);
    let rhs = q8_0_block(0x3800, 2);
    let lhs = Q5_1Row::from_bytes(&q5_1).unwrap();
    let rhs = Q8_0Row::from_bytes(&rhs).unwrap();
    let allocation = measure(|| {
        let mut output = [0.0_f32; BLOCK_VALUES];
        for _ in 0..256 {
            lhs.decode_block(0, &mut output).unwrap();
            assert_eq!(dot_q5_1_q8_0(lhs, rhs), Ok(168.0));
        }
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(allocation.bytes_total, 0);
}

#[test]
fn pinned_q5_k_q8_k_dot_matches_scale_min_and_high_bit_formula() {
    let lhs = q5_k_block(0x3c00, 0x3c00, 2, 3);
    let rhs = q8_k_block(2.0, 1);
    assert_eq!(
        dot_q5_k_q8_k(
            Q5KRow::from_bytes(&lhs).unwrap(),
            Q8KRow::from_bytes(&rhs).unwrap()
        ),
        Ok(768.0)
    );

    let mut high_lhs = lhs;
    high_lhs[16..48].fill(0xff);
    assert_eq!(
        dot_q5_k_q8_k(
            Q5KRow::from_bytes(&high_lhs).unwrap(),
            Q8KRow::from_bytes(&rhs).unwrap()
        ),
        Ok(8_960.0)
    );
}

#[test]
fn q5_k_q8_k_varied_qs_uses_each_64_value_chunk() {
    let mut lhs = q5_k_block(0x3c00, 0x3c00, 1, 2);
    for chunk in 0..4 {
        let chunk = u8::try_from(chunk).expect("four chunks fit in u8");
        for lane in 0..32 {
            let low = 1 + chunk;
            let high = 2 + chunk;
            lhs[48 + usize::from(chunk) * 32 + lane] = low | (high << 4);
        }
    }
    let rhs = q8_k_block(2.0, 1);
    assert_eq!(
        dot_q5_k_q8_k(
            Q5KRow::from_bytes(&lhs).unwrap(),
            Q8KRow::from_bytes(&rhs).unwrap()
        ),
        Ok(1_024.0)
    );
}

#[test]
fn q5_k_q8_k_rows_validate_lengths_and_block_counts() {
    assert!(matches!(
        Q5KRow::from_bytes(&[0; Q5_K_BLOCK_BYTES - 1]),
        Err(QuantMoreError::InvalidBlockLength {
            format: Format::Q5K,
            bytes,
        }) if bytes == Q5_K_BLOCK_BYTES - 1
    ));
    let lhs = q5_k_block(0x3c00, 0x3c00, 0, 0);
    let rhs = [q8_k_block(1.0, 1), q8_k_block(1.0, 1)].concat();
    assert_eq!(
        dot_q5_k_q8_k(
            Q5KRow::from_bytes(&lhs).unwrap(),
            Q8KRow::from_bytes(&rhs).unwrap()
        ),
        Err(QuantMoreError::MismatchedBlockCount { lhs: 1, rhs: 2 })
    );
}

#[test]
fn q5_k_q8_k_dot_is_allocation_free() {
    let lhs = q5_k_block(0x3c00, 0x3c00, 2, 3);
    let rhs = q8_k_block(2.0, 1);
    let lhs = Q5KRow::from_bytes(&lhs).unwrap();
    let rhs = Q8KRow::from_bytes(&rhs).unwrap();
    let allocation = measure(|| {
        for _ in 0..256 {
            assert_eq!(dot_q5_k_q8_k(lhs, rhs), Ok(768.0));
        }
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(allocation.bytes_total, 0);
}

#[test]
fn q2_k_q8_k_layout_and_scalar_formula_are_pinned() {
    let lhs = q2_k_block(0x3c00, 0x3c00, 1, 0, 0x55);
    let rhs = q8_k_block_with_sums(1.0, 1);
    let lhs_row = Q2KRow::from_bytes(&lhs).expect("valid q2_k row");
    let rhs_row = Q8KRow::from_bytes(&rhs).expect("valid q8_k row");
    assert_eq!(Q2_K_BLOCK_BYTES, 84);
    assert_eq!(lhs_row.block_count(), 1);
    assert_eq!(lhs_row.as_bytes(), lhs.as_slice());
    assert_eq!(rhs_row.block_count(), 1);
    assert_eq!(dot_q2_k_q8_k(lhs_row, rhs_row), Ok(256.0));

    let lhs_with_minimum = q2_k_block(0x3c00, 0x3c00, 1, 2, 0x55);
    assert_eq!(
        dot_q2_k_q8_k(
            Q2KRow::from_bytes(&lhs_with_minimum).unwrap(),
            Q8KRow::from_bytes(&rhs).unwrap()
        ),
        Ok(-256.0)
    );

    let lhs_two = [lhs, lhs].concat();
    let rhs_two = [rhs, rhs].concat();
    assert_eq!(
        dot_q2_k_q8_k(
            Q2KRow::from_bytes(&lhs_two).unwrap(),
            Q8KRow::from_bytes(&rhs_two).unwrap()
        ),
        Ok(512.0)
    );
}

#[test]
fn q2_k_q8_k_matches_reference_float_operation_order() {
    let mut lhs = q2_k_block(0xbf11, 0xbcae, 0, 0, 0x55);
    // One 16-lane group contributes both its low scale and minimum sum.
    lhs[0] = 0x11;
    let rhs = q8_k_block_with_sums(f32::from_bits(0xbf4b_6f2a), 1);
    let actual = dot_q2_k_q8_k(
        Q2KRow::from_bytes(&lhs).unwrap(),
        Q8KRow::from_bytes(&rhs).unwrap(),
    )
    .unwrap();
    // detail.hpp:2426-2427 computes the two scaled factors first and
    // detail.hpp:2456 subtracts their separately rounded products.
    assert_eq!(actual.to_bits(), 0x40f2_c528);
}

#[test]
fn q2_k_q8_k_varied_blocks_cover_all_packed_lanes() {
    let lhs = [
        varied_q2_k_block(0x3c00, 0x3400, 0),
        varied_q2_k_block(0x3800, 0x3000, 5),
    ]
    .concat();
    let rhs = [varied_q8_k_block(0.75, 2), varied_q8_k_block(-1.25, 9)].concat();
    let actual = dot_q2_k_q8_k(
        Q2KRow::from_bytes(&lhs).unwrap(),
        Q8KRow::from_bytes(&rhs).unwrap(),
    )
    .unwrap();
    assert_eq!(actual.to_bits(), 0xc498_9f00);
}

#[test]
fn q2_k_q8_k_rows_reject_bad_lengths_and_mismatches() {
    assert!(matches!(
        Q2KRow::from_bytes(&[0; Q2_K_BLOCK_BYTES - 1]),
        Err(QuantMoreError::InvalidBlockLength {
            format: Format::Q2K,
            bytes,
        }) if bytes == Q2_K_BLOCK_BYTES - 1
    ));
    let lhs = q2_k_block(0x3c00, 0, 1, 0, 0x55);
    let rhs = [q8_k_block_with_sums(1.0, 1), q8_k_block_with_sums(1.0, 1)].concat();
    assert_eq!(
        dot_q2_k_q8_k(
            Q2KRow::from_bytes(&lhs).unwrap(),
            Q8KRow::from_bytes(&rhs).unwrap()
        ),
        Err(QuantMoreError::MismatchedBlockCount { lhs: 1, rhs: 2 })
    );
}

#[test]
fn q2_k_q8_k_dot_is_allocation_free() {
    let lhs_bytes = q2_k_block(0x3c00, 0, 1, 0, 0x55);
    let rhs_bytes = q8_k_block_with_sums(1.0, 1);
    let lhs = Q2KRow::from_bytes(&lhs_bytes).unwrap();
    let rhs = Q8KRow::from_bytes(&rhs_bytes).unwrap();
    let allocation = measure(|| {
        for _ in 0..256 {
            assert_eq!(dot_q2_k_q8_k(lhs, rhs), Ok(256.0));
        }
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(allocation.bytes_total, 0);
}

#[test]
fn q3_k_q8_k_layout_and_scalar_formula_are_pinned() {
    let lhs = q3_k_block(0x3c00, 0, 0, [0; 12]);
    let rhs = q8_k_block(1.0, 1);
    let lhs_row = Q3KRow::from_bytes(&lhs).expect("valid q3_k row");
    let rhs_row = Q8KRow::from_bytes(&rhs).expect("valid q8_k row");
    assert_eq!(Q3_K_BLOCK_BYTES, 110);
    assert_eq!(lhs_row.block_count(), 1);
    assert_eq!(lhs_row.as_bytes(), lhs.as_slice());
    assert_eq!(dot_q3_k_q8_k(lhs_row, rhs_row), Ok(32_768.0));

    let high_bits = q3_k_block(0x3c00, 0, u8::MAX, [0; 12]);
    assert_eq!(
        dot_q3_k_q8_k(
            Q3KRow::from_bytes(&high_bits).unwrap(),
            Q8KRow::from_bytes(&rhs).unwrap(),
        ),
        Ok(0.0)
    );
}

#[test]
fn q3_k_q8_k_second_half_uses_high_mask_bits_four_through_seven() {
    let lhs = q3_k_block(0x3c00, 0, 0xf0, [0; 12]);
    let rhs = q8_k_block(1.0, 1);
    // The first 128 values use absent bits 0..3 and therefore decode to -4.
    // The second 128 values use present bits 4..7 and therefore decode to 0.
    assert_eq!(
        dot_q3_k_q8_k(
            Q3KRow::from_bytes(&lhs).unwrap(),
            Q8KRow::from_bytes(&rhs).unwrap(),
        ),
        Ok(16_384.0)
    );
}

#[test]
fn q3_k_q8_k_source_oracle_covers_both_halves_and_high_masks() {
    let mut lhs = q3_k_block(
        0x3800,
        0,
        0,
        [
            0x21, 0x43, 0x65, 0x87, 0xa9, 0xcb, 0xed, 0x0f, 0x12, 0x34, 0x56, 0x78,
        ],
    );
    for lane in 0..32 {
        lhs[lane] = u8::try_from((lane * 37 + 0x53) & 0xff).unwrap();
        lhs[32 + lane] = u8::try_from((lane * 29 + 0xa7) & 0xff).unwrap();
    }
    let mut rhs = q8_k_block(-0.75, 0);
    for lane in 0..256 {
        rhs[4 + lane] = (i8::try_from((lane * 11 + 7) % 127).unwrap() - 63).to_ne_bytes()[0];
    }
    let expected = reference_q3_k_q8_k_dot(&lhs, &rhs);
    let actual = dot_q3_k_q8_k(
        Q3KRow::from_bytes(&lhs).unwrap(),
        Q8KRow::from_bytes(&rhs).unwrap(),
    )
    .unwrap();
    assert_eq!(actual.to_bits(), expected.to_bits());
}

#[test]
#[allow(clippy::cast_precision_loss)]
fn q3_k_q8_k_packed_planes_scales_and_blocks_are_covered() {
    let mut lhs = q3_k_block(0x3800, 0, 0, [0; 12]);
    // Give each q3 plane a different low value and high-bit pattern. The
    // expected value below is calculated from the source's 16 signed scales.
    for plane in 0..4 {
        lhs[32..96].iter_mut().for_each(|byte| {
            *byte |= (u8::try_from(plane).unwrap() + 1) << (2 * plane);
        });
        lhs[..32].fill(1_u8 << plane);
    }
    lhs[96..108].copy_from_slice(&[
        0x21, 0x43, 0x65, 0x87, 0xa9, 0xcb, 0xed, 0x0f, 0x12, 0x34, 0x56, 0x78,
    ]);
    let rhs = q8_k_block(-0.5, 2);
    let actual = dot_q3_k_q8_k(
        Q3KRow::from_bytes(&lhs).unwrap(),
        Q8KRow::from_bytes(&rhs).unwrap(),
    )
    .unwrap();

    let mut unpacked = [0_u8; 16];
    let s0 = u32::from_le_bytes(lhs[96..100].try_into().unwrap());
    let s1 = u32::from_le_bytes(lhs[100..104].try_into().unwrap());
    let s2 = u32::from_le_bytes(lhs[104..108].try_into().unwrap());
    let words = [
        (s0 & 0x0f0f_0f0f) | ((s2 & 0x0303_0303) << 4),
        (s1 & 0x0f0f_0f0f) | (((s2 >> 2) & 0x0303_0303) << 4),
        ((s0 >> 4) & 0x0f0f_0f0f) | (((s2 >> 4) & 0x0303_0303) << 4),
        ((s1 >> 4) & 0x0f0f_0f0f) | (((s2 >> 6) & 0x0303_0303) << 4),
    ];
    for (word, bytes) in words
        .into_iter()
        .zip(unpacked.as_chunks_mut::<4>().0.iter_mut())
    {
        bytes.copy_from_slice(&word.to_le_bytes());
    }
    let mut integer = 0_i32;
    for (group, unpacked_scale) in unpacked.iter().enumerate() {
        let scale = i32::from(*unpacked_scale) - 32;
        let plane = (group % 8) / 2;
        let q3_base = (group / 8) * 32 + (group % 2) * 16;
        for lane in 0..16 {
            let value = i32::from((lhs[32 + q3_base + lane] >> (2 * plane)) & 3)
                - i32::from(u8::from(
                    (lhs[lane] & (1 << (plane + (group / 8) * 4))) == 0,
                )) * 4;
            integer += scale * value * 2;
        }
    }
    // d=0.5 and rhs d=-0.5, so the source's per-lane products scale by -0.25.
    assert_eq!(actual, integer as f32 * -0.25);
}

#[test]
fn q3_k_q8_k_rows_reject_bad_lengths_and_mismatches() {
    assert!(matches!(
        Q3KRow::from_bytes(&[0; Q3_K_BLOCK_BYTES - 1]),
        Err(QuantMoreError::InvalidBlockLength {
            format: Format::Q3K,
            bytes,
        }) if bytes == Q3_K_BLOCK_BYTES - 1
    ));
    let lhs = q3_k_block(0x3c00, 0, 0, [0; 12]);
    let rhs = [q8_k_block(1.0, 1), q8_k_block(1.0, 1)].concat();
    assert_eq!(
        dot_q3_k_q8_k(
            Q3KRow::from_bytes(&lhs).unwrap(),
            Q8KRow::from_bytes(&rhs).unwrap(),
        ),
        Err(QuantMoreError::MismatchedBlockCount { lhs: 1, rhs: 2 })
    );
}

#[test]
fn q3_k_q8_k_dot_is_allocation_free() {
    let lhs_bytes = q3_k_block(0x3c00, 0x55, 0, [0; 12]);
    let rhs_bytes = q8_k_block(1.0, 1);
    let lhs = Q3KRow::from_bytes(&lhs_bytes).unwrap();
    let rhs = Q8KRow::from_bytes(&rhs_bytes).unwrap();
    let allocation = measure(|| {
        for _ in 0..256 {
            let _ = dot_q3_k_q8_k(lhs, rhs).unwrap();
        }
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(allocation.bytes_total, 0);
}

#[test]
fn q4_k_q8_k_layout_and_asymmetric_scalar_formula_are_pinned() {
    let lhs = q4_k_block(0x3c00, 0x3c00, [1, 2, 3, 0, 2, 3, 4, 0, 0, 0, 0, 0], 0x21);
    let rhs = q8_k_block_with_sums(1.0, 1);
    let lhs_row = Q4KRow::from_bytes(&lhs).expect("valid q4_k row");
    let rhs_row = Q8KRow::from_bytes(&rhs).expect("valid q8_k row");
    assert_eq!(Q4_K_BLOCK_BYTES, 144);
    assert_eq!(lhs_row.block_count(), 1);
    assert_eq!(lhs_row.as_bytes(), lhs.as_slice());
    assert_eq!(dot_q4_k_q8_k(lhs_row, rhs_row), Ok(-32.0));

    let lhs_two = [lhs, lhs].concat();
    let rhs_two = [rhs, rhs].concat();
    assert_eq!(
        dot_q4_k_q8_k(
            Q4KRow::from_bytes(&lhs_two).unwrap(),
            Q8KRow::from_bytes(&rhs_two).unwrap(),
        ),
        Ok(-64.0)
    );
}

#[test]
fn q4_k_q8_k_source_oracle_covers_transformed_scales_and_varied_lanes() {
    let lhs = varied_q4_k_block();
    let rhs = varied_q8_k_oracle_block();
    let expected = reference_q4_k_q8_k_dot(&lhs, &rhs);
    let actual = dot_q4_k_q8_k(
        Q4KRow::from_bytes(&lhs).unwrap(),
        Q8KRow::from_bytes(&rhs).unwrap(),
    )
    .unwrap();

    assert_ne!(lhs[4] & 0xc0, 0);
    assert_ne!(lhs[8] & 0xc0, 0);
    assert_ne!(lhs[12] & 0xf0, 0);
    assert_ne!(rhs[260], rhs[262]);
    assert_ne!(actual.to_bits(), 0);
    assert_eq!(actual.to_bits(), expected.to_bits());
}

#[test]
fn q4_k_decode_block_matches_pinned_dequantize_row() {
    let encoded = varied_q4_k_block();
    let row = Q4KRow::from_bytes(&encoded).unwrap();
    let mut actual = [0.0_f32; QK_K_VALUES];
    row.decode_block(0, &mut actual).unwrap();
    let expected = reference_q4_k_dequant(&encoded);

    assert_eq!(actual, expected);
    assert!(actual.iter().any(|value| value.to_bits() != 0));
}

#[test]
fn q4_k_decode_block_rejects_out_of_range_without_allocating() {
    let encoded = varied_q4_k_block();
    let row = Q4KRow::from_bytes(&encoded).unwrap();
    let mut output = [7.0_f32; QK_K_VALUES];
    assert_eq!(
        row.decode_block(1, &mut output),
        Err(QuantMoreError::BlockOutOfRange {
            format: Format::Q4K,
            block: 1,
            block_count: 1,
        })
    );
    assert_eq!(output, [7.0_f32; QK_K_VALUES]);

    let allocation = measure(|| {
        for _ in 0..256 {
            row.decode_block(0, &mut output).unwrap();
        }
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(allocation.bytes_total, 0);
}

#[test]
fn q4_k_q8_k_rows_reject_bad_lengths_and_mismatches() {
    assert!(matches!(
        Q4KRow::from_bytes(&[0; Q4_K_BLOCK_BYTES - 1]),
        Err(QuantMoreError::InvalidBlockLength {
            format: Format::Q4K,
            bytes,
        }) if bytes == Q4_K_BLOCK_BYTES - 1
    ));
    let lhs = q4_k_block(0x3c00, 0, [0; 12], 0);
    let rhs = [q8_k_block(1.0, 1), q8_k_block(1.0, 1)].concat();
    assert_eq!(
        dot_q4_k_q8_k(
            Q4KRow::from_bytes(&lhs).unwrap(),
            Q8KRow::from_bytes(&rhs).unwrap(),
        ),
        Err(QuantMoreError::MismatchedBlockCount { lhs: 1, rhs: 2 })
    );
}

#[test]
fn q4_k_q8_k_dot_is_allocation_free() {
    let lhs_bytes = q4_k_block(0x3c00, 0x3c00, [1, 2, 3, 0, 2, 3, 4, 0, 0, 0, 0, 0], 0x21);
    let rhs_bytes = q8_k_block_with_sums(1.0, 1);
    let lhs = Q4KRow::from_bytes(&lhs_bytes).unwrap();
    let rhs = Q8KRow::from_bytes(&rhs_bytes).unwrap();
    let allocation = measure(|| {
        for _ in 0..256 {
            assert_eq!(dot_q4_k_q8_k(lhs, rhs), Ok(-32.0));
        }
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(allocation.bytes_total, 0);
}

#[test]
fn quant_more_public_rows_expose_borrowed_storage() {
    let q4 = q4_1_block(0x3c00, 0x3c00, 0, 1);
    let q4_k = q4_k_block(0x3c00, 0x3c00, [0; 12], 0);
    let q5_0 = q5_0_block(0x3c00, 0, 0, 1);
    let q5_1 = q5_1_block(0x3c00, 0x3c00, 0, 0, 1);
    let q8 = q8_1_block(0x3c00, 0x3c00, 2);
    let q4_row = Q4_1Row::from_bytes(&q4).unwrap();
    let q4k_borrowed = Q4KRow::from_bytes(&q4_k).unwrap();
    let q5_0_row = Q5_0Row::from_bytes(&q5_0).unwrap();
    let q5_1_row = Q5_1Row::from_bytes(&q5_1).unwrap();
    let q8_row = Q8_1Row::from_bytes(&q8).unwrap();
    assert_eq!(q4_row.block_count(), 1);
    assert_eq!(q4_row.as_bytes(), q4.as_slice());
    assert_eq!(q4k_borrowed.block_count(), 1);
    assert_eq!(q4k_borrowed.as_bytes(), q4_k.as_slice());
    assert_eq!(q5_0_row.block_count(), 1);
    assert_eq!(q5_0_row.as_bytes(), q5_0.as_slice());
    assert_eq!(q5_1_row.block_count(), 1);
    assert_eq!(q5_1_row.as_bytes(), q5_1.as_slice());
    assert_eq!(q8_row.block_count(), 1);
    assert_eq!(q8_row.as_bytes(), q8.as_slice());
}

#[test]
fn quant_more_invalid_blocks_and_errors_are_typed() {
    let q4 = q4_1_block(0x3c00, 0x3c00, 0, 1);
    let q5_0 = q5_0_block(0x3c00, 0, 0, 1);
    let q5_1 = q5_1_block(0x3c00, 0x3c00, 0, 0, 1);
    let q8 = q8_1_block(0x3c00, 0x3c00, 2);
    let q4_row = Q4_1Row::from_bytes(&q4).unwrap();
    let q5_0_row = Q5_0Row::from_bytes(&q5_0).unwrap();
    let q5_1_row = Q5_1Row::from_bytes(&q5_1).unwrap();
    let q8_row = Q8_1Row::from_bytes(&q8).unwrap();
    let mut output = [0.0_f32; BLOCK_VALUES];
    assert!(matches!(
        q4_row.decode_block(1, &mut output),
        Err(QuantMoreError::BlockOutOfRange {
            format: Format::Q4_1,
            block: 1,
            block_count: 1,
        })
    ));
    assert!(matches!(
        q5_0_row.decode_block(usize::MAX, &mut output),
        Err(QuantMoreError::BlockOutOfRange {
            format: Format::Q5_0,
            block: usize::MAX,
            block_count: 1,
        })
    ));
    assert!(matches!(
        q5_1_row.decode_block(1, &mut output),
        Err(QuantMoreError::BlockOutOfRange {
            format: Format::Q5_1,
            block: 1,
            block_count: 1,
        })
    ));
    assert_eq!(
        q8_row.sum_bits(1),
        Err(QuantMoreError::BlockOutOfRange {
            format: Format::Q8_1,
            block: 1,
            block_count: 1,
        })
    );
    assert!(matches!(
        q8_row.decode_block(usize::MAX, &mut output),
        Err(QuantMoreError::BlockOutOfRange {
            format: Format::Q8_1,
            block: usize::MAX,
            block_count: 1,
        })
    ));
}

#[test]
fn quant_more_invalid_lengths_and_errors_display() {
    assert!(matches!(
        Q4_1Row::from_bytes(&[0; Q4_1_BLOCK_BYTES - 1]),
        Err(QuantMoreError::InvalidBlockLength { format: Format::Q4_1, bytes })
            if bytes == Q4_1_BLOCK_BYTES - 1
    ));
    assert!(matches!(
        Q5_1Row::from_bytes(&[0; Q5_1_BLOCK_BYTES - 1]),
        Err(QuantMoreError::InvalidBlockLength { format: Format::Q5_1, bytes })
            if bytes == Q5_1_BLOCK_BYTES - 1
    ));
    assert!(matches!(
        Q8_1Row::from_bytes(&[0; Q8_1_BLOCK_BYTES - 1]),
        Err(QuantMoreError::InvalidBlockLength { format: Format::Q8_1, bytes })
            if bytes == Q8_1_BLOCK_BYTES - 1
    ));
    assert!(
        format!(
            "{}",
            QuantMoreError::InvalidBlockLength {
                format: Format::Q5_0,
                bytes: 1,
            }
        )
        .contains("Q5_0")
    );
    assert!(
        format!(
            "{}",
            QuantMoreError::BlockOutOfRange {
                format: Format::Q8_1,
                block: 2,
                block_count: 1,
            }
        )
        .contains("block 2")
    );
    assert!(
        format!(
            "{}",
            QuantMoreError::MismatchedBlockCount { lhs: 1, rhs: 2 }
        )
        .contains("1 != 2")
    );
}
