#![allow(missing_docs)]
#![allow(clippy::float_cmp)]

use allocation_counter::measure;
use emel_kernels::any::quant_more::{
    Format, Q6_K_BLOCK_BYTES, Q6KRow, Q8_K_BLOCK_BYTES, Q8KRow, QuantMoreError, dot_q6_k_q8_k,
};
use emel_tensor as _;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use pulp as _;
use sml as _;

const PINNED_EMEL_CPP: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";

fn fp16_to_f32(bits16: u16) -> f32 {
    let word = u32::from(bits16) << 16;
    let sign = word & 0x8000_0000;
    let doubled = word.wrapping_add(word);
    let exp_offset = 0xe0_u32 << 23;
    let normalized =
        f32::from_bits((doubled >> 4).wrapping_add(exp_offset)) * f32::from_bits(0x0780_0000);
    let magic_mask = 126_u32 << 23;
    let denormalized = f32::from_bits((doubled >> 17) | magic_mask) - 0.5;
    let result = if doubled < 1_u32 << 27 {
        sign | denormalized.to_bits()
    } else {
        sign | normalized.to_bits()
    };
    f32::from_bits(result)
}

fn q6_k_block(d: u16, scales: [i8; 16]) -> [u8; Q6_K_BLOCK_BYTES] {
    let mut block = [0_u8; Q6_K_BLOCK_BYTES];
    for (index, byte) in block[..128].iter_mut().enumerate() {
        let low = u8::try_from((5 * index + 3) % 16).unwrap();
        let high = u8::try_from((11 * index + 1) % 16).unwrap();
        *byte = low | (high << 4);
    }
    for (index, byte) in block[128..192].iter_mut().enumerate() {
        *byte = u8::try_from((13 * index + 5) % 256).unwrap();
    }
    for (byte, scale) in block[192..208].iter_mut().zip(scales) {
        *byte = scale.to_ne_bytes()[0];
    }
    block[208..].copy_from_slice(&d.to_le_bytes());
    block
}

fn q8_k_block(scale: f32, salt: usize) -> [u8; Q8_K_BLOCK_BYTES] {
    let mut block = [0_u8; Q8_K_BLOCK_BYTES];
    block[..4].copy_from_slice(&scale.to_le_bytes());
    for (index, byte) in block[4..260].iter_mut().enumerate() {
        let value = i8::try_from((17 * index + salt) % 127).unwrap() - 63;
        *byte = value.to_ne_bytes()[0];
    }
    block
}

#[allow(clippy::cast_precision_loss)]
fn oracle_q6_k_q8_k(lhs: &[u8; Q6_K_BLOCK_BYTES], rhs: &[u8; Q8_K_BLOCK_BYTES]) -> f32 {
    let mut decoded = [0_i8; 256];
    for chunk in 0..2 {
        let ql = &lhs[chunk * 64..chunk * 64 + 64];
        let qh = &lhs[128 + chunk * 32..128 + chunk * 32 + 32];
        for lane in 0..32 {
            let high = qh[lane];
            decoded[chunk * 128 + lane] =
                i8::from_ne_bytes([(ql[lane] & 0x0f) | ((high & 3) << 4)]) - 32;
            decoded[chunk * 128 + lane + 32] =
                i8::from_ne_bytes([(ql[lane + 32] & 0x0f) | (((high >> 2) & 3) << 4)]) - 32;
            decoded[chunk * 128 + lane + 64] =
                i8::from_ne_bytes([((ql[lane] >> 4) & 0x0f) | (((high >> 4) & 3) << 4)]) - 32;
            decoded[chunk * 128 + lane + 96] =
                i8::from_ne_bytes([((ql[lane + 32] >> 4) & 0x0f) | (((high >> 6) & 3) << 4)]) - 32;
        }
    }

    let rhs_scale = f32::from_le_bytes(rhs[..4].try_into().unwrap());
    let mut lanes = [0_i32; 8];
    for group in 0..16 {
        let scale = i32::from(i8::from_ne_bytes([lhs[192 + group]]));
        for offset in 0..16 {
            let q8 = i32::from(i8::from_ne_bytes([rhs[4 + group * 16 + offset]]));
            lanes[offset % 8] += scale * i32::from(decoded[group * 16 + offset]) * q8;
        }
    }
    let lhs_scale = fp16_to_f32(u16::from_le_bytes([lhs[208], lhs[209]]));
    (lhs_scale * rhs_scale) * lanes.into_iter().sum::<i32>() as f32
}

#[test]
fn q6_k_q8_k_row_matches_asymmetric_packed_oracle() {
    assert_eq!(PINNED_EMEL_CPP, "843a117386ef17dc5a50549bbfc821074c2141d6");
    let lhs_block = q6_k_block(
        0x3c00,
        [
            3, -5, 7, -9, 11, -13, 15, -17, 19, -21, 23, -25, 27, -29, 31, -32,
        ],
    );
    let rhs_block = q8_k_block(-0.75, 19);
    let lhs = Q6KRow::from_bytes(&lhs_block).unwrap();
    let rhs = Q8KRow::from_bytes(&rhs_block).unwrap();

    assert_eq!(lhs.block_count(), 1);
    assert_eq!(rhs.block_count(), 1);
    assert_eq!(
        dot_q6_k_q8_k(lhs, rhs),
        Ok(oracle_q6_k_q8_k(&lhs_block, &rhs_block))
    );
}

#[test]
fn q6_k_q8_k_row_accumulates_multiple_blocks_without_allocation() {
    let lhs_blocks = [q6_k_block(0x3c00, [1; 16]), q6_k_block(0x4000, [-2; 16])];
    let rhs_blocks = [q8_k_block(0.5, 7), q8_k_block(-1.25, 29)];
    let mut lhs_bytes = Vec::with_capacity(2 * Q6_K_BLOCK_BYTES);
    let mut rhs_bytes = Vec::with_capacity(2 * Q8_K_BLOCK_BYTES);
    for block in lhs_blocks {
        lhs_bytes.extend_from_slice(&block);
    }
    for block in rhs_blocks {
        rhs_bytes.extend_from_slice(&block);
    }
    let lhs = Q6KRow::from_bytes(&lhs_bytes).unwrap();
    let rhs = Q8KRow::from_bytes(&rhs_bytes).unwrap();
    let expected = oracle_q6_k_q8_k(&lhs_blocks[0], &rhs_blocks[0])
        + oracle_q6_k_q8_k(&lhs_blocks[1], &rhs_blocks[1]);
    let allocation = measure(|| assert_eq!(dot_q6_k_q8_k(lhs, rhs), Ok(expected)));
    assert_eq!(allocation.count_total, 0);
    assert_eq!(allocation.bytes_total, 0);
}

#[test]
fn q6_k_q8_k_row_rejects_invalid_lengths_and_block_counts() {
    assert!(matches!(
        Q6KRow::from_bytes(&[0; Q6_K_BLOCK_BYTES - 1]),
        Err(QuantMoreError::InvalidBlockLength {
            format: Format::Q6K,
            bytes,
        }) if bytes == Q6_K_BLOCK_BYTES - 1
    ));
    assert!(matches!(
        Q8KRow::from_bytes(&[0; Q8_K_BLOCK_BYTES - 1]),
        Err(QuantMoreError::InvalidBlockLength {
            format: Format::Q8K,
            bytes,
        }) if bytes == Q8_K_BLOCK_BYTES - 1
    ));
    let lhs_block = q6_k_block(0x3c00, [1; 16]);
    let rhs_blocks = [q8_k_block(1.0, 1), q8_k_block(1.0, 2)];
    let lhs = Q6KRow::from_bytes(&lhs_block).unwrap();
    let rhs_bytes = rhs_blocks.into_iter().flatten().collect::<Vec<_>>();
    let rhs = Q8KRow::from_bytes(&rhs_bytes).unwrap();
    assert_eq!(
        dot_q6_k_q8_k(lhs, rhs),
        Err(QuantMoreError::MismatchedBlockCount { lhs: 1, rhs: 2 })
    );
}
