//! Observer for the maintained scalar packed-kernel parity slice.

use emel_kernels::any::quant::{
    Q4_0_BLOCK_BYTES, Q4_0Row, Q8_0_BLOCK_BYTES, Q8_0Row, dot_q4_0_q8_0, dot_q8_0_q8_0,
};
use emel_kernels::any::quant_more::{
    Q2_K_BLOCK_BYTES, Q2KRow, Q3_K_BLOCK_BYTES, Q3KRow, Q4_K_BLOCK_BYTES, Q4KRow, Q5_K_BLOCK_BYTES,
    Q5KRow, Q6_K_BLOCK_BYTES, Q6KRow, Q8_K_BLOCK_BYTES, Q8KRow, dot_q2_k_q8_k, dot_q3_k_q8_k,
    dot_q4_k_q8_k, dot_q5_k_q8_k, dot_q6_k_q8_k,
};

const SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";

fn fixture_byte(value: usize) -> u8 {
    u8::try_from(value).expect("fixture arithmetic stays within a byte")
}

fn fixture_i8(value: usize) -> i8 {
    i8::try_from(value).expect("fixture arithmetic stays within i8")
}

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

#[allow(clippy::needless_range_loop)]
fn q5_k_varied_block() -> [u8; Q5_K_BLOCK_BYTES] {
    let mut block = [0_u8; Q5_K_BLOCK_BYTES];
    block[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
    block[2..4].copy_from_slice(&0x3800_u16.to_le_bytes());
    block[4..8].fill(1);
    block[8..12].fill(2);
    block[12..16].fill(1);
    for lane in 0..32 {
        block[16 + lane] = fixture_byte(lane).wrapping_mul(37).wrapping_add(0x5a);
    }
    for chunk in 0..4 {
        for lane in 0..32 {
            let low = (lane + chunk * 3) % 16;
            let high = (15 + chunk + 16 - lane) % 16;
            block[48 + chunk * 32 + lane] = fixture_byte(low) | (fixture_byte(high) << 4);
        }
    }
    block
}

#[allow(clippy::needless_range_loop)]
fn q8_k_varied_block() -> [u8; Q8_K_BLOCK_BYTES] {
    let mut block = [0_u8; Q8_K_BLOCK_BYTES];
    block[..4].copy_from_slice(&1.0_f32.to_le_bytes());
    for lane in 0..256 {
        block[4 + lane] = fixture_i8(lane % 23).wrapping_sub(11).to_ne_bytes()[0];
    }
    for group in 0..16 {
        let mut sum = 0_i16;
        for lane in 0..16 {
            sum += i16::from(i8::from_ne_bytes([block[4 + group * 16 + lane]]));
        }
        block[260 + group * 2..262 + group * 2].copy_from_slice(&sum.to_le_bytes());
    }
    block
}

#[allow(clippy::needless_range_loop)]
fn q8_k_seeded_block(seed: u8) -> [u8; Q8_K_BLOCK_BYTES] {
    let mut block = [0_u8; Q8_K_BLOCK_BYTES];
    block[..4].copy_from_slice(&1.0_f32.to_le_bytes());
    for lane in 0..256 {
        let value = (lane * 17 + usize::from(seed)) % 29;
        block[4 + lane] = fixture_i8(value).wrapping_sub(14).to_ne_bytes()[0];
    }
    for group in 0..16 {
        let mut sum = 0_i16;
        for lane in 0..16 {
            sum += i16::from(i8::from_ne_bytes([block[4 + group * 16 + lane]]));
        }
        block[260 + group * 2..262 + group * 2].copy_from_slice(&sum.to_le_bytes());
    }
    block
}

#[allow(clippy::needless_range_loop)]
fn q2_k_varied_block(seed: u8) -> [u8; Q2_K_BLOCK_BYTES] {
    let mut block = [0_u8; Q2_K_BLOCK_BYTES];
    for lane in 0..16 {
        let low = fixture_byte(lane).wrapping_mul(29).wrapping_add(seed) & 0x0f;
        let high = fixture_byte(lane).wrapping_mul(17).wrapping_add(3) & 0x0f;
        block[lane] = low | (high << 4);
    }
    for lane in 0..64 {
        block[16 + lane] = fixture_byte(lane).wrapping_mul(73).wrapping_add(seed);
    }
    block[80..82].copy_from_slice(&0x3c00_u16.to_le_bytes());
    block[82..84].copy_from_slice(&0x3800_u16.to_le_bytes());
    block
}

#[allow(clippy::needless_range_loop)]
fn q3_k_varied_block(seed: u8) -> [u8; Q3_K_BLOCK_BYTES] {
    let mut block = [0_u8; Q3_K_BLOCK_BYTES];
    for lane in 0..32 {
        block[lane] = fixture_byte(lane).wrapping_mul(41).wrapping_add(seed);
    }
    for lane in 0..64 {
        block[32 + lane] = fixture_byte(lane)
            .wrapping_mul(67)
            .wrapping_add(seed ^ 0xa5);
    }
    for lane in 0..12 {
        block[96 + lane] = fixture_byte(lane)
            .wrapping_mul(19)
            .wrapping_add(seed ^ 0x3c);
    }
    block[108..110].copy_from_slice(&0x3c00_u16.to_le_bytes());
    block
}

#[allow(clippy::needless_range_loop)]
fn q4_k_varied_block(seed: u8) -> [u8; Q4_K_BLOCK_BYTES] {
    let mut block = [0_u8; Q4_K_BLOCK_BYTES];
    block[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
    block[2..4].copy_from_slice(&0x3800_u16.to_le_bytes());
    for lane in 0..12 {
        block[4 + lane] = fixture_byte(lane).wrapping_mul(23).wrapping_add(seed);
    }
    for lane in 0..128 {
        let low = fixture_byte(lane).wrapping_mul(11).wrapping_add(seed) & 0x0f;
        let high = fixture_byte(lane).wrapping_mul(7).wrapping_add(seed ^ 0x55) & 0x0f;
        block[16 + lane] = low | (high << 4);
    }
    block
}

#[allow(clippy::needless_range_loop)]
fn q6_k_varied_block(seed: u8) -> [u8; Q6_K_BLOCK_BYTES] {
    let mut block = [0_u8; Q6_K_BLOCK_BYTES];
    for lane in 0..128 {
        block[lane] = fixture_byte(lane).wrapping_mul(13).wrapping_add(seed);
    }
    for lane in 0..64 {
        block[128 + lane] = fixture_byte(lane)
            .wrapping_mul(31)
            .wrapping_add(seed ^ 0xc3);
    }
    for lane in 0..16 {
        let scale = fixture_i8((lane * 5 + usize::from(seed)) % 15).wrapping_sub(7);
        block[192 + lane] = scale.to_ne_bytes()[0];
    }
    block[208..210].copy_from_slice(&0x3c00_u16.to_le_bytes());
    block
}

fn write_result(name: &str, result: Result<f32, impl std::fmt::Debug>) {
    match result {
        Ok(value) => println!("case={name} status=ok output_bits={:08x}", value.to_bits()),
        Err(error) => println!("case={name} status=error error={error:?}"),
    }
}

fn main() {
    println!("kernel-quantized-parity/v4");
    println!("source_repository=stateforward/emel.cpp");
    println!("source_commit={SOURCE_COMMIT}");
    println!(
        "scope=scalar_q4_0_q8_0_q5_k_q8_k_q2_k_row_aggregation_via_block_scalar_reference_q3_k_q4_k_q6_k_rows"
    );

    let q4 = [q4_block(0x3c00, 0, 15), q4_block(0x3800, 9, 7)].concat();
    let q8 = [q8_block(0x3800, 2), q8_block(0x3c00, -3)].concat();
    let q8_other = [q8_block(0x3c00, 3), q8_block(0x3800, -2)].concat();

    let q4_row = Q4_0Row::from_bytes(&q4).expect("q4 fixture is block aligned");
    let q8_row = Q8_0Row::from_bytes(&q8).expect("q8 fixture is block aligned");
    let q8_other_row = Q8_0Row::from_bytes(&q8_other).expect("q8 fixture is block aligned");

    write_result("dot_q4_0_q8_0", dot_q4_0_q8_0(q4_row, q8_row));
    write_result("dot_q8_0_q8_0", dot_q8_0_q8_0(q8_row, q8_other_row));

    let q5_k = q5_k_varied_block();
    let q8_k = q8_k_varied_block();
    let q5_k_row = Q5KRow::from_bytes(&q5_k).expect("varied q5_k fixture is block aligned");
    let q8_k_row = Q8KRow::from_bytes(&q8_k).expect("varied q8_k fixture is block aligned");
    write_result("dot_q5_k_q8_k_varied_qs", dot_q5_k_q8_k(q5_k_row, q8_k_row));

    let q2_k = [q2_k_varied_block(17), q2_k_varied_block(91)].concat();
    let q3_k = q3_k_varied_block(29);
    let q4_k = q4_k_varied_block(41);
    let q6_k = q6_k_varied_block(53);
    let q2_k_row = Q2KRow::from_bytes(&q2_k).expect("varied q2_k fixture is block aligned");
    let q3_k_row = Q3KRow::from_bytes(&q3_k).expect("varied q3_k fixture is block aligned");
    let q4_k_row = Q4KRow::from_bytes(&q4_k).expect("varied q4_k fixture is block aligned");
    let q6_k_row = Q6KRow::from_bytes(&q6_k).expect("varied q6_k fixture is block aligned");
    let q2_q8 = [q8_k_seeded_block(17), q8_k_seeded_block(91)].concat();
    let q3_q8 = q8_k_seeded_block(29);
    let q4_q8 = q8_k_seeded_block(41);
    let q6_q8 = q8_k_seeded_block(53);
    write_result(
        "dot_q2_k_q8_k_row_aggregation_via_block_scalar_reference",
        dot_q2_k_q8_k(
            q2_k_row,
            Q8KRow::from_bytes(&q2_q8).expect("q2 q8 fixture is block aligned"),
        ),
    );
    write_result(
        "dot_q3_k_q8_k_varied",
        dot_q3_k_q8_k(
            q3_k_row,
            Q8KRow::from_bytes(&q3_q8).expect("q3 q8 fixture is block aligned"),
        ),
    );
    write_result(
        "dot_q4_k_q8_k_varied",
        dot_q4_k_q8_k(
            q4_k_row,
            Q8KRow::from_bytes(&q4_q8).expect("q4 q8 fixture is block aligned"),
        ),
    );
    write_result(
        "dot_q6_k_q8_k_varied",
        dot_q6_k_q8_k(
            q6_k_row,
            Q8KRow::from_bytes(&q6_q8).expect("q6 q8 fixture is block aligned"),
        ),
    );
}
