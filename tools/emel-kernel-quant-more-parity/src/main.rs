//! Live public Rust observer for the pinned scalar `q5_k` by `q8_k` path.

use emel_kernels::any::quant_more::{
    Q5_K_BLOCK_BYTES, Q5KRow, Q8_K_BLOCK_BYTES, Q8KRow, dot_q5_k_q8_k,
};

const SOURCE_COMMIT: &str = "843a117386ef17dc5a50549bbfc821074c2141d6";

#[allow(clippy::cast_possible_truncation)]
fn q5_k_block() -> [u8; Q5_K_BLOCK_BYTES] {
    let mut block = [0_u8; Q5_K_BLOCK_BYTES];
    block[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
    block[2..4].copy_from_slice(&0x3800_u16.to_le_bytes());
    block[4..8].fill(1);
    block[8..12].fill(2);
    block[12..16].fill(1);
    for lane in 0..32 {
        block[16 + lane] = (lane as u8).wrapping_mul(37).wrapping_add(0x5a);
    }
    for chunk in 0..4 {
        for lane in 0..32 {
            let low = (lane + chunk * 3) % 16;
            let high = (15 + chunk + 16 - lane) % 16;
            block[48 + chunk * 32 + lane] = low as u8 | ((high as u8) << 4);
        }
    }
    block
}

#[allow(clippy::cast_possible_truncation)]
fn q8_k_block() -> [u8; Q8_K_BLOCK_BYTES] {
    let mut block = [0_u8; Q8_K_BLOCK_BYTES];
    block[..4].copy_from_slice(&1.0_f32.to_le_bytes());
    for lane in 0..256 {
        block[4 + lane] = ((lane % 23) as i8 - 11).to_ne_bytes()[0];
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

fn write_result(name: &str, result: Result<f32, impl std::fmt::Debug>) {
    match result {
        Ok(value) => println!("case={name} status=ok output_bits={:08x}", value.to_bits()),
        Err(error) => println!("case={name} status=error error={error:?}"),
    }
}

fn main() {
    println!("kernel-quant-more-parity/v1");
    println!("source_repository=stateforward/emel.cpp");
    println!("source_commit={SOURCE_COMMIT}");
    println!("source_layout=q5_k_176_bytes,q8_k_292_bytes,qk_k_256_values");
    println!("source_scalar_dot_span=2944-3000");
    println!("scope=borrowed_scalar_q5_k_q8_k_rows");

    let lhs = q5_k_block();
    let rhs = q8_k_block();
    let lhs_row = Q5KRow::from_bytes(&lhs).expect("q5_k fixture is block aligned");
    let rhs_row = Q8KRow::from_bytes(&rhs).expect("q8_k fixture is block aligned");
    write_result("dot_q5_k_q8_k_varied_qs", dot_q5_k_q8_k(lhs_row, rhs_row));

    let invalid = Q5KRow::from_bytes(&lhs[..Q5_K_BLOCK_BYTES - 1]);
    assert!(invalid.is_err());
    println!(
        "case=invalid_block_length status=error error=InvalidBlockLength {{ format: Q5K, bytes: {} }}",
        Q5_K_BLOCK_BYTES - 1
    );

    let two_rhs = [rhs, rhs].concat();
    let mismatch_rhs = Q8KRow::from_bytes(&two_rhs).expect("two q8_k blocks are aligned");
    let mismatch = dot_q5_k_q8_k(lhs_row, mismatch_rhs);
    assert!(mismatch.is_err());
    println!(
        "case=mismatched_block_count status=error error=MismatchedBlockCount {{ lhs: 1, rhs: 2 }}"
    );
}
