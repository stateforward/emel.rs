#![allow(clippy::float_cmp)]
#![allow(missing_docs)]
#![cfg(target_arch = "aarch64")]

use allocation_counter::measure;
use emel_kernels::aarch64::BroadcastF32Error;
use emel_kernels::aarch64::{
    BinaryAdd, BinaryDiv, BinaryF32Error, BinaryMul, BinarySub, BroadcastAdd, BroadcastMul, Dup,
    Gemv, Kernel, KernelError, MulMatArgmaxQ4PackedF32Bl4, MulMatArgmaxQ4PackedF32Bl8,
    MulMatArgmaxQ6Packed, MulMatF32, MulMatQ2K, MulMatQ2KVector, MulMatQ3K, MulMatQ3KVector,
    MulMatQ4_0Vector, MulMatQ4_1Vector, MulMatQ4K, MulMatQ4KVector, MulMatQ4PackedBl4,
    MulMatQ4PackedBl4MatrixX4, MulMatQ4PackedBl8, MulMatQ4PackedBl8MatrixX4,
    MulMatQ4PackedBl8MatrixX8, MulMatQ4PackedF32Bl4, MulMatQ4PackedF32Bl8, MulMatQ5_0Vector,
    MulMatQ6, MulMatQ6Packed, MulMatQ6PackedMatrixX4, MulMatQ6Prepared, MulMatQ6PreparedMatrixX4,
    MulMatQ6PreparedMatrixX8, MulMatQ6Vector, MulMatQ8_0PackedBl4, MulMatQ8_0PackedBl8,
    MulMatQ8_0PackedBl8MatrixX4, MulMatQ8_0Vector, MulMatQ8_0VectorQ8Rhs, OpScalarUnaryElu,
    OpScalarUnaryExp, OpScalarUnaryGelu, OpScalarUnarySilu, OpScalarUnaryTanh,
    TARGET_AARCH64_UNARY_SILU_RESIDUAL, TARGET_AARCH64_UNARY_SILU_SOURCE_ACTION_SPAN,
    TARGET_AARCH64_UNARY_SILU_SOURCE_COMMIT, TARGET_AARCH64_UNARY_SILU_SOURCE_GUARD_SPAN,
    TARGET_AARCH64_UNARY_SILU_SOURCE_SM_BLOB, UnarySilu, UnexpectedAarch64Kernel,
};
use emel_kernels::any::unary::UnaryError;

#[test]
fn router_dispatches_mul_mat_f32() {
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let lhs = [1.0_f32; 8];
    let rhs = [0.0_f32; 16];
    let mut output = [f32::NAN; 8];
    assert_eq!(
        kernel.process_event(MulMatF32::new(&lhs, &rhs, 2, 4, 4), &mut output),
        Ok(())
    );
    assert_eq!(output, [0.0; 8]);
}

#[test]
fn router_is_ready_and_dispatches_dup() {
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    assert!(kernel.is_ready());

    let input = [1.0_f32, -2.5, 4.0, 8.25];
    let mut output = [0.0_f32; 4];
    assert_eq!(kernel.process_event(Dup::new(&input), &mut output), Ok(()));
    assert_eq!(output, input);
    assert!(kernel.is_ready());
}

#[test]
fn router_dispatches_binary_add() {
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let lhs = [1.0_f32, -2.0, 5.5, 8.0];
    let rhs = [2.0_f32, 4.0, -0.5, 1.5];
    let mut output = [0.0_f32; 4];

    assert_eq!(
        kernel.process_event(BinaryAdd::new(&lhs, &rhs), &mut output),
        Ok(())
    );
    assert_eq!(output, [3.0, 2.0, 5.0, 9.5]);
}

#[test]
fn router_dispatches_all_native_binary_routes() {
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let lhs = [8.0_f32, -9.0, 6.0, 4.0];
    let rhs = [2.0_f32, 3.0, -2.0, 4.0];
    let mut output = [0.0_f32; 4];

    assert_eq!(
        kernel.process_event(BinaryAdd::new(&lhs, &rhs), &mut output),
        Ok(())
    );
    assert_eq!(output, [10.0, -6.0, 4.0, 8.0]);
    assert_eq!(
        kernel.process_event(BinarySub::new(&lhs, &rhs), &mut output),
        Ok(())
    );
    assert_eq!(output, [6.0, -12.0, 8.0, 0.0]);
    assert_eq!(
        kernel.process_event(BinaryMul::new(&lhs, &rhs), &mut output),
        Ok(())
    );
    assert_eq!(output, [16.0, -27.0, -12.0, 16.0]);
    assert_eq!(
        kernel.process_event(BinaryDiv::new(&lhs, &rhs), &mut output),
        Ok(())
    );
    assert_eq!(output, [4.0, -3.0, -3.0, 1.0]);
    assert!(kernel.is_ready());
}

#[test]
fn router_native_binary_rejection_preserves_sentinel_and_recovers() {
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let mut output = [f32::from_bits(0x7fc0_0001); 4];

    assert_eq!(
        kernel.process_event(BinaryAdd::new(&[1.0], &[2.0, 3.0]), &mut output,),
        Err(BinaryF32Error::InvalidShape)
    );
    assert!(output.iter().all(|value| value.to_bits() == 0x7fc0_0001));
    assert!(kernel.is_ready());

    assert_eq!(
        kernel.process_event(
            BinaryMul::new(&[2.0, 3.0, 4.0, 5.0], &[4.0, 3.0, 2.0, 1.0]),
            &mut output,
        ),
        Ok(())
    );
    assert_eq!(output, [8.0, 9.0, 8.0, 5.0]);
}

#[test]
fn router_native_binary_dispatch_is_allocation_free() {
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let lhs = [8.0_f32, -9.0, 6.0, 4.0];
    let rhs = [2.0_f32, 3.0, -2.0, 4.0];
    let mut output = [0.0_f32; 4];
    let allocation = measure(|| {
        for _ in 0..64 {
            assert_eq!(
                kernel.process_event(BinaryDiv::new(&lhs, &rhs), &mut output),
                Ok(())
            );
        }
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(output, [4.0, -3.0, -3.0, 1.0]);
}

#[test]
fn router_dispatches_binary_row_broadcast_add_and_mul() {
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let input = [8.0_f32, -9.0, 6.0, 4.0, -2.0, 3.0, 7.0, 11.0];
    let row = [2.0_f32, 3.0, -2.0, 4.0];
    let mut output = [0.0_f32; 8];

    assert_eq!(
        kernel.process_event(BroadcastAdd::new(&input, &row), &mut output),
        Ok(())
    );
    assert_eq!(output, [10.0, -6.0, 4.0, 8.0, 0.0, 6.0, 5.0, 15.0]);

    assert_eq!(
        kernel.process_event(BroadcastMul::new(&input, &row), &mut output),
        Ok(())
    );
    assert_eq!(output, [16.0, -27.0, -12.0, 16.0, -4.0, 9.0, -14.0, 44.0]);
    assert!(kernel.is_ready());
}

#[test]
fn router_rejects_binary_row_broadcast_without_mutation_and_recovers() {
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let mut output = [9.0_f32; 4];

    assert_eq!(
        kernel.process_event(
            BroadcastAdd::new(&[1.0, 2.0, 3.0], &[4.0, 5.0]),
            &mut output,
        ),
        Err(BroadcastF32Error::InvalidShape)
    );
    assert_eq!(output, [9.0; 4]);
    assert!(kernel.is_ready());

    assert_eq!(
        kernel.process_event(
            BroadcastMul::new(&[1.0, 2.0, 3.0, 4.0], &[2.0, 3.0]),
            &mut output,
        ),
        Ok(())
    );
    assert_eq!(output, [2.0, 6.0, 6.0, 12.0]);
}

#[test]
fn router_binary_row_broadcast_dispatch_is_allocation_free() {
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let input = [1.0_f32; 12];
    let row = [2.0_f32, 3.0, 4.0];
    let mut output = [0.0_f32; 12];
    let allocation = measure(|| {
        for _ in 0..64 {
            assert_eq!(
                kernel.process_event(BroadcastAdd::new(&input, &row), &mut output),
                Ok(())
            );
        }
    });
    assert_eq!(allocation.count_total, 0);
    assert_eq!(
        output,
        [3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0, 3.0, 4.0, 5.0]
    );
}

#[test]
fn router_dispatches_gemv() {
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let lhs = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs = [7.0_f32, 8.0, 9.0];
    let mut output = [f32::NAN; 2];

    assert_eq!(
        kernel.process_event(Gemv::new(&lhs, &rhs, 2, 3), &mut output),
        Ok(())
    );
    assert_eq!(output, [50.0, 122.0]);
}

#[test]
fn router_dispatches_q4_0_vector() {
    use emel_kernels::any::quant::Q4_0_BLOCK_BYTES;

    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let mut lhs = [0_u8; Q4_0_BLOCK_BYTES];
    lhs[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
    for packed in &mut lhs[2..] {
        *packed = 0x88;
    }
    let rhs = [1.0_f32; 32];
    let mut output = [f32::NAN; 1];
    assert_eq!(
        kernel.process_event(MulMatQ4_0Vector::new(&lhs, &rhs, 1, 32), &mut output),
        Ok(())
    );
    assert_eq!(output, [0.0]);
}

#[test]
fn router_dispatches_q4_1_vector() {
    const Q4_1_BLOCK_BYTES: usize = 20;
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let mut lhs = [0_u8; Q4_1_BLOCK_BYTES];
    lhs[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
    let rhs = [1.0_f32; 32];
    let mut output = [f32::NAN; 1];
    assert_eq!(
        kernel.process_event(MulMatQ4_1Vector::new(&lhs, &rhs, 1, 32), &mut output),
        Ok(())
    );
    assert_eq!(output, [0.0]);
}

#[test]
fn router_dispatches_q5_0_vector() {
    const Q5_0_BLOCK_BYTES: usize = 22;
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let mut lhs = [0_u8; Q5_0_BLOCK_BYTES];
    lhs[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
    lhs[2..6].copy_from_slice(&u32::MAX.to_le_bytes());
    let rhs = [1.0_f32; 32];
    let mut output = [f32::NAN; 1];
    assert_eq!(
        kernel.process_event(MulMatQ5_0Vector::new(&lhs, &rhs, 1, 32), &mut output),
        Ok(())
    );
    assert_eq!(output, [0.0]);
}

#[test]
fn router_dispatches_q8_0_vector() {
    use emel_kernels::any::quant::Q8_0_BLOCK_BYTES;
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let mut lhs = [0_u8; Q8_0_BLOCK_BYTES];
    lhs[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
    let rhs = [1.0_f32; 32];
    let mut output = [f32::NAN; 1];
    assert_eq!(
        kernel.process_event(MulMatQ8_0Vector::new(&lhs, &rhs, 1, 32), &mut output),
        Ok(())
    );
    assert_eq!(output, [0.0]);
}

#[test]
fn router_dispatches_q4_k_vector() {
    const Q4_K_BLOCK_BYTES: usize = 144;
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let lhs = [0_u8; Q4_K_BLOCK_BYTES];
    let rhs = [0.0_f32; 256];
    let mut output = [f32::NAN; 1];
    assert_eq!(
        kernel.process_event(MulMatQ4KVector::new(&lhs, &rhs, 1, 256), &mut output),
        Ok(())
    );
    assert_eq!(output, [0.0]);
}

#[test]
fn router_dispatches_q4_packed_bl4() {
    const Q4_K_X8_BLOCK_BYTES: usize = 1152;
    const Q8_K_BLOCK_BYTES: usize = 292;
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let lhs = [0_u8; Q4_K_X8_BLOCK_BYTES];
    let rhs = [0_u8; Q8_K_BLOCK_BYTES];
    let mut output = [f32::NAN; 1];
    assert_eq!(
        kernel.process_event(MulMatQ4PackedBl4::new(&lhs, &rhs, 1, 256), &mut output),
        Ok(())
    );
    assert_eq!(output, [0.0]);
}

#[test]
fn router_dispatches_q4_packed_bl4_matrix_x4() {
    const Q4_K_X8_BLOCK_BYTES: usize = 1152;
    const Q8_K_BLOCK_BYTES: usize = 292;
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let lhs = [0_u8; Q4_K_X8_BLOCK_BYTES];
    let rhs = [0_u8; 4 * Q8_K_BLOCK_BYTES];
    let mut output = [f32::NAN; 4];
    assert_eq!(
        kernel.process_event(
            MulMatQ4PackedBl4MatrixX4::new(&lhs, &rhs, 1, 256),
            &mut output
        ),
        Ok(())
    );
    assert_eq!(output, [0.0; 4]);
}

#[test]
fn router_dispatches_q4_packed_bl8() {
    const Q4_K_X8_BLOCK_BYTES: usize = 1152;
    const Q8_K_BLOCK_BYTES: usize = 292;
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let lhs = [0_u8; Q4_K_X8_BLOCK_BYTES];
    let rhs = [0_u8; Q8_K_BLOCK_BYTES];
    let mut output = [f32::NAN; 1];
    assert_eq!(
        kernel.process_event(MulMatQ4PackedBl8::new(&lhs, &rhs, 1, 256), &mut output),
        Ok(())
    );
    assert_eq!(output, [0.0]);
}

#[test]
fn router_dispatches_q4_packed_f32_bl4() {
    const Q4_K_X8_BLOCK_BYTES: usize = 1152;
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let lhs = [0_u8; Q4_K_X8_BLOCK_BYTES];
    let rhs = [0.0_f32; 256];
    let mut output = [f32::NAN; 1];
    assert_eq!(
        kernel.process_event(MulMatQ4PackedF32Bl4::new(&lhs, &rhs, 1, 256), &mut output),
        Ok(())
    );
    assert_eq!(output, [0.0]);
}

#[test]
fn router_dispatches_q4_packed_f32_bl8() {
    const Q4_K_X8_BLOCK_BYTES: usize = 1152;
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let lhs = [0_u8; Q4_K_X8_BLOCK_BYTES];
    let rhs = [0.0_f32; 256];
    let mut output = [f32::NAN; 1];
    assert_eq!(
        kernel.process_event(MulMatQ4PackedF32Bl8::new(&lhs, &rhs, 1, 256), &mut output),
        Ok(())
    );
    assert_eq!(output, [0.0]);
}

#[test]
fn router_dispatches_q4_packed_bl8_matrix_x4() {
    const Q4_K_X8_BLOCK_BYTES: usize = 1152;
    const Q8_K_BLOCK_BYTES: usize = 292;
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let lhs = [0_u8; Q4_K_X8_BLOCK_BYTES];
    let rhs = [0_u8; 4 * Q8_K_BLOCK_BYTES];
    let mut output = [f32::NAN; 4];
    assert_eq!(
        kernel.process_event(
            MulMatQ4PackedBl8MatrixX4::new(&lhs, &rhs, 1, 256),
            &mut output
        ),
        Ok(())
    );
    assert_eq!(output, [0.0; 4]);
}

#[test]
fn router_dispatches_q4_packed_bl8_matrix_x8() {
    const Q4_K_X8_BLOCK_BYTES: usize = 1152;
    const Q8_K_BLOCK_BYTES: usize = 292;
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let lhs = [0_u8; Q4_K_X8_BLOCK_BYTES];
    let rhs = [0_u8; 8 * Q8_K_BLOCK_BYTES];
    let mut output = [f32::NAN; 8];
    assert_eq!(
        kernel.process_event(
            MulMatQ4PackedBl8MatrixX8::new(&lhs, &rhs, 1, 256),
            &mut output
        ),
        Ok(())
    );
    assert_eq!(output, [0.0; 8]);
}

#[test]
fn router_dispatches_q4_k_gemm() {
    const Q4_K_BLOCK_BYTES: usize = 144;
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let lhs = [0_u8; 2 * Q4_K_BLOCK_BYTES];
    let rhs = [0.0_f32; 512];
    let mut output = [f32::NAN; 4];
    assert_eq!(
        kernel.process_event(MulMatQ4K::new(&lhs, &rhs, 2, 256, 2), &mut output),
        Ok(())
    );
    assert_eq!(output, [0.0; 4]);
}

#[test]
fn router_dispatches_q6_gemm() {
    const Q6_K_BLOCK_BYTES: usize = 210;
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let lhs = [0_u8; 2 * Q6_K_BLOCK_BYTES];
    let rhs = [0.0_f32; 512];
    let mut output = [f32::NAN; 4];
    assert_eq!(
        kernel.process_event(MulMatQ6::new(&lhs, &rhs, 2, 256, 2), &mut output),
        Ok(())
    );
    assert_eq!(output, [0.0; 4]);
}

#[test]
fn router_dispatches_q6_packed() {
    const Q6_K_X8_BLOCK_BYTES: usize = 1680;
    const Q8_K_BLOCK_BYTES: usize = 292;
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let lhs = [0_u8; Q6_K_X8_BLOCK_BYTES];
    let rhs = [0_u8; Q8_K_BLOCK_BYTES];
    let mut output = [f32::NAN; 1];
    assert_eq!(
        kernel.process_event(MulMatQ6Packed::new(&lhs, &rhs, 1, 256), &mut output),
        Ok(())
    );
    assert_eq!(output, [0.0]);
}

#[test]
fn router_dispatches_q6_packed_argmax() {
    const Q6_K_X8_BLOCK_BYTES: usize = 1680;
    const Q8_K_BLOCK_BYTES: usize = 292;
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let lhs = [0_u8; Q6_K_X8_BLOCK_BYTES];
    let rhs = [0_u8; Q8_K_BLOCK_BYTES];
    let mut output = [f32::NAN; 1];
    assert_eq!(
        kernel.process_event(MulMatArgmaxQ6Packed::new(&lhs, &rhs, 1, 256), &mut output),
        Ok(0)
    );
    assert_eq!(output, [0.0]);
}

#[test]
fn router_dispatches_q4_packed_f32_bl4_argmax() {
    const Q4_K_X8_BLOCK_BYTES: usize = 1152;
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let lhs = [0_u8; Q4_K_X8_BLOCK_BYTES];
    let rhs = [0.0_f32; 256];
    let mut output = [f32::NAN; 1];
    assert_eq!(
        kernel.process_event(
            MulMatArgmaxQ4PackedF32Bl4::new(&lhs, &rhs, 1, 256),
            &mut output
        ),
        Ok(0)
    );
    assert_eq!(output, [0.0]);
}

#[test]
fn router_dispatches_q4_packed_f32_bl8_argmax() {
    const Q4_K_X8_BLOCK_BYTES: usize = 1152;
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let lhs = [0_u8; Q4_K_X8_BLOCK_BYTES];
    let rhs = [0.0_f32; 256];
    let mut output = [f32::NAN; 1];
    assert_eq!(
        kernel.process_event(
            MulMatArgmaxQ4PackedF32Bl8::new(&lhs, &rhs, 1, 256),
            &mut output
        ),
        Ok(0)
    );
    assert_eq!(output, [0.0]);
}

#[test]
fn router_dispatches_q6_packed_matrix_x4() {
    const Q6_K_X8_BLOCK_BYTES: usize = 1680;
    const Q8_K_BLOCK_BYTES: usize = 292;
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let lhs = [0_u8; Q6_K_X8_BLOCK_BYTES];
    let rhs = [0_u8; 4 * Q8_K_BLOCK_BYTES];
    let mut output = [f32::NAN; 4];
    assert_eq!(
        kernel.process_event(MulMatQ6PackedMatrixX4::new(&lhs, &rhs, 1, 256), &mut output),
        Ok(())
    );
    assert_eq!(output, [0.0; 4]);
}

#[test]
fn router_dispatches_q6_prepared() {
    const Q6_K_X8_PREPARED_BLOCK_BYTES: usize = 2192;
    const Q8_K_BLOCK_BYTES: usize = 292;
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let lhs = [0_u8; Q6_K_X8_PREPARED_BLOCK_BYTES];
    let rhs = [0_u8; Q8_K_BLOCK_BYTES];
    let mut output = [f32::NAN; 1];
    assert_eq!(
        kernel.process_event(MulMatQ6Prepared::new(&lhs, &rhs, 1, 256), &mut output),
        Ok(())
    );
    assert_eq!(output, [0.0]);
}

#[test]
fn router_dispatches_q6_prepared_matrix_x4() {
    const Q6_K_X8_PREPARED_BLOCK_BYTES: usize = 2192;
    const Q8_K_BLOCK_BYTES: usize = 292;
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let lhs = [0_u8; Q6_K_X8_PREPARED_BLOCK_BYTES];
    let rhs = [0_u8; 4 * Q8_K_BLOCK_BYTES];
    let mut output = [f32::NAN; 4];
    assert_eq!(
        kernel.process_event(
            MulMatQ6PreparedMatrixX4::new(&lhs, &rhs, 1, 256),
            &mut output
        ),
        Ok(())
    );
    assert_eq!(output, [0.0; 4]);
}

#[test]
fn router_dispatches_q6_prepared_matrix_x8() {
    const Q6_K_X8_PREPARED_BLOCK_BYTES: usize = 2192;
    const Q8_K_BLOCK_BYTES: usize = 292;
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let lhs = [0_u8; Q6_K_X8_PREPARED_BLOCK_BYTES];
    let rhs = [0_u8; 8 * Q8_K_BLOCK_BYTES];
    let mut output = [f32::NAN; 8];
    assert_eq!(
        kernel.process_event(
            MulMatQ6PreparedMatrixX8::new(&lhs, &rhs, 1, 256),
            &mut output
        ),
        Ok(())
    );
    assert_eq!(output, [0.0; 8]);
}

#[test]
fn router_dispatches_q6_vector() {
    const Q6_K_BLOCK_BYTES: usize = 210;
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let lhs = [0_u8; Q6_K_BLOCK_BYTES];
    let rhs = [1.0_f32; 256];
    let mut output = [f32::NAN; 1];
    assert_eq!(
        kernel.process_event(MulMatQ6Vector::new(&lhs, &rhs, 1, 256), &mut output),
        Ok(())
    );
    assert_eq!(output, [0.0]);
}

#[test]
fn router_dispatches_q2_k_vector() {
    const Q2_K_BLOCK_BYTES: usize = 84;
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let lhs = [0_u8; Q2_K_BLOCK_BYTES];
    let rhs = [0.0_f32; 256];
    let mut output = [f32::NAN; 1];
    assert_eq!(
        kernel.process_event(MulMatQ2KVector::new(&lhs, &rhs, 1, 256), &mut output),
        Ok(())
    );
    assert_eq!(output, [0.0]);
}

#[test]
fn router_dispatches_q2_k_gemm() {
    const Q2_K_BLOCK_BYTES: usize = 84;
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let lhs = [0_u8; 2 * Q2_K_BLOCK_BYTES];
    let rhs = [0.0_f32; 512];
    let mut output = [f32::NAN; 4];
    assert_eq!(
        kernel.process_event(MulMatQ2K::new(&lhs, &rhs, 2, 256, 2), &mut output),
        Ok(())
    );
    assert_eq!(output, [0.0; 4]);
}

#[test]
fn router_dispatches_q3_k_vector() {
    const Q3_K_BLOCK_BYTES: usize = 110;
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let lhs = [0_u8; Q3_K_BLOCK_BYTES];
    let rhs = [0.0_f32; 256];
    let mut output = [f32::NAN; 1];
    assert_eq!(
        kernel.process_event(MulMatQ3KVector::new(&lhs, &rhs, 1, 256), &mut output),
        Ok(())
    );
    assert_eq!(output, [0.0]);
}

#[test]
fn router_dispatches_q3_k_gemm() {
    const Q3_K_BLOCK_BYTES: usize = 110;
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let lhs = [0_u8; 2 * Q3_K_BLOCK_BYTES];
    let rhs = [0.0_f32; 512];
    let mut output = [f32::NAN; 4];
    assert_eq!(
        kernel.process_event(MulMatQ3K::new(&lhs, &rhs, 2, 256, 2), &mut output),
        Ok(())
    );
    assert_eq!(output, [0.0; 4]);
}

#[test]
fn router_dispatches_q8_0_vector_q8_rhs() {
    use emel_kernels::any::quant::Q8_0_BLOCK_BYTES;
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let mut lhs = [0_u8; Q8_0_BLOCK_BYTES];
    let mut rhs = [0_u8; Q8_0_BLOCK_BYTES];
    lhs[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
    rhs[..2].copy_from_slice(&0x3800_u16.to_le_bytes());
    let mut output = [f32::NAN; 1];
    assert_eq!(
        kernel.process_event(MulMatQ8_0VectorQ8Rhs::new(&lhs, &rhs, 1, 32), &mut output),
        Ok(())
    );
    assert_eq!(output, [0.0]);
}

#[test]
fn router_dispatches_q8_0_packed_bl4() {
    const Q8_0_X4_BLOCK_BYTES: usize = 136;
    use emel_kernels::any::quant::Q8_0_BLOCK_BYTES;
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let lhs = [0_u8; Q8_0_X4_BLOCK_BYTES];
    let mut rhs = [0_u8; Q8_0_BLOCK_BYTES];
    rhs[..2].copy_from_slice(&0x3800_u16.to_le_bytes());
    let mut output = [f32::NAN; 1];
    assert_eq!(
        kernel.process_event(MulMatQ8_0PackedBl4::new(&lhs, &rhs, 1, 32), &mut output),
        Ok(())
    );
    assert_eq!(output, [0.0]);
}

#[test]
fn router_dispatches_q8_0_packed_bl8() {
    const Q8_0_X4_BLOCK_BYTES: usize = 136;
    use emel_kernels::any::quant::Q8_0_BLOCK_BYTES;
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let lhs = [0_u8; Q8_0_X4_BLOCK_BYTES];
    let mut rhs = [0_u8; Q8_0_BLOCK_BYTES];
    rhs[..2].copy_from_slice(&0x3800_u16.to_le_bytes());
    let mut output = [f32::NAN; 1];
    assert_eq!(
        kernel.process_event(MulMatQ8_0PackedBl8::new(&lhs, &rhs, 1, 32), &mut output),
        Ok(())
    );
    assert_eq!(output, [0.0]);
}

#[test]
fn router_dispatches_q8_0_packed_bl8_matrix_x4() {
    const Q8_0_X4_BLOCK_BYTES: usize = 136;
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let lhs = [0_u8; Q8_0_X4_BLOCK_BYTES];
    let rhs = [0_u8; Q8_0_X4_BLOCK_BYTES];
    let mut output = [f32::NAN; 16];
    assert_eq!(
        kernel.process_event(
            MulMatQ8_0PackedBl8MatrixX4::new(&lhs, &rhs, 4, 32),
            &mut output
        ),
        Ok(())
    );
    assert_eq!(output, [0.0; 16]);
}

#[test]
fn router_dispatches_neon_silu_and_preserves_vector_contract() {
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let input = [-4.0_f32, -1.0, 0.0, 4.0];
    let mut output = [f32::NAN; 4];

    assert_eq!(
        kernel.process_event(UnarySilu::new(&input), &mut output),
        Ok(())
    );
    assert_eq!(
        output.map(f32::to_bits),
        [0xbd93_57d2, 0xbe89_b2b0, 0x0000_0000, 0x407b_6541]
    );
    assert!(kernel.is_ready());
}

#[test]
fn router_neon_silu_executes_scalar_tail() {
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let mut output = [f32::NAN; 3];

    assert_eq!(
        kernel.process_event(UnarySilu::new(&[1.0, 2.0, 3.0]), &mut output),
        Ok(())
    );
    assert_eq!(
        output.map(f32::to_bits),
        [0x3f3b_26a8, 0x3fe1_7bea, 0x4036_e4ed]
    );
    assert!(kernel.is_ready());
}

#[test]
fn router_neon_silu_shape_mismatch_does_not_mutate() {
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let mut output = [7.0_f32; 2];
    assert!(
        kernel
            .process_event(UnarySilu::new(&[1.0, 2.0, 3.0]), &mut output)
            .is_err()
    );
    assert_eq!(output, [7.0; 2]);
    assert!(kernel.is_ready());
}

#[test]
fn router_neon_silu_dispatch_is_allocation_free() {
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let input = [0.25_f32; 4];
    let mut output = [0.0_f32; 4];
    let allocation = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                kernel.process_event(UnarySilu::new(&input), &mut output),
                Ok(())
            );
        }
    });
    assert_eq!(allocation.count_total, 0);
}

#[test]
fn router_neon_silu_source_identity_is_pinned() {
    assert_eq!(
        TARGET_AARCH64_UNARY_SILU_SOURCE_COMMIT,
        "843a117386ef17dc5a50549bbfc821074c2141d6"
    );
    assert_eq!(
        TARGET_AARCH64_UNARY_SILU_SOURCE_GUARD_SPAN,
        "src/emel/kernel/aarch64/guards.hpp:826-886"
    );
    assert_eq!(
        TARGET_AARCH64_UNARY_SILU_SOURCE_ACTION_SPAN,
        "src/emel/kernel/aarch64/actions.hpp:89-123,180-196"
    );
    assert_eq!(
        TARGET_AARCH64_UNARY_SILU_SOURCE_SM_BLOB,
        "865a9cc6ba6115382ed043c464f3d62bcd851357"
    );
    assert_eq!(
        TARGET_AARCH64_UNARY_SILU_RESIDUAL,
        "none for the pinned dense F32 SiLU route"
    );
}

#[test]
fn router_rejects_invalid_child_request_without_mutation() {
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let mut output = [7.0_f32; 3];
    let before = output;

    assert!(
        kernel
            .process_event(Dup::new(&[1.0, 2.0]), &mut output)
            .is_err()
    );
    assert_eq!(output, before);
    assert!(kernel.is_ready());
}

#[test]
fn router_unexpected_event_is_explicit() {
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    assert_eq!(
        kernel.process_event(UnexpectedAarch64Kernel, &mut []),
        Err(KernelError::UnexpectedEvent)
    );
}

#[test]
fn router_dispatch_is_allocation_free_after_construction() {
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let input = [1.0_f32; 17];
    let mut output = [0.0_f32; 17];
    let allocation = measure(|| {
        for _ in 0..128 {
            assert_eq!(kernel.process_event(Dup::new(&input), &mut output), Ok(()));
        }
    });
    assert_eq!(allocation.count_total, 0);
}

#[test]
fn router_dispatches_scalar_unary_slice() {
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let input = [-1.0_f32, 0.0, 1.0];
    let mut output = [0.0_f32; 3];

    assert_eq!(
        kernel.process_event(OpScalarUnaryExp::new(&input), &mut output),
        Ok(())
    );
    assert!((output[0] - (-1.0_f32).exp()).abs() < f32::EPSILON);
    assert!((output[2] - 1.0_f32.exp()).abs() < f32::EPSILON);

    assert_eq!(
        kernel.process_event(OpScalarUnaryTanh::new(&input), &mut output),
        Ok(())
    );
    assert!((output[0] - (-1.0_f32).tanh()).abs() < f32::EPSILON);
    assert!((output[2] - 1.0_f32.tanh()).abs() < f32::EPSILON);

    assert_eq!(
        kernel.process_event(OpScalarUnaryElu::new(&input), &mut output),
        Ok(())
    );
    assert!((output[0] - (-1.0_f32).exp_m1()).abs() < f32::EPSILON);
    assert_eq!(output[2], 1.0);

    assert_eq!(
        kernel.process_event(OpScalarUnaryGelu::new(&input), &mut output),
        Ok(())
    );
    assert!(output[0].is_finite() && output[2].is_finite());

    assert_eq!(
        kernel.process_event(OpScalarUnarySilu::new(&input), &mut output),
        Ok(())
    );
    assert!((output[0] - (-1.0_f32 / (1.0 + 1.0_f32.exp()))).abs() < f32::EPSILON);
    assert!((output[2] - (1.0_f32 / (1.0 + (-1.0_f32).exp()))).abs() < f32::EPSILON);
    assert!(kernel.is_ready());
}

#[test]
fn scalar_unary_invalid_shape_does_not_mutate_and_recovers() {
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let mut output = [7.0_f32; 2];
    let before = output;

    assert_eq!(
        kernel.process_event(OpScalarUnaryExp::new(&[1.0]), &mut output),
        Err(UnaryError::ShapeMismatch)
    );
    assert_eq!(output, before);

    assert_eq!(
        kernel.process_event(OpScalarUnaryTanh::new(&[]), &mut []),
        Err(UnaryError::InvalidView)
    );

    let input = [0.5_f32; 2];
    assert_eq!(
        kernel.process_event(OpScalarUnarySilu::new(&input), &mut output),
        Ok(())
    );
    assert!(kernel.is_ready());
}

#[test]
fn scalar_unary_dispatch_is_allocation_free_after_construction() {
    let Some(mut kernel) = Kernel::try_new() else {
        return;
    };
    let input = [0.25_f32; 17];
    let mut output = [0.0_f32; 17];
    let allocation = measure(|| {
        for _ in 0..128 {
            assert_eq!(
                kernel.process_event(OpScalarUnaryExp::new(&input), &mut output),
                Ok(())
            );
            assert_eq!(
                kernel.process_event(OpScalarUnaryTanh::new(&input), &mut output),
                Ok(())
            );
            assert_eq!(
                kernel.process_event(OpScalarUnaryElu::new(&input), &mut output),
                Ok(())
            );
            assert_eq!(
                kernel.process_event(OpScalarUnaryGelu::new(&input), &mut output),
                Ok(())
            );
            assert_eq!(
                kernel.process_event(OpScalarUnarySilu::new(&input), &mut output),
                Ok(())
            );
        }
    });
    assert_eq!(allocation.count_total, 0);
}

#[test]
fn scalar_unary_source_identity_is_pinned() {
    assert_eq!(
        emel_kernels::aarch64::PINNED_EMEL_CPP_COMMIT,
        "843a117386ef17dc5a50549bbfc821074c2141d6"
    );
    assert_eq!(
        emel_kernels::aarch64::PINNED_DETAIL_SPAN,
        "src/emel/kernel/detail.hpp:2392-2411,3271-3325"
    );
    assert_eq!(
        emel_kernels::aarch64::PINNED_AARCH64_GUARD_SPAN,
        "src/emel/kernel/aarch64/guards.hpp:826-834,848-860,877-886"
    );
    assert_eq!(
        emel_kernels::aarch64::PINNED_AARCH64_GUARD_BLOB,
        "c25714566ec9a02679daef85089544575123408e"
    );
    assert_eq!(
        emel_kernels::aarch64::PINNED_AARCH64_ACTION_BLOB,
        "267d4f74e6e7498155c8535920322ffef2c02fb6"
    );
    assert_eq!(
        emel_kernels::aarch64::PINNED_AARCH64_ACTION_SPAN,
        "src/emel/kernel/aarch64/actions.hpp:9194-9211,9376-9380"
    );
    assert_eq!(
        emel_kernels::aarch64::PINNED_AARCH64_TRANSITION_BLOB,
        "865a9cc6ba6115382ed043c464f3d62bcd851357"
    );
    assert_eq!(
        emel_kernels::aarch64::PINNED_AARCH64_TRANSITION_SPAN,
        "src/emel/kernel/aarch64/sm.hpp:1195-1223"
    );
}
