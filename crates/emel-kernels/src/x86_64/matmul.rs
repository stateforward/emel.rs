//! Safe x86 packed matrix arithmetic for the pinned target contract.
//!
//! The pinned target routes are the AVX2/FMA packed-row branches at
//! `src/emel/kernel/x86_64/actions.hpp:1110-1326`. This module keeps their
//! operand class and quantization order, while using Pulp's safe V3 SIMD
//! interface for the integer dot products. The owning SML router selects one
//! operation family before calling these family-specific actions; this module
//! contains no runtime algorithm or dtype selection.

#![cfg(target_arch = "x86_64")]
#![allow(private_interfaces)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::needless_pass_by_ref_mut)]
#![allow(clippy::similar_names)]
#![allow(clippy::suboptimal_flops)]

use core::cell::Cell;
use core::fmt;

use super::fma::run_fma;
use super::gemv::run_gemv;
use crate::any::matmul::{
    MatmulArgmaxResult, MatmulError, MatmulResult, OpMulMat, OpMulMatArgmaxQ2K, OpMulMatArgmaxQ3K,
    OpMulMatArgmaxQ4_0, OpMulMatArgmaxQ4_1, OpMulMatArgmaxQ4K, OpMulMatArgmaxQ5_0,
    OpMulMatArgmaxQ6K, OpMulMatArgmaxQ8_0, OpMulMatQ2K, OpMulMatQ3K, OpMulMatQ4_0, OpMulMatQ4_1,
    OpMulMatQ4K, OpMulMatQ5_0, OpMulMatQ6K, OpMulMatQ8_0, Q8_0Scratch, Q8KScratch, quantize_q8_0,
    quantize_q8_0_strided, quantize_q8_k, quantize_q8_k_strided,
};
use crate::any::quant::fp16_to_f32;
use pulp::{Simd, WithSimd};
use sml::sml;

const QK_32: usize = 32;
const QK_K: usize = 256;
const Q4_0_BYTES: usize = 18;
const Q4_1_BYTES: usize = 20;
const Q5_0_BYTES: usize = 22;
const Q8_0_BYTES: usize = 34;
const Q2_K_BYTES: usize = 84;
const Q3_K_BYTES: usize = 110;
const Q4_K_BYTES: usize = 144;
const Q6_K_BYTES: usize = 210;
const MAX_Q8_0_BLOCKS: usize = 1024;
const MAX_Q8_K_BLOCKS: usize = 128;

macro_rules! impl_argmax_32 {
    ($method:ident, $event:ident, $bytes:expr, $dot:ident) => {
        pub(crate) fn $method(&mut self, event: $event<'_>) -> i32 {
            let (lhs, rhs, destination) = event.into_parts();
            let shape = lhs.layout().ne();
            let k = usize::try_from(shape[0]).expect("guard-proven k fits usize");
            let rows = usize::try_from(shape[1]).expect("guard-proven rows fit usize");
            let blocks = k / QK_32;
            quantize_q8_0(&rhs, &mut self.q8_0, k);
            let row_bytes = blocks * $bytes;
            let lhs_bytes = lhs.as_bytes();
            let mut best = f32::NEG_INFINITY;
            let mut best_index = 0_i32;
            let mut row = 0;
            while row < rows {
                let value = self.$dot(
                    &lhs_bytes[row * row_bytes..(row + 1) * row_bytes],
                    &self.q8_0,
                    blocks,
                );
                if value > best || row == 0 {
                    best = value;
                    best_index = i32::try_from(row).expect("guard-proven row fits i32");
                }
                row += 1;
            }
            let (output, _) = destination.into_parts();
            output[0] = best;
            best_index
        }
    };
}

macro_rules! impl_argmax_k {
    ($method:ident, $event:ident, $bytes:expr, $dot:ident) => {
        pub(crate) fn $method(&mut self, event: $event<'_>) -> i32 {
            let (lhs, rhs, destination) = event.into_parts();
            let shape = lhs.layout().ne();
            let k = usize::try_from(shape[0]).expect("guard-proven k fits usize");
            let rows = usize::try_from(shape[1]).expect("guard-proven rows fit usize");
            let blocks = k / QK_K;
            quantize_q8_k(&rhs, &mut self.q8_k, k);
            let row_bytes = blocks * $bytes;
            let lhs_bytes = lhs.as_bytes();
            let mut best = f32::NEG_INFINITY;
            let mut best_index = 0_i32;
            let mut row = 0;
            while row < rows {
                let value = self.$dot(
                    &lhs_bytes[row * row_bytes..(row + 1) * row_bytes],
                    &self.q8_k,
                    blocks,
                );
                if value > best || row == 0 {
                    best = value;
                    best_index = i32::try_from(row).expect("guard-proven row fits i32");
                }
                row += 1;
            }
            let (output, _) = destination.into_parts();
            output[0] = best;
            best_index
        }
    };
}

macro_rules! process_argmax {
    ($name:ident, $event:ident, $runtime:ident, $variant:ident) => {
        pub(crate) fn $name(&mut self, event: $event<'_>) -> MatmulArgmaxResult {
            let result = Cell::new(Err(MatmulError::Internal));
            self.machine
                .process_event(X86MatmulMachineEvents::$variant($runtime {
                    event,
                    result: &result,
                }))
                .map_err(|_| MatmulError::Internal)?;
            result.get()
        }
    };
}

/// Actor-owned reusable workspace for the x86 packed target routes.
pub(crate) struct X86MatmulBackend {
    backend: pulp::x86::V3,
    q8_0: Box<[Q8_0Scratch]>,
    q8_k: Box<[Q8KScratch]>,
}

impl X86MatmulBackend {
    /// Constructs all one-time workspace before any dispatch occurs.
    pub(crate) fn new(backend: pulp::x86::V3) -> Self {
        Self {
            backend,
            q8_0: vec![Q8_0Scratch::ZERO; MAX_Q8_0_BLOCKS].into_boxed_slice(),
            q8_k: vec![Q8KScratch::ZERO; MAX_Q8_K_BLOCKS].into_boxed_slice(),
        }
    }

    pub(crate) fn mul_mat_f32_vector(&mut self, event: OpMulMat<'_>) {
        let (lhs, rhs, destination) = event.into_parts();
        let lhs_shape = lhs.layout().ne();
        let k = usize::try_from(lhs_shape[0]).expect("guard-proven k fits usize");
        let m = usize::try_from(lhs_shape[1]).expect("guard-proven m fits usize");
        let (output, _) = destination.into_parts();
        run_gemv(self.backend, lhs.as_slice(), rhs.as_slice(), output, m, k);
    }

    pub(crate) fn mul_mat_f32_matrix(&mut self, event: OpMulMat<'_>) {
        let (lhs, rhs, destination) = event.into_parts();
        let lhs_shape = lhs.layout().ne();
        let rhs_shape = rhs.layout().ne();
        let k = usize::try_from(lhs_shape[0]).expect("guard-proven k fits usize");
        let m = usize::try_from(lhs_shape[1]).expect("guard-proven m fits usize");
        let n = usize::try_from(rhs_shape[0]).expect("guard-proven n fits usize");
        let (output, _) = destination.into_parts();
        run_fma(
            self.backend,
            lhs.as_slice(),
            rhs.as_slice(),
            output,
            m,
            n,
            k,
        );
    }

    pub(crate) fn mul_mat_q4_0(&mut self, event: OpMulMatQ4_0<'_>) {
        let (lhs, rhs, destination) = event.into_parts();
        let shape = lhs.layout().ne();
        let k = usize::try_from(shape[0]).expect("guard-proven k fits usize");
        let rows = usize::try_from(shape[1]).expect("guard-proven rows fit usize");
        let columns =
            usize::try_from(rhs.layout().ne()[0]).expect("guard-proven columns fit usize");
        let blocks = k / QK_32;
        let row_bytes = blocks * Q4_0_BYTES;
        let lhs_bytes = lhs.as_bytes();
        let (output, _) = destination.into_parts();
        let mut column = 0;
        while column < columns {
            quantize_q8_0_strided(&rhs, &mut self.q8_0, k, columns, column);
            let mut row = 0;
            while row < rows {
                output[column + columns * row] = self.dot_q4_0(
                    &lhs_bytes[row * row_bytes..(row + 1) * row_bytes],
                    &self.q8_0,
                    blocks,
                );
                row += 1;
            }
            column += 1;
        }
    }

    pub(crate) fn mul_mat_q4_1(&mut self, event: OpMulMatQ4_1<'_>) {
        let (lhs, rhs, destination) = event.into_parts();
        let shape = lhs.layout().ne();
        let k = usize::try_from(shape[0]).expect("guard-proven k fits usize");
        let rows = usize::try_from(shape[1]).expect("guard-proven rows fit usize");
        let columns =
            usize::try_from(rhs.layout().ne()[0]).expect("guard-proven columns fit usize");
        let blocks = k / QK_32;
        let row_bytes = blocks * Q4_1_BYTES;
        let lhs_bytes = lhs.as_bytes();
        let (output, _) = destination.into_parts();
        let mut column = 0;
        while column < columns {
            quantize_q8_0_strided(&rhs, &mut self.q8_0, k, columns, column);
            let mut row = 0;
            while row < rows {
                output[column + columns * row] = self.dot_q4_1(
                    &lhs_bytes[row * row_bytes..(row + 1) * row_bytes],
                    &self.q8_0,
                    blocks,
                );
                row += 1;
            }
            column += 1;
        }
    }

    pub(crate) fn mul_mat_q5_0(&mut self, event: OpMulMatQ5_0<'_>) {
        let (lhs, rhs, destination) = event.into_parts();
        let shape = lhs.layout().ne();
        let k = usize::try_from(shape[0]).expect("guard-proven k fits usize");
        let rows = usize::try_from(shape[1]).expect("guard-proven rows fit usize");
        let columns =
            usize::try_from(rhs.layout().ne()[0]).expect("guard-proven columns fit usize");
        let blocks = k / QK_32;
        let row_bytes = blocks * Q5_0_BYTES;
        let lhs_bytes = lhs.as_bytes();
        let (output, _) = destination.into_parts();
        let mut column = 0;
        while column < columns {
            quantize_q8_0_strided(&rhs, &mut self.q8_0, k, columns, column);
            let mut row = 0;
            while row < rows {
                output[column + columns * row] = self.dot_q5_0(
                    &lhs_bytes[row * row_bytes..(row + 1) * row_bytes],
                    &self.q8_0,
                    blocks,
                );
                row += 1;
            }
            column += 1;
        }
    }

    pub(crate) fn mul_mat_q8_0(&mut self, event: OpMulMatQ8_0<'_>) {
        let (lhs, rhs, destination) = event.into_parts();
        let shape = lhs.layout().ne();
        let k = usize::try_from(shape[0]).expect("guard-proven k fits usize");
        let rows = usize::try_from(shape[1]).expect("guard-proven rows fit usize");
        let columns =
            usize::try_from(rhs.layout().ne()[0]).expect("guard-proven columns fit usize");
        let blocks = k / QK_32;
        let row_bytes = blocks * Q8_0_BYTES;
        let lhs_bytes = lhs.as_bytes();
        let (output, _) = destination.into_parts();
        let mut column = 0;
        while column < columns {
            quantize_q8_0_strided(&rhs, &mut self.q8_0, k, columns, column);
            let mut row = 0;
            while row < rows {
                output[column + columns * row] = self.dot_q8_0(
                    &lhs_bytes[row * row_bytes..(row + 1) * row_bytes],
                    &self.q8_0,
                    blocks,
                );
                row += 1;
            }
            column += 1;
        }
    }

    pub(crate) fn mul_mat_q2_k(&mut self, event: OpMulMatQ2K<'_>) {
        let (lhs, rhs, destination) = event.into_parts();
        let shape = lhs.layout().ne();
        let k = usize::try_from(shape[0]).expect("guard-proven k fits usize");
        let rows = usize::try_from(shape[1]).expect("guard-proven rows fit usize");
        let columns =
            usize::try_from(rhs.layout().ne()[0]).expect("guard-proven columns fit usize");
        let blocks = k / QK_K;
        let row_bytes = blocks * Q2_K_BYTES;
        let lhs_bytes = lhs.as_bytes();
        let (output, _) = destination.into_parts();
        let mut column = 0;
        while column < columns {
            quantize_q8_k_strided(&rhs, &mut self.q8_k, k, columns, column);
            let mut row = 0;
            while row < rows {
                output[column + columns * row] = self.dot_q2_k(
                    &lhs_bytes[row * row_bytes..(row + 1) * row_bytes],
                    &self.q8_k,
                    blocks,
                );
                row += 1;
            }
            column += 1;
        }
    }

    pub(crate) fn mul_mat_q3_k(&mut self, event: OpMulMatQ3K<'_>) {
        let (lhs, rhs, destination) = event.into_parts();
        let shape = lhs.layout().ne();
        let k = usize::try_from(shape[0]).expect("guard-proven k fits usize");
        let rows = usize::try_from(shape[1]).expect("guard-proven rows fit usize");
        let columns =
            usize::try_from(rhs.layout().ne()[0]).expect("guard-proven columns fit usize");
        let blocks = k / QK_K;
        let row_bytes = blocks * Q3_K_BYTES;
        let lhs_bytes = lhs.as_bytes();
        let (output, _) = destination.into_parts();
        let mut column = 0;
        while column < columns {
            quantize_q8_k_strided(&rhs, &mut self.q8_k, k, columns, column);
            let mut row = 0;
            while row < rows {
                output[column + columns * row] = self.dot_q3_k(
                    &lhs_bytes[row * row_bytes..(row + 1) * row_bytes],
                    &self.q8_k,
                    blocks,
                );
                row += 1;
            }
            column += 1;
        }
    }

    pub(crate) fn mul_mat_q4_k(&mut self, event: OpMulMatQ4K<'_>) {
        let (lhs, rhs, destination) = event.into_parts();
        let shape = lhs.layout().ne();
        let k = usize::try_from(shape[0]).expect("guard-proven k fits usize");
        let rows = usize::try_from(shape[1]).expect("guard-proven rows fit usize");
        let columns =
            usize::try_from(rhs.layout().ne()[0]).expect("guard-proven columns fit usize");
        let blocks = k / QK_K;
        let row_bytes = blocks * Q4_K_BYTES;
        let lhs_bytes = lhs.as_bytes();
        let (output, _) = destination.into_parts();
        let mut column = 0;
        while column < columns {
            quantize_q8_k_strided(&rhs, &mut self.q8_k, k, columns, column);
            let mut row = 0;
            while row < rows {
                output[column + columns * row] = self.dot_q4_k(
                    &lhs_bytes[row * row_bytes..(row + 1) * row_bytes],
                    &self.q8_k,
                    blocks,
                );
                row += 1;
            }
            column += 1;
        }
    }

    pub(crate) fn mul_mat_q6_k(&mut self, event: OpMulMatQ6K<'_>) {
        let (lhs, rhs, destination) = event.into_parts();
        let shape = lhs.layout().ne();
        let k = usize::try_from(shape[0]).expect("guard-proven k fits usize");
        let rows = usize::try_from(shape[1]).expect("guard-proven rows fit usize");
        let columns =
            usize::try_from(rhs.layout().ne()[0]).expect("guard-proven columns fit usize");
        let blocks = k / QK_K;
        let row_bytes = blocks * Q6_K_BYTES;
        let lhs_bytes = lhs.as_bytes();
        let (output, _) = destination.into_parts();
        let mut column = 0;
        while column < columns {
            quantize_q8_k_strided(&rhs, &mut self.q8_k, k, columns, column);
            let mut row = 0;
            while row < rows {
                output[column + columns * row] = self.dot_q6_k(
                    &lhs_bytes[row * row_bytes..(row + 1) * row_bytes],
                    &self.q8_k,
                    blocks,
                );
                row += 1;
            }
            column += 1;
        }
    }

    impl_argmax_32!(argmax_q4_0, OpMulMatArgmaxQ4_0, Q4_0_BYTES, dot_q4_0);
    impl_argmax_32!(argmax_q4_1, OpMulMatArgmaxQ4_1, Q4_1_BYTES, dot_q4_1);
    impl_argmax_32!(argmax_q5_0, OpMulMatArgmaxQ5_0, Q5_0_BYTES, dot_q5_0);
    impl_argmax_32!(argmax_q8_0, OpMulMatArgmaxQ8_0, Q8_0_BYTES, dot_q8_0);

    impl_argmax_k!(argmax_q2_k, OpMulMatArgmaxQ2K, Q2_K_BYTES, dot_q2_k);
    impl_argmax_k!(argmax_q3_k, OpMulMatArgmaxQ3K, Q3_K_BYTES, dot_q3_k);
    impl_argmax_k!(argmax_q4_k, OpMulMatArgmaxQ4K, Q4_K_BYTES, dot_q4_k);
    impl_argmax_k!(argmax_q6_k, OpMulMatArgmaxQ6K, Q6_K_BYTES, dot_q6_k);

    fn dot_q4_0(&self, row: &[u8], rhs: &[Q8_0Scratch], blocks: usize) -> f32 {
        let mut sum = 0.0_f32;
        let mut block = 0;
        while block < blocks {
            let offset = block * Q4_0_BYTES;
            let mut lhs = [0_i16; QK_32];
            let mut rhs_values = [0_i16; QK_32];
            let mut index = 0;
            while index < QK_32 / 2 {
                let packed = row[offset + 2 + index];
                lhs[index] = i16::from(packed & 0x0f) - 8;
                lhs[index + QK_32 / 2] = i16::from(packed >> 4) - 8;
                index += 1;
            }
            fill_i16(&rhs[block].qs, &mut rhs_values);
            let integer = dot_i16(self.backend, &lhs, &rhs_values);
            sum +=
                integer as f32 * (fp16_to_f32(packed_u16(row, offset)) * fp16_to_f32(rhs[block].d));
            block += 1;
        }
        sum
    }

    fn dot_q4_1(&self, row: &[u8], rhs: &[Q8_0Scratch], blocks: usize) -> f32 {
        let mut sum = 0.0_f32;
        let mut block = 0;
        while block < blocks {
            let offset = block * Q4_1_BYTES;
            let mut lhs = [0_i16; QK_32];
            let mut rhs_values = [0_i16; QK_32];
            let mut rhs_sum = 0_i32;
            let mut index = 0;
            while index < QK_32 / 2 {
                let packed = row[offset + 4 + index];
                let low = i16::from(rhs[block].qs[index]);
                let high = i16::from(rhs[block].qs[index + QK_32 / 2]);
                lhs[index] = i16::from(packed & 0x0f);
                lhs[index + QK_32 / 2] = i16::from(packed >> 4);
                rhs_sum += i32::from(low) + i32::from(high);
                index += 1;
            }
            fill_i16(&rhs[block].qs, &mut rhs_values);
            let integer = dot_i16(self.backend, &lhs, &rhs_values);
            let rhs_scale = fp16_to_f32(rhs[block].d);
            sum += rhs_scale
                * (fp16_to_f32(packed_u16(row, offset)) * integer as f32
                    + fp16_to_f32(packed_u16(row, offset + 2)) * rhs_sum as f32);
            block += 1;
        }
        sum
    }

    fn dot_q5_0(&self, row: &[u8], rhs: &[Q8_0Scratch], blocks: usize) -> f32 {
        let mut sum = 0.0_f32;
        let mut block = 0;
        while block < blocks {
            let offset = block * Q5_0_BYTES;
            let high = u32::from_le_bytes([
                row[offset + 2],
                row[offset + 3],
                row[offset + 4],
                row[offset + 5],
            ]);
            let mut lhs = [0_i16; QK_32];
            let mut rhs_values = [0_i16; QK_32];
            let mut index = 0;
            while index < QK_32 / 2 {
                let low_high = (((high >> index) & 1) as i16) << 4;
                let high_high = (((high >> (index + QK_32 / 2)) & 1) as i16) << 4;
                lhs[index] = i16::from(row[offset + 6 + index] & 0x0f) + low_high - 16;
                lhs[index + QK_32 / 2] = i16::from(row[offset + 6 + index] >> 4) + high_high - 16;
                index += 1;
            }
            fill_i16(&rhs[block].qs, &mut rhs_values);
            let integer = dot_i16(self.backend, &lhs, &rhs_values);
            sum +=
                integer as f32 * (fp16_to_f32(packed_u16(row, offset)) * fp16_to_f32(rhs[block].d));
            block += 1;
        }
        sum
    }

    fn dot_q8_0(&self, row: &[u8], rhs: &[Q8_0Scratch], blocks: usize) -> f32 {
        let mut sum = 0.0_f32;
        let mut block = 0;
        while block < blocks {
            let offset = block * Q8_0_BYTES;
            let mut lhs = [0_i16; QK_32];
            let mut rhs_values = [0_i16; QK_32];
            fill_i16_bytes(&row[offset + 2..offset + 2 + QK_32], &mut lhs);
            fill_i16(&rhs[block].qs, &mut rhs_values);
            let integer = dot_i16(self.backend, &lhs, &rhs_values);
            sum +=
                integer as f32 * (fp16_to_f32(packed_u16(row, offset)) * fp16_to_f32(rhs[block].d));
            block += 1;
        }
        sum
    }

    fn dot_q2_k(&self, row: &[u8], rhs: &[Q8KScratch], blocks: usize) -> f32 {
        let mut total = 0.0_f32;
        let mut block = 0;
        while block < blocks {
            let offset = block * Q2_K_BYTES;
            let scales = &row[offset..offset + 16];
            let q = &row[offset + 16..offset + 80];
            let d = fp16_to_f32(packed_u16(row, offset + 80));
            let dmin = fp16_to_f32(packed_u16(row, offset + 82));
            let mut coefficients = [0_i16; QK_K];
            let mut rhs_values = [0_i16; QK_K];
            let mut group_scales = [0_i16; QK_K / 16];
            let mut sum_mins = 0_i32;
            let mut group = 0;
            while group < 16 {
                sum_mins += i32::from(rhs[block].bsums[group]) * i32::from(scales[group] >> 4);
                group += 1;
            }
            let mut scale_index = 0;
            let mut chunk = 0;
            while chunk < 2 {
                let mut shift = 0;
                while shift < 8 {
                    let scale0 = i16::from(scales[scale_index] & 0x0f);
                    let scale1 = i16::from(scales[scale_index + 1] & 0x0f);
                    let rhs_base = chunk * 128 + (shift / 2) * 32;
                    let mut index = 0;
                    while index < 16 {
                        let qbase = chunk * 32;
                        coefficients[rhs_base + index] = i16::from((q[qbase + index] >> shift) & 3);
                        coefficients[rhs_base + 16 + index] =
                            i16::from((q[qbase + 16 + index] >> shift) & 3);
                        rhs_values[rhs_base + index] = i16::from(rhs[block].qs[rhs_base + index]);
                        rhs_values[rhs_base + 16 + index] =
                            i16::from(rhs[block].qs[rhs_base + 16 + index]);
                        index += 1;
                    }
                    group_scales[rhs_base / 16] = scale0;
                    group_scales[rhs_base / 16 + 1] = scale1;
                    scale_index += 2;
                    shift += 2;
                }
                chunk += 1;
            }
            let mut sum = 0_i32;
            let mut group = 0;
            while group < QK_K / 16 {
                let base = group * 16;
                sum += i32::from(group_scales[group])
                    * dot_i16(
                        self.backend,
                        &coefficients[base..base + 16],
                        &rhs_values[base..base + 16],
                    );
                group += 1;
            }
            let d_all = rhs[block].d * d;
            let d_min = rhs[block].d * dmin;
            total += d_all * sum as f32 - d_min * sum_mins as f32;
            block += 1;
        }
        total
    }

    fn dot_q3_k(&self, row: &[u8], rhs: &[Q8KScratch], blocks: usize) -> f32 {
        let mut total = 0.0_f32;
        let mut block = 0;
        while block < blocks {
            let offset = block * Q3_K_BYTES;
            let hmask = &row[offset..offset + 32];
            let q = &row[offset + 32..offset + 96];
            let scales = unpack_q3_scales(&row[offset + 96..offset + 108]);
            let d = fp16_to_f32(packed_u16(row, offset + 108));
            let mut coefficients = [0_i16; QK_K];
            let mut rhs_values = [0_i16; QK_K];
            let mut group_scales = [0_i16; QK_K / 16];
            let mut group = 0;
            while group < 16 {
                let scale = i16::from(scales[group]);
                let qbase = (group / 8) * 32;
                let rhsbase = group * 16;
                let mask_shift = group / 2;
                let mask = 1_u8 << mask_shift;
                let half = (group % 2) * 16;
                let mut index = 0;
                while index < 16 {
                    let qbyte = q[qbase + half + index];
                    let offset_value = if hmask[half + index] & mask == 0 {
                        4
                    } else {
                        0
                    };
                    coefficients[rhsbase + index] =
                        i16::from((qbyte >> (((group / 2) % 4) * 2)) & 3) - offset_value;
                    rhs_values[rhsbase + index] = i16::from(rhs[block].qs[rhsbase + index]);
                    index += 1;
                }
                group_scales[group] = scale;
                group += 1;
            }
            let mut sum = 0_i32;
            group = 0;
            while group < QK_K / 16 {
                let base = group * 16;
                sum += i32::from(group_scales[group])
                    * dot_i16(
                        self.backend,
                        &coefficients[base..base + 16],
                        &rhs_values[base..base + 16],
                    );
                group += 1;
            }
            total += rhs[block].d * d * sum as f32;
            block += 1;
        }
        total
    }

    fn dot_q4_k(&self, row: &[u8], rhs: &[Q8KScratch], blocks: usize) -> f32 {
        let mut total = 0.0_f32;
        let mut block = 0;
        while block < blocks {
            let offset = block * Q4_K_BYTES;
            let mut coefficients = [0_i16; QK_K];
            let mut rhs_values = [0_i16; QK_K];
            let mut group_scales = [0_i16; QK_K / 32];
            let mut group = 0;
            while group < QK_K / 64 {
                let q4_offset = group * 32;
                let output_offset = group * 64;
                let mut lane = 0;
                while lane < 32 {
                    let packed = row[offset + 16 + q4_offset + lane];
                    coefficients[output_offset + lane] = i16::from(packed & 0x0f);
                    coefficients[output_offset + 32 + lane] = i16::from(packed >> 4);
                    lane += 1;
                }
                group += 1;
            }
            let scales = &row[offset + 4..offset + 16];
            let mut unpacked_scales = [0_u8; 12];
            let mut minimums = [0_u8; 8];
            let mut lane = 0;
            while lane < 4 {
                let scale_word0 = scales[lane];
                let scale_word1 = scales[4 + lane];
                let scale_word2 = scales[8 + lane];
                unpacked_scales[lane] = scale_word0 & 0x3f;
                unpacked_scales[4 + lane] =
                    (scale_word2 & 0x0f) | (((scale_word0 >> 6) & 0x03) << 4);
                unpacked_scales[8 + lane] = scale_word1 & 0x3f;
                minimums[lane] = scale_word1 & 0x3f;
                minimums[4 + lane] =
                    ((scale_word2 >> 4) & 0x0f) | (((scale_word1 >> 6) & 0x03) << 4);
                lane += 1;
            }
            let mut minimum_sum = 0_i32;
            let mut group = 0;
            while group < QK_K / 16 {
                minimum_sum += i32::from(rhs[block].bsums[group]) * i32::from(minimums[group / 2]);
                group += 1;
            }
            group = 0;
            while group < QK_K / 32 {
                let scale = i16::from(unpacked_scales[group]);
                let base = group * 32;
                let mut index = 0;
                while index < 32 {
                    rhs_values[base + index] = i16::from(rhs[block].qs[base + index]);
                    index += 1;
                }
                group_scales[group] = scale;
                group += 1;
            }
            let mut sum = 0_i32;
            group = 0;
            while group < QK_K / 32 {
                let base = group * 32;
                sum += i32::from(group_scales[group])
                    * dot_i16(
                        self.backend,
                        &coefficients[base..base + 32],
                        &rhs_values[base..base + 32],
                    );
                group += 1;
            }
            let d = fp16_to_f32(packed_u16(row, offset)) * rhs[block].d;
            let dmin = fp16_to_f32(packed_u16(row, offset + 2)) * rhs[block].d;
            total += d * sum as f32 - dmin * minimum_sum as f32;
            block += 1;
        }
        total
    }

    fn dot_q6_k(&self, row: &[u8], rhs: &[Q8KScratch], blocks: usize) -> f32 {
        let mut total = 0.0_f32;
        let mut block = 0;
        while block < blocks {
            let offset = block * Q6_K_BYTES;
            let ql = &row[offset..offset + 128];
            let qh = &row[offset + 128..offset + 192];
            let scales = &row[offset + 192..offset + 208];
            let d = fp16_to_f32(packed_u16(row, offset + 208));
            let mut coefficients = [0_i16; QK_K];
            let mut rhs_values = [0_i16; QK_K];
            let mut group_scales = [0_i16; QK_K / 16];
            let mut chunk = 0;
            while chunk < 2 {
                let mut index = 0;
                while index < 32 {
                    let high = qh[chunk * 32 + index];
                    coefficients[chunk * 128 + index] =
                        i16::from((ql[chunk * 64 + index] & 0x0f) | ((high & 3) << 4)) - 32;
                    coefficients[chunk * 128 + index + 32] =
                        i16::from((ql[chunk * 64 + index + 32] & 0x0f) | (((high >> 2) & 3) << 4))
                            - 32;
                    coefficients[chunk * 128 + index + 64] = i16::from(
                        ((ql[chunk * 64 + index] >> 4) & 0x0f) | (((high >> 4) & 3) << 4),
                    ) - 32;
                    coefficients[chunk * 128 + index + 96] = i16::from(
                        ((ql[chunk * 64 + index + 32] >> 4) & 0x0f) | (((high >> 6) & 3) << 4),
                    ) - 32;
                    index += 1;
                }
                chunk += 1;
            }
            let mut group = 0;
            while group < 16 {
                let scale = i16::from(i8::from_ne_bytes([scales[group]]));
                let base = group * 16;
                group_scales[group] = scale;
                let mut index = 0;
                while index < 16 {
                    rhs_values[base + index] = i16::from(rhs[block].qs[base + index]);
                    index += 1;
                }
                group += 1;
            }
            let mut sum = 0_i32;
            group = 0;
            while group < QK_K / 16 {
                let base = group * 16;
                sum += i32::from(group_scales[group])
                    * dot_i16(
                        self.backend,
                        &coefficients[base..base + 16],
                        &rhs_values[base..base + 16],
                    );
                group += 1;
            }
            total += rhs[block].d * d * sum as f32;
            block += 1;
        }
        total
    }
}

struct MatmulQ4_0Runtime<'a> {
    event: OpMulMatQ4_0<'a>,
    result: &'a Cell<MatmulResult>,
}

struct MatmulRuntime<'a> {
    event: OpMulMat<'a>,
    result: &'a Cell<MatmulResult>,
}

struct MatmulQ4_1Runtime<'a> {
    event: OpMulMatQ4_1<'a>,
    result: &'a Cell<MatmulResult>,
}

struct MatmulQ5_0Runtime<'a> {
    event: OpMulMatQ5_0<'a>,
    result: &'a Cell<MatmulResult>,
}

struct MatmulQ8_0Runtime<'a> {
    event: OpMulMatQ8_0<'a>,
    result: &'a Cell<MatmulResult>,
}

struct MatmulQ2KRuntime<'a> {
    event: OpMulMatQ2K<'a>,
    result: &'a Cell<MatmulResult>,
}

struct MatmulQ3KRuntime<'a> {
    event: OpMulMatQ3K<'a>,
    result: &'a Cell<MatmulResult>,
}

struct MatmulQ4KRuntime<'a> {
    event: OpMulMatQ4K<'a>,
    result: &'a Cell<MatmulResult>,
}

struct MatmulQ6KRuntime<'a> {
    event: OpMulMatQ6K<'a>,
    result: &'a Cell<MatmulResult>,
}

struct MatmulArgmaxQ4_0Runtime<'a> {
    event: OpMulMatArgmaxQ4_0<'a>,
    result: &'a Cell<MatmulArgmaxResult>,
}

struct MatmulArgmaxQ4_1Runtime<'a> {
    event: OpMulMatArgmaxQ4_1<'a>,
    result: &'a Cell<MatmulArgmaxResult>,
}

struct MatmulArgmaxQ5_0Runtime<'a> {
    event: OpMulMatArgmaxQ5_0<'a>,
    result: &'a Cell<MatmulArgmaxResult>,
}

struct MatmulArgmaxQ8_0Runtime<'a> {
    event: OpMulMatArgmaxQ8_0<'a>,
    result: &'a Cell<MatmulArgmaxResult>,
}

struct MatmulArgmaxQ2KRuntime<'a> {
    event: OpMulMatArgmaxQ2K<'a>,
    result: &'a Cell<MatmulArgmaxResult>,
}

struct MatmulArgmaxQ3KRuntime<'a> {
    event: OpMulMatArgmaxQ3K<'a>,
    result: &'a Cell<MatmulArgmaxResult>,
}

struct MatmulArgmaxQ4KRuntime<'a> {
    event: OpMulMatArgmaxQ4K<'a>,
    result: &'a Cell<MatmulArgmaxResult>,
}

struct MatmulArgmaxQ6KRuntime<'a> {
    event: OpMulMatArgmaxQ6K<'a>,
    result: &'a Cell<MatmulArgmaxResult>,
}

struct UnexpectedRuntime<'a> {
    result: &'a Cell<MatmulResult>,
}

struct Context {
    backend: X86MatmulBackend,
}

sml! {
    X86MatmulMachine<'dispatch> {
        "ready"_s <= *"ready"_s + Matmul(MatmulRuntime<'dispatch>)
            [guard_matmul_vector] / effect_matmul_vector,
        "ready"_s <= "ready"_s + Matmul(MatmulRuntime<'dispatch>)
            [guard_matmul_matrix] / effect_matmul_matrix,
        "ready"_s <= "ready"_s + Matmul(MatmulRuntime<'dispatch>)
            [guard_matmul_shape] / effect_matmul_shape,
        "ready"_s <= "ready"_s + Matmul(MatmulRuntime<'dispatch>)
            [guard_matmul_view] / effect_matmul_view,

        "ready"_s <= "ready"_s + MatmulQ4_0(MatmulQ4_0Runtime<'dispatch>)
            [guard_q4_0_valid] / effect_q4_0,
        "ready"_s <= "ready"_s + MatmulQ4_0(MatmulQ4_0Runtime<'dispatch>)
            [guard_q4_0_shape] / effect_q4_0_shape,
        "ready"_s <= "ready"_s + MatmulQ4_0(MatmulQ4_0Runtime<'dispatch>)
            [guard_q4_0_view] / effect_q4_0_view,

        "ready"_s <= "ready"_s + MatmulQ4_1(MatmulQ4_1Runtime<'dispatch>)
            [guard_q4_1_valid] / effect_q4_1,
        "ready"_s <= "ready"_s + MatmulQ4_1(MatmulQ4_1Runtime<'dispatch>)
            [guard_q4_1_shape] / effect_q4_1_shape,
        "ready"_s <= "ready"_s + MatmulQ4_1(MatmulQ4_1Runtime<'dispatch>)
            [guard_q4_1_view] / effect_q4_1_view,

        "ready"_s <= "ready"_s + MatmulQ5_0(MatmulQ5_0Runtime<'dispatch>)
            [guard_q5_0_valid] / effect_q5_0,
        "ready"_s <= "ready"_s + MatmulQ5_0(MatmulQ5_0Runtime<'dispatch>)
            [guard_q5_0_shape] / effect_q5_0_shape,
        "ready"_s <= "ready"_s + MatmulQ5_0(MatmulQ5_0Runtime<'dispatch>)
            [guard_q5_0_view] / effect_q5_0_view,

        "ready"_s <= "ready"_s + MatmulQ8_0(MatmulQ8_0Runtime<'dispatch>)
            [guard_q8_0_valid] / effect_q8_0,
        "ready"_s <= "ready"_s + MatmulQ8_0(MatmulQ8_0Runtime<'dispatch>)
            [guard_q8_0_shape] / effect_q8_0_shape,
        "ready"_s <= "ready"_s + MatmulQ8_0(MatmulQ8_0Runtime<'dispatch>)
            [guard_q8_0_view] / effect_q8_0_view,

        "ready"_s <= "ready"_s + MatmulQ2K(MatmulQ2KRuntime<'dispatch>)
            [guard_q2_k_valid] / effect_q2_k,
        "ready"_s <= "ready"_s + MatmulQ2K(MatmulQ2KRuntime<'dispatch>)
            [guard_q2_k_shape] / effect_q2_k_shape,
        "ready"_s <= "ready"_s + MatmulQ2K(MatmulQ2KRuntime<'dispatch>)
            [guard_q2_k_view] / effect_q2_k_view,

        "ready"_s <= "ready"_s + MatmulQ3K(MatmulQ3KRuntime<'dispatch>)
            [guard_q3_k_valid] / effect_q3_k,
        "ready"_s <= "ready"_s + MatmulQ3K(MatmulQ3KRuntime<'dispatch>)
            [guard_q3_k_shape] / effect_q3_k_shape,
        "ready"_s <= "ready"_s + MatmulQ3K(MatmulQ3KRuntime<'dispatch>)
            [guard_q3_k_view] / effect_q3_k_view,

        "ready"_s <= "ready"_s + MatmulQ4K(MatmulQ4KRuntime<'dispatch>)
            [guard_q4_k_valid] / effect_q4_k,
        "ready"_s <= "ready"_s + MatmulQ4K(MatmulQ4KRuntime<'dispatch>)
            [guard_q4_k_shape] / effect_q4_k_shape,
        "ready"_s <= "ready"_s + MatmulQ4K(MatmulQ4KRuntime<'dispatch>)
            [guard_q4_k_view] / effect_q4_k_view,

        "ready"_s <= "ready"_s + MatmulQ6K(MatmulQ6KRuntime<'dispatch>)
            [guard_q6_k_valid] / effect_q6_k,
        "ready"_s <= "ready"_s + MatmulQ6K(MatmulQ6KRuntime<'dispatch>)
            [guard_q6_k_shape] / effect_q6_k_shape,
        "ready"_s <= "ready"_s + MatmulQ6K(MatmulQ6KRuntime<'dispatch>)
            [guard_q6_k_view] / effect_q6_k_view,

        "ready"_s <= "ready"_s + MatmulArgmaxQ4_0(MatmulArgmaxQ4_0Runtime<'dispatch>)
            [guard_argmax_q4_0_valid] / effect_argmax_q4_0,
        "ready"_s <= "ready"_s + MatmulArgmaxQ4_0(MatmulArgmaxQ4_0Runtime<'dispatch>)
            [guard_argmax_q4_0_shape] / effect_argmax_q4_0_shape,
        "ready"_s <= "ready"_s + MatmulArgmaxQ4_0(MatmulArgmaxQ4_0Runtime<'dispatch>)
            [guard_argmax_q4_0_view] / effect_argmax_q4_0_view,

        "ready"_s <= "ready"_s + MatmulArgmaxQ4_1(MatmulArgmaxQ4_1Runtime<'dispatch>)
            [guard_argmax_q4_1_valid] / effect_argmax_q4_1,
        "ready"_s <= "ready"_s + MatmulArgmaxQ4_1(MatmulArgmaxQ4_1Runtime<'dispatch>)
            [guard_argmax_q4_1_shape] / effect_argmax_q4_1_shape,
        "ready"_s <= "ready"_s + MatmulArgmaxQ4_1(MatmulArgmaxQ4_1Runtime<'dispatch>)
            [guard_argmax_q4_1_view] / effect_argmax_q4_1_view,

        "ready"_s <= "ready"_s + MatmulArgmaxQ5_0(MatmulArgmaxQ5_0Runtime<'dispatch>)
            [guard_argmax_q5_0_valid] / effect_argmax_q5_0,
        "ready"_s <= "ready"_s + MatmulArgmaxQ5_0(MatmulArgmaxQ5_0Runtime<'dispatch>)
            [guard_argmax_q5_0_shape] / effect_argmax_q5_0_shape,
        "ready"_s <= "ready"_s + MatmulArgmaxQ5_0(MatmulArgmaxQ5_0Runtime<'dispatch>)
            [guard_argmax_q5_0_view] / effect_argmax_q5_0_view,

        "ready"_s <= "ready"_s + MatmulArgmaxQ8_0(MatmulArgmaxQ8_0Runtime<'dispatch>)
            [guard_argmax_q8_0_valid] / effect_argmax_q8_0,
        "ready"_s <= "ready"_s + MatmulArgmaxQ8_0(MatmulArgmaxQ8_0Runtime<'dispatch>)
            [guard_argmax_q8_0_shape] / effect_argmax_q8_0_shape,
        "ready"_s <= "ready"_s + MatmulArgmaxQ8_0(MatmulArgmaxQ8_0Runtime<'dispatch>)
            [guard_argmax_q8_0_view] / effect_argmax_q8_0_view,

        "ready"_s <= "ready"_s + MatmulArgmaxQ2K(MatmulArgmaxQ2KRuntime<'dispatch>)
            [guard_argmax_q2_k_valid] / effect_argmax_q2_k,
        "ready"_s <= "ready"_s + MatmulArgmaxQ2K(MatmulArgmaxQ2KRuntime<'dispatch>)
            [guard_argmax_q2_k_shape] / effect_argmax_q2_k_shape,
        "ready"_s <= "ready"_s + MatmulArgmaxQ2K(MatmulArgmaxQ2KRuntime<'dispatch>)
            [guard_argmax_q2_k_view] / effect_argmax_q2_k_view,

        "ready"_s <= "ready"_s + MatmulArgmaxQ3K(MatmulArgmaxQ3KRuntime<'dispatch>)
            [guard_argmax_q3_k_valid] / effect_argmax_q3_k,
        "ready"_s <= "ready"_s + MatmulArgmaxQ3K(MatmulArgmaxQ3KRuntime<'dispatch>)
            [guard_argmax_q3_k_shape] / effect_argmax_q3_k_shape,
        "ready"_s <= "ready"_s + MatmulArgmaxQ3K(MatmulArgmaxQ3KRuntime<'dispatch>)
            [guard_argmax_q3_k_view] / effect_argmax_q3_k_view,

        "ready"_s <= "ready"_s + MatmulArgmaxQ4K(MatmulArgmaxQ4KRuntime<'dispatch>)
            [guard_argmax_q4_k_valid] / effect_argmax_q4_k,
        "ready"_s <= "ready"_s + MatmulArgmaxQ4K(MatmulArgmaxQ4KRuntime<'dispatch>)
            [guard_argmax_q4_k_shape] / effect_argmax_q4_k_shape,
        "ready"_s <= "ready"_s + MatmulArgmaxQ4K(MatmulArgmaxQ4KRuntime<'dispatch>)
            [guard_argmax_q4_k_view] / effect_argmax_q4_k_view,

        "ready"_s <= "ready"_s + MatmulArgmaxQ6K(MatmulArgmaxQ6KRuntime<'dispatch>)
            [guard_argmax_q6_k_valid] / effect_argmax_q6_k,
        "ready"_s <= "ready"_s + MatmulArgmaxQ6K(MatmulArgmaxQ6KRuntime<'dispatch>)
            [guard_argmax_q6_k_shape] / effect_argmax_q6_k_shape,
        "ready"_s <= "ready"_s + MatmulArgmaxQ6K(MatmulArgmaxQ6KRuntime<'dispatch>)
            [guard_argmax_q6_k_view] / effect_argmax_q6_k_view,

        "ready"_s <= "ready"_s + Unexpected(UnexpectedRuntime<'dispatch>)
            / effect_unexpected,
        "ready"_s <= "ready"_s + unexpected_event<_> / effect_generic_unexpected,
    }
}

macro_rules! impl_regular_route {
    ($event:ident, $runtime:ident, $valid:ident, $shape:ident, $view:ident,
        $effect:ident, $effect_shape:ident, $effect_view:ident, $method:ident) => {
        fn $valid(&self, event: &$runtime<'_>) -> Result<bool, ()> {
            Ok(event.event.target_views_valid() && event.event.target_shape_valid())
        }

        fn $shape(&self, event: &$runtime<'_>) -> Result<bool, ()> {
            Ok(event.event.target_views_valid() && !event.event.target_shape_valid())
        }

        fn $view(&self, event: &$runtime<'_>) -> Result<bool, ()> {
            Ok(!event.event.target_views_valid())
        }

        fn $effect(&mut self, event: $runtime<'_>) -> Result<(), ()> {
            self.backend.$method(event.event);
            event.result.set(Ok(()));
            Ok(())
        }

        fn $effect_shape(&mut self, event: $runtime<'_>) -> Result<(), ()> {
            event.result.set(Err(MatmulError::ShapeMismatch));
            Ok(())
        }

        fn $effect_view(&mut self, event: $runtime<'_>) -> Result<(), ()> {
            event.result.set(Err(MatmulError::InvalidView));
            Ok(())
        }
    };
}

macro_rules! impl_argmax_route {
    ($event:ident, $runtime:ident, $valid:ident, $shape:ident, $view:ident,
        $effect:ident, $effect_shape:ident, $effect_view:ident, $method:ident) => {
        fn $valid(&self, event: &$runtime<'_>) -> Result<bool, ()> {
            Ok(event.event.target_views_valid() && event.event.target_shape_valid())
        }

        fn $shape(&self, event: &$runtime<'_>) -> Result<bool, ()> {
            Ok(event.event.target_views_valid() && !event.event.target_shape_valid())
        }

        fn $view(&self, event: &$runtime<'_>) -> Result<bool, ()> {
            Ok(!event.event.target_views_valid())
        }

        fn $effect(&mut self, event: $runtime<'_>) -> Result<(), ()> {
            let index = self.backend.$method(event.event);
            event.result.set(Ok(index));
            Ok(())
        }

        fn $effect_shape(&mut self, event: $runtime<'_>) -> Result<(), ()> {
            event.result.set(Err(MatmulError::ShapeMismatch));
            Ok(())
        }

        fn $effect_view(&mut self, event: $runtime<'_>) -> Result<(), ()> {
            event.result.set(Err(MatmulError::InvalidView));
            Ok(())
        }
    };
}

impl X86MatmulMachineStateMachineContext for Context {
    fn guard_matmul_vector(&self, event: &MatmulRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.target_dense_views_valid()
            && event.event.target_shape_valid()
            && event.event.target_rhs_columns() == 1)
    }

    fn guard_matmul_matrix(&self, event: &MatmulRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.target_dense_views_valid()
            && event.event.target_shape_valid()
            && event.event.target_rhs_columns() != 1)
    }

    fn guard_matmul_shape(&self, event: &MatmulRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.target_dense_views_valid() && !event.event.target_shape_valid())
    }

    fn guard_matmul_view(&self, event: &MatmulRuntime<'_>) -> Result<bool, ()> {
        Ok(!event.event.target_dense_views_valid())
    }

    fn effect_matmul_vector(&mut self, event: MatmulRuntime<'_>) -> Result<(), ()> {
        self.backend.mul_mat_f32_vector(event.event);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_matmul_matrix(&mut self, event: MatmulRuntime<'_>) -> Result<(), ()> {
        self.backend.mul_mat_f32_matrix(event.event);
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_matmul_shape(&mut self, event: MatmulRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(MatmulError::ShapeMismatch));
        Ok(())
    }

    fn effect_matmul_view(&mut self, event: MatmulRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(MatmulError::InvalidView));
        Ok(())
    }

    impl_regular_route!(
        OpMulMatQ4_0,
        MatmulQ4_0Runtime,
        guard_q4_0_valid,
        guard_q4_0_shape,
        guard_q4_0_view,
        effect_q4_0,
        effect_q4_0_shape,
        effect_q4_0_view,
        mul_mat_q4_0
    );
    impl_regular_route!(
        OpMulMatQ4_1,
        MatmulQ4_1Runtime,
        guard_q4_1_valid,
        guard_q4_1_shape,
        guard_q4_1_view,
        effect_q4_1,
        effect_q4_1_shape,
        effect_q4_1_view,
        mul_mat_q4_1
    );
    impl_regular_route!(
        OpMulMatQ5_0,
        MatmulQ5_0Runtime,
        guard_q5_0_valid,
        guard_q5_0_shape,
        guard_q5_0_view,
        effect_q5_0,
        effect_q5_0_shape,
        effect_q5_0_view,
        mul_mat_q5_0
    );
    impl_regular_route!(
        OpMulMatQ8_0,
        MatmulQ8_0Runtime,
        guard_q8_0_valid,
        guard_q8_0_shape,
        guard_q8_0_view,
        effect_q8_0,
        effect_q8_0_shape,
        effect_q8_0_view,
        mul_mat_q8_0
    );
    impl_regular_route!(
        OpMulMatQ2K,
        MatmulQ2KRuntime,
        guard_q2_k_valid,
        guard_q2_k_shape,
        guard_q2_k_view,
        effect_q2_k,
        effect_q2_k_shape,
        effect_q2_k_view,
        mul_mat_q2_k
    );
    impl_regular_route!(
        OpMulMatQ3K,
        MatmulQ3KRuntime,
        guard_q3_k_valid,
        guard_q3_k_shape,
        guard_q3_k_view,
        effect_q3_k,
        effect_q3_k_shape,
        effect_q3_k_view,
        mul_mat_q3_k
    );
    impl_regular_route!(
        OpMulMatQ4K,
        MatmulQ4KRuntime,
        guard_q4_k_valid,
        guard_q4_k_shape,
        guard_q4_k_view,
        effect_q4_k,
        effect_q4_k_shape,
        effect_q4_k_view,
        mul_mat_q4_k
    );
    impl_regular_route!(
        OpMulMatQ6K,
        MatmulQ6KRuntime,
        guard_q6_k_valid,
        guard_q6_k_shape,
        guard_q6_k_view,
        effect_q6_k,
        effect_q6_k_shape,
        effect_q6_k_view,
        mul_mat_q6_k
    );

    impl_argmax_route!(
        OpMulMatArgmaxQ4_0,
        MatmulArgmaxQ4_0Runtime,
        guard_argmax_q4_0_valid,
        guard_argmax_q4_0_shape,
        guard_argmax_q4_0_view,
        effect_argmax_q4_0,
        effect_argmax_q4_0_shape,
        effect_argmax_q4_0_view,
        argmax_q4_0
    );
    impl_argmax_route!(
        OpMulMatArgmaxQ4_1,
        MatmulArgmaxQ4_1Runtime,
        guard_argmax_q4_1_valid,
        guard_argmax_q4_1_shape,
        guard_argmax_q4_1_view,
        effect_argmax_q4_1,
        effect_argmax_q4_1_shape,
        effect_argmax_q4_1_view,
        argmax_q4_1
    );
    impl_argmax_route!(
        OpMulMatArgmaxQ5_0,
        MatmulArgmaxQ5_0Runtime,
        guard_argmax_q5_0_valid,
        guard_argmax_q5_0_shape,
        guard_argmax_q5_0_view,
        effect_argmax_q5_0,
        effect_argmax_q5_0_shape,
        effect_argmax_q5_0_view,
        argmax_q5_0
    );
    impl_argmax_route!(
        OpMulMatArgmaxQ8_0,
        MatmulArgmaxQ8_0Runtime,
        guard_argmax_q8_0_valid,
        guard_argmax_q8_0_shape,
        guard_argmax_q8_0_view,
        effect_argmax_q8_0,
        effect_argmax_q8_0_shape,
        effect_argmax_q8_0_view,
        argmax_q8_0
    );
    impl_argmax_route!(
        OpMulMatArgmaxQ2K,
        MatmulArgmaxQ2KRuntime,
        guard_argmax_q2_k_valid,
        guard_argmax_q2_k_shape,
        guard_argmax_q2_k_view,
        effect_argmax_q2_k,
        effect_argmax_q2_k_shape,
        effect_argmax_q2_k_view,
        argmax_q2_k
    );
    impl_argmax_route!(
        OpMulMatArgmaxQ3K,
        MatmulArgmaxQ3KRuntime,
        guard_argmax_q3_k_valid,
        guard_argmax_q3_k_shape,
        guard_argmax_q3_k_view,
        effect_argmax_q3_k,
        effect_argmax_q3_k_shape,
        effect_argmax_q3_k_view,
        argmax_q3_k
    );
    impl_argmax_route!(
        OpMulMatArgmaxQ4K,
        MatmulArgmaxQ4KRuntime,
        guard_argmax_q4_k_valid,
        guard_argmax_q4_k_shape,
        guard_argmax_q4_k_view,
        effect_argmax_q4_k,
        effect_argmax_q4_k_shape,
        effect_argmax_q4_k_view,
        argmax_q4_k
    );
    impl_argmax_route!(
        OpMulMatArgmaxQ6K,
        MatmulArgmaxQ6KRuntime,
        guard_argmax_q6_k_valid,
        guard_argmax_q6_k_shape,
        guard_argmax_q6_k_view,
        effect_argmax_q6_k,
        effect_argmax_q6_k_shape,
        effect_argmax_q6_k_view,
        argmax_q6_k
    );

    fn effect_unexpected(&mut self, event: UnexpectedRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(MatmulError::UnexpectedEvent));
        Ok(())
    }

    fn effect_generic_unexpected(&mut self) -> Result<(), ()> {
        Err(())
    }
}

/// Single-writer x86 packed matmul actor.
pub(crate) struct X86MatmulKernel {
    machine: X86MatmulMachineStateMachine<Context>,
}

impl fmt::Debug for X86MatmulKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("X86MatmulKernel")
            .finish_non_exhaustive()
    }
}

impl X86MatmulKernel {
    /// Resolves V3 and allocates reusable packed scratch before dispatch.
    pub(crate) fn try_new() -> Option<Self> {
        let pulp::x86::Arch::V3(backend) = pulp::x86::Arch::new() else {
            return None;
        };
        Some(Self {
            machine: X86MatmulMachineStateMachine::new(Context {
                backend: X86MatmulBackend::new(backend),
            }),
        })
    }

    /// Dispatches one validated dense F32 request through the target actor.
    pub(crate) fn process_f32(&mut self, event: OpMulMat<'_>) -> MatmulResult {
        let result = Cell::new(Err(MatmulError::Internal));
        self.machine
            .process_event(X86MatmulMachineEvents::Matmul(MatmulRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| MatmulError::Internal)?;
        result.get()
    }

    pub(crate) fn process_q4_0(&mut self, event: OpMulMatQ4_0<'_>) -> MatmulResult {
        let result = Cell::new(Err(MatmulError::Internal));
        self.machine
            .process_event(X86MatmulMachineEvents::MatmulQ4_0(MatmulQ4_0Runtime {
                event,
                result: &result,
            }))
            .map_err(|_| MatmulError::Internal)?;
        result.get()
    }

    pub(crate) fn process_q4_1(&mut self, event: OpMulMatQ4_1<'_>) -> MatmulResult {
        let result = Cell::new(Err(MatmulError::Internal));
        self.machine
            .process_event(X86MatmulMachineEvents::MatmulQ4_1(MatmulQ4_1Runtime {
                event,
                result: &result,
            }))
            .map_err(|_| MatmulError::Internal)?;
        result.get()
    }

    pub(crate) fn process_q5_0(&mut self, event: OpMulMatQ5_0<'_>) -> MatmulResult {
        let result = Cell::new(Err(MatmulError::Internal));
        self.machine
            .process_event(X86MatmulMachineEvents::MatmulQ5_0(MatmulQ5_0Runtime {
                event,
                result: &result,
            }))
            .map_err(|_| MatmulError::Internal)?;
        result.get()
    }

    pub(crate) fn process_q8_0(&mut self, event: OpMulMatQ8_0<'_>) -> MatmulResult {
        let result = Cell::new(Err(MatmulError::Internal));
        self.machine
            .process_event(X86MatmulMachineEvents::MatmulQ8_0(MatmulQ8_0Runtime {
                event,
                result: &result,
            }))
            .map_err(|_| MatmulError::Internal)?;
        result.get()
    }

    pub(crate) fn process_q2_k(&mut self, event: OpMulMatQ2K<'_>) -> MatmulResult {
        let result = Cell::new(Err(MatmulError::Internal));
        self.machine
            .process_event(X86MatmulMachineEvents::MatmulQ2K(MatmulQ2KRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| MatmulError::Internal)?;
        result.get()
    }

    pub(crate) fn process_q3_k(&mut self, event: OpMulMatQ3K<'_>) -> MatmulResult {
        let result = Cell::new(Err(MatmulError::Internal));
        self.machine
            .process_event(X86MatmulMachineEvents::MatmulQ3K(MatmulQ3KRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| MatmulError::Internal)?;
        result.get()
    }

    pub(crate) fn process_q4_k(&mut self, event: OpMulMatQ4K<'_>) -> MatmulResult {
        let result = Cell::new(Err(MatmulError::Internal));
        self.machine
            .process_event(X86MatmulMachineEvents::MatmulQ4K(MatmulQ4KRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| MatmulError::Internal)?;
        result.get()
    }

    pub(crate) fn process_q6_k(&mut self, event: OpMulMatQ6K<'_>) -> MatmulResult {
        let result = Cell::new(Err(MatmulError::Internal));
        self.machine
            .process_event(X86MatmulMachineEvents::MatmulQ6K(MatmulQ6KRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| MatmulError::Internal)?;
        result.get()
    }

    process_argmax!(
        process_argmax_q4_0,
        OpMulMatArgmaxQ4_0,
        MatmulArgmaxQ4_0Runtime,
        MatmulArgmaxQ4_0
    );
    process_argmax!(
        process_argmax_q4_1,
        OpMulMatArgmaxQ4_1,
        MatmulArgmaxQ4_1Runtime,
        MatmulArgmaxQ4_1
    );
    process_argmax!(
        process_argmax_q5_0,
        OpMulMatArgmaxQ5_0,
        MatmulArgmaxQ5_0Runtime,
        MatmulArgmaxQ5_0
    );
    process_argmax!(
        process_argmax_q8_0,
        OpMulMatArgmaxQ8_0,
        MatmulArgmaxQ8_0Runtime,
        MatmulArgmaxQ8_0
    );
    process_argmax!(
        process_argmax_q2_k,
        OpMulMatArgmaxQ2K,
        MatmulArgmaxQ2KRuntime,
        MatmulArgmaxQ2K
    );
    process_argmax!(
        process_argmax_q3_k,
        OpMulMatArgmaxQ3K,
        MatmulArgmaxQ3KRuntime,
        MatmulArgmaxQ3K
    );
    process_argmax!(
        process_argmax_q4_k,
        OpMulMatArgmaxQ4K,
        MatmulArgmaxQ4KRuntime,
        MatmulArgmaxQ4K
    );
    process_argmax!(
        process_argmax_q6_k,
        OpMulMatArgmaxQ6K,
        MatmulArgmaxQ6KRuntime,
        MatmulArgmaxQ6K
    );
}

#[inline]
fn packed_u16(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

#[inline]
fn fill_i16(source: &[i8], destination: &mut [i16]) {
    let mut index = 0;
    while index < source.len() {
        destination[index] = i16::from(source[index]);
        index += 1;
    }
}

#[inline]
fn fill_i16_bytes(source: &[u8], destination: &mut [i16]) {
    let mut index = 0;
    while index < source.len() {
        destination[index] = i16::from(i8::from_ne_bytes([source[index]]));
        index += 1;
    }
}

struct I16Dot<'a> {
    lhs: &'a [i16],
    rhs: &'a [i16],
    output: &'a mut i32,
}

impl WithSimd for I16Dot<'_> {
    type Output = ();

    fn with_simd<S: Simd>(self, simd: S) {
        let Self { lhs, rhs, output } = self;
        let (lhs_vectors, lhs_tail) = S::as_simd_i16s(lhs);
        let (rhs_vectors, rhs_tail) = S::as_simd_i16s(rhs);
        let mut total = 0_i32;
        for (lhs_vector, rhs_vector) in lhs_vectors.iter().zip(rhs_vectors) {
            let product = simd.mul_i16s(*lhs_vector, *rhs_vector);
            let mut lanes = [0_i16; 16];
            simd.partial_store_i16s(&mut lanes, product);
            let mut lane = 0;
            while lane < 16 {
                total += i32::from(lanes[lane]);
                lane += 1;
            }
        }
        for (lhs_value, rhs_value) in lhs_tail.iter().zip(rhs_tail) {
            total += i32::from(*lhs_value) * i32::from(*rhs_value);
        }
        *output = total;
    }
}

#[inline]
fn dot_i16(backend: pulp::x86::V3, lhs: &[i16], rhs: &[i16]) -> i32 {
    let mut output = 0_i32;
    Simd::vectorize(
        backend,
        I16Dot {
            lhs,
            rhs,
            output: &mut output,
        },
    );
    output
}

const fn unpack_q3_scales(bytes: &[u8]) -> [i8; 16] {
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0f0f_0f0f;
    let mut aux = [0_u32; 4];
    aux[0] = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    aux[1] = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    aux[2] = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    let temporary = aux[2];
    aux[2] = ((aux[0] >> 4) & KMASK2) | (((temporary >> 4) & KMASK1) << 4);
    aux[3] = ((aux[1] >> 4) & KMASK2) | (((temporary >> 6) & KMASK1) << 4);
    aux[0] = (aux[0] & KMASK2) | ((temporary & KMASK1) << 4);
    aux[1] = (aux[1] & KMASK2) | (((temporary >> 2) & KMASK1) << 4);
    let mut scales = [0_i8; 16];
    let mut index = 0;
    while index < 4 {
        let word = aux[index].to_le_bytes();
        scales[index * 4] = i8::from_ne_bytes([word[0]]).wrapping_sub(32);
        scales[index * 4 + 1] = i8::from_ne_bytes([word[1]]).wrapping_sub(32);
        scales[index * 4 + 2] = i8::from_ne_bytes([word[2]]).wrapping_sub(32);
        scales[index * 4 + 3] = i8::from_ne_bytes([word[3]]).wrapping_sub(32);
        index += 1;
    }
    scales
}
