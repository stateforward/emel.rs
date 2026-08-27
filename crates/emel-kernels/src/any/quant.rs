//! Safe scalar kernels for the packed `q4_0` and `q8_0` formats.
//!
//! The byte layout and arithmetic in this module are pinned to
//! `emel.cpp@843a117386ef17dc5a50549bbfc821074c2141d6`,
//! `src/emel/kernel/detail.hpp`: `QK4_0 == QK8_0 == 32`, little-endian
//! IEEE-754 binary16 scales, and the nibble ordering used by the reference
//! `dot_q4_0_q8_0_row_scalar` implementation.  The API borrows encoded model
//! storage and never allocates or materializes a whole dequantized tensor.

use core::fmt;

/// Number of values represented by one `q4_0` or `q8_0` block.
pub const BLOCK_VALUES: usize = 32;
/// Number of bytes in one packed `q4_0` block (`d` plus 16 packed nibbles).
pub const Q4_0_BLOCK_BYTES: usize = 18;
/// Number of bytes in one packed `q8_0` block (`d` plus 32 signed values).
pub const Q8_0_BLOCK_BYTES: usize = 34;

/// Packed format used by a validated quantized row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Format {
    /// Symmetric four-bit values with a binary16 scale.
    Q4_0,
    /// Symmetric eight-bit values with a binary16 scale.
    Q8_0,
}

/// Errors returned before a quantized kernel touches packed input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuantError {
    /// The encoded row is not an integral number of blocks.
    InvalidBlockLength {
        /// Format whose block size was required.
        format: Format,
        /// Number of bytes supplied by the caller.
        bytes: usize,
    },
    /// The two rows do not have the same number of blocks.
    MismatchedBlockCount {
        /// Number of left-hand blocks.
        lhs: usize,
        /// Number of right-hand blocks.
        rhs: usize,
    },
}

impl fmt::Display for QuantError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBlockLength { format, bytes } => {
                write!(formatter, "invalid {format:?} block length: {bytes} bytes")
            }
            Self::MismatchedBlockCount { lhs, rhs } => {
                write!(
                    formatter,
                    "mismatched quantized block counts: {lhs} != {rhs}"
                )
            }
        }
    }
}

impl std::error::Error for QuantError {}

/// A borrowed, validated row of packed `q4_0` blocks.
#[derive(Clone, Copy, Debug)]
pub struct Q4_0Row<'a> {
    bytes: &'a [u8],
}

impl<'a> Q4_0Row<'a> {
    /// Validates and borrows an encoded `q4_0` row.
    ///
    /// # Errors
    ///
    /// Returns [`QuantError::InvalidBlockLength`] when `bytes` is not an
    /// integral number of packed blocks.
    pub const fn from_bytes(bytes: &'a [u8]) -> Result<Self, QuantError> {
        if !bytes.len().is_multiple_of(Q4_0_BLOCK_BYTES) {
            return Err(QuantError::InvalidBlockLength {
                format: Format::Q4_0,
                bytes: bytes.len(),
            });
        }
        Ok(Self { bytes })
    }

    /// Returns the number of packed blocks in this row.
    #[must_use]
    pub const fn block_count(self) -> usize {
        self.bytes.len() / Q4_0_BLOCK_BYTES
    }

    /// Returns the encoded bytes borrowed by this row.
    #[must_use]
    pub const fn as_bytes(self) -> &'a [u8] {
        self.bytes
    }

    /// Decodes one block into exactly 32 values without allocating.
    ///
    /// # Errors
    ///
    /// Returns [`QuantError::InvalidBlockLength`] when `block` is outside the
    /// validated row.
    pub fn decode_block(
        self,
        block: usize,
        output: &mut [f32; BLOCK_VALUES],
    ) -> Result<(), QuantError> {
        let offset = block
            .checked_mul(Q4_0_BLOCK_BYTES)
            .ok_or(QuantError::InvalidBlockLength {
                format: Format::Q4_0,
                bytes: self.bytes.len(),
            })?;
        if offset
            .checked_add(Q4_0_BLOCK_BYTES)
            .is_none_or(|end| end > self.bytes.len())
        {
            return Err(QuantError::InvalidBlockLength {
                format: Format::Q4_0,
                bytes: self.bytes.len(),
            });
        }
        let block = &self.bytes[offset..offset + Q4_0_BLOCK_BYTES];
        let scale = fp16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for index in 0..Q4_0_BLOCK_BYTES - 2 {
            let packed = block[index + 2];
            output[index] = (f32::from(packed & 0x0f) - 8.0) * scale;
            output[index + BLOCK_VALUES / 2] = (f32::from(packed >> 4) - 8.0) * scale;
        }
        Ok(())
    }
}

/// A borrowed, validated row of packed `q8_0` blocks.
#[derive(Clone, Copy, Debug)]
pub struct Q8_0Row<'a> {
    bytes: &'a [u8],
}

impl<'a> Q8_0Row<'a> {
    /// Validates and borrows an encoded `q8_0` row.
    ///
    /// # Errors
    ///
    /// Returns [`QuantError::InvalidBlockLength`] when `bytes` is not an
    /// integral number of packed blocks.
    pub const fn from_bytes(bytes: &'a [u8]) -> Result<Self, QuantError> {
        if !bytes.len().is_multiple_of(Q8_0_BLOCK_BYTES) {
            return Err(QuantError::InvalidBlockLength {
                format: Format::Q8_0,
                bytes: bytes.len(),
            });
        }
        Ok(Self { bytes })
    }

    /// Returns the number of packed blocks in this row.
    #[must_use]
    pub const fn block_count(self) -> usize {
        self.bytes.len() / Q8_0_BLOCK_BYTES
    }

    /// Returns the encoded bytes borrowed by this row.
    #[must_use]
    pub const fn as_bytes(self) -> &'a [u8] {
        self.bytes
    }

    /// Decodes one block into exactly 32 values without allocating.
    ///
    /// # Errors
    ///
    /// Returns [`QuantError::InvalidBlockLength`] when `block` is outside the
    /// validated row.
    pub fn decode_block(
        self,
        block: usize,
        output: &mut [f32; BLOCK_VALUES],
    ) -> Result<(), QuantError> {
        let offset = block
            .checked_mul(Q8_0_BLOCK_BYTES)
            .ok_or(QuantError::InvalidBlockLength {
                format: Format::Q8_0,
                bytes: self.bytes.len(),
            })?;
        if offset
            .checked_add(Q8_0_BLOCK_BYTES)
            .is_none_or(|end| end > self.bytes.len())
        {
            return Err(QuantError::InvalidBlockLength {
                format: Format::Q8_0,
                bytes: self.bytes.len(),
            });
        }
        let block = &self.bytes[offset..offset + Q8_0_BLOCK_BYTES];
        let scale = fp16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for index in 0..BLOCK_VALUES {
            let quantized = i8::from_ne_bytes([block[index + 2]]);
            output[index] = f32::from(quantized) * scale;
        }
        Ok(())
    }
}

/// Computes the reference scalar `q4_0` by `q8_0` dot product.
///
/// This consumes packed values directly.  It does not dequantize either row
/// into an intermediate tensor and performs no allocation.
///
/// # Errors
///
/// Returns [`QuantError::MismatchedBlockCount`] when the rows have different
/// numbers of blocks.
// Keep separate scale products and accumulation to preserve reference bits.
#[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
pub fn dot_q4_0_q8_0(lhs: Q4_0Row<'_>, rhs: Q8_0Row<'_>) -> Result<f32, QuantError> {
    if lhs.block_count() != rhs.block_count() {
        return Err(QuantError::MismatchedBlockCount {
            lhs: lhs.block_count(),
            rhs: rhs.block_count(),
        });
    }

    let mut sum = 0.0_f32;
    for block in 0..lhs.block_count() {
        let lhs_offset = block * Q4_0_BLOCK_BYTES;
        let rhs_offset = block * Q8_0_BLOCK_BYTES;
        let lhs_block = &lhs.bytes[lhs_offset..lhs_offset + Q4_0_BLOCK_BYTES];
        let rhs_block = &rhs.bytes[rhs_offset..rhs_offset + Q8_0_BLOCK_BYTES];
        let lhs_scale = fp16_to_f32(u16::from_le_bytes([lhs_block[0], lhs_block[1]]));
        let rhs_scale = fp16_to_f32(u16::from_le_bytes([rhs_block[0], rhs_block[1]]));
        let mut integer_sum = 0_i32;
        for index in 0..Q4_0_BLOCK_BYTES - 2 {
            let packed = lhs_block[index + 2];
            let lhs_low = i32::from(packed & 0x0f) - 8;
            let lhs_high = i32::from(packed >> 4) - 8;
            let rhs_low = i32::from(i8::from_ne_bytes([rhs_block[index + 2]]));
            let rhs_high = i32::from(i8::from_ne_bytes([rhs_block[index + 2 + BLOCK_VALUES / 2]]));
            integer_sum += lhs_low * rhs_low + lhs_high * rhs_high;
        }
        sum += integer_sum as f32 * (lhs_scale * rhs_scale);
    }
    Ok(sum)
}

/// Computes the reference scalar `q8_0` by `q8_0` dot product.
///
/// # Errors
///
/// Returns [`QuantError::MismatchedBlockCount`] when the rows have different
/// numbers of blocks.
// Keep separate scale products and accumulation to preserve reference bits.
#[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
pub fn dot_q8_0_q8_0(lhs: Q8_0Row<'_>, rhs: Q8_0Row<'_>) -> Result<f32, QuantError> {
    if lhs.block_count() != rhs.block_count() {
        return Err(QuantError::MismatchedBlockCount {
            lhs: lhs.block_count(),
            rhs: rhs.block_count(),
        });
    }

    let mut sum = 0.0_f32;
    for block in 0..lhs.block_count() {
        let lhs_offset = block * Q8_0_BLOCK_BYTES;
        let rhs_offset = block * Q8_0_BLOCK_BYTES;
        let lhs_block = &lhs.bytes[lhs_offset..lhs_offset + Q8_0_BLOCK_BYTES];
        let rhs_block = &rhs.bytes[rhs_offset..rhs_offset + Q8_0_BLOCK_BYTES];
        let lhs_scale = fp16_to_f32(u16::from_le_bytes([lhs_block[0], lhs_block[1]]));
        let rhs_scale = fp16_to_f32(u16::from_le_bytes([rhs_block[0], rhs_block[1]]));
        let mut integer_sum = 0_i32;
        for index in 0..BLOCK_VALUES {
            let lhs_value = i32::from(i8::from_ne_bytes([lhs_block[index + 2]]));
            let rhs_value = i32::from(i8::from_ne_bytes([rhs_block[index + 2]]));
            integer_sum += lhs_value * rhs_value;
        }
        sum += integer_sum as f32 * (lhs_scale * rhs_scale);
    }
    Ok(sum)
}

/// Converts a packed binary16 value using the same arithmetic as the pinned
/// reference's `fp16_to_fp32` helper.
#[inline]
pub(crate) fn fp16_to_f32(bits16: u16) -> f32 {
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

/// Converts binary32 to binary16 using the pinned `fp32_to_fp16` bit path.
///
/// This is shared by kernels whose compile-time destination is F16. Keeping
/// the conversion here prevents each actor from introducing a subtly
/// different rounding or subnormal boundary.
#[inline]
#[allow(clippy::assign_op_pattern)]
pub(crate) fn fp32_to_fp16(value: f32) -> u16 {
    let scale_to_inf = f32::from_bits(0x7780_0000);
    let scale_to_zero = f32::from_bits(0x0880_0000);
    let mut base = (value.abs() * scale_to_inf) * scale_to_zero;

    let word = value.to_bits();
    let doubled = word.wrapping_add(word);
    let sign = word & 0x8000_0000;
    let mut bias = doubled & 0xff00_0000;
    if bias < 0x7100_0000 {
        bias = 0x7100_0000;
    }

    base = f32::from_bits((bias >> 1).wrapping_add(0x0780_0000)) + base;
    let encoded_bits = base.to_bits();
    let exponent_bits = (encoded_bits >> 13) & 0x0000_7c00;
    let mantissa_bits = encoded_bits & 0x0000_0fff;
    let nonsign = exponent_bits + mantissa_bits;
    let encoded = if doubled > 0xff00_0000 {
        0x7e00
    } else {
        nonsign
    };
    u16::try_from((sign >> 16) | encoded).expect("binary16 encoding fits u16")
}
