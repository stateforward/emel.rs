//! Safe scalar kernels for the legacy packed `q4_1`, `q5_0`, and `q5_1`, plus
//! the canonical packed `q8_1` format.
//!
//! The `q2_k`, `q3_k`, `q4_k`, `q4_1`, `q5_0`, and `q5_1` layouts and their direct
//! dot paths are pinned to `emel.cpp@843a117386ef17dc5a50549bbfc821074c2141d6`,
//! `src/emel/kernel/detail.hpp` lines 236-256, 504-521, 631-677, and
//! 3155-3249. The scalar `q2_k` by `q8_k` block path is pinned to lines
//! 2414-2450. The `q3_k` by `q8_k` path is pinned to lines 2567-2760. The
//! `q4_k` by `q8_k` row path is pinned to lines 2846-2942. The
//! `q5_k` by `q8_k` path is pinned to lines 257-302 and 2944-3000. The `q8_1`
//! layout is the canonical GGML layout represented by
//! the pinned source's `dtype::q8_1` enum: two little-endian binary16 values
//! (`d`, `s`) followed by 32 signed bytes.  The pinned EMEL detail source
//! does not define a `block_q8_1` or a `q8_1` dot routine, so this module only
//! exposes its decode surface and does not claim EMEL source parity for it.
//!
//! All APIs borrow encoded model storage, validate block boundaries before
//! reading, and use fixed-size stack output or direct packed accumulation.
//! They never allocate or materialize a whole dequantized tensor.

use core::fmt;

use crate::any::quant::fp16_to_f32;

/// Number of values represented by one legacy quantized block.
pub const BLOCK_VALUES: usize = 32;
/// Number of bytes in one packed `q4_1` block (`d`, `m`, and 16 nibbles).
pub const Q4_1_BLOCK_BYTES: usize = 20;
/// Number of bytes in one packed `q5_0` block (`d`, `qh`, and 16 nibbles).
pub const Q5_0_BLOCK_BYTES: usize = 22;
/// Number of bytes in one packed `q5_1` block (`d`, `m`, `qh`, and 16 nibbles).
pub const Q5_1_BLOCK_BYTES: usize = 24;
/// Number of bytes in one packed `q2_k` block.
pub const Q2_K_BLOCK_BYTES: usize = 84;
/// Number of bytes in one packed `q3_k` block.
pub const Q3_K_BLOCK_BYTES: usize = 110;
/// Number of bytes in one packed `q4_k` block.
pub const Q4_K_BLOCK_BYTES: usize = 144;
/// Number of bytes in one packed `q5_k` block.
pub const Q5_K_BLOCK_BYTES: usize = 176;
/// Number of bytes in one packed `q6_k` block.
pub const Q6_K_BLOCK_BYTES: usize = 210;
/// Number of bytes in one canonical packed `q8_1` block (`d`, `s`, and 32 bytes).
pub const Q8_1_BLOCK_BYTES: usize = 36;
/// Number of bytes in one packed `q8_k` block (`d`, 256 values, and 16 sums).
pub const Q8_K_BLOCK_BYTES: usize = 292;
/// Number of values represented by one K-family block.
pub const QK_K_VALUES: usize = 256;

/// Packed format used by a validated legacy quantized row.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Format {
    /// Asymmetric four-bit values with binary16 scale and minimum.
    Q4_1,
    /// Symmetric five-bit values with a binary16 scale.
    Q5_0,
    /// Asymmetric five-bit values with binary16 scale and minimum.
    Q5_1,
    /// K-family asymmetric two-bit values with binary16 scale and minimum.
    Q2K,
    /// K-family signed three-bit values with binary16 scale.
    Q3K,
    /// K-family asymmetric four-bit values with binary16 scale and minimum.
    Q4K,
    /// Eight-bit values with binary16 scale and stored sum.
    Q8_1,
    /// K-family asymmetric five-bit values with binary16 scale and minimum.
    Q5K,
    /// K-family signed six-bit values with signed per-group scales.
    Q6K,
    /// K-family eight-bit values with binary32 scale and block sums.
    Q8K,
}

/// Validation errors returned before a packed kernel reads input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QuantMoreError {
    /// The encoded row is not an integral number of blocks.
    InvalidBlockLength {
        /// Format whose block size was required.
        format: Format,
        /// Number of bytes supplied by the caller.
        bytes: usize,
    },
    /// A requested block is outside a validated row.
    BlockOutOfRange {
        /// Format whose block was requested.
        format: Format,
        /// Requested block index.
        block: usize,
        /// Number of blocks in the row.
        block_count: usize,
    },
    /// Two rows do not have the same number of blocks.
    MismatchedBlockCount {
        /// Number of left-hand blocks.
        lhs: usize,
        /// Number of right-hand blocks.
        rhs: usize,
    },
}

impl fmt::Display for QuantMoreError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBlockLength { format, bytes } => {
                write!(formatter, "invalid {format:?} block length: {bytes} bytes")
            }
            Self::BlockOutOfRange {
                format,
                block,
                block_count,
            } => write!(
                formatter,
                "{format:?} block {block} is outside row with {block_count} blocks"
            ),
            Self::MismatchedBlockCount { lhs, rhs } => {
                write!(
                    formatter,
                    "mismatched quantized block counts: {lhs} != {rhs}"
                )
            }
        }
    }
}

impl std::error::Error for QuantMoreError {}

const fn block_count(
    bytes: usize,
    block_bytes: usize,
    format: Format,
) -> Result<usize, QuantMoreError> {
    if bytes.is_multiple_of(block_bytes) {
        Ok(bytes / block_bytes)
    } else {
        Err(QuantMoreError::InvalidBlockLength { format, bytes })
    }
}

fn block(
    bytes: &[u8],
    index: usize,
    block_bytes: usize,
    format: Format,
) -> Result<&[u8], QuantMoreError> {
    let count = bytes.len() / block_bytes;
    let offset = index
        .checked_mul(block_bytes)
        .ok_or(QuantMoreError::BlockOutOfRange {
            format,
            block: index,
            block_count: count,
        })?;
    let end = offset
        .checked_add(block_bytes)
        .ok_or(QuantMoreError::BlockOutOfRange {
            format,
            block: index,
            block_count: count,
        })?;
    bytes
        .get(offset..end)
        .ok_or(QuantMoreError::BlockOutOfRange {
            format,
            block: index,
            block_count: count,
        })
}

/// A borrowed, validated row of packed `q4_1` blocks.
#[derive(Clone, Copy, Debug)]
pub struct Q4_1Row<'a> {
    bytes: &'a [u8],
}

impl<'a> Q4_1Row<'a> {
    /// Validates and borrows an encoded `q4_1` row.
    ///
    /// # Errors
    ///
    /// Returns [`QuantMoreError::InvalidBlockLength`] when `bytes` is not an
    /// integral number of packed blocks.
    pub const fn from_bytes(bytes: &'a [u8]) -> Result<Self, QuantMoreError> {
        match block_count(bytes.len(), Q4_1_BLOCK_BYTES, Format::Q4_1) {
            Ok(_) => Ok(Self { bytes }),
            Err(error) => Err(error),
        }
    }

    /// Returns the number of packed blocks in this row.
    #[must_use]
    pub const fn block_count(self) -> usize {
        self.bytes.len() / Q4_1_BLOCK_BYTES
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
    /// Returns [`QuantMoreError::BlockOutOfRange`] when `index` is outside
    /// this validated row.
    pub fn decode_block(
        self,
        index: usize,
        output: &mut [f32; BLOCK_VALUES],
    ) -> Result<(), QuantMoreError> {
        let bytes = block(self.bytes, index, Q4_1_BLOCK_BYTES, Format::Q4_1)?;
        let scale = fp16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
        let minimum = fp16_to_f32(u16::from_le_bytes([bytes[2], bytes[3]]));
        for j in 0..BLOCK_VALUES / 2 {
            let packed = bytes[j + 4];
            output[j] = f32::from(packed & 0x0f).mul_add(scale, minimum);
            output[j + BLOCK_VALUES / 2] = f32::from(packed >> 4).mul_add(scale, minimum);
        }
        Ok(())
    }
}

/// A borrowed, validated row of packed `q5_0` blocks.
#[derive(Clone, Copy, Debug)]
pub struct Q5_0Row<'a> {
    bytes: &'a [u8],
}

impl<'a> Q5_0Row<'a> {
    /// Validates and borrows an encoded `q5_0` row.
    ///
    /// # Errors
    ///
    /// Returns [`QuantMoreError::InvalidBlockLength`] when `bytes` is not an
    /// integral number of packed blocks.
    pub const fn from_bytes(bytes: &'a [u8]) -> Result<Self, QuantMoreError> {
        match block_count(bytes.len(), Q5_0_BLOCK_BYTES, Format::Q5_0) {
            Ok(_) => Ok(Self { bytes }),
            Err(error) => Err(error),
        }
    }

    /// Returns the number of packed blocks in this row.
    #[must_use]
    pub const fn block_count(self) -> usize {
        self.bytes.len() / Q5_0_BLOCK_BYTES
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
    /// Returns [`QuantMoreError::BlockOutOfRange`] when `index` is outside
    /// this validated row.
    pub fn decode_block(
        self,
        index: usize,
        output: &mut [f32; BLOCK_VALUES],
    ) -> Result<(), QuantMoreError> {
        let bytes = block(self.bytes, index, Q5_0_BLOCK_BYTES, Format::Q5_0)?;
        let scale = fp16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
        let high = u32::from_le_bytes([bytes[2], bytes[3], bytes[4], bytes[5]]);
        for j in 0..BLOCK_VALUES / 2 {
            let packed = bytes[j + 6];
            let low_high_bit = ((high >> j) & 1) as u8;
            let high_high_bit = ((high >> (j + BLOCK_VALUES / 2)) & 1) as u8;
            let low = (packed & 0x0f) | (low_high_bit << 4);
            let high_value = (packed >> 4) | (high_high_bit << 4);
            output[j] = (f32::from(low) - 16.0) * scale;
            output[j + BLOCK_VALUES / 2] = (f32::from(high_value) - 16.0) * scale;
        }
        Ok(())
    }
}

/// A borrowed, validated row of packed `q5_1` blocks.
#[derive(Clone, Copy, Debug)]
pub struct Q5_1Row<'a> {
    bytes: &'a [u8],
}

impl<'a> Q5_1Row<'a> {
    /// Validates and borrows an encoded `q5_1` row.
    ///
    /// # Errors
    ///
    /// Returns [`QuantMoreError::InvalidBlockLength`] when `bytes` is not an
    /// integral number of packed blocks.
    pub const fn from_bytes(bytes: &'a [u8]) -> Result<Self, QuantMoreError> {
        match block_count(bytes.len(), Q5_1_BLOCK_BYTES, Format::Q5_1) {
            Ok(_) => Ok(Self { bytes }),
            Err(error) => Err(error),
        }
    }

    /// Returns the number of packed blocks in this row.
    #[must_use]
    pub const fn block_count(self) -> usize {
        self.bytes.len() / Q5_1_BLOCK_BYTES
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
    /// Returns [`QuantMoreError::BlockOutOfRange`] when `index` is outside
    /// this validated row.
    pub fn decode_block(
        self,
        index: usize,
        output: &mut [f32; BLOCK_VALUES],
    ) -> Result<(), QuantMoreError> {
        let bytes = block(self.bytes, index, Q5_1_BLOCK_BYTES, Format::Q5_1)?;
        let scale = fp16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
        let minimum = fp16_to_f32(u16::from_le_bytes([bytes[2], bytes[3]]));
        let high = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        for j in 0..BLOCK_VALUES / 2 {
            let packed = bytes[j + 8];
            let low_high_bit = ((high >> j) & 1) as u8;
            let high_high_bit = ((high >> (j + BLOCK_VALUES / 2)) & 1) as u8;
            let low = (packed & 0x0f) | (low_high_bit << 4);
            let high_value = (packed >> 4) | (high_high_bit << 4);
            output[j] = f32::from(low).mul_add(scale, minimum);
            output[j + BLOCK_VALUES / 2] = f32::from(high_value).mul_add(scale, minimum);
        }
        Ok(())
    }
}

/// A borrowed, validated row of packed `q2_k` blocks.
///
/// Each block contains 16 packed scale/minimum bytes, 64 packed two-bit
/// values, and binary16 `d`/`dmin` fields. The byte layout is the pinned
/// `block_q2_k` layout from `detail.hpp:236-244`.
#[derive(Clone, Copy, Debug)]
pub struct Q2KRow<'a> {
    bytes: &'a [u8],
}

impl<'a> Q2KRow<'a> {
    /// Validates and borrows an encoded `q2_k` row.
    ///
    /// # Errors
    ///
    /// Returns [`QuantMoreError::InvalidBlockLength`] when `bytes` is not an
    /// integral number of packed blocks.
    pub const fn from_bytes(bytes: &'a [u8]) -> Result<Self, QuantMoreError> {
        match block_count(bytes.len(), Q2_K_BLOCK_BYTES, Format::Q2K) {
            Ok(_) => Ok(Self { bytes }),
            Err(error) => Err(error),
        }
    }

    /// Returns the number of packed blocks in this row.
    #[must_use]
    pub const fn block_count(self) -> usize {
        self.bytes.len() / Q2_K_BLOCK_BYTES
    }

    /// Returns the encoded bytes borrowed by this row.
    #[must_use]
    pub const fn as_bytes(self) -> &'a [u8] {
        self.bytes
    }
}

/// A borrowed, validated row of packed `q3_k` blocks.
///
/// Each block contains 32 high-bit masks, 64 packed low-bit bytes, 12 packed
/// signed six-bit scales, and a binary16 `d` field. This is the pinned
/// `block_q3_k` layout from `detail.hpp:247-253`.
#[derive(Clone, Copy, Debug)]
pub struct Q3KRow<'a> {
    bytes: &'a [u8],
}

/// A borrowed, validated row of packed `q4_k` blocks.
///
/// Each block contains binary16 `d`/`dmin` fields, 12 packed six-bit
/// scale/minimum values, and 128 bytes containing 256 four-bit values. This
/// is the pinned `block_q4_k` layout from `detail.hpp:250-256`.
#[derive(Clone, Copy, Debug)]
pub struct Q4KRow<'a> {
    bytes: &'a [u8],
}

impl<'a> Q4KRow<'a> {
    /// Validates and borrows an encoded `q4_k` row.
    ///
    /// # Errors
    ///
    /// Returns [`QuantMoreError::InvalidBlockLength`] when `bytes` is not an
    /// integral number of packed blocks.
    pub const fn from_bytes(bytes: &'a [u8]) -> Result<Self, QuantMoreError> {
        match block_count(bytes.len(), Q4_K_BLOCK_BYTES, Format::Q4K) {
            Ok(_) => Ok(Self { bytes }),
            Err(error) => Err(error),
        }
    }

    /// Returns the number of packed blocks in this row.
    #[must_use]
    pub const fn block_count(self) -> usize {
        self.bytes.len() / Q4_K_BLOCK_BYTES
    }

    /// Returns the encoded bytes borrowed by this row.
    #[must_use]
    pub const fn as_bytes(self) -> &'a [u8] {
        self.bytes
    }

    /// Decodes one block into exactly 256 values without allocating.
    ///
    /// The scale/minimum unpacking and low/high nibble order match the pinned
    /// `dequantize_row_q4_k` implementation in `detail.hpp:523-550`.
    ///
    /// # Errors
    ///
    /// Returns [`QuantMoreError::BlockOutOfRange`] when `index` is outside
    /// this validated row.
    #[allow(clippy::suboptimal_flops)]
    pub fn decode_block(
        self,
        index: usize,
        output: &mut [f32; QK_K_VALUES],
    ) -> Result<(), QuantMoreError> {
        let bytes = block(self.bytes, index, Q4_K_BLOCK_BYTES, Format::Q4K)?;
        let scale = fp16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
        let minimum = fp16_to_f32(u16::from_le_bytes([bytes[2], bytes[3]]));
        let mut scales_data = [0_u8; 12];
        scales_data.copy_from_slice(&bytes[4..16]);

        let mut scale_index = 0;
        let mut output_offset = 0;
        while output_offset < QK_K_VALUES {
            let (scale_a, minimum_a) = q4_k_scale_min(scale_index, &scales_data);
            let (scale_b, minimum_b) = q4_k_scale_min(scale_index + 1, &scales_data);
            let d0 = scale * f32::from(scale_a);
            let d1 = scale * f32::from(scale_b);
            let m0 = minimum * f32::from(minimum_a);
            let m1 = minimum * f32::from(minimum_b);
            let packed_offset = 16 + output_offset / 2;
            for lane in 0..QK_K_VALUES / 8 {
                let packed = bytes[packed_offset + lane];
                output[output_offset + lane] = d0 * f32::from(packed & 0x0f) - m0;
                output[output_offset + lane + QK_K_VALUES / 8] = d1 * f32::from(packed >> 4) - m1;
            }
            scale_index += 2;
            output_offset += QK_K_VALUES / 4;
        }
        Ok(())
    }
}

impl<'a> Q3KRow<'a> {
    /// Validates and borrows an encoded `q3_k` row.
    ///
    /// # Errors
    ///
    /// Returns [`QuantMoreError::InvalidBlockLength`] when `bytes` is not an
    /// integral number of packed blocks.
    pub const fn from_bytes(bytes: &'a [u8]) -> Result<Self, QuantMoreError> {
        match block_count(bytes.len(), Q3_K_BLOCK_BYTES, Format::Q3K) {
            Ok(_) => Ok(Self { bytes }),
            Err(error) => Err(error),
        }
    }

    /// Returns the number of packed blocks in this row.
    #[must_use]
    pub const fn block_count(self) -> usize {
        self.bytes.len() / Q3_K_BLOCK_BYTES
    }

    /// Returns the encoded bytes borrowed by this row.
    #[must_use]
    pub const fn as_bytes(self) -> &'a [u8] {
        self.bytes
    }
}

/// A borrowed, validated row of packed `q5_k` blocks.
#[derive(Clone, Copy, Debug)]
pub struct Q5KRow<'a> {
    bytes: &'a [u8],
}

/// A borrowed, validated row of packed `q6_k` blocks.
///
/// Each block contains 128 low quantization bytes, 64 high-bit bytes, 16
/// signed per-16-value scales, and a binary16 `d` field. This is the pinned
/// `block_q6_k` layout from `detail.hpp:265-272`.
#[derive(Clone, Copy, Debug)]
pub struct Q6KRow<'a> {
    bytes: &'a [u8],
}

impl<'a> Q5KRow<'a> {
    /// Validates and borrows an encoded `q5_k` row.
    ///
    /// # Errors
    ///
    /// Returns [`QuantMoreError::InvalidBlockLength`] when `bytes` is not an
    /// integral number of packed blocks.
    pub const fn from_bytes(bytes: &'a [u8]) -> Result<Self, QuantMoreError> {
        match block_count(bytes.len(), Q5_K_BLOCK_BYTES, Format::Q5K) {
            Ok(_) => Ok(Self { bytes }),
            Err(error) => Err(error),
        }
    }

    /// Returns the number of packed blocks in this row.
    #[must_use]
    pub const fn block_count(self) -> usize {
        self.bytes.len() / Q5_K_BLOCK_BYTES
    }

    /// Returns the encoded bytes borrowed by this row.
    #[must_use]
    pub const fn as_bytes(self) -> &'a [u8] {
        self.bytes
    }
}

impl<'a> Q6KRow<'a> {
    /// Validates and borrows an encoded `q6_k` row.
    ///
    /// # Errors
    ///
    /// Returns [`QuantMoreError::InvalidBlockLength`] when `bytes` is not an
    /// integral number of packed blocks.
    pub const fn from_bytes(bytes: &'a [u8]) -> Result<Self, QuantMoreError> {
        match block_count(bytes.len(), Q6_K_BLOCK_BYTES, Format::Q6K) {
            Ok(_) => Ok(Self { bytes }),
            Err(error) => Err(error),
        }
    }

    /// Returns the number of packed blocks in this row.
    #[must_use]
    pub const fn block_count(self) -> usize {
        self.bytes.len() / Q6_K_BLOCK_BYTES
    }

    /// Returns the encoded bytes borrowed by this row.
    #[must_use]
    pub const fn as_bytes(self) -> &'a [u8] {
        self.bytes
    }
}

/// A borrowed, validated row of packed `q8_k` blocks.
#[derive(Clone, Copy, Debug)]
pub struct Q8KRow<'a> {
    bytes: &'a [u8],
}

impl<'a> Q8KRow<'a> {
    /// Validates and borrows an encoded `q8_k` row.
    ///
    /// # Errors
    ///
    /// Returns [`QuantMoreError::InvalidBlockLength`] when `bytes` is not an
    /// integral number of packed blocks.
    pub const fn from_bytes(bytes: &'a [u8]) -> Result<Self, QuantMoreError> {
        match block_count(bytes.len(), Q8_K_BLOCK_BYTES, Format::Q8K) {
            Ok(_) => Ok(Self { bytes }),
            Err(error) => Err(error),
        }
    }

    /// Returns the number of packed blocks in this row.
    #[must_use]
    pub const fn block_count(self) -> usize {
        self.bytes.len() / Q8_K_BLOCK_BYTES
    }

    /// Returns the encoded bytes borrowed by this row.
    #[must_use]
    pub const fn as_bytes(self) -> &'a [u8] {
        self.bytes
    }
}

/// A borrowed, validated canonical GGML `q8_1` row.
#[derive(Clone, Copy, Debug)]
pub struct Q8_1Row<'a> {
    bytes: &'a [u8],
}

impl<'a> Q8_1Row<'a> {
    /// Validates and borrows an encoded `q8_1` row.
    ///
    /// # Errors
    ///
    /// Returns [`QuantMoreError::InvalidBlockLength`] when `bytes` is not an
    /// integral number of packed blocks.
    pub const fn from_bytes(bytes: &'a [u8]) -> Result<Self, QuantMoreError> {
        match block_count(bytes.len(), Q8_1_BLOCK_BYTES, Format::Q8_1) {
            Ok(_) => Ok(Self { bytes }),
            Err(error) => Err(error),
        }
    }

    /// Returns the number of packed blocks in this row.
    #[must_use]
    pub const fn block_count(self) -> usize {
        self.bytes.len() / Q8_1_BLOCK_BYTES
    }

    /// Returns the encoded bytes borrowed by this row.
    #[must_use]
    pub const fn as_bytes(self) -> &'a [u8] {
        self.bytes
    }

    /// Returns the stored binary16 sum field for one block.
    ///
    /// # Errors
    ///
    /// Returns [`QuantMoreError::BlockOutOfRange`] when `index` is outside
    /// this validated row.
    pub fn sum_bits(self, index: usize) -> Result<u16, QuantMoreError> {
        let bytes = block(self.bytes, index, Q8_1_BLOCK_BYTES, Format::Q8_1)?;
        Ok(u16::from_le_bytes([bytes[2], bytes[3]]))
    }

    /// Decodes one block into exactly 32 values without allocating.
    ///
    /// # Errors
    ///
    /// Returns [`QuantMoreError::BlockOutOfRange`] when `index` is outside
    /// this validated row.
    pub fn decode_block(
        self,
        index: usize,
        output: &mut [f32; BLOCK_VALUES],
    ) -> Result<(), QuantMoreError> {
        let bytes = block(self.bytes, index, Q8_1_BLOCK_BYTES, Format::Q8_1)?;
        let scale = fp16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
        for j in 0..BLOCK_VALUES {
            output[j] = f32::from(i8::from_ne_bytes([bytes[j + 4]])) * scale;
        }
        Ok(())
    }
}

const fn equal_blocks(lhs: usize, rhs: usize) -> Result<(), QuantMoreError> {
    if lhs == rhs {
        Ok(())
    } else {
        Err(QuantMoreError::MismatchedBlockCount { lhs, rhs })
    }
}

/// Computes the pinned scalar `q4_1` by `q8_0` row dot product.
///
/// # Errors
///
/// Returns [`QuantMoreError::MismatchedBlockCount`] when the rows have
/// different numbers of blocks.
// Preserve the pinned packed scale and accumulation order; fused arithmetic
// changes the observable floating-point result.
#[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
pub fn dot_q4_1_q8_0(
    lhs: Q4_1Row<'_>,
    rhs: crate::any::quant::Q8_0Row<'_>,
) -> Result<f32, QuantMoreError> {
    equal_blocks(lhs.block_count(), rhs.block_count())?;
    let mut sum = 0.0_f32;
    for index in 0..lhs.block_count() {
        let lhs_block = block(lhs.bytes, index, Q4_1_BLOCK_BYTES, Format::Q4_1)?;
        let rhs_block = rhs.as_bytes();
        let rhs_block = &rhs_block[index * crate::any::quant::Q8_0_BLOCK_BYTES
            ..(index + 1) * crate::any::quant::Q8_0_BLOCK_BYTES];
        let lhs_scale = fp16_to_f32(u16::from_le_bytes([lhs_block[0], lhs_block[1]]));
        let minimum = fp16_to_f32(u16::from_le_bytes([lhs_block[2], lhs_block[3]]));
        let rhs_scale = fp16_to_f32(u16::from_le_bytes([rhs_block[0], rhs_block[1]]));
        let mut packed_sum = 0_i32;
        let mut rhs_sum = 0_i32;
        for j in 0..BLOCK_VALUES / 2 {
            let packed = lhs_block[j + 4];
            let rhs_low = i32::from(i8::from_ne_bytes([rhs_block[j + 2]]));
            let rhs_high = i32::from(i8::from_ne_bytes([rhs_block[j + 18]]));
            packed_sum += i32::from(packed & 0x0f) * rhs_low;
            packed_sum += i32::from(packed >> 4) * rhs_high;
            rhs_sum += rhs_low + rhs_high;
        }
        let scaled_sum = lhs_scale * packed_sum as f32;
        let minimum_sum = minimum * rhs_sum as f32;
        let affine = scaled_sum + minimum_sum;
        sum += rhs_scale * affine;
    }
    Ok(sum)
}

/// Computes the pinned scalar `q5_0` by `q8_0` row dot product.
///
/// # Errors
///
/// Returns [`QuantMoreError::MismatchedBlockCount`] when the rows have
/// different numbers of blocks.
// Preserve the pinned packed scale and accumulation order; fused arithmetic
// changes the observable floating-point result.
#[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
pub fn dot_q5_0_q8_0(
    lhs: Q5_0Row<'_>,
    rhs: crate::any::quant::Q8_0Row<'_>,
) -> Result<f32, QuantMoreError> {
    equal_blocks(lhs.block_count(), rhs.block_count())?;
    let mut sum = 0.0_f32;
    for index in 0..lhs.block_count() {
        let lhs_block = block(lhs.bytes, index, Q5_0_BLOCK_BYTES, Format::Q5_0)?;
        let rhs_bytes = rhs.as_bytes();
        let rhs_block = &rhs_bytes[index * crate::any::quant::Q8_0_BLOCK_BYTES
            ..(index + 1) * crate::any::quant::Q8_0_BLOCK_BYTES];
        let scale = fp16_to_f32(u16::from_le_bytes([lhs_block[0], lhs_block[1]]));
        let rhs_scale = fp16_to_f32(u16::from_le_bytes([rhs_block[0], rhs_block[1]]));
        let high = u32::from_le_bytes([lhs_block[2], lhs_block[3], lhs_block[4], lhs_block[5]]);
        let mut packed_sum = 0_i32;
        for j in 0..BLOCK_VALUES / 2 {
            let packed = lhs_block[j + 6];
            let low = i32::from((packed & 0x0f) | ((((high >> j) & 1) as u8) << 4)) - 16;
            let high_value =
                i32::from((packed >> 4) | ((((high >> (j + 16)) & 1) as u8) << 4)) - 16;
            let rhs_low = i32::from(i8::from_ne_bytes([rhs_block[j + 2]]));
            let rhs_high = i32::from(i8::from_ne_bytes([rhs_block[j + 18]]));
            packed_sum += low * rhs_low + high_value * rhs_high;
        }
        sum += packed_sum as f32 * (scale * rhs_scale);
    }
    Ok(sum)
}

/// Computes the pinned scalar `q5_1` by `q8_0` row dot product.
///
/// # Errors
///
/// Returns [`QuantMoreError::MismatchedBlockCount`] when the rows have
/// different numbers of blocks.
// Preserve the pinned packed scale and accumulation order; fused arithmetic
// changes the observable floating-point result.
#[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
pub fn dot_q5_1_q8_0(
    lhs: Q5_1Row<'_>,
    rhs: crate::any::quant::Q8_0Row<'_>,
) -> Result<f32, QuantMoreError> {
    equal_blocks(lhs.block_count(), rhs.block_count())?;
    let mut sum = 0.0_f32;
    for index in 0..lhs.block_count() {
        let lhs_block = block(lhs.bytes, index, Q5_1_BLOCK_BYTES, Format::Q5_1)?;
        let rhs_bytes = rhs.as_bytes();
        let rhs_block = &rhs_bytes[index * crate::any::quant::Q8_0_BLOCK_BYTES
            ..(index + 1) * crate::any::quant::Q8_0_BLOCK_BYTES];
        let scale = fp16_to_f32(u16::from_le_bytes([lhs_block[0], lhs_block[1]]));
        let minimum = fp16_to_f32(u16::from_le_bytes([lhs_block[2], lhs_block[3]]));
        let rhs_scale = fp16_to_f32(u16::from_le_bytes([rhs_block[0], rhs_block[1]]));
        let high = u32::from_le_bytes([lhs_block[4], lhs_block[5], lhs_block[6], lhs_block[7]]);
        let mut packed_sum = 0_i32;
        let mut rhs_sum = 0_i32;
        for j in 0..BLOCK_VALUES / 2 {
            let packed = lhs_block[j + 8];
            let low = i32::from((packed & 0x0f) | ((((high >> j) & 1) as u8) << 4));
            let high_value = i32::from((packed >> 4) | ((((high >> (j + 16)) & 1) as u8) << 4));
            let rhs_low = i32::from(i8::from_ne_bytes([rhs_block[j + 2]]));
            let rhs_high = i32::from(i8::from_ne_bytes([rhs_block[j + 18]]));
            packed_sum += low * rhs_low + high_value * rhs_high;
            rhs_sum += rhs_low + rhs_high;
        }
        let scaled_sum = scale * packed_sum as f32;
        let minimum_sum = minimum * rhs_sum as f32;
        let affine = scaled_sum + minimum_sum;
        sum += rhs_scale * affine;
    }
    Ok(sum)
}

/// Computes the pinned scalar `q3_k` by `q8_k` row dot product.
///
/// This is the direct packed contract from `detail.hpp:2658-2760`. The low
/// bits, high-bit masks, and packed signed six-bit scales are decoded into
/// bounded stack accumulators; no dequantized tensor is materialized.
///
/// # Errors
///
/// Returns [`QuantMoreError::MismatchedBlockCount`] when the rows have
/// different numbers of blocks.
// Preserve the pinned packed scale and accumulation order; fused arithmetic
// changes the observable floating-point result.
#[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
pub fn dot_q3_k_q8_k(lhs: Q3KRow<'_>, rhs: Q8KRow<'_>) -> Result<f32, QuantMoreError> {
    equal_blocks(lhs.block_count(), rhs.block_count())?;
    let mut lane_sums = [0.0_f32; 8];
    for index in 0..lhs.block_count() {
        let lhs_block = block(lhs.bytes, index, Q3_K_BLOCK_BYTES, Format::Q3K)?;
        let rhs_block = block(rhs.bytes, index, Q8_K_BLOCK_BYTES, Format::Q8K)?;

        let scale_word0 =
            u32::from_le_bytes([lhs_block[96], lhs_block[97], lhs_block[98], lhs_block[99]]);
        let scale_word1 = u32::from_le_bytes([
            lhs_block[100],
            lhs_block[101],
            lhs_block[102],
            lhs_block[103],
        ]);
        let scale_word2 = u32::from_le_bytes([
            lhs_block[104],
            lhs_block[105],
            lhs_block[106],
            lhs_block[107],
        ]);
        let mask1 = 0x0303_0303_u32;
        let mask2 = 0x0f0f_0f0f_u32;
        let unpacked_words = [
            (scale_word0 & mask2) | ((scale_word2 & mask1) << 4),
            (scale_word1 & mask2) | (((scale_word2 >> 2) & mask1) << 4),
            ((scale_word0 >> 4) & mask2) | (((scale_word2 >> 4) & mask1) << 4),
            ((scale_word1 >> 4) & mask2) | (((scale_word2 >> 6) & mask1) << 4),
        ];

        let mut accumulators = [0_i32; 8];
        for group in 0..16 {
            let scale_bytes = unpacked_words[group / 4].to_le_bytes();
            let scale = i32::from(scale_bytes[group % 4]) - 32;
            let plane = (group % 8) / 2;
            let q3_block_offset = (group / 8) * 32;
            let lane_offset = (group % 2) * 16;
            let q8_offset = 4 + group * 16;
            let high_mask = 1_u8 << (group / 2);
            for lane in 0..16 {
                let q3 =
                    (lhs_block[32 + q3_block_offset + lane_offset + lane] >> (plane * 2)) & 0x03;
                let high =
                    i32::from(u8::from((lhs_block[lane_offset + lane] & high_mask) == 0)) * 4;
                let value = i32::from(q3) - high;
                let rhs_value = i32::from(i8::from_ne_bytes([rhs_block[q8_offset + lane]]));
                accumulators[lane & 7] += scale * (value * rhs_value);
            }
        }

        let rhs_scale =
            f32::from_le_bytes([rhs_block[0], rhs_block[1], rhs_block[2], rhs_block[3]]);
        let d = fp16_to_f32(u16::from_le_bytes([lhs_block[108], lhs_block[109]])) * rhs_scale;
        for lane in 0..8 {
            lane_sums[lane] += d * accumulators[lane] as f32;
        }
    }
    Ok(lane_sums.into_iter().sum())
}

/// Computes the pinned scalar `q4_k` by `q8_k` row dot product.
///
/// This is the direct packed contract from `detail.hpp:2846-2942`. The `Q4_K`
/// nibbles are expanded into a fixed-size stack buffer only to preserve the
/// source's scalar accumulation order; no dequantized tensor is materialized.
/// Both encoded rows remain borrowed for the complete operation.
///
/// # Errors
///
/// Returns [`QuantMoreError::MismatchedBlockCount`] when the rows have
/// different numbers of blocks.
// Preserve the pinned packed scale and accumulation order; fused arithmetic
// changes the observable floating-point result.
#[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
pub fn dot_q4_k_q8_k(lhs: Q4KRow<'_>, rhs: Q8KRow<'_>) -> Result<f32, QuantMoreError> {
    equal_blocks(lhs.block_count(), rhs.block_count())?;
    let mut lane_sums = [0.0_f32; 8];
    let mut sum = 0.0_f32;
    for index in 0..lhs.block_count() {
        let lhs_block = block(lhs.bytes, index, Q4_K_BLOCK_BYTES, Format::Q4K)?;
        let rhs_block = block(rhs.bytes, index, Q8_K_BLOCK_BYTES, Format::Q8K)?;

        let mut unpacked_q4 = [0_i8; QK_K_VALUES];
        for group in 0..(QK_K_VALUES / 64) {
            let q4_offset = group * 32;
            let output_offset = group * 64;
            for lane in 0..32 {
                let packed = lhs_block[16 + q4_offset + lane];
                unpacked_q4[output_offset + lane] = i8::from_ne_bytes([packed & 0x0f]);
                unpacked_q4[output_offset + 32 + lane] = i8::from_ne_bytes([packed >> 4]);
            }
        }

        let mut scales = [0_u8; 12];
        let mut minimums = [0_u8; 8];
        for lane in 0..4 {
            let scale_word0 = lhs_block[4 + lane];
            let scale_word1 = lhs_block[8 + lane];
            let scale_word2 = lhs_block[12 + lane];
            scales[lane] = scale_word0 & 0x3f;
            scales[4 + lane] = (scale_word2 & 0x0f) | (((scale_word0 >> 6) & 0x03) << 4);
            scales[8 + lane] = scale_word1 & 0x3f;
            minimums[lane] = scale_word1 & 0x3f;
            minimums[4 + lane] = ((scale_word2 >> 4) & 0x0f) | (((scale_word1 >> 6) & 0x03) << 4);
        }

        let rhs_scale =
            f32::from_le_bytes([rhs_block[0], rhs_block[1], rhs_block[2], rhs_block[3]]);
        let mut minimum_sum = 0_i32;
        for group in 0..(QK_K_VALUES / 16) {
            let rhs_sum =
                i16::from_le_bytes([rhs_block[260 + group * 2], rhs_block[261 + group * 2]]);
            minimum_sum += i32::from(rhs_sum) * i32::from(minimums[group / 2]);
        }

        let mut accumulators = [0_i32; 8];
        for (group, &scale_byte) in scales.iter().take(QK_K_VALUES / 32).enumerate() {
            let scale = i32::from(scale_byte);
            let value_offset = group * 32;
            for lane in 0..32 {
                let lhs_value = i32::from(unpacked_q4[value_offset + lane]);
                let rhs_value = i32::from(i8::from_ne_bytes([rhs_block[4 + value_offset + lane]]));
                accumulators[lane % 8] += scale * (lhs_value * rhs_value);
            }
        }

        let lhs_scale = fp16_to_f32(u16::from_le_bytes([lhs_block[0], lhs_block[1]]));
        let lhs_minimum = fp16_to_f32(u16::from_le_bytes([lhs_block[2], lhs_block[3]]));
        let d = lhs_scale * rhs_scale;
        for lane in 0..8 {
            lane_sums[lane] += d * accumulators[lane] as f32;
        }
        let dmin = lhs_minimum * rhs_scale;
        // Keep the pinned source order: the minimum term is subtracted from
        // the running scalar sum after the scaled lane accumulators.
        sum -= dmin * minimum_sum as f32;
    }

    for lane in lane_sums {
        sum += lane;
    }
    Ok(sum)
}

#[inline]
const fn q4_k_scale_min(index: usize, scales: &[u8; 12]) -> (u8, u8) {
    if index < 4 {
        (scales[index] & 0x3f, scales[index + 4] & 0x3f)
    } else {
        (
            (scales[index + 4] & 0x0f) | ((scales[index - 4] >> 6) << 4),
            (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4),
        )
    }
}

#[inline]
const fn q5_k_scale_min(index: usize, scales: &[u8; 12]) -> (u8, u8) {
    q4_k_scale_min(index, scales)
}

/// Computes the pinned scalar `q2_k` by `q8_k` row dot product.
///
/// This is the direct packed contract from `detail.hpp:2414-2450`. It keeps
/// both operands borrowed, accumulates packed integer products, and applies
/// the binary16 scale/minimum fields only after the block accumulator is
/// complete. It does not materialize a dequantized tensor.
///
/// # Errors
///
/// Returns [`QuantMoreError::MismatchedBlockCount`] when the rows have
/// different numbers of blocks.
// Preserve the pinned packed scale and accumulation order; fused arithmetic
// changes the observable floating-point result.
#[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
pub fn dot_q2_k_q8_k(lhs: Q2KRow<'_>, rhs: Q8KRow<'_>) -> Result<f32, QuantMoreError> {
    equal_blocks(lhs.block_count(), rhs.block_count())?;
    let mut sum = 0.0_f32;
    for index in 0..lhs.block_count() {
        let lhs_block = block(lhs.bytes, index, Q2_K_BLOCK_BYTES, Format::Q2K)?;
        let rhs_block = block(rhs.bytes, index, Q8_K_BLOCK_BYTES, Format::Q8K)?;
        let scale_bytes = &lhs_block[..16];
        let q2 = &lhs_block[16..80];
        let d = fp16_to_f32(u16::from_le_bytes([lhs_block[80], lhs_block[81]]));
        let minimum_scale = fp16_to_f32(u16::from_le_bytes([lhs_block[82], lhs_block[83]]));
        let rhs_scale =
            f32::from_le_bytes([rhs_block[0], rhs_block[1], rhs_block[2], rhs_block[3]]);

        let mut sum_mins = 0_i32;
        for j in 0..16 {
            let rhs_sum = i16::from_le_bytes([rhs_block[260 + 2 * j], rhs_block[261 + 2 * j]]);
            sum_mins += i32::from(rhs_sum) * i32::from(scale_bytes[j] >> 4);
        }

        let mut packed_sum = 0_i32;
        for half in 0..2 {
            let q2_offset = half * 32;
            for group in 0..4 {
                let shift = group * 2;
                let subscale0 = i32::from(scale_bytes[half * 8 + group * 2] & 0x0f);
                let subscale1 = i32::from(scale_bytes[half * 8 + group * 2 + 1] & 0x0f);
                let q8_offset = 4 + half * 128 + group * 32;
                let mut dot0 = 0_i32;
                let mut dot1 = 0_i32;
                for lane in 0..16 {
                    let q8_0 = i32::from(i8::from_ne_bytes([rhs_block[q8_offset + lane]]));
                    let q8_1 = i32::from(i8::from_ne_bytes([rhs_block[q8_offset + 16 + lane]]));
                    let q2_0 = i32::from((q2[q2_offset + lane] >> shift) & 0x03);
                    let q2_1 = i32::from((q2[q2_offset + 16 + lane] >> shift) & 0x03);
                    dot0 += q8_0 * q2_0;
                    dot1 += q8_1 * q2_1;
                }
                packed_sum += subscale0 * dot0 + subscale1 * dot1;
            }
        }

        // Keep the pinned operation order: detail.hpp:2426-2427 computes
        // `d_all` and `d_min` first, then detail.hpp:2456 subtracts their
        // separately rounded products. In particular, do not replace this
        // with `mul_add`; the fused path changes observable result bits.
        let d_all = rhs_scale * d;
        let d_min = rhs_scale * minimum_scale;
        let scaled_sum = d_all * packed_sum as f32;
        let minimum_sum = d_min * sum_mins as f32;
        sum += scaled_sum - minimum_sum;
    }
    Ok(sum)
}

/// Computes the pinned scalar `q5_k` by `q8_k` row dot product.
///
/// This is the direct packed contract from `detail.hpp:2944-3000`. It keeps
/// the packed operands borrowed and accumulates the integer products directly;
/// it does not materialize a dequantized tensor.
///
/// # Errors
///
/// Returns [`QuantMoreError::MismatchedBlockCount`] when the rows have
/// different numbers of blocks.
// Preserve the pinned packed scale and accumulation order; fused arithmetic
// changes the observable floating-point result.
#[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
pub fn dot_q5_k_q8_k(lhs: Q5KRow<'_>, rhs: Q8KRow<'_>) -> Result<f32, QuantMoreError> {
    equal_blocks(lhs.block_count(), rhs.block_count())?;
    let mut sum = 0.0_f32;
    for index in 0..lhs.block_count() {
        let lhs_block = block(lhs.bytes, index, Q5_K_BLOCK_BYTES, Format::Q5K)?;
        let rhs_block = block(rhs.bytes, index, Q8_K_BLOCK_BYTES, Format::Q8K)?;
        let d = fp16_to_f32(u16::from_le_bytes([lhs_block[0], lhs_block[1]]));
        let minimum = fp16_to_f32(u16::from_le_bytes([lhs_block[2], lhs_block[3]]));
        let rhs_scale =
            f32::from_le_bytes([rhs_block[0], rhs_block[1], rhs_block[2], rhs_block[3]]);
        let mut scales_data = [0_u8; 12];
        scales_data.copy_from_slice(&lhs_block[4..16]);
        let mut scale_index = 0;
        let mut low_mask = 1_u8;
        let mut high_mask = 2_u8;
        for offset in (0..QK_K_VALUES).step_by(64) {
            let (scale0, minimum0) = q5_k_scale_min(scale_index, &scales_data);
            let (scale1, minimum1) = q5_k_scale_min(scale_index + 1, &scales_data);
            let mut dot0 = 0_i32;
            let mut dot1 = 0_i32;
            let mut rhs_sum0 = 0_i32;
            let mut rhs_sum1 = 0_i32;
            let ql_offset = offset / 2;
            for lane in 0..32 {
                let rhs0 = i32::from(i8::from_ne_bytes([rhs_block[4 + offset + lane]]));
                let rhs1 = i32::from(i8::from_ne_bytes([rhs_block[4 + offset + 32 + lane]]));
                let high0 = if lhs_block[16 + lane] & low_mask != 0 {
                    16
                } else {
                    0
                };
                let high1 = if lhs_block[16 + lane] & high_mask != 0 {
                    16
                } else {
                    0
                };
                dot0 += (i32::from(lhs_block[48 + ql_offset + lane] & 0x0f) + high0) * rhs0;
                dot1 += (i32::from(lhs_block[48 + ql_offset + lane] >> 4) + high1) * rhs1;
                rhs_sum0 += rhs0;
                rhs_sum1 += rhs1;
            }
            sum += rhs_scale
                * (d * f32::from(scale0)).mul_add(
                    dot0 as f32,
                    -(minimum * f32::from(minimum0) * rhs_sum0 as f32),
                );
            sum += rhs_scale
                * (d * f32::from(scale1)).mul_add(
                    dot1 as f32,
                    -(minimum * f32::from(minimum1) * rhs_sum1 as f32),
                );
            scale_index += 2;
            low_mask <<= 2;
            high_mask <<= 2;
        }
    }
    Ok(sum)
}

/// Computes the pinned scalar `q6_k` by `q8_k` row dot product.
///
/// This is the direct packed contract from `detail.hpp:3002-3140`. The
/// six-bit values are expanded into a bounded stack buffer for each block,
/// preserving the source's scalar accumulation order; no whole-tensor
/// dequantization or heap allocation occurs.
///
/// # Errors
///
/// Returns [`QuantMoreError::MismatchedBlockCount`] when the rows have
/// different numbers of blocks.
// Preserve the pinned packed scale and accumulation order; fused arithmetic
// changes the observable floating-point result.
#[allow(clippy::cast_precision_loss, clippy::suboptimal_flops)]
pub fn dot_q6_k_q8_k(lhs: Q6KRow<'_>, rhs: Q8KRow<'_>) -> Result<f32, QuantMoreError> {
    equal_blocks(lhs.block_count(), rhs.block_count())?;
    let mut aux8 = [0_i8; QK_K_VALUES];
    let mut aux16 = [0_i16; 8];
    let mut sums = [0.0_f32; 8];
    let mut aux32 = [0_i32; 8];

    for index in 0..lhs.block_count() {
        let lhs_block = block(lhs.bytes, index, Q6_K_BLOCK_BYTES, Format::Q6K)?;
        let rhs_block = block(rhs.bytes, index, Q8_K_BLOCK_BYTES, Format::Q8K)?;
        let q8 = &rhs_block[4..4 + QK_K_VALUES];

        aux32.fill(0);
        for chunk in 0..2 {
            let ql = &lhs_block[chunk * 64..chunk * 64 + 64];
            let qh = &lhs_block[128 + chunk * 32..128 + chunk * 32 + 32];
            let output = &mut aux8[chunk * 128..chunk * 128 + 128];
            for lane in 0..32 {
                let high = qh[lane];
                output[lane] = i8::from_ne_bytes([((ql[lane] & 0x0f) | ((high & 0x03) << 4))]) - 32;
                output[lane + 32] =
                    i8::from_ne_bytes([((ql[lane + 32] & 0x0f) | (((high >> 2) & 0x03) << 4))])
                        - 32;
                output[lane + 64] =
                    i8::from_ne_bytes([(((ql[lane] >> 4) & 0x0f) | (((high >> 4) & 0x03) << 4))])
                        - 32;
                output[lane + 96] = i8::from_ne_bytes([
                    (((ql[lane + 32] >> 4) & 0x0f) | (((high >> 6) & 0x03) << 4))
                ]) - 32;
            }
        }

        for group in 0..16 {
            let scale = i32::from(i8::from_ne_bytes([lhs_block[192 + group]]));
            let q8_offset = group * 16;
            let values = group * 16;
            for lane in 0..8 {
                aux16[lane] = i16::from(i8::from_ne_bytes([q8[q8_offset + lane]]))
                    * i16::from(aux8[values + lane]);
                aux32[lane] += scale * i32::from(aux16[lane]);
            }
            for lane in 0..8 {
                aux16[lane] = i16::from(i8::from_ne_bytes([q8[q8_offset + 8 + lane]]))
                    * i16::from(aux8[values + 8 + lane]);
                aux32[lane] += scale * i32::from(aux16[lane]);
            }
        }

        let lhs_scale = fp16_to_f32(u16::from_le_bytes([lhs_block[208], lhs_block[209]]));
        let rhs_scale =
            f32::from_le_bytes([rhs_block[0], rhs_block[1], rhs_block[2], rhs_block[3]]);
        let scale = lhs_scale * rhs_scale;
        for lane in 0..8 {
            sums[lane] += scale * aux32[lane] as f32;
        }
    }

    Ok(sums.into_iter().sum())
}
