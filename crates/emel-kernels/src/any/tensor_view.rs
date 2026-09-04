//! Safe tensor-view contracts and strided F32 kernel operations.
//!
//! The pinned C++ kernel contract represents a view with a dtype, four
//! extents (`ne`) and four byte strides (`nb`).  This module keeps that wire
//! shape while binding the data to ordinary Rust slices.  A view is accepted
//! by the actor only after its complete byte span has been proven to fit the
//! slice; no raw pointer or unchecked indexing is required.

#![allow(clippy::derive_partial_eq_without_eq)]

use core::cell::Cell;
use core::fmt;

use sml::sml;

/// Element type carried by a kernel tensor view.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DType {
    /// IEEE 754 binary32.
    F32,
    /// IEEE 754 binary16.
    F16,
    /// Brain floating point 16-bit.
    Bf16,
    /// 4-bit quantized blocks with a 32-value block.
    Q4_0,
    /// 4-bit quantized blocks with a 32-value block and offset.
    Q4_1,
    /// 5-bit quantized blocks with a 32-value block.
    Q5_0,
    /// 5-bit quantized blocks with a 32-value block and offset.
    Q5_1,
    /// 8-bit quantized blocks with a 32-value block.
    Q8_0,
    /// 8-bit quantized blocks with a 32-value block and offset.
    Q8_1,
    /// 2-bit K-family quantized blocks.
    Q2K,
    /// 3-bit K-family quantized blocks.
    Q3K,
    /// 4-bit K-family quantized blocks.
    Q4K,
    /// 5-bit K-family quantized blocks.
    Q5K,
    /// 6-bit K-family quantized blocks.
    Q6K,
    /// 8-bit K-family quantized blocks.
    Q8K,
    /// Importance-matrix 2-bit quantized blocks, XXS variant.
    Iq2Xxs,
    /// Importance-matrix 2-bit quantized blocks, XS variant.
    Iq2Xs,
    /// Importance-matrix 3-bit quantized blocks, XXS variant.
    Iq3Xxs,
    /// Importance-matrix 1-bit quantized blocks, S variant.
    Iq1S,
    /// Importance-matrix 4-bit non-linear quantized blocks.
    Iq4Nl,
    /// Importance-matrix 3-bit quantized blocks, S variant.
    Iq3S,
    /// Importance-matrix 2-bit quantized blocks, S variant.
    Iq2S,
    /// Importance-matrix 4-bit quantized blocks, XS variant.
    Iq4Xs,
    /// Signed 8-bit integer storage.
    I8,
    /// Signed 16-bit integer storage.
    I16,
    I32,
    /// Signed 64-bit integer storage.
    I64,
    /// IEEE 754 binary64.
    F64,
    /// Importance-matrix 1-bit quantized blocks, M variant.
    Iq1M,
    /// 4-bit quantized blocks with 4-by-4 packing.
    Q4_0_4_4,
    /// 4-bit quantized blocks with 4-by-8 packing.
    Q4_0_4_8,
    /// 4-bit quantized blocks with 8-by-8 packing.
    Q4_0_8_8,
    /// Ternary quantized blocks, 1.0 variant.
    Tq1_0,
    /// Ternary quantized blocks, 2.0 variant.
    Tq2_0,
    /// 6-bit K-family blocks prepared for eight-wide multiplication.
    Q6KX8,
    /// 6-bit K-family blocks prepared for eight-wide Q8 multiplication.
    Q6KX8Q8Prepared,
    /// 6-bit K-family blocks prepared for eight-wide argmax multiplication.
    Q6KX8Q8ArgmaxPrepared,
    /// 8-bit blocks packed for four-wide block-length-four multiplication.
    Q8_0X4Bl4,
    /// 8-bit blocks packed for four-wide block-length-eight multiplication.
    Q8_0X4Bl8,
    /// 4-bit K-family blocks packed for eight-wide block-length-four multiplication.
    Q4KX8Bl4,
    /// 4-bit K-family blocks packed for eight-wide block-length-eight multiplication.
    Q4KX8Bl8,
    /// 8-bit K-family blocks packed for four-wide multiplication.
    Q8KX4,
    /// 8-bit K-family blocks packed for eight-wide multiplication.
    Q8KX8,
    /// Any other pinned wire dtype, retained without lossy conversion.
    Other(u8),
}

impl DType {
    /// Converts the pinned C++ wire value to a Rust view dtype.
    #[must_use]
    pub const fn from_code(code: u8) -> Self {
        match code {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            16 => Self::Iq2Xxs,
            17 => Self::Iq2Xs,
            18 => Self::Iq3Xxs,
            19 => Self::Iq1S,
            20 => Self::Iq4Nl,
            21 => Self::Iq3S,
            22 => Self::Iq2S,
            23 => Self::Iq4Xs,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::Iq1M,
            30 => Self::Bf16,
            31 => Self::Q4_0_4_4,
            32 => Self::Q4_0_4_8,
            33 => Self::Q4_0_8_8,
            34 => Self::Tq1_0,
            35 => Self::Tq2_0,
            36 => Self::Q6KX8,
            37 => Self::Q6KX8Q8Prepared,
            38 => Self::Q6KX8Q8ArgmaxPrepared,
            39 => Self::Q8_0X4Bl4,
            40 => Self::Q8_0X4Bl8,
            41 => Self::Q4KX8Bl4,
            42 => Self::Q4KX8Bl8,
            43 => Self::Q8KX4,
            44 => Self::Q8KX8,
            value => Self::Other(value),
        }
    }

    /// Returns the pinned C++ wire value.
    #[must_use]
    pub const fn code(self) -> u8 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2K => 10,
            Self::Q3K => 11,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
            Self::Q8K => 15,
            Self::Iq2Xxs => 16,
            Self::Iq2Xs => 17,
            Self::Iq3Xxs => 18,
            Self::Iq1S => 19,
            Self::Iq4Nl => 20,
            Self::Iq3S => 21,
            Self::Iq2S => 22,
            Self::Iq4Xs => 23,
            Self::I8 => 24,
            Self::I16 => 25,
            Self::I32 => 26,
            Self::I64 => 27,
            Self::F64 => 28,
            Self::Iq1M => 29,
            Self::Bf16 => 30,
            Self::Q4_0_4_4 => 31,
            Self::Q4_0_4_8 => 32,
            Self::Q4_0_8_8 => 33,
            Self::Tq1_0 => 34,
            Self::Tq2_0 => 35,
            Self::Q6KX8 => 36,
            Self::Q6KX8Q8Prepared => 37,
            Self::Q6KX8Q8ArgmaxPrepared => 38,
            Self::Q8_0X4Bl4 => 39,
            Self::Q8_0X4Bl8 => 40,
            Self::Q4KX8Bl4 => 41,
            Self::Q4KX8Bl8 => 42,
            Self::Q8KX4 => 43,
            Self::Q8KX8 => 44,
            Self::Other(value) => value,
        }
    }
}

/// Four-dimensional tensor metadata matching `tensor_view` in the pinned
/// kernel events contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Layout {
    dtype: DType,
    ne: [u64; 4],
    nb: [u64; 4],
}

impl Layout {
    /// Creates a layout from the pinned dtype, extents and byte strides.
    #[must_use]
    pub const fn new(dtype: DType, ne: [u64; 4], nb: [u64; 4]) -> Self {
        Self { dtype, ne, nb }
    }

    /// Creates a row-major F32 layout when its stride arithmetic fits.
    #[must_use]
    pub fn contiguous(dtype: DType, ne: [u64; 4]) -> Option<Self> {
        let mut nb = [0u64; 4];
        nb[0] = 4;
        let mut dimension = 1;
        while dimension < 4 {
            nb[dimension] = nb[dimension - 1].checked_mul(ne[dimension - 1])?;
            dimension += 1;
        }
        Some(Self::new(dtype, ne, nb))
    }

    /// Returns the element type.
    #[must_use]
    pub const fn dtype(self) -> DType {
        self.dtype
    }

    /// Returns the four logical extents.
    #[must_use]
    pub const fn ne(self) -> [u64; 4] {
        self.ne
    }

    /// Returns the four byte strides.
    #[must_use]
    pub const fn nb(self) -> [u64; 4] {
        self.nb
    }

    /// Returns effective F32 byte strides, resolving the pinned implicit
    /// contiguous form represented by `nb[0] == 0`.
    #[must_use]
    pub(crate) fn effective_f32_strides(self) -> Option<[u64; 4]> {
        if self.dtype != DType::F32 {
            return None;
        }
        if self.nb[0] != 0 {
            return Some(self.nb);
        }
        let mut strides = [0_u64; 4];
        strides[0] = 4;
        let mut dimension = 1;
        while dimension < 4 {
            strides[dimension] = strides[dimension - 1].checked_mul(self.ne[dimension - 1])?;
            dimension += 1;
        }
        Some(strides)
    }

    /// Returns whether this F32 layout is dense row-major storage.
    #[must_use]
    pub fn is_dense_contiguous(self) -> bool {
        self.is_dense_contiguous_for(DType::F32, 4)
    }

    /// Returns whether this layout is dense row-major I32 storage.
    #[must_use]
    pub fn is_dense_contiguous_i32(self) -> bool {
        self.is_dense_contiguous_for(DType::I32, 4)
    }

    /// Returns whether this layout is dense row-major F16 storage.
    #[must_use]
    pub fn is_dense_contiguous_f16(self) -> bool {
        self.is_dense_contiguous_for(DType::F16, 2)
    }

    fn is_dense_contiguous_for(self, dtype: DType, element_size: u64) -> bool {
        if self.dtype != dtype {
            return false;
        }
        let mut expected = element_size;
        let mut dimension = 0;
        while dimension < 4 {
            if self.nb[dimension] != expected {
                return false;
            }
            expected = match expected.checked_mul(self.ne[dimension]) {
                Some(value) => value,
                None => return false,
            };
            dimension += 1;
        }
        true
    }

    /// Returns a layout with dimensions and byte strides permuted together.
    #[must_use]
    pub const fn permuted(self, axes: [usize; 4]) -> Option<Self> {
        let mut ne = [0u64; 4];
        let mut nb = [0u64; 4];
        let mut dimension = 0;
        while dimension < 4 {
            let source = axes[dimension];
            if source >= 4 {
                return None;
            }
            ne[dimension] = self.ne[source];
            nb[dimension] = self.nb[source];
            dimension += 1;
        }
        Some(Self::new(self.dtype, ne, nb))
    }

    /// Returns the transposed two-dimensional layout.
    #[must_use]
    pub const fn transposed(self) -> Self {
        Self::new(
            self.dtype,
            [self.ne[1], self.ne[0], self.ne[2], self.ne[3]],
            [self.nb[1], self.nb[0], self.nb[2], self.nb[3]],
        )
    }

    /// Returns whether the logical shape has a representable, non-zero size.
    #[must_use]
    pub fn element_count(self) -> Option<usize> {
        let mut count = 1usize;
        let mut dimension = 0;
        while dimension < 4 {
            let extent = usize::try_from(self.ne[dimension]).ok()?;
            if extent == 0 {
                return None;
            }
            count = count.checked_mul(extent)?;
            dimension += 1;
        }
        Some(count)
    }

    /// Validates dtype, shape, stride and complete byte-span invariants.
    ///
    /// # Errors
    ///
    /// Returns the first failed dtype, shape, stride, or bounds invariant.
    pub fn validate(self, data_len: usize) -> Result<usize, ViewError> {
        self.validate_element_storage(DType::F32, 4, data_len, false)
    }

    /// Validates an F32 layout using the binary-kernel tensor contract.
    ///
    /// The pinned kernel permits an explicit zero stride for a singleton
    /// dimension. That representation is intentionally scoped to binary
    /// dispatch so the stricter general view contract remains unchanged.
    pub(crate) fn validate_binary(self, data_len: usize) -> Result<usize, ViewError> {
        self.validate_element_storage(DType::F32, 4, data_len, true)
    }

    /// Validates a signed 32-bit integer layout against its backing storage.
    ///
    /// This is used by kernels whose reference contract carries integer
    /// metadata alongside F32 data. It intentionally remains separate from
    /// [`Self::validate`] so ordinary F32 views cannot silently accept an
    /// integer dtype.
    ///
    /// # Errors
    ///
    /// Returns the first failed dtype, shape, stride, or bounds invariant.
    pub fn validate_i32(self, data_len: usize) -> Result<usize, ViewError> {
        self.validate_element_storage(DType::I32, 4, data_len, false)
    }

    fn validate_element_storage(
        self,
        expected_dtype: DType,
        element_size: u64,
        data_len: usize,
        allow_singleton_zero_stride: bool,
    ) -> Result<usize, ViewError> {
        if self.dtype != expected_dtype {
            return Err(ViewError::UnsupportedDType(self.dtype));
        }

        let count = self.element_count().ok_or(ViewError::InvalidShape)?;
        let strides = if expected_dtype == DType::F32 {
            self.effective_f32_strides()
                .ok_or(ViewError::InvalidShape)?
        } else {
            self.nb
        };
        let mut dimension = 0;
        while dimension < 4 {
            let stride = strides[dimension];
            let singleton_zero =
                allow_singleton_zero_stride && self.ne[dimension] == 1 && stride == 0;
            if !singleton_zero && (stride < element_size || !stride.is_multiple_of(element_size)) {
                return Err(ViewError::InvalidStride);
            }
            dimension += 1;
        }

        let mut max_offset = 0u128;
        dimension = 0;
        while dimension < 4 {
            let extent = u128::from(self.ne[dimension]);
            let stride = u128::from(strides[dimension]);
            max_offset = max_offset
                .checked_add(
                    (extent - 1)
                        .checked_mul(stride)
                        .ok_or(ViewError::InvalidShape)?,
                )
                .ok_or(ViewError::InvalidShape)?;
            dimension += 1;
        }

        let end = max_offset
            .checked_add(u128::from(element_size))
            .ok_or(ViewError::InvalidShape)?;
        let capacity = (data_len as u128)
            .checked_mul(4)
            .ok_or(ViewError::InvalidShape)?;
        if end > capacity {
            return Err(ViewError::OutOfBounds);
        }

        Ok(count)
    }
}

/// A borrowed packed tensor view with a validated row-major byte contract.
///
/// Packed rows are represented by `ne[0] / block_values` blocks, with each
/// block occupying `block_bytes` bytes. The remaining dimensions must use
/// exact row-major strides. This shared boundary lets packed kernels borrow
/// rows without exposing pointers or requiring byte reinterpretation.
#[derive(Clone, Copy, Debug)]
pub struct PackedRowView<'a> {
    data: &'a [u8],
    layout: Layout,
}

impl<'a> PackedRowView<'a> {
    /// Creates a borrowed packed view. Validation is performed by the owner.
    #[must_use]
    pub const fn new(data: &'a [u8], layout: Layout) -> Self {
        Self { data, layout }
    }

    /// Returns packed tensor metadata.
    #[must_use]
    pub const fn layout(self) -> Layout {
        self.layout
    }

    /// Borrows the packed bytes after an owning guard proves their complete
    /// row span. This remains crate-private so packed arithmetic cannot escape
    /// the actor boundary.
    pub(crate) const fn as_bytes(self) -> &'a [u8] {
        self.data
    }

    /// Validates dtype, packed row strides, and the complete byte span.
    pub(crate) fn validate(self, code: u8, block_values: usize, block_bytes: usize) -> bool {
        if self.layout.dtype().code() != code || block_values == 0 {
            return false;
        }
        let ne = self.layout.ne();
        let nb = self.layout.nb();
        if ne[0] == 0 || ne[1] == 0 || ne[2] == 0 || ne[3] == 0 {
            return false;
        }
        if !ne[0].is_multiple_of(block_values as u64) || nb[0] != 1 {
            return false;
        }
        let Some(row_bytes) = (ne[0] / block_values as u64).checked_mul(block_bytes as u64) else {
            return false;
        };
        let Some(row_stride_2) = row_bytes.checked_mul(ne[1]) else {
            return false;
        };
        let Some(row_stride_3) = row_stride_2.checked_mul(ne[2]) else {
            return false;
        };
        if row_bytes == 0 || nb[1] != row_bytes || nb[2] != row_stride_2 || nb[3] != row_stride_3 {
            return false;
        }
        let mut max_offset = 0u128;
        let mut dimension = 1;
        while dimension < 4 {
            let extent = u128::from(ne[dimension]);
            let stride = u128::from(nb[dimension]);
            let Some(contribution) = (extent - 1).checked_mul(stride) else {
                return false;
            };
            let Some(next) = max_offset.checked_add(contribution) else {
                return false;
            };
            max_offset = next;
            dimension += 1;
        }
        let Some(end) = max_offset.checked_add(u128::from(row_bytes)) else {
            return false;
        };
        end <= self.data.len() as u128
    }

    /// Returns one row after its owner's validation has proven its bounds.
    #[inline]
    pub(crate) fn row(self, row: usize, row_bytes: usize) -> &'a [u8] {
        let start = row
            .checked_mul(row_bytes)
            .expect("guard-proven packed row offset fits usize");
        &self.data[start..start + row_bytes]
    }
}

/// A safe immutable tensor view backed by an F32 slice.
#[derive(Clone, Copy, Debug)]
pub struct TensorView<'a> {
    data: &'a [f32],
    layout: Layout,
    count: usize,
    strides: [u64; 4],
}

/// A safe immutable F32 tensor view backed by raw bytes.
///
/// This view exists for reference operations whose row conversion copies
/// `ne[0] * sizeof(float)` bytes from a row base.  The row base may therefore
/// be at an arbitrary byte offset even though each copied F32 value occupies
/// four bytes. Values are decoded with [`f32::from_le_bytes`], so the view
/// needs no alignment assumption and never exposes a raw pointer.
#[derive(Debug)]
pub struct ByteTensorView<'a> {
    data: &'a [u8],
    layout: Layout,
    strides: [u64; 4],
}

impl<'a> ByteTensorView<'a> {
    /// Creates a byte-backed F32 view. Layout validity is classified by the
    /// owning actor guard at dispatch time.
    #[must_use]
    pub fn new(data: &'a [u8], layout: Layout) -> Self {
        let strides = layout.effective_f32_strides().unwrap_or(layout.nb);
        Self {
            data,
            layout,
            strides,
        }
    }

    /// Returns the layout without exposing the backing storage.
    #[must_use]
    pub const fn layout(&self) -> Layout {
        self.layout
    }

    /// Validates the F32 metadata and every byte accessed by row conversion.
    ///
    /// Unlike [`TensorView::validate`], this accepts arbitrary outer byte
    /// strides. The selected row operation ignores the source element stride
    /// after locating a row and copies the first four bytes of each column.
    ///
    /// # Errors
    ///
    /// Returns the first failed dtype, shape, stride, or byte-span invariant.
    pub fn validate(&self) -> Result<usize, ViewError> {
        if self.layout.dtype != DType::F32 {
            return Err(ViewError::UnsupportedDType(self.layout.dtype));
        }
        let ne = self.layout.ne;
        let mut count = 1usize;
        let mut dimension = 0;
        while dimension < 4 {
            let extent = usize::try_from(ne[dimension]).map_err(|_| ViewError::InvalidShape)?;
            if extent == 0 {
                return Err(ViewError::InvalidShape);
            }
            count = count.checked_mul(extent).ok_or(ViewError::InvalidShape)?;
            dimension += 1;
        }

        let nb = self.layout.nb;
        if nb[0] != 0 && (nb[0] < 4 || !nb[0].is_multiple_of(4)) {
            return Err(ViewError::InvalidStride);
        }
        if nb[0] != 0 {
            dimension = 0;
            while dimension < 4 {
                if ne[dimension] > 1 && nb[dimension] == 0 {
                    return Err(ViewError::InvalidStride);
                }
                dimension += 1;
            }
        }

        let strides = self
            .layout
            .effective_f32_strides()
            .ok_or(ViewError::InvalidShape)?;
        let row_bytes = u128::from(ne[0])
            .checked_mul(4)
            .ok_or(ViewError::InvalidShape)?;
        let mut max_offset = 0u128;
        dimension = 1;
        while dimension < 4 {
            let contribution = u128::from(ne[dimension] - 1)
                .checked_mul(u128::from(strides[dimension]))
                .ok_or(ViewError::InvalidShape)?;
            max_offset = max_offset
                .checked_add(contribution)
                .ok_or(ViewError::InvalidShape)?;
            dimension += 1;
        }
        let end = max_offset
            .checked_add(row_bytes)
            .ok_or(ViewError::InvalidShape)?;
        if end > self.data.len() as u128 {
            return Err(ViewError::OutOfBounds);
        }
        Ok(count)
    }

    /// Reads one F32 value from a guard-proven row byte range.
    #[inline]
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn read_row_contiguous(
        &self,
        row: usize,
        outer_1: usize,
        outer_2: usize,
        column: usize,
    ) -> f32 {
        let mut byte_offset = (row as u64)
            .checked_mul(self.strides[1])
            .expect("guard-proven row offset fits u64");
        byte_offset = byte_offset
            .checked_add(
                (outer_1 as u64)
                    .checked_mul(self.strides[2])
                    .expect("guard-proven outer row offset fits u64"),
            )
            .expect("guard-proven outer row sum fits u64");
        byte_offset = byte_offset
            .checked_add(
                (outer_2 as u64)
                    .checked_mul(self.strides[3])
                    .expect("guard-proven outer batch offset fits u64"),
            )
            .expect("guard-proven outer batch sum fits u64");
        byte_offset = byte_offset
            .checked_add(
                (column as u64)
                    .checked_mul(4)
                    .expect("guard-proven column offset fits u64"),
            )
            .expect("guard-proven byte offset fits u64");
        let start = usize::try_from(byte_offset).expect("guard-proven byte offset fits usize");
        let bytes = &self.data[start..start + 4];
        f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }
}

impl<'a> TensorView<'a> {
    /// Creates a view. Layout validity is classified by [`TensorKernel`]
    /// guards at dispatch time.
    #[must_use]
    pub fn new(data: &'a [f32], layout: Layout) -> Self {
        let count = layout.element_count().unwrap_or_default();
        let strides = layout.effective_f32_strides().unwrap_or(layout.nb);
        Self {
            data,
            layout,
            count,
            strides,
        }
    }

    /// Returns the layout without exposing the backing storage.
    #[must_use]
    pub const fn layout(&self) -> Layout {
        self.layout
    }

    /// Validates the layout against this view's backing slice.
    ///
    /// # Errors
    ///
    /// Returns the first failed dtype, shape, stride, or bounds invariant.
    pub fn validate(&self) -> Result<usize, ViewError> {
        self.layout.validate(self.data.len())
    }

    /// Validates this view with the binary-kernel singleton-zero-stride
    /// contract.
    pub(crate) fn validate_binary(&self) -> Result<usize, ViewError> {
        self.layout.validate_binary(self.data.len())
    }

    /// Returns the precomputed logical element count for a guard-proven view.
    pub(crate) const fn count(&self) -> usize {
        self.count
    }

    /// Returns backing storage length for same-RTC derived-view validation.
    pub(crate) const fn storage_len(&self) -> usize {
        self.data.len()
    }

    /// Returns whether this view uses the dense F32 layout required by the
    /// direct contiguous kernel actions.
    pub(crate) fn is_dense_f32(&self) -> bool {
        self.layout.is_dense_contiguous()
    }

    /// Borrows the backing F32 storage after a dense-layout guard succeeds.
    pub(crate) const fn as_slice(&self) -> &[f32] {
        self.data
    }

    /// Reads one element by logical ordinal after the owning machine's guard
    /// has proven the view's complete span.
    #[inline]
    pub(crate) const fn read(&self, ordinal: usize) -> f32 {
        self.read_f32(ordinal)
    }

    /// Reads a contiguous F32 row prefix from a strided tensor row.
    ///
    /// The row and outer-dimension strides locate the row start, while the
    /// selected operation's F32 conversion contract treats the first
    /// `cols` bytes as contiguous regardless of the source element stride.
    /// The owning guard proves the layout and complete byte span before this
    /// accessor is used by a hot-path action.
    pub(crate) const fn read_row_contiguous(
        &self,
        row: usize,
        outer_1: usize,
        outer_2: usize,
        column: usize,
    ) -> f32 {
        let strides = self.strides;
        let byte_offset = (row as u64) * strides[1]
            + (outer_1 as u64) * strides[2]
            + (outer_2 as u64) * strides[3]
            + (column as u64) * 4;
        self.data[(byte_offset / 4) as usize]
    }

    #[inline]
    const fn read_f32(&self, ordinal: usize) -> f32 {
        self.data[self.element_index(ordinal)]
    }

    #[inline]
    #[allow(clippy::cast_possible_truncation)]
    const fn element_index(&self, ordinal: usize) -> usize {
        let mut remaining = ordinal as u64;
        let mut byte_offset = 0u64;
        let mut dimension = 0;
        while dimension < 4 {
            let extent = self.layout.ne[dimension];
            byte_offset += (remaining % extent) * self.strides[dimension];
            remaining /= extent;
            dimension += 1;
        }
        (byte_offset / 4) as usize
    }
}

/// A safe mutable tensor view backed by an F32 slice.
#[derive(Debug)]
pub struct TensorViewMut<'a> {
    data: &'a mut [f32],
    layout: Layout,
    count: usize,
    strides: [u64; 4],
}

impl<'a> TensorViewMut<'a> {
    /// Creates a mutable view. Layout validity is classified by
    /// [`TensorKernel`] guards at dispatch time.
    #[must_use]
    pub fn new(data: &'a mut [f32], layout: Layout) -> Self {
        let count = layout.element_count().unwrap_or_default();
        let strides = layout.effective_f32_strides().unwrap_or(layout.nb);
        Self {
            data,
            layout,
            count,
            strides,
        }
    }

    /// Returns the layout without exposing the backing storage.
    #[must_use]
    pub const fn layout(&self) -> Layout {
        self.layout
    }

    /// Validates the layout against this view's backing slice.
    ///
    /// # Errors
    ///
    /// Returns the first failed dtype, shape, stride, or bounds invariant.
    pub fn validate(&self) -> Result<usize, ViewError> {
        self.layout.validate(self.data.len())
    }

    /// Validates this view with the binary-kernel singleton-zero-stride
    /// contract.
    pub(crate) fn validate_binary(&self) -> Result<usize, ViewError> {
        self.layout.validate_binary(self.data.len())
    }

    /// Returns the precomputed logical element count for a guard-proven view.
    pub(crate) const fn count(&self) -> usize {
        self.count
    }

    /// Returns backing storage length for same-RTC derived-view validation.
    pub(crate) const fn storage_len(&self) -> usize {
        self.data.len()
    }

    /// Returns whether this view uses the dense F32 layout required by the
    /// direct contiguous kernel actions.
    pub(crate) fn is_dense_f32(&self) -> bool {
        self.layout.is_dense_contiguous()
    }

    /// Mutably borrows the backing F32 storage after a dense-layout guard
    /// succeeds.
    pub(crate) const fn as_mut_slice(&mut self) -> &mut [f32] {
        self.data
    }

    /// Transfers the backing storage and layout to a same-RTC owner.
    pub(crate) const fn into_parts(self) -> (&'a mut [f32], Layout) {
        (self.data, self.layout)
    }

    /// Writes one element by logical ordinal after the owning machine's guard
    /// has proven the view's complete span.
    #[inline]
    pub(crate) const fn write(&mut self, ordinal: usize, value: f32) {
        self.write_f32(ordinal, value);
    }

    /// Reads one element from a guard-proven mutable view during a same-actor
    /// multi-pass operation.
    #[inline]
    pub(crate) const fn read(&self, ordinal: usize) -> f32 {
        self.data[self.element_index(ordinal)]
    }

    #[inline]
    const fn write_f32(&mut self, ordinal: usize, value: f32) {
        let index = self.element_index(ordinal);
        self.data[index] = value;
    }

    #[inline]
    #[allow(clippy::cast_possible_truncation)]
    const fn element_index(&self, ordinal: usize) -> usize {
        let mut remaining = ordinal as u64;
        let mut byte_offset = 0u64;
        let mut dimension = 0;
        while dimension < 4 {
            let extent = self.layout.ne[dimension];
            byte_offset += (remaining % extent) * self.strides[dimension];
            remaining /= extent;
            dimension += 1;
        }
        (byte_offset / 4) as usize
    }
}

/// Errors found while validating a view independently of actor dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ViewError {
    /// The maintained operation accepts F32 views only.
    UnsupportedDType(DType),
    /// An extent is zero, overflows, or has an unrepresentable element count.
    InvalidShape,
    /// A byte stride is smaller than one F32 or is not F32-aligned.
    InvalidStride,
    /// The complete strided byte span exceeds the backing slice.
    OutOfBounds,
}

impl fmt::Display for ViewError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedDType(dtype) => {
                write!(formatter, "unsupported tensor dtype {dtype:?}")
            }
            Self::InvalidShape => formatter.write_str("invalid tensor shape"),
            Self::InvalidStride => formatter.write_str("invalid tensor stride"),
            Self::OutOfBounds => formatter.write_str("tensor view is out of bounds"),
        }
    }
}

impl std::error::Error for ViewError {}

/// Errors returned by strided tensor operations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorError {
    /// One or more views failed dtype, shape, stride or bounds validation.
    InvalidView,
    /// Valid views do not have identical logical extents.
    ShapeMismatch,
    /// The generated machine rejected an unexpected event.
    UnexpectedEvent,
    /// The generated machine failed to dispatch an event.
    Internal,
}

impl fmt::Display for TensorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidView => formatter.write_str("invalid tensor view"),
            Self::ShapeMismatch => formatter.write_str("tensor view shapes differ"),
            Self::UnexpectedEvent => formatter.write_str("unexpected tensor event"),
            Self::Internal => formatter.write_str("internal tensor dispatch error"),
        }
    }
}

impl std::error::Error for TensorError {}

/// Copies one strided F32 view into another strided F32 view.
#[derive(Debug)]
pub struct CopyTensor<'a> {
    src: TensorView<'a>,
    dst: TensorViewMut<'a>,
}

impl<'a> CopyTensor<'a> {
    /// Creates a copy event; layout validation occurs in an actor guard.
    #[must_use]
    pub const fn new(src: TensorView<'a>, dst: TensorViewMut<'a>) -> Self {
        Self { src, dst }
    }
}

/// Adds two strided F32 views into a third strided F32 view.
#[derive(Debug)]
pub struct AddTensor<'a> {
    lhs: TensorView<'a>,
    rhs: TensorView<'a>,
    dst: TensorViewMut<'a>,
}

impl<'a> AddTensor<'a> {
    /// Creates an addition event; layout validation occurs in an actor guard.
    #[must_use]
    pub const fn new(lhs: TensorView<'a>, rhs: TensorView<'a>, dst: TensorViewMut<'a>) -> Self {
        Self { lhs, rhs, dst }
    }
}

#[derive(Debug)]
pub struct CopyRuntime<'a> {
    event: CopyTensor<'a>,
    result: &'a Cell<Result<(), TensorError>>,
}

#[derive(Debug)]
pub struct AddRuntime<'a> {
    event: AddTensor<'a>,
    result: &'a Cell<Result<(), TensorError>>,
}

#[derive(Default)]
struct Context;

sml! {
    TensorViewMachine<'dispatch> {
        "ready"_s <= *"ready"_s + Copy(CopyRuntime<'dispatch>) [guard_copy_valid] / effect_copy_execute,
        "ready"_s <= "ready"_s + Copy(CopyRuntime<'dispatch>) [guard_copy_shape_mismatch] / effect_copy_shape_reject,
        "ready"_s <= "ready"_s + Copy(CopyRuntime<'dispatch>) [guard_copy_invalid_view] / effect_copy_view_reject,

        "ready"_s <= "ready"_s + Add(AddRuntime<'dispatch>) [guard_add_valid] / effect_add_execute,
        "ready"_s <= "ready"_s + Add(AddRuntime<'dispatch>) [guard_add_shape_mismatch] / effect_add_shape_reject,
        "ready"_s <= "ready"_s + Add(AddRuntime<'dispatch>) [guard_add_invalid_view] / effect_add_view_reject,

        "ready"_s <= "ready"_s + unexpected_event<_> / effect_unexpected,
    }
}

/// Single-writer, RTC actor for validated strided F32 view operations.
pub struct TensorKernel {
    machine: TensorViewMachineStateMachine<Context>,
}

impl fmt::Debug for TensorKernel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TensorKernel")
            .finish_non_exhaustive()
    }
}

impl Default for TensorKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl TensorKernel {
    /// Constructs an independent tensor kernel actor.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            machine: TensorViewMachineStateMachine::new(Context),
        }
    }

    /// Dispatches one tensor event synchronously to completion.
    pub fn process_event<E: TensorEvent>(&mut self, event: E) -> E::Output {
        event.dispatch(self)
    }

    fn copy(&mut self, event: CopyTensor<'_>) -> Result<(), TensorError> {
        let result = Cell::new(Err(TensorError::UnexpectedEvent));
        self.machine
            .process_event(TensorViewMachineEvents::Copy(CopyRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| TensorError::Internal)?;
        result.get()
    }

    fn add(&mut self, event: AddTensor<'_>) -> Result<(), TensorError> {
        let result = Cell::new(Err(TensorError::UnexpectedEvent));
        self.machine
            .process_event(TensorViewMachineEvents::Add(AddRuntime {
                event,
                result: &result,
            }))
            .map_err(|_| TensorError::Internal)?;
        result.get()
    }
}

/// Typed event accepted by [`TensorKernel`].
pub trait TensorEvent: Sized {
    /// Result returned after RTC dispatch.
    type Output;

    #[doc(hidden)]
    fn dispatch(self, actor: &mut TensorKernel) -> Self::Output;
}

impl TensorEvent for CopyTensor<'_> {
    type Output = Result<(), TensorError>;

    fn dispatch(self, actor: &mut TensorKernel) -> Self::Output {
        actor.copy(self)
    }
}

impl TensorEvent for AddTensor<'_> {
    type Output = Result<(), TensorError>;

    fn dispatch(self, actor: &mut TensorKernel) -> Self::Output {
        actor.add(self)
    }
}

impl TensorViewMachineStateMachineContext for Context {
    fn guard_copy_valid(&self, event: &CopyRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.src.validate().is_ok()
            && event.event.dst.validate().is_ok()
            && event.event.src.layout.ne == event.event.dst.layout.ne)
    }

    fn guard_copy_shape_mismatch(&self, event: &CopyRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.src.validate().is_ok()
            && event.event.dst.validate().is_ok()
            && event.event.src.layout.ne != event.event.dst.layout.ne)
    }

    fn guard_copy_invalid_view(&self, event: &CopyRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.src.validate().is_err() || event.event.dst.validate().is_err())
    }

    fn effect_copy_execute(&mut self, mut event: CopyRuntime<'_>) -> Result<(), ()> {
        let count = event.event.src.count;
        let mut ordinal = 0;
        while ordinal < count {
            let value = event.event.src.read_f32(ordinal);
            event.event.dst.write_f32(ordinal, value);
            ordinal += 1;
        }
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_copy_shape_reject(&mut self, event: CopyRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(TensorError::ShapeMismatch));
        Ok(())
    }

    fn effect_copy_view_reject(&mut self, event: CopyRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(TensorError::InvalidView));
        Ok(())
    }

    fn guard_add_valid(&self, event: &AddRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.lhs.validate().is_ok()
            && event.event.rhs.validate().is_ok()
            && event.event.dst.validate().is_ok()
            && event.event.lhs.layout.ne == event.event.rhs.layout.ne
            && event.event.lhs.layout.ne == event.event.dst.layout.ne)
    }

    fn guard_add_shape_mismatch(&self, event: &AddRuntime<'_>) -> Result<bool, ()> {
        let valid = event.event.lhs.validate().is_ok()
            && event.event.rhs.validate().is_ok()
            && event.event.dst.validate().is_ok();
        Ok(valid
            && (event.event.lhs.layout.ne != event.event.rhs.layout.ne
                || event.event.lhs.layout.ne != event.event.dst.layout.ne))
    }

    fn guard_add_invalid_view(&self, event: &AddRuntime<'_>) -> Result<bool, ()> {
        Ok(event.event.lhs.validate().is_err()
            || event.event.rhs.validate().is_err()
            || event.event.dst.validate().is_err())
    }

    fn effect_add_execute(&mut self, mut event: AddRuntime<'_>) -> Result<(), ()> {
        let count = event.event.lhs.count;
        let mut ordinal = 0;
        while ordinal < count {
            let value = event.event.lhs.read_f32(ordinal) + event.event.rhs.read_f32(ordinal);
            event.event.dst.write_f32(ordinal, value);
            ordinal += 1;
        }
        event.result.set(Ok(()));
        Ok(())
    }

    fn effect_add_shape_reject(&mut self, event: AddRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(TensorError::ShapeMismatch));
        Ok(())
    }

    fn effect_add_view_reject(&mut self, event: AddRuntime<'_>) -> Result<(), ()> {
        event.result.set(Err(TensorError::InvalidView));
        Ok(())
    }

    fn effect_unexpected(&mut self) -> Result<(), ()> {
        Ok(())
    }
}
