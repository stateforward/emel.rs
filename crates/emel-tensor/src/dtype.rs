//! Serialized tensor representations and the distinct kernel execution domain.

use core::fmt;

/// A tensor type accepted from the serialized GGUF wire format.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum SerializedType {
    #[default]
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    Iq2Xxs,
    Iq2Xs,
    Iq3Xxs,
    Iq1S,
    Iq4Nl,
    Iq3S,
    Iq2S,
    Iq4Xs,
    I8,
    I16,
    I32,
    I64,
    F64,
    Iq1M,
    Bf16,
    Tq1_0,
    Tq2_0,
    Mxfp4,
    Q4Kx8Bl4,
    Q4Kx8Bl8,
}

/// Family of one serialized tensor layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SerializedLayoutKind {
    /// Ordinary GGML blocks.
    Block,
    /// EMEL's row-interleaved eight-row packed `Q4_K` representation.
    PackedQ4Kx8,
}

/// Exact geometry for one serialized tensor type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SerializedLayout {
    kind: SerializedLayoutKind,
    columns: u16,
    rows: u8,
    bytes: u16,
    minimum_dimensions: u8,
}

impl SerializedLayout {
    /// Returns the semantic layout family.
    #[must_use]
    pub const fn kind(self) -> SerializedLayoutKind {
        self.kind
    }

    /// Returns the number of columns represented by one serialized group.
    #[must_use]
    pub const fn columns(self) -> u16 {
        self.columns
    }

    /// Returns the number of rows represented by one serialized group.
    #[must_use]
    pub const fn rows(self) -> u8 {
        self.rows
    }

    /// Returns the bytes occupied by one serialized group.
    #[must_use]
    pub const fn bytes(self) -> u16 {
        self.bytes
    }
}

/// Kernel execution types. This domain is deliberately distinct from wire codes.
#[allow(non_camel_case_types)]
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[non_exhaustive]
pub enum KernelType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    Iq2Xxs,
    Iq2Xs,
    Iq3Xxs,
    Iq1S,
    Iq4Nl,
    Iq3S,
    Iq2S,
    Iq4Xs,
    I8,
    I16,
    I32,
    I64,
    F64,
    Iq1M,
    Bf16,
    Tq1_0,
    Tq2_0,
    Q4_0_4_4,
    Q4_0_4_8,
    Q4_0_8_8,
    Q6Kx8,
    Q6Kx8Q8Prepared,
    Q6Kx8Q8ArgmaxPrepared,
    Q8_0X4Bl4,
    Q8_0X4Bl8,
    Q4Kx8Bl4,
    Q4Kx8Bl8,
    Q8Kx4,
    Q8Kx8,
}

/// Failure converting or sizing a serialized tensor representation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ConversionError {
    /// The `u32` value is not an accepted serialized wire type.
    UnknownSerializedType(u32),
    /// The active dimension count or shape is invalid for the representation.
    InvalidShape,
    /// Shape or byte-size arithmetic overflowed `u64`.
    Capacity,
    /// The serialized representation has no same-semantics kernel execution type.
    NoKernelEquivalent(SerializedType),
}

impl fmt::Display for ConversionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownSerializedType(code) => {
                write!(formatter, "unknown serialized tensor type {code}")
            }
            Self::InvalidShape => formatter.write_str("invalid serialized tensor shape"),
            Self::Capacity => formatter.write_str("serialized tensor size overflow"),
            Self::NoKernelEquivalent(tensor_type) => {
                write!(formatter, "no kernel equivalent for {tensor_type:?}")
            }
        }
    }
}

impl std::error::Error for ConversionError {}

impl TryFrom<u32> for SerializedType {
    type Error = ConversionError;

    fn try_from(code: u32) -> Result<Self, Self::Error> {
        match code {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            10 => Ok(Self::Q2K),
            11 => Ok(Self::Q3K),
            12 => Ok(Self::Q4K),
            13 => Ok(Self::Q5K),
            14 => Ok(Self::Q6K),
            15 => Ok(Self::Q8K),
            16 => Ok(Self::Iq2Xxs),
            17 => Ok(Self::Iq2Xs),
            18 => Ok(Self::Iq3Xxs),
            19 => Ok(Self::Iq1S),
            20 => Ok(Self::Iq4Nl),
            21 => Ok(Self::Iq3S),
            22 => Ok(Self::Iq2S),
            23 => Ok(Self::Iq4Xs),
            24 => Ok(Self::I8),
            25 => Ok(Self::I16),
            26 => Ok(Self::I32),
            27 => Ok(Self::I64),
            28 => Ok(Self::F64),
            29 => Ok(Self::Iq1M),
            30 => Ok(Self::Bf16),
            34 => Ok(Self::Tq1_0),
            35 => Ok(Self::Tq2_0),
            39 => Ok(Self::Mxfp4),
            41 => Ok(Self::Q4Kx8Bl4),
            42 => Ok(Self::Q4Kx8Bl8),
            _ => Err(ConversionError::UnknownSerializedType(code)),
        }
    }
}

impl SerializedType {
    /// Returns the exact `u32` GGUF wire code without truncation.
    #[must_use]
    pub const fn wire_code(self) -> u32 {
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
            Self::Tq1_0 => 34,
            Self::Tq2_0 => 35,
            Self::Mxfp4 => 39,
            Self::Q4Kx8Bl4 => 41,
            Self::Q4Kx8Bl8 => 42,
        }
    }

    /// Returns the exact serialized block or packed-row layout.
    #[must_use]
    pub const fn layout(self) -> SerializedLayout {
        match self {
            Self::F32 | Self::I32 => block(1, 4),
            Self::F16 | Self::I16 | Self::Bf16 => block(1, 2),
            Self::Q4_0 | Self::Iq4Nl => block(32, 18),
            Self::Q4_1 => block(32, 20),
            Self::Q5_0 => block(32, 22),
            Self::Q5_1 => block(32, 24),
            Self::Q8_0 => block(32, 34),
            Self::Q8_1 => block(32, 36),
            Self::Q2K => block(256, 84),
            Self::Q3K | Self::Iq3S => block(256, 110),
            Self::Q4K => block(256, 144),
            Self::Q5K => block(256, 176),
            Self::Q6K => block(256, 210),
            Self::Q8K => block(256, 292),
            Self::Iq2Xxs | Self::Tq2_0 => block(256, 66),
            Self::Iq2Xs => block(256, 74),
            Self::Iq3Xxs => block(256, 98),
            Self::Iq1S => block(256, 50),
            Self::Iq2S => block(256, 82),
            Self::Iq4Xs => block(256, 136),
            Self::I8 => block(1, 1),
            Self::I64 | Self::F64 => block(1, 8),
            Self::Iq1M => block(256, 56),
            Self::Tq1_0 => block(256, 54),
            Self::Mxfp4 => block(32, 17),
            Self::Q4Kx8Bl4 | Self::Q4Kx8Bl8 => packed_q4_k_x8(),
        }
    }

    /// Computes the exact unpadded serialized size for up to four dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`ConversionError::InvalidShape`] for invalid dimensionality or
    /// block alignment and [`ConversionError::Capacity`] on arithmetic overflow.
    pub fn data_size(
        self,
        dimensions: [u64; 4],
        dimension_count: u32,
    ) -> Result<u64, ConversionError> {
        serialized_data_size(dimensions, dimension_count, self.layout())
    }
}

const fn block(elements: u16, bytes: u16) -> SerializedLayout {
    SerializedLayout {
        kind: SerializedLayoutKind::Block,
        columns: elements,
        rows: 1,
        bytes,
        minimum_dimensions: 0,
    }
}

const fn packed_q4_k_x8() -> SerializedLayout {
    SerializedLayout {
        kind: SerializedLayoutKind::PackedQ4Kx8,
        columns: 256,
        rows: 8,
        bytes: 1152,
        minimum_dimensions: 2,
    }
}

fn active_dimensions(
    dimensions: &[u64; 4],
    dimension_count: u32,
) -> Result<&[u64], ConversionError> {
    let count = usize::try_from(dimension_count).map_err(|_| ConversionError::InvalidShape)?;
    dimensions.get(..count).ok_or(ConversionError::InvalidShape)
}

fn serialized_data_size(
    dimensions: [u64; 4],
    dimension_count: u32,
    layout: SerializedLayout,
) -> Result<u64, ConversionError> {
    let active = active_dimensions(&dimensions, dimension_count)?;
    if dimension_count < u32::from(layout.minimum_dimensions) {
        return Err(ConversionError::InvalidShape);
    }
    let columns = active.first().copied().unwrap_or(1);
    let columns_per_group = u64::from(layout.columns);
    if !columns.is_multiple_of(columns_per_group) {
        return Err(ConversionError::InvalidShape);
    }
    let rows = active
        .get(1..)
        .unwrap_or_default()
        .iter()
        .try_fold(1_u64, |rows, dimension| {
            rows.checked_mul(*dimension)
                .ok_or(ConversionError::Capacity)
        })?;
    let rows_per_group = u64::from(layout.rows);
    let row_groups = rows / rows_per_group + u64::from(!rows.is_multiple_of(rows_per_group));
    (columns / columns_per_group)
        .checked_mul(row_groups)
        .and_then(|groups| groups.checked_mul(u64::from(layout.bytes)))
        .ok_or(ConversionError::Capacity)
}

impl TryFrom<SerializedType> for KernelType {
    type Error = ConversionError;

    fn try_from(serialized: SerializedType) -> Result<Self, Self::Error> {
        match serialized {
            SerializedType::F32 => Ok(Self::F32),
            SerializedType::F16 => Ok(Self::F16),
            SerializedType::Q4_0 => Ok(Self::Q4_0),
            SerializedType::Q4_1 => Ok(Self::Q4_1),
            SerializedType::Q5_0 => Ok(Self::Q5_0),
            SerializedType::Q5_1 => Ok(Self::Q5_1),
            SerializedType::Q8_0 => Ok(Self::Q8_0),
            SerializedType::Q8_1 => Ok(Self::Q8_1),
            SerializedType::Q2K => Ok(Self::Q2K),
            SerializedType::Q3K => Ok(Self::Q3K),
            SerializedType::Q4K => Ok(Self::Q4K),
            SerializedType::Q5K => Ok(Self::Q5K),
            SerializedType::Q6K => Ok(Self::Q6K),
            SerializedType::Q8K => Ok(Self::Q8K),
            SerializedType::Iq2Xxs => Ok(Self::Iq2Xxs),
            SerializedType::Iq2Xs => Ok(Self::Iq2Xs),
            SerializedType::Iq3Xxs => Ok(Self::Iq3Xxs),
            SerializedType::Iq1S => Ok(Self::Iq1S),
            SerializedType::Iq4Nl => Ok(Self::Iq4Nl),
            SerializedType::Iq3S => Ok(Self::Iq3S),
            SerializedType::Iq2S => Ok(Self::Iq2S),
            SerializedType::Iq4Xs => Ok(Self::Iq4Xs),
            SerializedType::I8 => Ok(Self::I8),
            SerializedType::I16 => Ok(Self::I16),
            SerializedType::I32 => Ok(Self::I32),
            SerializedType::I64 => Ok(Self::I64),
            SerializedType::F64 => Ok(Self::F64),
            SerializedType::Iq1M => Ok(Self::Iq1M),
            SerializedType::Bf16 => Ok(Self::Bf16),
            SerializedType::Tq1_0 => Ok(Self::Tq1_0),
            SerializedType::Tq2_0 => Ok(Self::Tq2_0),
            SerializedType::Mxfp4 => Err(ConversionError::NoKernelEquivalent(serialized)),
            SerializedType::Q4Kx8Bl4 => Ok(Self::Q4Kx8Bl4),
            SerializedType::Q4Kx8Bl8 => Ok(Self::Q4Kx8Bl8),
        }
    }
}
